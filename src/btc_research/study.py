"""The research protocol used by the notebooks, the pipeline script, and the report.

Protocol (chronological 60/20/20 split of 2024 five-minute bars):
    1. Train: grid-search signal parameters (window, entry z, exit band) for the
       mean-reversion signal; keep the highest net Sharpe at the base cost.
    2. Train: rank UTC hours by the chosen signal's mean gross return; the top
       ``N_SESSION_HOURS`` form a candidate session filter.
    3. Validation: choose the overlay combination (session filter on/off,
       volatility target, daily stop) with the highest net Sharpe.
    4. Test: evaluate the chosen configuration once, in both directions.
Signals are always computed on the full series and then sliced, so each split
starts with a warm signal.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .metrics import buy_and_hold_metrics, returns_by_hour_of_day, summarize_backtest
from .param_search import grid_search
from .preprocess import PROCESSED_PATH, chronological_split, load_processed_ohlcv
from .signals import Direction, rolling_zscore_signal

# Cost assumption: 10 bp of notional per unit of turnover. This is about the Binance
# spot taker fee for a regular (non-VIP) account without the BNB discount (0.10%
# per side at the time of writing). The BTCUSDT half-spread (~0.01 bp) and market
# impact for small size are ignored.
BASE_COST: float = 0.0010
BINANCE_TAKER_FEE_BP: float = 10.0
COST_GRID_BP: tuple[float, ...] = tuple(float(c) for c in range(0, 16))

SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")
SPLIT_FRACTIONS: tuple[float, float, float] = (0.6, 0.2, 0.2)

WINDOWS: tuple[int, ...] = (10, 20, 40)
Z_ENTRIES: tuple[float, ...] = (1.0, 1.5, 2.0)
EXIT_BANDWIDTHS: tuple[float | None, ...] = (None, 0.5, 0.0)

# Overlay grid. Positions are capped at 1.0 (no leverage), so volatility targeting
# can only scale exposure down. Targets sit near the quartiles of the 96-bar rolling
# volatility in the training block (0.097%, 0.132%, 0.179% per bar).
MAX_POSITION: float = 1.0
VOL_LOOKBACK: int = 96
VOL_TARGETS: tuple[float | None, ...] = (None, 0.0010, 0.0013, 0.0016)
DAILY_STOPS: tuple[float | None, ...] = (None, 0.0075, 0.01)
N_SESSION_HOURS: int = 8


@dataclass(frozen=True)
class StrategyConfig:
    """A full strategy specification: signal parameters plus overlays."""

    window: int = 20
    z_entry: float = 1.0
    exit_bandwidth: float | None = None
    active_hours: frozenset[int] | None = None
    target_volatility: float | None = None
    daily_stop_loss: float | None = None
    direction: Direction = "mean_reversion"

    def signal(self, df: pd.DataFrame) -> pd.DataFrame:
        return rolling_zscore_signal(
            df,
            window=self.window,
            z_entry=self.z_entry,
            exit_bandwidth=self.exit_bandwidth,
            active_hours=self.active_hours,
            direction=self.direction,
        )

    def risk_params(self) -> dict[str, Any]:
        return {
            "max_position": MAX_POSITION,
            "target_volatility": self.target_volatility,
            "volatility_lookback": VOL_LOOKBACK,
            "daily_stop_loss": self.daily_stop_loss,
        }

    def signal_only(self) -> StrategyConfig:
        """Same signal parameters, no session filter and no overlays."""
        return StrategyConfig(self.window, self.z_entry, self.exit_bandwidth, direction=self.direction)

    def mirrored(self) -> StrategyConfig:
        flip: Direction = "momentum" if self.direction == "mean_reversion" else "mean_reversion"
        return replace(self, direction=flip)

    def describe(self) -> str:
        parts = [f"W={self.window}", f"z={self.z_entry:g}"]
        if self.exit_bandwidth is not None:
            parts.append(f"exit={self.exit_bandwidth:g}")
        if self.active_hours is not None:
            parts.append(f"hours={sorted(self.active_hours)}")
        if self.target_volatility is not None:
            parts.append(f"vol target={self.target_volatility:.2%}")
        if self.daily_stop_loss is not None:
            parts.append(f"stop={self.daily_stop_loss:.2%}")
        return ", ".join(parts)


@dataclass
class Splits:
    """Full series plus the three chronological blocks."""

    full: pd.DataFrame
    blocks: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.blocks[name]

    def index(self, name: str) -> pd.Index:
        return self.blocks[name].index

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                name: {"start": b.index[0], "end": b.index[-1], "bars": len(b)}
                for name, b in self.blocks.items()
            }
        ).T


def load_splits(path: str | Path = PROCESSED_PATH) -> Splits:
    """Load the processed 5-minute data and split it 60/20/20 by time."""
    df = load_processed_ohlcv(path)
    blocks = dict(zip(SPLIT_NAMES, chronological_split(df, SPLIT_FRACTIONS), strict=True))
    return Splits(full=df, blocks=blocks)


def backtest(
    splits: Splits, config: StrategyConfig, split: str, cost: float = BASE_COST
) -> BacktestResult:
    """Compute the signal on the full series and backtest it on one split."""
    sig = config.signal(splits.full).loc[splits.index(split)]
    return run_backtest(sig, cost_per_unit_turnover=cost, **config.risk_params())


# ── Step 1: signal parameters on train ───────────────────────────────────────


def signal_grid(splits: Splits, cost: float = BASE_COST) -> pd.DataFrame:
    """Signal-parameter grid on train and validation (test is never touched here).

    Sorted by train net Sharpe. Validation columns are shown for information;
    the choice in ``select_signal`` uses train only.
    """
    return grid_search(
        splits.full,
        {"train": splits.index("train"), "validation": splits.index("validation")},
        windows=WINDOWS,
        z_entries=Z_ENTRIES,
        exit_bandwidths=EXIT_BANDWIDTHS,
        cost_per_unit_turnover=cost,
    )


def select_signal(grid: pd.DataFrame) -> StrategyConfig:
    """Top row of the train-sorted grid."""
    best = grid.sort_values("train_sharpe_net", ascending=False).iloc[0]
    eb = best["exit_bandwidth"]
    return StrategyConfig(
        window=int(best["window"]),
        z_entry=float(best["z_entry"]),
        exit_bandwidth=None if pd.isna(eb) else float(eb),
    )


# ── Step 2: session hours on train ───────────────────────────────────────────


def train_hourly_gross(splits: Splits, config: StrategyConfig) -> pd.Series:
    """Mean gross return by UTC hour on the training block."""
    bt = backtest(splits, config.signal_only(), "train", cost=0.0)
    return returns_by_hour_of_day(bt.df["strategy_return_gross"])


def select_session_hours(hourly: pd.Series, n: int = N_SESSION_HOURS) -> frozenset[int]:
    return frozenset(int(h) for h in hourly.nlargest(n).index)


# ── Step 3: overlays on validation ───────────────────────────────────────────


def overlay_grid(
    splits: Splits,
    base: StrategyConfig,
    session_hours: frozenset[int],
    cost: float = BASE_COST,
) -> pd.DataFrame:
    """Evaluate session filter x vol target x daily stop on train and validation."""
    rows = []
    for hours, vol, stop in itertools.product(
        (None, session_hours), VOL_TARGETS, DAILY_STOPS
    ):
        cfg = replace(
            base.signal_only(), active_hours=hours, target_volatility=vol, daily_stop_loss=stop
        )
        row: dict[str, Any] = {
            "session_filter": hours is not None,
            "target_volatility": vol,
            "daily_stop_loss": stop,
        }
        for split in ("train", "validation"):
            stats = summarize_backtest(backtest(splits, cfg, split, cost))
            row.update({f"{split}_{k}": v for k, v in stats.items()})
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("validation_sharpe_net", ascending=False)
        .reset_index(drop=True)
    )


def select_overlay(
    grid: pd.DataFrame, base: StrategyConfig, session_hours: frozenset[int]
) -> StrategyConfig:
    best = grid.iloc[0]
    vol, stop = best["target_volatility"], best["daily_stop_loss"]
    return replace(
        base.signal_only(),
        active_hours=session_hours if bool(best["session_filter"]) else None,
        target_volatility=None if pd.isna(vol) else float(vol),
        daily_stop_loss=None if pd.isna(stop) else float(stop),
    )


# ── Step 4: test, once ───────────────────────────────────────────────────────


@dataclass
class StudyResult:
    splits: Splits
    signal_grid: pd.DataFrame
    signal_config: StrategyConfig
    hourly_train: pd.Series
    session_hours: frozenset[int]
    overlay_grid: pd.DataFrame
    final_config: StrategyConfig
    results: pd.DataFrame
    cost_curves: pd.DataFrame
    test_runs: dict[str, BacktestResult]
    buy_and_hold: dict[str, dict[str, float]]


def reported_variants(signal_config: StrategyConfig, final_config: StrategyConfig) -> dict[str, StrategyConfig]:
    """The four configurations reported on test: {signal only, selected} x {MR, momentum}."""
    variants = {
        "Mean reversion, signal only": signal_config.signal_only(),
        "Momentum (mirror), signal only": signal_config.signal_only().mirrored(),
    }
    if final_config != signal_config.signal_only():
        variants["Mean reversion, selected overlays"] = final_config
        variants["Momentum (mirror), selected overlays"] = final_config.mirrored()
    return variants


def results_table(
    splits: Splits, variants: dict[str, StrategyConfig], split: str = "test", cost: float = BASE_COST
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    rows, runs = [], {}
    for label, cfg in variants.items():
        bt = backtest(splits, cfg, split, cost)
        runs[label] = bt
        rows.append({"variant": label, "config": cfg.describe(), **summarize_backtest(bt, with_ci=True)})
    return pd.DataFrame(rows).set_index("variant"), runs


def cost_curves(
    splits: Splits, variants: dict[str, StrategyConfig], split: str = "test"
) -> pd.DataFrame:
    """Net log return and net Sharpe for each variant across ``COST_GRID_BP``."""
    rows = []
    for label, cfg in variants.items():
        for bp in COST_GRID_BP:
            stats = summarize_backtest(backtest(splits, cfg, split, bp / 1e4))
            rows.append(
                {
                    "variant": label,
                    "cost_bp": bp,
                    "net_log_return": stats["net_log_return"],
                    "sharpe_net": stats["sharpe_net"],
                }
            )
    return pd.DataFrame(rows)


def run_study(path: str | Path = PROCESSED_PATH, cost: float = BASE_COST) -> StudyResult:
    """Run steps 1-4 end to end."""
    splits = load_splits(path)
    sgrid = signal_grid(splits, cost)
    signal_cfg = select_signal(sgrid)
    hourly = train_hourly_gross(splits, signal_cfg)
    hours = select_session_hours(hourly)
    ogrid = overlay_grid(splits, signal_cfg, hours, cost)
    final_cfg = select_overlay(ogrid, signal_cfg, hours)

    variants = reported_variants(signal_cfg, final_cfg)
    table, runs = results_table(splits, variants, "test", cost)
    curves = cost_curves(splits, variants, "test")
    bh = {name: buy_and_hold_metrics(splits[name]) for name in SPLIT_NAMES}
    return StudyResult(
        splits=splits,
        signal_grid=sgrid,
        signal_config=signal_cfg,
        hourly_train=hourly,
        session_hours=hours,
        overlay_grid=ogrid,
        final_config=final_cfg,
        results=table,
        cost_curves=curves,
        test_runs=runs,
        buy_and_hold=bh,
    )


__all__ = [
    "BASE_COST",
    "BINANCE_TAKER_FEE_BP",
    "COST_GRID_BP",
    "DAILY_STOPS",
    "EXIT_BANDWIDTHS",
    "MAX_POSITION",
    "N_SESSION_HOURS",
    "SPLIT_FRACTIONS",
    "SPLIT_NAMES",
    "StrategyConfig",
    "Splits",
    "StudyResult",
    "VOL_LOOKBACK",
    "VOL_TARGETS",
    "WINDOWS",
    "Z_ENTRIES",
    "backtest",
    "cost_curves",
    "load_splits",
    "overlay_grid",
    "results_table",
    "run_study",
    "select_overlay",
    "select_session_hours",
    "select_signal",
    "signal_grid",
    "reported_variants",
    "train_hourly_gross",
]
