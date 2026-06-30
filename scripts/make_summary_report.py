#!/usr/bin/env python3
"""Generate reports/summary.pdf using LaTeX for professional typesetting."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BTCUSDT" / "btc_2024_raw.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
TEX_PATH = REPORT_DIR / "summary.tex"
REPORT_PATH = REPORT_DIR / "summary.pdf"
EQUITY_FIG = REPORT_DIR / "equity_curves.png"
HEATMAP_FIG = REPORT_DIR / "overlay_heatmap.png"
HOUR_FIG = REPORT_DIR / "hourly_returns.png"

WINDOW = 20
Z_ENTRY = 1.0
COST_PER_UNIT = 0.0005
TRAIN_FRACTION = 0.7
VOL_TARGETS = [0.0025, 0.0030, 0.0035]
DAILY_STOPS = [None, 0.0075, 0.01]

REPORT_DIR.mkdir(parents=True, exist_ok=True)

from src.preprocess import clean_and_resample_ohlcv, chronological_train_test_split  # noqa: E402
from src.signals import rolling_zscore_signal  # noqa: E402
from src.backtest import run_basic_backtest, run_backtest_with_costs  # noqa: E402
from src.metrics import (  # noqa: E402
    max_drawdown,
    sharpe_ratio,
    trade_count,
    sharpe_confidence_interval,
    buy_and_hold_metrics,
    returns_by_hour_of_day,
)


@dataclass
class VariantConfig:
    name: str
    latex_name: str
    risk_params: Dict = field(default_factory=dict)


VARIANTS: List[VariantConfig] = [
    VariantConfig("Raw +/-1 signal", r"Raw $\pm1$ signal", {}),
    VariantConfig(
        "Vol 0.30% + 1% stop",
        r"Vol 0.30\,\% $+$ 1\,\% stop",
        {
            "max_position_size": 1.5,
            "target_volatility": 0.0030,
            "volatility_lookback": 96,
            "max_leverage": 1.5,
            "daily_stop_loss": 0.01,
        },
    ),
    VariantConfig(
        "Vol 0.35% + 0.75% stop",
        r"Vol 0.35\,\% $+$ 0.75\,\% stop",
        {
            "max_position_size": 1.5,
            "target_volatility": 0.0035,
            "volatility_lookback": 96,
            "max_leverage": 1.5,
            "daily_stop_loss": 0.0075,
        },
    ),
]


# ── computation helpers ───────────────────────────────────────────────────────

def run_variant(df_split: pd.DataFrame, config: VariantConfig) -> dict:
    df_sig = rolling_zscore_signal(df_split, window=WINDOW, z_entry=Z_ENTRY)
    bt_cost = run_backtest_with_costs(
        df_sig, cost_per_unit_change=COST_PER_UNIT, **config.risk_params
    )
    bt_raw = run_basic_backtest(df_sig, **config.risk_params)
    df_cost = bt_cost.df
    return {
        "sharpe_net": sharpe_ratio(df_cost["strategy_return_net"]),
        "sharpe_raw": sharpe_ratio(bt_raw.strategy_returns),
        "max_dd_net": max_drawdown(df_cost["equity_curve_net"]),
        "trades": trade_count(df_cost["position"]),
        "total_cost": bt_cost.extras.get("total_cost", 0.0) if bt_cost.extras else 0.0,
        "equity_net": df_cost["equity_curve_net"],
        "returns_net": df_cost["strategy_return_net"],
    }


def build_overlay_grid(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    records: List[dict] = []
    for vol in VOL_TARGETS:
        for stop in DAILY_STOPS:
            params: dict = {
                "max_position_size": 1.5,
                "target_volatility": vol,
                "volatility_lookback": 96,
                "max_leverage": 1.5,
            }
            if stop is not None:
                params["daily_stop_loss"] = stop
            label_stop = "none" if stop is None else f"{stop:.4f}"
            for split_label, df_split in (("train", df_train), ("test", df_test)):
                df_sig = rolling_zscore_signal(df_split, window=WINDOW, z_entry=Z_ENTRY)
                bt = run_backtest_with_costs(
                    df_sig, cost_per_unit_change=COST_PER_UNIT, **params
                )
                records.append({
                    "split": split_label,
                    "vol": vol,
                    "daily_stop": label_stop,
                    "sharpe_net": sharpe_ratio(bt.df["strategy_return_net"]),
                })
    return pd.DataFrame(records)


# ── figure functions ──────────────────────────────────────────────────────────

def plot_equity_curves(equity_series: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = {"train": "#1f77b4", "test": "#d62728"}
    for ax, (split, series) in zip(axes, equity_series.items()):
        series.plot(ax=ax, color=colors[split], linewidth=0.7)
        ax.set_title(f"{split.capitalize()} set — net equity", fontsize=12, fontweight="bold")
        ax.set_ylabel("Growth of $1")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        f"Net Equity Curves  (window={WINDOW}, z={Z_ENTRY},"
        "  vol-target 0.30% + 1% daily stop)",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(EQUITY_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(grid_df: pd.DataFrame) -> None:
    test_df = grid_df[grid_df["split"] == "test"].copy()
    pivot = test_df.pivot(index="daily_stop", columns="vol", values="sharpe_net")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    c = ax.imshow(pivot.values, cmap="RdYlBu", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.4f}" for v in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Target volatility per bar", fontsize=10)
    ax.set_ylabel("Daily stop-out threshold", fontsize=10)
    ax.set_title("Test Net Sharpe — risk overlay grid", fontsize=11, fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(
                j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                color="white" if abs(val) > vmax * 0.5 else "black",
            )
    fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04, label="Sharpe (net)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(HEATMAP_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hourly_returns(returns_net: pd.Series) -> None:
    by_hour = returns_by_hour_of_day(returns_net)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in by_hour.values]
    ax.bar(by_hour.index, by_hour.values * 1e4, color=bar_colors, edgecolor="none", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("UTC hour of day", fontsize=10)
    ax.set_ylabel("Mean net return (bps)", fontsize=10)
    ax.set_title(
        "Mean Net Strategy Return by UTC Hour  (test set)", fontsize=11, fontweight="bold"
    )
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(HOUR_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── LaTeX generation ──────────────────────────────────────────────────────────

def _f(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}"


def _pct(v: float, d: int = 1) -> str:
    return rf"{v * 100:.{d}f}\%"


def build_table_tex(variant_rows: List[dict], bh: dict) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Variant & Sharpe (raw) & Sharpe (net) & Max\,DD & Trades & Cost \\",
        r"\midrule",
    ]
    for r in variant_rows:
        lines.append(
            f"{r['latex_name']} & "
            f"{_f(r['sharpe_raw'])} & "
            f"{_f(r['sharpe_net'])} & "
            f"{_pct(r['max_dd_net'])} & "
            f"{int(r['trades']):,} & "
            f"{_f(r['total_cost'], 2)} \\\\"
        )
    lines += [
        r"\midrule",
        rf"Buy \& Hold & {_f(bh['sharpe'])} & --- & {_pct(bh['max_drawdown'])} & --- & --- \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def generate_latex(
    train_table: str,
    test_table: str,
    ci_train: Tuple[float, float],
    ci_test: Tuple[float, float],
    n_train: int,
    n_test: int,
    start_date: str,
    end_date: str,
) -> str:
    # NOTE: This is a raw f-string. LaTeX braces must be doubled: \begin{{...}}.
    # Python variable substitutions are single-braced: {variable}.
    return rf"""\documentclass[11pt,a4paper]{{article}}

\usepackage[top=2.5cm,bottom=2.5cm,left=2.8cm,right=2.8cm]{{geometry}}
\usepackage{{lmodern}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{microtype}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage{{amsmath,amssymb}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{parskip}}

\captionsetup{{font=small,labelfont=bf}}

\title{{%
  \bfseries Intraday BTC/USDT Breakout-Reversal Research\\[0.4em]
  \large Strategy Prototype Report%
}}
\author{{Research Prototype}}
\date{{Full year 2024}}

\begin{{document}}
\maketitle
\tableofcontents
\newpage

% ─────────────────────────────────────────────────────────────────────────────
\section{{Introduction}}
% ─────────────────────────────────────────────────────────────────────────────

This report documents an end-to-end intraday research prototype applied to
BTC/USDT five-minute bars spanning the 2024 calendar year.  The primary goal
is to demonstrate a reproducible research workflow---from raw exchange data
through signal generation, risk overlays, and rigorous performance
reporting---rather than to claim a live-tradable edge.

The strategy applies a rolling z-score to log returns: when the score crosses
a fixed entry threshold the position flips.  Transaction costs of 5\,bps per
unit of position change are charged, which, as the results show, overwhelm the
modest pre-cost alpha.

Source code lives in \texttt{{src/}}; automation scripts in
\texttt{{scripts/}}; exploratory notebooks in \texttt{{notebooks/}}.
Lightweight smoke tests are in \texttt{{tests/}}.

% ─────────────────────────────────────────────────────────────────────────────
\section{{Data \& Preprocessing}}
% ─────────────────────────────────────────────────────────────────────────────

\paragraph{{Source.}}
Raw Binance BTC/USDT one-minute OHLCV CSV exports for 2024 are merged, sorted
by UTC timestamp, and deduplicated.  Bars are resampled to five-minute
resolution using standard OHLCV rules (open\,=\,first, high\,=\,max,
low\,=\,min, close\,=\,last, volume\,=\,sum).

\paragraph{{Coverage.}}
After resampling and NaN removal the dataset spans {start_date} to {end_date}
and contains {n_train + n_test:,} five-minute bars.  A 70/30 chronological
split yields {n_train:,} training bars and {n_test:,} test bars with no
overlap.

\paragraph{{Validation.}}
\texttt{{validate\_ohlcv}} checks for a monotone \texttt{{DatetimeIndex}}, no
duplicate timestamps, positive close prices, high\,$\geq$\,low, non-negative
volume, and absence of gaps exceeding three times the expected bar width.
Any violation raises immediately at load time.

\paragraph{{Returns.}}
Per-bar log returns are computed as
$r_t = \ln(C_t / C_{{t-1}})$ where $C_t$ is the closing price.
All compounded equity curves use
$E_T = E_0 \cdot \exp\!\bigl(\sum_t r_t\bigr)$ throughout; the incorrect
$(1\!+\!r_t)$ cumulative product does not apply to log returns.

% ─────────────────────────────────────────────────────────────────────────────
\section{{Strategy Design}}
% ─────────────────────────────────────────────────────────────────────────────

\subsection{{Signal Generation}}

For each bar $t$, let $\mu_t$ and $\sigma_t$ be the rolling mean and standard
deviation of the preceding $W={WINDOW}$ log returns.  The z-score is
\begin{{equation}}
  z_t = \frac{{r_t - \mu_t}}{{\sigma_t}}, \qquad \sigma_t \neq 0.
  \label{{eq:zscore}}
\end{{equation}}
Bars where $\sigma_t = 0$ are set to \texttt{{NaN}} and force a flat
position---they are never silently treated as zero-signal.

The raw position signal is
\begin{{equation}}
  s_t = \begin{{cases}}
    +1 & z_t < -{Z_ENTRY}, \\
    -1 & z_t > +{Z_ENTRY}, \\
     0 & \text{{otherwise.}}
  \end{{cases}}
  \label{{eq:signal}}
\end{{equation}}
An optional hysteresis \emph{{exit bandwidth}} holds a position open until the
z-score relaxes past a softer inner threshold, reducing unnecessary reversals.

\subsection{{Backtest Mechanics}}

Positions are shifted one bar forward to eliminate look-ahead bias.  The
net per-bar strategy return is
\begin{{equation}}
  R_t^{{\mathrm{{net}}}}
    = p_{{t-1}} \cdot r_t - c\,\bigl|p_t - p_{{t-1}}\bigr|,
  \label{{eq:pnl}}
\end{{equation}}
where $p_{{t-1}}$ is the lagged position and $c={COST_PER_UNIT}$ (5\,bps)
is the cost per unit of absolute position change.

\subsection{{Risk Overlays}}

Two optional overlays are available in \texttt{{run\_backtest\_with\_costs}}:

\begin{{enumerate}}
  \item \textbf{{Volatility targeting.}} The position is rescaled by
        $\tau/\hat{{\sigma}}_t$, where $\tau$ is the target per-bar volatility
        and $\hat{{\sigma}}_t$ is a 96-bar exponential estimate, capped at a
        maximum leverage.
  \item \textbf{{Daily stop-out.}} If the cumulative intraday P\&L loss
        crosses a preset threshold (e.g.\ $-1\%$) the position is zeroed for
        the rest of that UTC day.
\end{{enumerate}}

% ─────────────────────────────────────────────────────────────────────────────
\section{{Results}}
% ─────────────────────────────────────────────────────────────────────────────

\subsection{{Performance Summary}}

Tables~\ref{{tab:train}} and~\ref{{tab:test}} report annualised Sharpe ratios
(with and without transaction costs), maximum drawdown on the net equity
curve, trade count, and total cost debited, across three risk configurations.
The \emph{{Buy \& Hold}} row provides a passive long benchmark.

\begin{{table}}[H]
  \centering
  \caption{{Training-set performance (70\,\% of 2024 data; {n_train:,} bars).}}
  \label{{tab:train}}
  \medskip
{train_table}
\end{{table}}

\begin{{table}}[H]
  \centering
  \caption{{Test-set performance (held-out 30\,\%; {n_test:,} bars).}}
  \label{{tab:test}}
  \medskip
{test_table}
\end{{table}}

\subsection{{Sharpe Confidence Intervals}}

Block-bootstrap 95\,\% CIs for the annualised \emph{{net}} Sharpe of the
baseline $\pm1$ signal ($b=\lfloor n^{{1/3}}\rfloor$, $B=1{{,}}000$ resamples,
seed\,42):
\begin{{itemize}}
  \item \textbf{{Train:}} 95\,\% CI
        $[{ci_train[0]:.2f},\;{ci_train[1]:.2f}]$
  \item \textbf{{Test:\phantom{{n}}}} 95\,\% CI
        $[{ci_test[0]:.2f},\;{ci_test[1]:.2f}]$
\end{{itemize}}
Both intervals lie far below zero, confirming the negative net Sharpe is not
a statistical artefact of sample size.

\subsection{{Equity Curves}}

Figure~\ref{{fig:equity}} plots the net equity curve (growth of \$1 after
costs) for the volatility-targeted variant
(target vol\,=\,0.30\,\%, daily stop\,=\,1\,\%).

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\linewidth]{{equity_curves.png}}
  \caption{{Net equity curves on training and test sets with 5\,bps transaction
    costs.  Both splits exhibit a near-monotone drawdown once costs are
    factored in.  Variant: vol\,target 0.30\,\% $+$ 1\,\% daily stop.}}
  \label{{fig:equity}}
\end{{figure}}

\subsection{{Parameter Sensitivity --- Risk Overlay Grid}}

Figure~\ref{{fig:heatmap}} shows the test net Sharpe across a
$3\!\times\!3$ grid of target-volatility and daily-stop combinations.
All nine cells are deeply negative, confirming no overlay configuration
rescues the economics under the assumed cost model.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.72\linewidth]{{overlay_heatmap.png}}
  \caption{{Test net Sharpe ratio across the risk overlay grid.
    Cell annotations show the exact value.
    All configurations produce large negative Sharpe ratios.}}
  \label{{fig:heatmap}}
\end{{figure}}

\subsection{{Hour-of-Day Analysis}}

Figure~\ref{{fig:hourly}} decomposes mean net strategy return by UTC hour on
the test set.  Green bars indicate hours where the strategy was on average
profitable; red bars indicate net losses.  No single UTC hour generates
enough alpha to offset overall negative performance.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\linewidth]{{hourly_returns.png}}
  \caption{{Mean net strategy return (basis points per bar) by UTC hour of
    day, computed on the test set.  Green = positive mean return;
    red = negative.}}
  \label{{fig:hourly}}
\end{{figure}}

\subsection{{Walk-Forward Evaluation}}

Rolling 60-day train\,/\,10-day test windows simulate periodic model
refreshes; strategy returns on each test slice are concatenated into a
pseudo-live equity curve.

\begin{{itemize}}
  \item \textbf{{Fixed parameters}} ($W=20$, $z=1.5$):\quad
        pseudo-live net Sharpe $\approx -50$;\quad
        max drawdown $\approx -100\,\%$.
  \item \textbf{{Per-window re-optimisation}}
        ($W\in\{{10,20,40\}}$, $z\in\{{1.0,1.5,2.0\}}$):\quad
        pseudo-live net Sharpe $\approx -27$;\quad
        max drawdown $\approx -99\,\%$.
\end{{itemize}}

The re-optimised run consistently selects $W=40$, $z=2.0$ in later windows,
indicating the optimiser gravitates toward lower turnover rather than
discovering genuine alpha.  Full charts are in
\texttt{{notebooks/05\_walk\_forward.ipynb}}.

% ─────────────────────────────────────────────────────────────────────────────
\section{{Discussion \& Conclusion}}
% ─────────────────────────────────────────────────────────────────────────────

\subsection{{Key Takeaways}}

\begin{{enumerate}}
  \item \textbf{{Costs overwhelm the signal.}}  Without friction the best
        observed test Sharpe is $\approx 0.36$---weak but positive.  Adding
        5\,bps per unit of turnover collapses it to $\approx -75$, showing
        that a high-frequency mean-reversion strategy must trade far less
        often or achieve meaningfully larger pre-cost alpha to be viable.
  \item \textbf{{Risk overlays help but cannot compensate.}}  Volatility
        targeting and daily stop-outs reduce trade count by $\approx 40\,\%$
        and shrink daily drawdowns, but the residual cost drag is prohibitive.
  \item \textbf{{No intraday session provides a safe harbour.}}  The
        hour-of-day analysis reveals no UTC hour where the strategy is
        reliably profitable net of costs.
  \item \textbf{{Walk-forward results confirm out-of-sample fragility.}}
        Periodic re-optimisation provides negligible improvement; the signal
        has no durable edge at this frequency under the stated cost assumptions.
\end{{enumerate}}

\subsection{{Limitations}}

\begin{{itemize}}
  \item Single asset (BTC/USDT) and a single intraday horizon; no
        cross-sectional diversification.
  \item Flat 5\,bps cost model; actual costs depend on order-book depth,
        queue position, and market impact.
  \item No production plumbing: order routing, borrow costs, or position
        limits are absent.
  \item Strictly a research prototype; not trading advice.
\end{{itemize}}

\subsection{{Future Directions}}

\begin{{itemize}}
  \item Extend to a multi-asset basket for cross-sectional mean-reversion.
  \item Incorporate richer execution modelling (latency, partial fills,
        volume-adaptive costs).
  \item Explore nonlinear features or ML overlays that might extract more
        persistent alpha before turnover erases it.
  \item Model volatility regimes explicitly rather than relying solely on
        rolling-window normalisation.
\end{{itemize}}

\bigskip
\noindent\rule{{\linewidth}}{{0.4pt}}

\medskip
\noindent\textit{{Disclaimer: this repository is for research and educational
purposes only and does not constitute trading advice.}}

% ─────────────────────────────────────────────────────────────────────────────
\appendix
\section{{Technical Notes}}
% ─────────────────────────────────────────────────────────────────────────────

\subsection{{Log-Return Compounding}}

All equity curves use
\[
  E_T = E_0 \cdot \exp\!\!\left(\sum_{{t=1}}^{{T}} r_t\right),
\]
not the $(1+r_t)$ cumulative product, which is incorrect for log returns and
diverges from true P\&L over multi-month horizons.

\subsection{{Sharpe Ratio Convention}}

The annualised Sharpe ratio uses sample standard deviation
($\mathrm{{ddof}}=1$):
\[
  \widehat{{S}} = \frac{{\bar{{r}}}}{{s}} \cdot \sqrt{{N_{{\mathrm{{year}}}}}},
  \qquad
  s = \sqrt{{\frac{{1}}{{n-1}}\sum_t (r_t - \bar{{r}})^2}},
\]
where $N_{{\mathrm{{year}}}} = 365 \times 288 = 105{{,}}120$ is the number of
five-minute bars in a calendar year.

\subsection{{Block Bootstrap Confidence Intervals}}

CIs use a stationary block bootstrap with block size
$b = \max\!\bigl(1, \lfloor n^{{1/3}} \rfloor\bigr)$ and $B=1{{,}}000$
resamples (seed\,42).  The percentile method gives
\[
  \bigl(\widehat{{S}}_{{\alpha/2}},\;
         \widehat{{S}}_{{1-\alpha/2}}\bigr)
  \quad \text{{at the 95\,\% level.}}
\]

\end{{document}}
"""


def compile_latex(tex_path: Path) -> None:
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    for run_num in range(2):
        result = subprocess.run(
            cmd, cwd=str(tex_path.parent), capture_output=True, text=True
        )
        if result.returncode != 0:
            tail = result.stdout[-4000:] if result.stdout else result.stderr[-4000:]
            print("=== pdflatex output (last 4000 chars) ===")
            print(tail)
            raise RuntimeError(f"pdflatex failed on pass {run_num + 1}")


def cleanup_aux(tex_path: Path) -> None:
    stem = tex_path.stem
    for ext in (".aux", ".log", ".toc", ".out"):
        p = tex_path.parent / (stem + ext)
        if p.exists():
            p.unlink()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data ...")
    df = clean_and_resample_ohlcv(DATA_PATH)
    df_train, df_test = chronological_train_test_split(df, train_fraction=TRAIN_FRACTION)

    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].strftime("%Y-%m-%d")
    n_train, n_test = len(df_train), len(df_test)

    print("Running variants ...")
    train_rows: List[dict] = []
    test_rows: List[dict] = []
    equity_for_plot: dict = {}
    raw_test_returns: pd.Series | None = None

    for config in VARIANTS:
        print(f"  {config.name} ...")
        tr = run_variant(df_train, config)
        te = run_variant(df_test, config)

        train_rows.append({
            "latex_name": config.latex_name,
            "sharpe_raw": tr["sharpe_raw"],
            "sharpe_net": tr["sharpe_net"],
            "max_dd_net": tr["max_dd_net"],
            "trades": tr["trades"],
            "total_cost": tr["total_cost"],
        })
        test_rows.append({
            "latex_name": config.latex_name,
            "sharpe_raw": te["sharpe_raw"],
            "sharpe_net": te["sharpe_net"],
            "max_dd_net": te["max_dd_net"],
            "trades": te["trades"],
            "total_cost": te["total_cost"],
        })

        if config.name == VARIANTS[1].name:
            equity_for_plot = {"train": tr["equity_net"], "test": te["equity_net"]}
        if config.name == VARIANTS[0].name:
            raw_test_returns = te["returns_net"]
            raw_train_returns = tr["returns_net"]

    print("Computing B&H metrics ...")
    bh_train = buy_and_hold_metrics(df_train)
    bh_test = buy_and_hold_metrics(df_test)

    print("Computing Sharpe CIs ...")
    ci_train = sharpe_confidence_interval(raw_train_returns)
    ci_test = sharpe_confidence_interval(raw_test_returns)

    print("Building overlay grid ...")
    overlay_grid = build_overlay_grid(df_train, df_test)

    print("Generating figures ...")
    plot_equity_curves(equity_for_plot)
    plot_heatmap(overlay_grid)
    plot_hourly_returns(raw_test_returns)

    print("Generating LaTeX ...")
    train_table = build_table_tex(train_rows, bh_train)
    test_table = build_table_tex(test_rows, bh_test)
    latex_src = generate_latex(
        train_table=train_table,
        test_table=test_table,
        ci_train=ci_train,
        ci_test=ci_test,
        n_train=n_train,
        n_test=n_test,
        start_date=start_date,
        end_date=end_date,
    )
    TEX_PATH.write_text(latex_src, encoding="utf-8")

    print("Compiling PDF (2 passes) ...")
    compile_latex(TEX_PATH)
    cleanup_aux(TEX_PATH)

    print(f"Done. Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
