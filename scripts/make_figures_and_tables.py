#!/usr/bin/env python3
"""Run the study and the walk-forward check, and write the figures and tables in reports/.

Outputs (under reports/):
    figures/cost_sensitivity.png   test net return against cost per unit turnover
    figures/equity_curves.png      test cumulative gross and net log return
    figures/break_even_grid.png    train break-even cost across the signal grid
    figures/hourly_returns.png     train mean gross return by UTC hour
    results.csv                    test results table
    signal_grid.csv                train/validation signal grid (selection input)
    overlay_grid.csv               train/validation overlay grid (selection input)
    walk_forward.csv               per-window walk-forward results

The test results and the walk-forward summary are also printed.

Usage:
    uv run python scripts/make_figures_and_tables.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btc_research import study
from btc_research.metrics import max_drawdown, sharpe_confidence_interval, sharpe_ratio
from btc_research.preprocess import PROJECT_ROOT
from btc_research.walk_forward import (
    generate_rolling_windows,
    walk_forward_fixed_params,
    walk_forward_retrained_params,
)

REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

WF_TRAIN_DAYS, WF_TEST_DAYS = 60, 10
WF_FIXED = (20, 1.5)  # fixed in advance

BLUE, ORANGE, GRAY, INK, MUTED = "#257de0", "#f47230", "#8a8a8a", "#191919", "#666666"
STYLE: dict[str, Any] = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#bdbdbd",
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 9.5,
    "legend.frameon": False,
    "figure.dpi": 150,
}
VARIANT_STYLE = {
    "Mean reversion, signal only": {"color": BLUE, "ls": "-"},
    "Momentum (mirror), signal only": {"color": ORANGE, "ls": "-"},
    "Mean reversion, selected overlays": {"color": BLUE, "ls": "--"},
    "Momentum (mirror), selected overlays": {"color": ORANGE, "ls": "--"},
}


def walk_forward_tables(splits: study.Splits) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both walk-forward variants. Returns a summary per method and the per-window rows."""
    df = splits.full
    windows = generate_rolling_windows(df, WF_TRAIN_DAYS, WF_TEST_DAYS)
    runs = {
        f"Fixed W={WF_FIXED[0]}, z={WF_FIXED[1]:g}": walk_forward_fixed_params(
            df, windows, *WF_FIXED, cost_per_unit_turnover=study.BASE_COST
        ),
        "Re-selected per window": walk_forward_retrained_params(
            df, windows, study.WINDOWS, study.Z_ENTRIES, cost_per_unit_turnover=study.BASE_COST
        ),
    }
    summary, per_window = [], []
    for label, res in runs.items():
        pw = pd.DataFrame(res["per_window_metrics"]).assign(method=label)
        per_window.append(pw)
        net = res["test_returns"]
        gross, to = pw["test_gross_log_return"].sum(), pw["test_turnover"].sum()
        lo, hi = sharpe_confidence_interval(net)
        summary.append(
            {
                "method": label,
                "windows": len(pw),
                "gross_log_return": gross,
                "net_log_return": float(net.sum()),
                "turnover": to,
                "break_even_bp": gross / to * 1e4,
                "sharpe_net": sharpe_ratio(net),
                "sharpe_net_ci_low": lo,
                "sharpe_net_ci_high": hi,
                "max_drawdown_net": max_drawdown(res["test_equity_curve"]),
                "first_test_day": pw["test_start"].min(),
                "last_test_day": pw["test_end"].max(),
            }
        )
    return pd.DataFrame(summary), pd.concat(per_window, ignore_index=True)


def plot_cost_sensitivity(curves: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for label, grp in curves.groupby("variant", sort=False):
        st = VARIANT_STYLE[label]
        ax.plot(grp["cost_bp"], grp["net_log_return"] * 100, lw=2, label=label, **st)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(study.BINANCE_TAKER_FEE_BP, color=GRAY, lw=1, ls=":")
    ax.text(
        study.BINANCE_TAKER_FEE_BP + 0.2,
        -12,
        "Binance spot taker fee\n(about 10 bp, assumed)",
        color=MUTED,
        fontsize=8.5,
        va="top",
    )
    ax.set_xlabel("Cost per unit of turnover (bp of notional)")
    ax.set_ylabel("Net cumulative log return, test (%)")
    ax.set_title("Test-period net return against trading cost")
    ax.set_xlim(0, max(study.COST_GRID_BP))
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_equity(runs: dict[str, Any], bh_test: pd.Series, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = ["Mean reversion, signal only", "Momentum (mirror), signal only"]
    for ax, col, title in (
        (axes[0], "strategy_return_gross", "Gross of costs"),
        (
            axes[1],
            "strategy_return_net",
            f"Net of {study.BASE_COST * 1e4:.0f} bp per unit turnover",
        ),
    ):
        for label in labels:
            r = runs[label].df[col]
            ax.plot(
                r.index, r.cumsum() * 100, lw=1.4, label=label, color=VARIANT_STYLE[label]["color"]
            )
        if col == "strategy_return_gross":
            ax.plot(bh_test.index, bh_test.cumsum() * 100, lw=1.2, color=GRAY, label="Buy and hold")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(title)
        ax.set_ylabel("Cumulative log return (%)")
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Test period (Oct 19 to Dec 31, 2024)", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_break_even_grid(grid: pd.DataFrame, path: Path) -> None:
    exits = study.EXIT_BANDWIDTHS
    vals = [
        grid[grid["exit_bandwidth"].isna() if e is None else grid["exit_bandwidth"] == e].pivot(
            index="window", columns="z_entry", values="train_break_even_bp"
        )
        for e in exits
    ]
    vmax = max(float(np.nanmax(np.abs(v.values))) for v in vals)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, e, piv in zip(axes, exits, vals, strict=True):
        im = ax.imshow(piv.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_xlabel("Entry threshold z")
        ax.set_title("No exit band" if e is None else f"Exit band {e:g}", fontsize=10)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(v) > 0.6 * vmax else INK,
                )
    axes[0].set_ylabel("Window (bars)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Break-even cost (bp)")
    fig.suptitle(
        "Training block: break-even cost per unit turnover, mean-reversion signal",
        fontsize=10,
        y=1.05,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_hourly(hourly: pd.Series, hours: frozenset[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = [BLUE if h in hours else "#c9c9c9" for h in hourly.index]
    ax.bar(hourly.index, hourly.values * 1e4, color=colors, width=0.75)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("Mean gross return per bar (bp)")
    ax.set_title("Training block: gross return by hour (blue = hours kept by the session filter)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Running study ...")
    res = study.run_study()
    print("Running walk-forward ...")
    wf, wf_windows = walk_forward_tables(res.splits)

    reselected = wf_windows[wf_windows["method"] == "Re-selected per window"]
    counts = (
        reselected.groupby(["chosen_window", "chosen_z_entry"]).size().sort_values(ascending=False)
    )
    wf_choices = "Choices across {} windows: {}.".format(
        len(reselected),
        "; ".join(f"W={w}, k={z:g} ({n})" for (w, z), n in counts.items()),
    )

    print("Writing tables and figures ...")
    res.results.to_csv(REPORT_DIR / "results.csv", float_format="%.6g")
    res.signal_grid.to_csv(REPORT_DIR / "signal_grid.csv", index=False, float_format="%.6g")
    res.overlay_grid.to_csv(REPORT_DIR / "overlay_grid.csv", index=False, float_format="%.6g")
    wf_windows.to_csv(REPORT_DIR / "walk_forward.csv", index=False, float_format="%.6g")

    with plt.rc_context(STYLE):
        plot_cost_sensitivity(res.cost_curves, FIG_DIR / "cost_sensitivity.png")
        plot_equity(res.test_runs, res.splits["test"]["log_return"], FIG_DIR / "equity_curves.png")
        plot_break_even_grid(res.signal_grid, FIG_DIR / "break_even_grid.png")
        plot_hourly(res.hourly_train, res.session_hours, FIG_DIR / "hourly_returns.png")

    pd.set_option("display.width", 200)
    print("\nSelected signal:", res.signal_config.describe())
    print("Selected overlays:", res.final_config.describe())
    cols = [
        "gross_log_return",
        "net_log_return",
        "turnover",
        "entries",
        "break_even_bp",
        "sharpe_gross",
        "sharpe_gross_ci_low",
        "sharpe_gross_ci_high",
        "sharpe_net",
    ]
    print(res.results[cols].to_string(float_format=lambda v: f"{v:.4g}"))
    print("\nBuy and hold:", {k: round(v["total_return"], 4) for k, v in res.buy_and_hold.items()})
    print("\nWalk-forward:\n", wf.to_string(float_format=lambda v: f"{v:.4g}"))
    print("\n" + wf_choices)


if __name__ == "__main__":
    main()
