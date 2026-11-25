#!/usr/bin/env python3
"""Run the study and write figures, CSV tables and reports/summary.tex.

Outputs (all under reports/):
    figures/cost_sensitivity.png   test net return vs cost per unit turnover
    figures/equity_curves.png      test cumulative gross and net log return
    figures/break_even_grid.png    train break-even cost across the signal grid
    figures/hourly_returns.png     train mean gross return by UTC hour
    results.csv                    test-set results table
    signal_grid.csv                train/validation grid (selection input)
    overlay_grid.csv               train/validation overlay grid (selection input)
    walk_forward.csv               per-window walk-forward results
    summary.tex                    LaTeX write-up; compiled to summary.pdf only if
                                   pdflatex is on PATH (the PDF is not committed)

Usage:
    uv run python scripts/make_summary_report.py
"""

from __future__ import annotations

import shutil
import subprocess
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
TEX_PATH = REPORT_DIR / "summary.tex"

WF_TRAIN_DAYS, WF_TEST_DAYS = 60, 10
WF_FIXED = (20, 1.5)  # pre-specified, not chosen from data

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


# ── walk-forward ─────────────────────────────────────────────────────────────


def run_walk_forward(splits: study.Splits) -> tuple[pd.DataFrame, pd.DataFrame]:
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


# ── figures ──────────────────────────────────────────────────────────────────


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
        (axes[1], "strategy_return_net", f"Net of {study.BASE_COST * 1e4:.0f} bp per unit turnover"),
    ):
        for label in labels:
            r = runs[label].df[col]
            ax.plot(r.index, r.cumsum() * 100, lw=1.4, label=label, color=VARIANT_STYLE[label]["color"])
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
    exits = [None, 0.5, 0.0]
    vals = [
        grid[grid["exit_bandwidth"].isna() if e is None else grid["exit_bandwidth"] == e]
        .pivot(index="window", columns="z_entry", values="train_break_even_bp")
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
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.6 * vmax else INK)
    axes[0].set_ylabel("Window (bars)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Break-even cost (bp)")
    fig.suptitle("Training block: break-even cost per unit turnover, mean-reversion signal", fontsize=10, y=1.05)
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


# ── LaTeX ────────────────────────────────────────────────────────────────────


def _num(v: float, d: int = 2) -> str:
    return f"{v:,.{d}f}".replace("-", "$-$")


def results_tex(table: pd.DataFrame, bh: dict[str, float]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Variant & Gross (\%) & Net (\%) & Turnover & Break-even (bp) & Gross Sharpe [95\% CI] & Net Sharpe \\",
        r"\midrule",
    ]
    for label, r in table.iterrows():
        lines.append(
            f"{label} & {_num(r.gross_log_return * 100)} & {_num(r.net_log_return * 100, 1)} & "
            f"{r.turnover:,.0f} & {_num(r.break_even_bp)} & "
            f"{_num(r.sharpe_gross, 1)} [{_num(r.sharpe_gross_ci_low, 1)}, {_num(r.sharpe_gross_ci_high, 1)}] & "
            f"{_num(r.sharpe_net, 1)} \\\\"
        )
    lines += [
        r"\midrule",
        f"Buy and hold & {_num(bh['log_return'] * 100)} & --- & --- & --- & {_num(bh['sharpe'], 1)} & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def walk_forward_tex(wf: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Gross (\%) & Net (\%) & Turnover & Break-even (bp) & Net Sharpe [95\% CI] \\",
        r"\midrule",
    ]
    for _, r in wf.iterrows():
        lines.append(
            f"{r.method.replace('=', '$=$')} & {_num(r.gross_log_return * 100)} & {_num(r.net_log_return * 100, 1)} & "
            f"{r.turnover:,.0f} & {_num(r.break_even_bp)} & "
            f"{_num(r.sharpe_net, 1)} [{_num(r.sharpe_net_ci_low, 1)}, {_num(r.sharpe_net_ci_high, 1)}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_latex(res: study.StudyResult, wf: pd.DataFrame, wf_choices: str) -> str:
    sp = res.splits.describe()
    span = {k: (v["start"].strftime("%b %d"), v["end"].strftime("%b %d"), int(v["bars"])) for k, v in sp.iterrows()}
    sig, fin = res.signal_config, res.final_config
    mr = res.results.loc["Mean reversion, signal only"]
    grid_max_be = res.signal_grid["train_break_even_bp"].max()
    ogrid = res.overlay_grid
    cap = ogrid.dropna(subset=["target_volatility"]).groupby("target_volatility")[
        "validation_share_of_bars_at_cap"
    ].first()
    cap_text = ", ".join(f"{v * 100:.2f}\\,\\%: {s:.0%}".replace("%", "\\%") for v, s in cap.items())
    hours = ", ".join(str(h) for h in sorted(res.session_hours))
    final_overlays = []
    if fin.active_hours is not None:
        final_overlays.append(f"the session filter (UTC hours {hours})")
    if fin.target_volatility is not None:
        final_overlays.append(f"a {fin.target_volatility * 100:.2f}\\,\\% volatility target")
    if fin.daily_stop_loss is not None:
        final_overlays.append(f"a {fin.daily_stop_loss * 100:g}\\,\\% daily stop")
    final_text = " and ".join(final_overlays) if final_overlays else "no overlays"
    exit_text = "no exit band" if sig.exit_bandwidth is None else f"exit band {sig.exit_bandwidth:g}"

    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[top=2.5cm,bottom=2.5cm,left=2.5cm,right=2.5cm]{{geometry}}
\usepackage{{lmodern}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{microtype}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage{{amsmath}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{parskip}}
\graphicspath{{{{figures/}}}}
\captionsetup{{font=small,labelfont=bf}}

\title{{Short-horizon mean reversion in BTC/USDT:\\ a test on 5-minute bars, 2024}}
\date{{}}

\begin{{document}}
\maketitle

\section*{{Result}}

A rolling z-score mean-reversion signal on 5-minute BTC/USDT returns does not
cover trading costs. On the held-out test block its gross return is
{_num(mr.gross_log_return * 100)}\,\% over {mr.turnover:,.0f} units of turnover,
a break-even cost of {_num(mr.break_even_bp)}\,bp per unit turnover. Binance spot
taker fees are about 10\,bp per side for a regular account (our assumption).
The best break-even cost anywhere in the 27-point training grid was
{grid_max_be:.2f}\,bp. The momentum version of the same signal (its exact
negative) has the opposite gross return and the same costs, so it fails too.

\section{{Data}}

Binance spot BTCUSDT 1-minute klines for 2024, from the public archive at
\url{{https://data.binance.vision}}, aggregated to 5-minute bars
(open first, high max, low min, close last, volume sum). The year has
{len(res.splits.full) + 1:,} complete 5-minute bars and no gaps.
\texttt{{validate\_ohlcv}} runs when the raw files are loaded (sorted unique
timestamps, positive close, high $\geq$ low, non-negative volume) and again
on the 5-minute bars, where it also rejects gaps longer than three bars.

The series is split by time into train ({span['train'][0]}--{span['train'][1]},
{span['train'][2]:,} bars), validation ({span['validation'][0]}--{span['validation'][1]},
{span['validation'][2]:,} bars) and test ({span['test'][0]}--{span['test'][1]},
{span['test'][2]:,} bars), a 60/20/20 split. Signals are computed on the full
series and then sliced, which is valid because they only use past bars.

\section{{Signal and backtest}}

With $r_t = \ln(C_t/C_{{t-1}})$ and $\mu_t,\sigma_t$ the mean and population
standard deviation of $r_{{t-W+1}},\dots,r_t$,
\begin{{equation}}
  z_t = \frac{{r_t-\mu_t}}{{\sigma_t}}, \qquad
  s_t = \begin{{cases}} +1 & z_t < -k\\ -1 & z_t > k\\ 0 & \text{{otherwise}} \end{{cases}}
\end{{equation}}
for the mean-reversion direction; the momentum direction is $-s_t$. An optional
exit band $e<k$ holds a position until $z$ relaxes past $\mp e$. Bars with
$\sigma_t=0$ get no signal.

The position for bar $t$ is decided at the close of bar $t-1$:
$p_t = s_{{t-1}}\cdot m_{{t-1}}$, where $m$ is the volatility scaler (1 if
off). Per-bar net return is
\begin{{equation}}
  R_t = p_t r_t - c\,|p_t - p_{{t-1}}|,
\end{{equation}}
with $c = {study.BASE_COST}$ (10\,bp) in the main results. Positions are
capped at $|p|\le 1$, so there is no leverage.

Volatility targeting sets $m_{{t-1}} = \min(1, \tau/\hat\sigma_{{t-1}})$ where
$\hat\sigma_{{t-1}}$ is the equal-weighted standard deviation of the
{study.VOL_LOOKBACK} returns up to and including bar $t-1$. The daily stop sets
the position to zero for the rest of the UTC day once that day's cumulative
\emph{{net}} return reaches $-L$.

\section{{Selection protocol}}

\begin{{enumerate}}
  \item Train: a grid of $W\in\{{10,20,40\}}$, $k\in\{{1,1.5,2\}}$ and exit band
        $\in\{{\text{{none}},0.5,0\}}$ (27 points). The highest net Sharpe at 10\,bp
        was $W={sig.window}$, $k={sig.z_entry:g}$, {exit_text}.
  \item Train: the {study.N_SESSION_HOURS} UTC hours with the highest mean gross
        return of that signal form a candidate session filter (hours {hours}).
  \item Validation: session filter on/off $\times$ volatility target
        $\in\{{\text{{off}}, 0.10, 0.13, 0.16\}}\,\%$ per bar $\times$ daily stop
        $\in\{{\text{{off}}, 0.75, 1\}}\,\%$ (24 points). The highest net Sharpe was
        {final_text}.
  \item Test: the selected signal, with and without its overlays, in both
        directions, evaluated once.
\end{{enumerate}}

With the position cap at 1, volatility targeting can only reduce exposure. The
share of validation bars at the cap was {cap_text} for the three targets.

\section{{Results on the test block}}

\begin{{table}}[H]
\centering\small
\caption{{Test block ({span['test'][2]:,} bars). Returns are cumulative log returns.
Turnover is $\sum_t|p_t-p_{{t-1}}|$. Break-even is gross return divided by turnover.
Sharpe ratios are annualised with $365\times288$ bars per year; intervals are
moving-block bootstrap percentiles (block $\lfloor n^{{1/3}}\rfloor$, 1{{,}}000
resamples). Net figures use 10\,bp per unit turnover.}}
\resizebox{{\linewidth}}{{!}}{{%
{results_tex(res.results, res.buy_and_hold['test'])}
}}
\end{{table}}

The signal-only rows are exact mirrors: same turnover, opposite gross return.
With a daily stop the mirror is not exact, because the stop fires on different
days for the two directions. The gross Sharpe intervals include zero for the
signal-only rows; the annualised net Sharpe is large and negative only because
costs are charged on almost every trade, and it says little beyond that.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.85\linewidth]{{cost_sensitivity.png}}
\caption{{Test net return as a function of cost per unit turnover.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{equity_curves.png}}
\caption{{Cumulative gross and net log return on the test block, both directions, signal only.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{break_even_grid.png}}
\caption{{Break-even cost on the training block for every grid point.}}
\end{{figure}}

\section{{Walk-forward check}}

Rolling {WF_TRAIN_DAYS}-day train and {WF_TEST_DAYS}-day test windows over the
whole year, at 10\,bp. The fixed run uses $W={WF_FIXED[0]}$, $k={WF_FIXED[1]:g}$ in
every window; the re-selected run picks $(W,k)$ by net Sharpe on each training
window. {wf_choices}

\begin{{table}}[H]
\centering\small
{walk_forward_tex(wf)}
\end{{table}}

\section{{Limitations}}

\begin{{itemize}}
  \item One asset, one year, one bar size. Q4 2024 was a strong uptrend
        (buy and hold returned {res.buy_and_hold['test']['total_return']:.0%} on the test block).
  \item The cost model is linear: a flat fee per unit turnover, no spread, impact,
        latency or partial fills. Maker execution or VIP fee tiers would be cheaper
        than 10\,bp, but every grid point breaks even well below 1\,bp.
  \item The parameter and overlay grids, and the choice to test this signal at all,
        were informed by an earlier pass over the full 2024 sample (with the
        signal sign reversed). The train/validation/test protocol removes direct
        selection on test data, but not that earlier exposure.
  \item Annualised Sharpe ratios at 5-minute frequency are noisy; the bootstrap
        intervals above are wide.
\end{{itemize}}

\end{{document}}
"""


def compile_latex(tex_path: Path) -> bool:
    if shutil.which("pdflatex") is None:
        print("pdflatex not found; wrote summary.tex only.")
        return False
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    for _ in range(2):
        out = subprocess.run(cmd, cwd=tex_path.parent, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stdout[-3000:])
            raise RuntimeError("pdflatex failed")
    for ext in (".aux", ".log", ".out", ".toc"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)
    return True


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Running study ...")
    res = study.run_study()
    print("Running walk-forward ...")
    wf, wf_windows = run_walk_forward(res.splits)

    reselected = wf_windows[wf_windows["method"] == "Re-selected per window"]
    counts = reselected.groupby(["chosen_window", "chosen_z_entry"]).size().sort_values(ascending=False)
    wf_choices = "Choices across {} windows: {}.".format(
        len(reselected),
        "; ".join(f"$W={w}, k={z:g}$ ({n})" for (w, z), n in counts.items()),
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

    TEX_PATH.write_text(build_latex(res, wf, wf_choices), encoding="utf-8")
    compile_latex(TEX_PATH)

    pd.set_option("display.width", 200)
    print("\nSelected signal:", res.signal_config.describe())
    print("Selected overlays:", res.final_config.describe())
    cols = ["gross_log_return", "net_log_return", "turnover", "entries", "break_even_bp",
            "sharpe_gross", "sharpe_gross_ci_low", "sharpe_gross_ci_high", "sharpe_net"]
    print(res.results[cols].to_string(float_format=lambda v: f"{v:.4g}"))
    print("\nBuy and hold:", {k: round(v["total_return"], 4) for k, v in res.buy_and_hold.items()})
    print("\nWalk-forward:\n", wf.to_string(float_format=lambda v: f"{v:.4g}"))
    print("\n" + wf_choices)


if __name__ == "__main__":
    main()
