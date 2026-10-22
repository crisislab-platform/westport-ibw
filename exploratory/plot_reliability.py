"""Calibration of the BuildProp-Max exceedance probabilities.

Reads out/verification_pairs_buildprop_max.csv -- the per-building (forecast
probability, observed class) pairs written by
forecast_runner.write_verification_pairs -- and writes

  fig_reliability   reliability curve + sharpness, for the two class boundaries

WHAT THE FIGURE ANSWERS. The ranked probability skill score says how good the
forecasts are overall; it cannot say whether a stated 70% actually happens 70%
of the time. That is calibration, and it is only visible by binning the
forecasts and reading the outcome frequency off each bin. A warning system that
is skilful but overconfident is a different thing to fix than one that is
underconfident, and the skill score reports the same number for both.

The curve and its inset answer the two halves of the question the Brier
decomposition splits (Murphy 1973):

  reliability (curve)  distance from the diagonal -- do the probabilities mean
                       what they say. A curve FLATTER than the diagonal is
                       overconfidence, the usual failure of a small ensemble;
                       steeper is underconfidence.
  sharpness (inset)    how often each probability is used at all. Not a
                       diagnostic of correctness but of usefulness: a forecast
                       that only ever says the base rate is perfectly reliable
                       and worthless, and that failure appears here as all the
                       mass in one central bin, not in the curve.

Both are needed -- neither alone separates a useful forecast from a vacuous
one -- which is why the inset shares the curve's x axis rather than becoming a
second figure.

THE BINS. The plotted curve uses equal-width bins because that is what makes a
reliability diagram readable. The printed decomposition uses the ATTAINABLE
values instead -- an M-member ensemble can only produce M+1 distinct
probabilities -- because the identity BS = REL - RES + UNC holds exactly only
when the forecast is constant within a bin. Coarse bins leave a within-bin
variance term and the identity fails by a small amount. Both are reported, and
they answer the same question at different resolutions.

THE CONSISTENCY BARS. A bin whose observed frequency sits off the diagonal has
not necessarily failed: with a finite sample it would scatter around the
diagonal even if the forecast were perfect. The bars are the 5th-95th
percentile of that scatter under a BLOCK bootstrap that resamples whole
timesteps, not individual buildings. Buildings in one street flood together, so
resampling them independently would treat thousands of correlated rows as
thousands of independent ones and give bars several times too tight. See
Broecker and Smith (2007) on why a reliability diagram without them invites
reading noise as miscalibration.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
PAIRS_FILE = os.path.join(
    PROJECT_ROOT, "out", "verification_pairs_buildprop_max.csv")
PLOT_DIR = os.path.join(PROJECT_ROOT, "out")

# The two INTERIOR class boundaries. P(>= H1) is not plotted: every building is
# at least H1, so the event is certain and its reliability curve is a point.
#
# Colours are the hazard-class ramp from plot_miss_distribution.py, unchanged
# and for the same reason -- these curves ARE H2 and H3+, so the colour that
# means H3+ in one figure must mean H3+ in the other. That ramp is sequential
# (light -> dark with severity) and its adjacent normal-vision dE is 26.6, well
# clear of the floor, so the two curves separate for CVD readers and in
# greyscale print. Line style differs as well, so identity never rests on
# colour alone.
EVENTS = (
    {"column": "prob_at_least_H2", "level": 2, "label": r"$P(\geq$ H2$)$",
     "color": "#e6550d", "marker": "o", "linestyle": "-"},
    {"column": "prob_at_least_H3_above", "level": 3, "label": r"$P(\geq$ H3+$)$",
     "color": "#7f2704", "marker": "s", "linestyle": (0, (4, 1.6))},
)

N_BINS = 10
N_BOOTSTRAP = 1000
BOOTSTRAP_INTERVAL = (5, 95)

INK = "#1a1a1a"
RULE = "#4d4d4d"
GRID = "#cccccc"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,  # embed as TrueType so the PDF is editable/searchable
    "ps.fonttype": 42,
})


def load(scenario_id=None, lead_h=None):
    """The verification pairs, optionally restricted to one scenario / lead.

    Pooling ACROSS timesteps is the default and is deliberate: the upper bins
    hold only buildings the ensemble was confident about, and a single timestep
    rarely has enough of them to estimate a frequency. Stratify by lead only
    when there is enough data to, since calibration genuinely degrades with lead
    and pooling hides that.
    """
    df = pd.read_csv(PAIRS_FILE)
    if scenario_id is not None:
        df = df[df["scenario_id"] == scenario_id]
    if lead_h is not None:
        df = df[df["lead_h"] == lead_h]
    if df.empty:
        raise SystemExit(f"no rows in {PAIRS_FILE} for the requested subset")
    return df


def decompose(forecast, observed, group_codes):
    """Murphy's (1973) partition BS = REL - RES + UNC over the given bins.

    ``group_codes`` is an integer bin label per row. Exact when the forecast is
    constant within every bin -- see the module docstring on bins.

    Aggregated with bincount rather than by iterating boolean masks. The bins
    that make the partition exact are the DISTINCT FORECAST VALUES, of which a
    well-behaved M-member ensemble has M+1; but pooling runs of different M, or
    any future continuous probability, makes that a per-row mask over the whole
    array and the cost quadratic. This form does not care how many bins there
    are.
    """
    n = len(forecast)
    base_rate = observed.mean()
    counts = np.bincount(group_codes)
    forecast_sum = np.bincount(group_codes, weights=forecast)
    observed_sum = np.bincount(group_codes, weights=observed)

    filled = counts > 0
    counts = counts[filled]
    mean_forecast = forecast_sum[filled] / counts
    frequency = observed_sum[filled] / counts
    return (
        float((counts * (mean_forecast - frequency) ** 2).sum() / n),
        float((counts * (frequency - base_rate) ** 2).sum() / n),
        float(base_rate * (1.0 - base_rate)),
    )


def bin_pairs(forecast, observed, blocks, edges):
    """Per-bin count, mean forecast, observed frequency and bootstrap interval.

    ``blocks`` labels the timestep each row came from; the bootstrap resamples
    those labels, so every row of a timestep travels together.
    """
    index = np.clip(np.digitize(forecast, edges[1:-1]), 0, len(edges) - 2)
    codes, unique_blocks = pd.factorize(blocks)
    n_blocks = len(unique_blocks)
    rng = np.random.default_rng(0)

    # A resampled frequency is a ratio of SUMS over the drawn timesteps, so the
    # bootstrap only ever needs each timestep's hit count and row count in this
    # bin -- never the rows themselves. Reducing to those two per-block vectors
    # first turns the whole thing into two gathers per bin; resampling rows
    # directly re-scanned the full array once per draw per block, which is
    # minutes rather than milliseconds at the row counts a real event produces.
    counts, mean_forecast, frequency, lower, upper = [], [], [], [], []
    for k in range(len(edges) - 1):
        mask = index == k
        counts.append(int(mask.sum()))
        if not mask.any():
            mean_forecast.append(np.nan); frequency.append(np.nan)
            lower.append(np.nan); upper.append(np.nan)
            continue
        mean_forecast.append(float(forecast[mask].mean()))
        frequency.append(float(observed[mask].mean()))

        block_hits = np.bincount(codes[mask], weights=observed[mask],
                                 minlength=n_blocks)
        block_rows = np.bincount(codes[mask], minlength=n_blocks)
        drawn = rng.integers(0, n_blocks, size=(N_BOOTSTRAP, n_blocks))
        hits = block_hits[drawn].sum(axis=1)
        rows = block_rows[drawn].sum(axis=1)
        # A draw that happened to miss this bin entirely says nothing about its
        # frequency, so it is dropped rather than counted as zero.
        usable = rows > 0
        if usable.any():
            lo, hi = np.percentile(hits[usable] / rows[usable], BOOTSTRAP_INTERVAL)
        else:
            lo = hi = np.nan
        lower.append(float(lo)); upper.append(float(hi))

    return (np.array(counts), np.array(mean_forecast), np.array(frequency),
            np.array(lower), np.array(upper))


def style_axis(ax):
    ax.grid(True, color=GRID, linewidth=0.5, linestyle=(0, (3, 3)), zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=RULE, labelcolor=INK, length=3.5, width=0.8)


def save(fig, stem):
    for ext in ("png", "pdf"):
        out = os.path.join(PLOT_DIR, f"{stem}.{ext}")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"saved {out}")


def plot_reliability(df, stem="fig_reliability"):
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    blocks = (df["scenario_id"].astype(str) + "|"
              + df["valid_time"].astype(str)).to_numpy()

    # A square panel with the sharpness as a corner inset, rather than two
    # stacked panels. Equal aspect is not decoration here -- the diagonal has to
    # sit at 45 degrees for the eye to judge distance from it -- and enforcing it
    # on a stacked layout shrinks the upper axes so the two x axes no longer
    # line up, which is worse than any gain from the larger histogram. The
    # lower right of a reliability plot is empty by construction (the curve runs
    # bottom-left to top-right), so the inset costs no space.
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax_sharp = ax.inset_axes([0.575, 0.085, 0.395, 0.235])

    # Perfect calibration, and the base rate a no-resolution forecast would sit
    # on. Drawn first so every mark sits over them.
    ax.plot([0, 1], [0, 1], color=RULE, linewidth=0.9, linestyle=(0, (2, 2)),
            zorder=1, label="perfect calibration")

    summary = []
    for event in EVENTS:
        forecast = df[event["column"]].to_numpy(dtype=float)
        observed = (df["observed_level"].to_numpy(dtype=int) >= event["level"]
                    ).astype(float)
        counts, mean_forecast, frequency, lower, upper = bin_pairs(
            forecast, observed, blocks, edges)

        drawn = counts > 0
        ax.errorbar(
            mean_forecast[drawn], frequency[drawn],
            yerr=[frequency[drawn] - lower[drawn], upper[drawn] - frequency[drawn]],
            marker=event["marker"], markersize=4.5, linewidth=1.5,
            linestyle=event["linestyle"],
            color=event["color"], ecolor=event["color"], elinewidth=0.8,
            capsize=0, zorder=3,
            markeredgecolor="white", markeredgewidth=0.6,
            label=event["label"],
        )

        # The exact partition, on the values the ensemble can actually produce.
        attainable = np.unique(forecast, return_inverse=True)[1]
        reliability, resolution, uncertainty = decompose(
            forecast, observed, attainable)
        brier = float(((forecast - observed) ** 2).mean())
        summary.append({
            "label": event["label"], "n": len(forecast),
            "base_rate": float(observed.mean()), "brier": brier,
            "rel": reliability, "res": resolution, "unc": uncertainty,
        })

        # stairs, not step: it takes the bin EDGES, so the last bin is closed
        # rather than run flat to the axis limit as a step plot does.
        ax_sharp.stairs(
            np.maximum(counts, 0.7), edges, color=event["color"],
            linewidth=1.1, linestyle=event["linestyle"], zorder=3,
        )

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration of the BuildProp-Max exceedance probabilities",
                 color=INK, pad=8)
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False, handlelength=2.4,
              borderaxespad=0.9)

    # No statistics block on the figure. Ten numbers beside a curve is a second
    # thing to read, and none of them is what the reader comes here for -- the
    # question a reliability diagram answers is "how far off the diagonal", and
    # the answer is the picture. BS/REL/RES/UNC still go to stdout, for the
    # caption.

    # The inset is a shape, not a table: how often each probability gets used.
    # Its y values are never read off, so the tick labels go and one word says
    # what it is. Log scale stays -- the distribution spans two decades, and on
    # a linear axis every bin but the two end ones would read as zero.
    ax_sharp.set_yscale("log")
    ax_sharp.set_xlim(0, 1)
    style_axis(ax_sharp)
    ax_sharp.grid(False)
    ax_sharp.set_yticks([])
    ax_sharp.tick_params(labelsize=7.5, length=2.5, width=0.7)
    ax_sharp.set_xticks([0, 0.5, 1.0])
    ax_sharp.text(0.02, 0.88, "how often each probability is used",
                  transform=ax_sharp.transAxes, fontsize=7.5, color=RULE,
                  ha="left", va="top", style="italic")

    save(fig, stem)
    plt.close(fig)

    print(f"\n{'event':>16} {'n':>9} {'base':>7} {'BS':>8} {'REL':>8} "
          f"{'RES':>8} {'UNC':>8} {'BSS':>7}")
    for s in summary:
        skill = 1.0 - s["brier"] / s["unc"] if s["unc"] > 0 else float("nan")
        print(f"{s['label']:>16} {s['n']:>9,} {s['base_rate']:>7.3f} "
              f"{s['brier']:>8.4f} {s['rel']:>8.4f} {s['res']:>8.4f} "
              f"{s['unc']:>8.4f} {skill:>7.3f}")
    print("\nREL -> 0 is calibrated; RES higher is better; "
          "BSS = 1 - BS/UNC is skill against the base rate.")


if __name__ == "__main__":
    plot_reliability(load())
