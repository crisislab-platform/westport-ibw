"""Per-event forecast skill: QWK and RPS, one row per event.

Replaces the per-event metrics table, which is unreadable as a grid of numbers:
the eye cannot rank ten events on several columns at once, and the one thing the
table is FOR -- which events are hard, and whether the ensemble helps on them --
is exactly what a grid of decimals hides.

The figure is a row per event, ordered by QWK, and two panels sharing that row
order:

    QWK | RPS

The two panels are THE SAME FORECAST SCORED TWO WAYS, which is why they belong
in one figure rather than two. QWK is the DETERMINISTIC reading -- the class of
the ensemble-mean indicator against the simulated map's class -- and RPS is the
PROBABILISTIC one, the full exceedance distribution against that same observed
class. An event can be easy for one and hard for the other, and the pairing is
what makes that visible.

They point in OPPOSITE DIRECTIONS (QWK higher is better, RPS lower is better)
and carry different units, so they get separate value axes rather than a shared
one, and each panel says its direction under its own axis. Reading them as one
0-1 scale would be wrong, so the figure does not offer that reading.

The RPS panel draws the ENSEMBLE ALONE, one mark per event on an axis from zero,
so it reads as an absolute score against a perfect probabilistic forecast rather
than as a per-event comparison. The deterministic reference is still read from
the file -- main() prints the pooled RPSS over all events, and that ratio needs
it -- but it is not drawn: what this figure ranks is how hard each event was,
and a second mark on every row asked the reader to do a different job.

Hue means WHICH READING: orange is the deterministic score (QWK), blue the
ensemble (RPS). The blue is the one fig_miss_rate_by_damage uses for its
sequential ramp, so a reader moving between the paper's figures is not asked to
relearn it.

When the source file still carries per-class hit rate and CSI, three dumbbell
panels are appended -- POD against CSI per hazard class, where the span is the
false-alarm penalty. The current writer does not emit them; see load().

Sorted by QWK descending, so the vertical axis carries information rather than
scenario-id order, and the RPS panel is read against the overall ranking -- a
row whose RPS sits out of step with its QWK position is an event where the two
readings disagree.

CLASS NAMES: the source CSV predates the 2026-08-18 move to H1/H2/H3+ bands, so
its tiers 2 and 3 still mean H2-H3 (0.30-1.20 m) and H4+ (>1.20 m). CLASS_LABELS
below is the single place that says so -- after the simulated validation is rerun
under the current bands, change it to ("H1", "H2", "H3+") and nothing else moves.
"""
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.impact_module.risk_module import REPORTED_PREFIX  # noqa: E402

METRICS_FILE = os.path.join(PROJECT_ROOT, "simulated_validation_metrics.csv")
PLOT_DIR = os.path.join(PROJECT_ROOT, "out")

# The metrics carry one QWK and one RPS set PER CONFIGURATION (see
# forecast_runner.compute_metrics), so a figure has to say which one it draws.
# Set CONFIGURATION to any of forecast_runner.CONFIGURATION_PREFIXES; the
# figure's file name does NOT change with it, so change the stem in main() too
# if two configurations are wanted side by side.
#
# BuildProp-Max, which is deliberately NOT the operationally reported
# configuration (REPORTED_PREFIX, currently buildpropstreet_max). It is the one
# the verification pairs and exceedance files are written for
# (out/verification_pairs_buildprop_max.csv), so this figure and
# fig_reliability score the same configuration. main() prints a note whenever
# the two differ, so the divergence cannot go unnoticed.
QWK_SUFFIX = "qwk"
CONFIGURATION = "buildprop_max"

# The probabilistic pair, both off the same prefix: the ensemble's RPS over the
# exceedance distribution, and the RPS of the deterministic reading used as the
# RPSS reference. Only the first is DRAWN; the reference is carried for the
# pooled RPSS main() prints, so a file may have one without the other.
RPS_SUFFIX = "rps"
DET_RPS_SUFFIX = "rps_ensemble_mean"

# The CSV names the three tiers by warning colour; these are tiers 1/2/3, and
# what those tiers MEAN depends on the bands the run used -- see the module
# docstring. Ordered safe -> severe.
CLASS_KEYS = ("Yellow", "Orange", "Red")
CLASS_LABELS = ("H1", "H2-H3", "H4+")

# Palette: dataviz skill defaults, validated all-pairs against a white surface
# (lightness band / chroma floor / CVD separation / normal-vision floor /
# contrast all pass; worst all-pairs CVD dE 18.9 deutan, worst normal-vision
# dE 19.2, minimum contrast 3.64:1).
#
# The blue is the sequential ramp fig_miss_rate_by_damage uses (steps 5 and 7 of
# ColorBrewer Blues); one blue means one thing across the paper's figures.
BLUE_MID = "#4292c6"
BLUE_DARK = "#08519c"

# POD and CSI are ONE HUE IN TWO SHADES, not two hues: they are an ordered pair
# on one measure, and a shade step says "same thing, further along" where a
# second hue would say "different thing". QWK is a different measure, so it
# takes the next categorical slot rather than a third step of the blue.
POD_COLOR = BLUE_MID
CSI_COLOR = BLUE_DARK
QWK_COLOR = "#eb6834"   # categorical slot 2
# Hue = which reading: orange marks the deterministic score (QWK), blue the
# ensemble (RPS).
ENSEMBLE_COLOR = BLUE_MID
# The connector is chrome, not identity -- it spans the two marks and must not
# compete with either, so it stays in the neutral rule colour.
LINK_COLOR = "#c3c2b7"

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,  # embed as TrueType so the PDF is editable/searchable
    "ps.fonttype": 42,
})


def qwk_column(df):
    """This figure's configuration's QWK column, or the file's only one.

    Files written before the metrics carried all six configurations have a
    single QWK column under a different name; falling back to it keeps an older
    sweep plottable rather than failing on a name that was correct when it was
    written.
    """
    wanted = f"building_{CONFIGURATION}_{QWK_SUFFIX}"
    if wanted in df.columns:
        return wanted
    candidates = [c for c in df.columns
                  if c.startswith("building_") and c.endswith(QWK_SUFFIX)]
    if len(candidates) == 1:
        print(f"{os.path.basename(METRICS_FILE)} has no {wanted}; "
              f"drawing its only QWK column, {candidates[0]}")
        return candidates[0]
    raise KeyError(
        f"{os.path.basename(METRICS_FILE)} has no {wanted}; its QWK columns "
        f"are {candidates}")


def load():
    """One row per event: QWK, and POD/CSI per class WHERE THE FILE HAS THEM.

    The per-class hit rate and CSI left the schema when the fuzzy engine did:
    the current writer publishes one QWK plus the probabilistic scores (RPS,
    RPSS), and no per-class contingency. The class panels are therefore drawn
    only if the file still carries them -- an older run, or a writer configured
    to emit them -- and the figure is one panel wide otherwise. Silently
    plotting nothing there would be worse than either.

    Each event contributes many timesteps (35-49 here), all at lead 0. They are
    averaged UNWEIGHTED, so the number reported for an event is the skill of its
    typical timestep rather than of its worst or its busiest -- a long quiet
    event and a short violent one count the same. Whether that is the right
    summary is a question for the caption, not for this function; what matters
    here is that it is one summary, stated.

    The unweighted mean is also the CORRECT pooling for RPS here, not merely the
    consistent one: every timestep scores the same building population (the `_n`
    column is constant across the file), so weighting by n would multiply every
    term by the same number. Were that to stop being true -- a configuration
    that drops buildings on some timesteps -- RPS would need an n-weighted mean
    while QWK kept this one, and the two panels would no longer be pooled alike.
    RPSS is deliberately NOT averaged here: it is a ratio, and the mean of
    per-timestep ratios is not the pooled skill. main() pools the two RPS sums
    into one skill score instead, which is the ratio of the means.
    """
    df = pd.read_csv(METRICS_FILE)
    prefix = qwk_column(df)[:-len(QWK_SUFFIX)]
    columns = {"QWK": f"{prefix}{QWK_SUFFIX}"}

    if f"{prefix}{RPS_SUFFIX}" in df.columns:
        columns["RPS"] = f"{prefix}{RPS_SUFFIX}"
    else:
        print(f"{os.path.basename(METRICS_FILE)} carries no {RPS_SUFFIX}; "
              f"drawing QWK alone")
    # Read but NOT drawn: the deterministic reference is the denominator of the
    # pooled RPSS main() prints, and nothing else here needs it.
    if f"{prefix}{DET_RPS_SUFFIX}" in df.columns:
        columns["RPS (det.)"] = f"{prefix}{DET_RPS_SUFFIX}"

    per_class = {}
    for key, label in zip(CLASS_KEYS, CLASS_LABELS):
        per_class[f"POD {label}"] = f"{prefix}hit_rate_{key}"
        per_class[f"CSI {label}"] = f"{prefix}csi_{key}"
    if all(column in df.columns for column in per_class.values()):
        columns.update(per_class)
    else:
        print(f"{os.path.basename(METRICS_FILE)} carries no per-class hit "
              f"rate/CSI; omitting the class panels")

    events = df.groupby("scenario_id")[list(columns.values())].mean()
    events.columns = list(columns)
    events["n_steps"] = df.groupby("scenario_id").size()
    # Best at the top: the y axis then carries the overall ranking, and the
    # class panels are read against it.
    return events.sort_values("QWK", ascending=True)


def style(ax, y, labels, xlim, title, show_y, xticks=None):
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(y) - 0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    # NOT set_yticklabels([]): sharey means the four panels share one tick
    # formatter, so emptying it here would blank the labelled panel as well.
    ax.tick_params(axis="y", labelleft=show_y)
    ax.set_title(title, color=INK, pad=8)
    # Three ticks, not five: panels side by side put one panel's right tick
    # next to the next panel's left tick, and at five they collide. The 0-1
    # panels name their three explicitly; RPS has no fixed range, so it asks
    # for three chosen ticks instead.
    if xticks is None:
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    else:
        ax.set_xticks(xticks)

    # Solid hairlines: a dashed rule reads as data at this weight, and there are
    # already two mark shapes in the panel competing for the same ink.
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY, length=3, width=0.8)
    ax.tick_params(axis="y", length=0)


def plot(events):
    y = np.arange(len(events))
    labels = [f"{int(i)}" for i in events.index]
    qwk_xlim = (0.42, 1.05)

    # One panel per score present: QWK always, RPS and the three class panels
    # only if load() found their columns (see its docstring).
    has_rps = "RPS" in events.columns
    class_labels = [label for label in CLASS_LABELS
                    if f"POD {label}" in events.columns]
    n_panels = 1 + int(has_rps) + len(class_labels)

    # EVERY dimension here is in INCHES, converted to fractions only where
    # matplotlib insists. Fractional margins are what clipped the caption and
    # the legend when the second panel arrived: they scale the chrome with the
    # figure, when what the chrome actually needs is a fixed number of points
    # for a line of 9pt text. Sized in inches, the panel can grow by rows or by
    # panels and the caption still fits.
    row_pitch, top_in = 0.26, 0.62
    left_in, right_in, panel_in, wspace = 0.75, 0.12, 2.35, 0.18
    # Only the class panels put a legend under the figure; with QWK and RPS one
    # series each, the bottom margin holds nothing but the axis label.
    bottom_in = 1.05 if class_labels else 0.60

    height = top_in + bottom_in + row_pitch * len(events)
    # The gaps are wspace * the mean panel width, so they enter the total the
    # same way matplotlib will lay them out.
    width = (left_in + right_in
             + panel_in * (n_panels + wspace * (n_panels - 1)))

    fig, axes = plt.subplots(
        1, n_panels, figsize=(width, height), sharey=True, squeeze=False,
        gridspec_kw={"width_ratios": [1] * n_panels, "wspace": wspace},
    )
    axes = axes[0]

    # --- QWK: one series, so no legend -- the panel title names it ----------
    ax = axes[0]
    ax.scatter(events["QWK"], y, s=42, color=QWK_COLOR, zorder=3,
               edgecolor="white", linewidth=0.8)
    style(ax, y, labels, qwk_xlim, "QWK", show_y=True, xticks=(0.5, 0.75, 1.0))
    ax.set_ylabel("Event (scenario id)", color=INK, labelpad=6)
    ax.set_xlabel("higher is better", color=MUTED, fontsize=8.5,
                  style="italic", labelpad=5)

    # --- RPS: the ensemble's probabilistic score, one mark per event -------
    if has_rps:
        ax = axes[1]
        ensemble = events["RPS"].to_numpy()
        # From zero, unlike the QWK panel: RPS has a meaningful origin -- a
        # perfect probabilistic forecast -- so the distances from the left edge
        # are themselves readable, and starting somewhere convenient would
        # exaggerate the spread between events.
        top = float(np.nanmax(ensemble))
        rps_xlim = (-0.02 * top, top * 1.12)

        ax.scatter(ensemble, y, s=42, color=ENSEMBLE_COLOR, zorder=3,
                   edgecolor="white", linewidth=0.8)
        style(ax, y, labels, rps_xlim, "RPS (ensemble)", show_y=False)
        ax.set_xlabel("lower is better", color=MUTED, fontsize=8.5,
                      style="italic", labelpad=5)

    # --- the three class panels, when the file carried them ----------------
    for ax, label in zip(axes[1 + int(has_rps):], class_labels):
        pod, csi = events[f"POD {label}"].to_numpy(), events[f"CSI {label}"].to_numpy()
        # Drawn first and behind, so it reads as the span between the marks
        # rather than as a third series lying on top of them.
        ax.hlines(y, csi, pod, color=LINK_COLOR, linewidth=1.8, zorder=2)
        # A white ring on each mark, so a row where POD and CSI nearly coincide
        # still resolves as two marks instead of one muddy blob.
        ax.scatter(pod, y, s=42, color=POD_COLOR, zorder=3,
                   edgecolor="white", linewidth=0.8, label="POD")
        ax.scatter(csi, y, s=42, color=CSI_COLOR, zorder=4,
                   edgecolor="white", linewidth=0.8, label="CSI")
        style(ax, y, labels, qwk_xlim, label, show_y=False,
              xticks=(0.5, 0.75, 1.0))

    fig.subplots_adjust(left=left_in / width, right=1 - right_in / width,
                        top=1 - top_in / height, bottom=bottom_in / height)

    note = f"{CONFIGURATION}; ordered by QWK"
    if class_labels:
        # The dumbbell panels are the only ones that need a key, and their own
        # handles are the legend, so the entries cannot drift from the marks
        # they describe.
        handles, handle_labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles, handle_labels, loc="lower center", ncol=2, frameon=False,
            bbox_to_anchor=(0.5, 0.005), handletextpad=0.4, columnspacing=1.8,
            labelcolor=INK,
        )
        # Says what the gap between the two marks means, since that span is the
        # reason each pair is drawn as a dumbbell rather than two dot columns.
        note += ", the gap between POD and CSI is the false-alarm penalty"
    # The configuration belongs ON the figure: six of them are scored, the
    # marks do not say which, and the file name does not either.
    fig.text(0.5, 1 - 0.16 / height, note,
             ha="center", va="top", color=SECONDARY, fontsize=9, style="italic")
    return fig


def save(fig, stem):
    os.makedirs(PLOT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(PLOT_DIR, f"{stem}.{ext}")
        fig.savefig(out, dpi=600)
        print(f"saved {out}")


def main():
    if CONFIGURATION != REPORTED_PREFIX:
        print(f"drawing {CONFIGURATION}, which is NOT the reported "
              f"configuration ({REPORTED_PREFIX})")
    events = load()
    fig = plot(events)
    save(fig, "fig_event_skill")

    shown = events.drop(columns="n_steps").iloc[::-1]
    print(f"\n{len(events)} events, {int(events['n_steps'].sum())} timesteps "
          f"({events['n_steps'].min()}-{events['n_steps'].max()} per event), "
          f"mean over timesteps:")
    print(shown.round(4).to_string())
    if "RPS" in shown.columns and "RPS (det.)" in shown.columns:
        # Pooled over every timestep of every event, which is NOT the mean of
        # the per-event skills printed above -- stated here so the two numbers
        # are not read as a discrepancy.
        gain = 1.0 - events["RPS"].sum() / events["RPS (det.)"].sum()
        print(f"\npooled RPSS over all events: {gain:.3f} "
              f"(1 - sum RPS / sum RPS_det)")


if __name__ == "__main__":
    main()
