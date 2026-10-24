"""Error rates of the hazard indicator configurations, July 2021 event.

Reads out/binary_validation_metrics.csv (surveyed depths) and writes two companion
figures:

  fig_miss_rate_by_damage   missed buildings by observed MBIE damage class
  fig_false_alarm_rate      false alarms by assigned hazard class

Both are plotted as counts. The class size is carried in the legend rather than by
normalising the axis, which matters most in the miss figure: its three damage classes
differ in size (179 / 356 / 112), so bar heights there are NOT comparable across
classes without reading n off the legend.

The two are split rather than combined because the error types live in disjoint
classes: misses can only occur among damaged buildings (L1-L3) and false alarms only
among undamaged ones (L0), which is why the metrics file reports zero false alarms in
every damage class above 0. Damage severity is ordinal, so it is encoded as a
single-hue sequential ramp (light to dark with increasing severity) rather than as
unordered categories. The ramp is monotonic in lightness, which keeps it legible
under colour-vision deficiency and in greyscale print.
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.forecast_runner.forecast_runner_helper import (  # noqa: E402
    SCORED_CONFIGURATIONS,
)

METRICS_FILE = os.path.join(PROJECT_ROOT, "out", "binary_validation_metrics.csv")
PLOT_DIR = os.path.join(PROJECT_ROOT, "out")

# Sequential blue ramp, light -> dark with increasing severity.
# Verified: OKLab lightness 0.815 / 0.633 / 0.439 (monotonic); adjacent normal-vision
# dE 19.1 and 19.8; deutan dE 18.9 / 19.5; protan dE 17.9 / 20.5; relative luminance
# 0.549 / 0.258 / 0.083 so the classes stay ordered in greyscale.
DAMAGE_LEVELS = {
    1: {"label": "Slight (L1)", "color": "#9ecae1"},
    2: {"label": "Moderate (L2)", "color": "#4292c6"},
    3: {"label": "Severe (L3)", "color": "#08519c"},
}

# Assigned hazard class is a different ordinal variable from observed damage class, so
# it gets its own sequential ramp rather than reusing the blue -- one colour must not
# mean two things across a pair of companion figures.
# Verified: OKLab lightness 0.796 / 0.596 / 0.352 (monotonic); adjacent normal-vision
# dE 19.7 and 26.6; worst adjacent CVD dE 17.1; relative luminance 0.451 / 0.277 /
# 0.049; minimum dE 18.1 against any step of the blue ramp above, so the two ramps
# cannot be mistaken for each other.
# H1 is not plotted. It is the negative prediction -- _hazards_to_affected collapses
# it to not_affected, the same call that observed damage class 0 is the negative
# observation of -- so an H1 building is never flagged and cannot be a false alarm.
# plot_false_alarm_rate asserts the column really is empty rather than trusting that
# argument, since dropping a class that turned out to be populated would hide data.
HAZARD_CLASSES = {
    "H2": "#e6550d",
    "H3+": "#7f2704",
}
EMPTY_HAZARD_CLASS = "H1"

# No total is drawn: as an aggregate of the two classes it repeats them, and with H3+
# a small minority everywhere it would sit almost on top of the H2 bar. The totals
# stay in the printed table and in binary_validation_metrics.csv.

# The metrics CSV publishes the paper name in its `indicator` column, so this is
# an ORDER and a FILTER, not a mapping -- the bar labels are the index itself.
# Filtering matters: the file also holds diagnostic rows, which are exactly the
# rows with no paper name.
#
# building_hazard_level is one of them, and would be worth excluding even if it
# had one: it is identical to BuildingOnly-Max in the metrics file (same
# hits/misses/false alarms and same per-damage-level breakdown), so plotting it
# would repeat that group.
#
# TAKEN FROM THE SCORING REGISTRY, not restated here: a list of its own went
# stale the moment a configuration was added (WetMean, 2026-08-21) and dropped
# those rows from both figures without saying so, since a name the CSV has and
# the list does not simply never matches.
CONFIGURATION_ORDER = tuple(name for name, _ in SCORED_CONFIGURATIONS.values())

INK = "#1a1a1a"
RULE = "#4d4d4d"

# Below the xtick.labelsize the rcParams set, and set here rather than there so
# the y axis keeps the larger size. Configurations with names as long as
# "BuildPropStreet-WetMean" crowd each other at the default. Applied in
# style_axis, so both figures move together, and style_axis also breaks the name
# at its hyphen so a label is two short lines rather than one long one.
XTICK_LABEL_SIZE = 8

# The axis is sized BY THE GROUP COUNT rather than fixed: six configurations fit
# the 7.6in figure these were drawn at, nine do not, and it is the tick label
# that runs out of room first. The constants reproduce that 7.6in at six, so
# widening the ablation widens the figure instead of squeezing it.
GROUP_WIDTH_IN = 1.12
AXIS_MARGIN_IN = 0.88

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


def figure_size(n_configurations, height=4.4):
    """Figure size for a bar chart of ``n_configurations`` groups."""
    return (AXIS_MARGIN_IN + GROUP_WIDTH_IN * n_configurations, height)


def load(scenario_id=0, model="surveyed", damage_threshold=1):
    df = pd.read_csv(METRICS_FILE)
    df = df[
        (df["scenario_id"] == scenario_id)
        & (df["model"] == model)
        & (df["damage_threshold"] == damage_threshold)
    ]
    order = [i for i in CONFIGURATION_ORDER if i in set(df["indicator"])]
    return df.set_index("indicator").loc[order]


def label_bars(ax, bars, values, fmt="{:.1f}"):
    for bar, value in zip(bars, values):
        ax.annotate(
            fmt.format(value) if value > 0 else "0",
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
            zorder=4,
        )


def style_axis(ax, x, labels, ymax, tick_step=10):
    ax.set_xlabel("Hazard indicator configuration", labelpad=34)
    ax.set_xticks(x)
    # Broken at the hyphen: the context and the statistic are two different
    # things anyway, and stacking them keeps a nine-group axis readable without
    # rotating the labels or shrinking them further.
    ax.set_xticklabels([label.replace("-", "\n") for label in labels],
                       fontsize=XTICK_LABEL_SIZE)
    ax.set_xlim(-0.55, len(x) - 0.45)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_step))

    ax.yaxis.grid(True, color="#cccccc", linewidth=0.5, linestyle=(0, (3, 3)), zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=RULE, labelcolor=INK, length=3.5, width=0.8)
    ax.tick_params(axis="x", length=0)

    # Annotate the ordering of the x axis: configurations are arranged left to right by
    # the extent of the context the depth is aggregated over (footprint -> parcel -> street).
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    y_arrow = -0.175
    ax.annotate(
        "",
        xy=(x[-1] + 0.42, y_arrow),
        xytext=(x[0] - 0.42, y_arrow),
        xycoords=trans,
        textcoords=trans,
        arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.4",
                        color=RULE, linewidth=0.9,
                        shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.text(
        (x[0] + x[-1]) / 2,
        y_arrow + 0.022,
        "expanding spatial context",
        transform=trans,
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
        color=INK,
        clip_on=False,
    )


def save(fig, stem):
    for ext in ("png", "pdf"):
        out = os.path.join(PLOT_DIR, f"{stem}.{ext}")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"saved {out}")


def plot_miss_rate(df):
    levels = list(DAMAGE_LEVELS)
    labels = list(df.index)
    x = np.arange(len(df), dtype=float)

    misses = np.array([df[f"misses_damage_{lvl}"].to_numpy() for lvl in levels])
    totals = np.array([df[f"n_damage_{lvl}"].to_numpy() for lvl in levels])
    rates = 100.0 * misses / totals

    fig, ax = plt.subplots(figsize=figure_size(len(df)))

    width = 0.25
    for row, lvl in enumerate(levels):
        bars = ax.bar(
            x + (row - 1) * width,
            misses[row],
            width=width * 0.9,
            color=DAMAGE_LEVELS[lvl]["color"],
            edgecolor=RULE,
            linewidth=0.6,
            label=f"{DAMAGE_LEVELS[lvl]['label']}, $n$ = {int(totals[row][0])}",
            zorder=3,
        )
        label_bars(ax, bars, misses[row], fmt="{:.0f}")

    # Counts, with each class's n in the legend. The classes are unequal in size, so
    # a bar is only interpretable against its own n -- see the module docstring.
    ax.set_ylabel("Missed buildings (count)")
    style_axis(ax, x, labels, ymax=np.ceil(misses.max() / 20) * 20 + 30,
               tick_step=20)

    ax.legend(
        title="Observed damage class",
        title_fontsize=9.5,
        loc="upper right",
        frameon=True,
        framealpha=1.0,
        edgecolor=RULE,
        borderpad=0.7,
        handlelength=1.4,
        handleheight=1.0,
    ).get_frame().set_linewidth(0.6)

    fig.tight_layout()
    save(fig, "fig_miss_rate_by_damage")

    return pd.DataFrame(
        {
            DAMAGE_LEVELS[lvl]["label"]: [
                f"{m}/{t} ({r:.1f}%)"
                for m, t, r in zip(misses[i], totals[i], rates[i])
            ]
            for i, lvl in enumerate(levels)
        },
        index=labels,
    )


def plot_false_alarm_rate(df):
    """False alarm rate, split by the hazard class the FIS assigned.

    All false alarms fall in damage class 0 -- a flagged building with L1-L3 damage
    is a hit, not a false alarm -- so the breakdown that carries information is by
    PREDICTED class rather than observed one. H1 is left off the axis: it is the
    negative prediction, equivalent to observed damage class 0, so it cannot be a
    false alarm. State that in the caption so its absence reads as a property of the
    scoring rather than a gap in the data.

    The per-class counts come from false_alarm_hazard_classes.csv, which
    error_analysis_by_configuration.py writes; the counts CSV carries only totals.
    """
    classes = list(HAZARD_CLASSES)
    labels = list(df.index)
    x = np.arange(len(df), dtype=float)

    split_path = os.path.join(PLOT_DIR, "false_alarm_hazard_classes.csv")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"{split_path} not found; run exploratory/error_analysis_by_configuration.py "
            "first -- the hazard-class split is not in binary_validation_metrics.csv"
        )
    split = pd.read_csv(split_path).set_index("configuration").loc[labels]

    # Plotted as counts, not rates: every configuration is scored against the same
    # 1597 undamaged buildings, so a single denominator stated once in the corner
    # carries what a percentage axis would, and counts stay comparable with the
    # contingency table. The miss figure cannot do this -- its three damage classes
    # have very different sizes, so only a rate is comparable across them.
    n_undamaged = df["n_damage_0"].to_numpy()
    counts = np.array([split[f"n_{name}"].to_numpy() for name in classes])
    totals = counts.sum(axis=0)

    empty = split[f"n_{EMPTY_HAZARD_CLASS}"].to_numpy()
    if empty.any():
        raise AssertionError(
            f"{EMPTY_HAZARD_CLASS} is off the axis as the negative prediction, but "
            f"carries false alarms: {dict(zip(labels, empty))}. Plot it or find out "
            "why a not-affected prediction was scored as a false alarm."
        )

    published = df["false_alarms_damage_0"].to_numpy()
    if not np.array_equal(totals, published):
        raise AssertionError(
            "false alarm totals disagree between the per-configuration record and "
            f"binary_validation_metrics.csv: {totals} vs {published}. Regenerate both."
        )

    fig, ax = plt.subplots(figsize=figure_size(len(df)))

    # Centred on the tick whatever the class count, so dropping H1 recentres the
    # group instead of leaving it hanging left of its label.
    width = 0.3
    offset = (len(classes) - 1) / 2
    for row, name in enumerate(classes):
        bars = ax.bar(
            x + (row - offset) * width,
            counts[row],
            width=width * 0.9,
            color=HAZARD_CLASSES[name],
            edgecolor=RULE,
            linewidth=0.6,
            label=name,
            zorder=3,
        )
        label_bars(ax, bars, counts[row], fmt="{:.0f}")

    ax.set_ylabel("Undamaged buildings falsely flagged (count)")
    style_axis(ax, x, labels, ymax=np.ceil(counts.max() / 100) * 100 + 200,
               tick_step=200)

    # Note right, legend left: the tallest group sits on the right of this axis, so
    # the legend would land on top of it in the usual upper-right corner.
    ax.text(
        0.985,
        0.97,
        f"Undamaged class (L0), $n$ = {int(n_undamaged[0]):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color=INK,
    )

    ax.legend(
        title="Assigned hazard class",
        title_fontsize=9.5,
        loc="upper left",
        frameon=True,
        framealpha=1.0,
        edgecolor=RULE,
        borderpad=0.7,
        handlelength=1.4,
        handleheight=1.0,
    ).get_frame().set_linewidth(0.6)

    fig.tight_layout()
    save(fig, "fig_false_alarm_rate")

    return pd.DataFrame(
        {name: [f"{c}/{n} ({100.0 * c / n:.1f}%)"
                for c, n in zip(counts[i], n_undamaged)]
         for i, name in enumerate(classes)},
        index=labels,
    ).assign(Total=[f"{t}/{n} ({100.0 * t / n:.1f}%)"
                    for t, n in zip(totals, n_undamaged)])


def main():
    df = load()

    stray = df[[c for c in df.columns if c.startswith("false_alarms_damage_")
                and c != "false_alarms_damage_0"]].to_numpy().sum()
    if stray:
        raise AssertionError(
            f"expected all false alarms in damage class 0, found {stray} elsewhere"
        )

    print(plot_miss_rate(df).to_string(), end="\n\n")
    print(plot_false_alarm_rate(df).to_string())


if __name__ == "__main__":
    main()
