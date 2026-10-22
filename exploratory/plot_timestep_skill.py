"""Skill through an event, timestep by timestep, for one risk-module configuration.

Reads simulated_validation_metrics.csv -- one row per (issue, valid) pair
written by forecast_runner.write_results_simulated -- and writes

  fig_qwk_by_timestep   the DETERMINISTIC score, QWK, against time
  fig_rps_by_timestep   the PROBABILISTIC score, RPS, against time, with the
                        deterministic forecast's own RPS beside it

WHY TWO FIGURES AND NOT ONE PANEL. QWK and RPS are not two views of one
quantity and must not share an axis (they do not share units, a direction of
"good", or a scale). QWK compares the class of the ensemble-MEAN indicator with
the class the simulated map gets: the ensemble is collapsed before the
comparison, which is the reading a warning that must name a single class makes,
and higher is better. RPS scores the whole exceedance distribution against that
same observed class, so it is the only one of the two that can see the
ensemble's spread, and LOWER is better. Putting them on one pair of axes would
be the dual-axis mistake.

WHY RPS IS DRAWN AGAINST A SECOND LINE. An RPS is an absolute score whose
magnitude depends on how many buildings were wet at that timestep; 0.01 at the
peak and 0.01 in the tail are not comparable, so a single RPS trace says almost
nothing on its own. The second line is the RPS of the deterministic forecast --
the same ensemble-mean class the QWK figure scores, treated as a distribution
with all its mass on one class. The gap between the two lines IS the ranked
probability skill score, plotted rather than reduced to a number, and it is the
one comparison the method's claim rests on: does reading the ensemble
probabilistically beat collapsing it first.

WHY SMALL MULTIPLES. Events differ in length, in forcing regime and in base
rate, so their traces are not comparable point by point along a shared x. One
panel per event keeps each trace readable and keeps the y axis shared, which is
what makes the panels comparable at all. Eight events would also be eight
series in one axes, which no categorical palette should be asked to carry.

    python exploratory/plot_timestep_skill.py

The configuration drawn is CONFIGURATION below, defaulting to the one the
system reports operationally; every configuration the sweep scored is in the
file under its own prefix.
"""

import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.impact_module.risk_module import REPORTED_PREFIX  # noqa: E402

METRICS_FILE = os.path.join(PROJECT_ROOT, "simulated_validation_metrics.csv")
PLOT_DIR = os.path.join(PROJECT_ROOT, "out")

# Which configuration's scores are drawn. Any of
# forecast_runner.CONFIGURATION_PREFIXES; the sweep writes all six.
CONFIGURATION = REPORTED_PREFIX

# Which lead time. None takes the shortest lead present, which is what a sweep
# run at forecast_horizon 1 produces -- and mixing leads in one trace would put
# two different forecasts at the same x.
LEAD_H = None

# Which run produced the rows. The simulated validation only ever writes
# flow_matching, but the column exists and an unfiltered read would silently
# pool anything else added later.
MODEL = "flow_matching"

# Categorical slots 1 and 2 of the reference palette, unmodified and in slot
# order. Blue is the ENSEMBLE and orange the DETERMINISTIC reading, and that
# pairing is held across figures: the orange line in the RPS figure is the same
# forecast the QWK figure scores, so a reader can carry the colour between them.
# Line style differs as well, so identity never rests on colour alone -- which
# is what keeps the figure readable in greyscale print and for CVD readers.
ENSEMBLE_COLOR = "#2a78d6"
DETERMINISTIC_COLOR = "#eb6834"

# Inches. Wide enough for the longest title and the legend at 10 pt serif; the
# panels expand to fill whatever is left over.
MIN_FIGURE_WIDTH = 6.9

INK = "#1a1a1a"
RULE = "#4d4d4d"
GRID = "#cccccc"
MUTED = "#898781"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,  # embed as TrueType so the PDF is editable/searchable
    "ps.fonttype": 42,
})


def columns(configuration=CONFIGURATION):
    """The metric columns this configuration writes, as written by compute_metrics."""
    return {
        "qwk": f"building_{configuration}_qwk",
        "rps": f"building_{configuration}_rps",
        "reference": f"building_{configuration}_rps_ensemble_mean",
        "n": f"building_{configuration}_n",
    }


def load(configuration=CONFIGURATION, lead_h=LEAD_H, model=MODEL):
    """One frame per event, indexed by hours from that event's first scored step.

    The x axis is hours from the START OF THE SCORED EVENT rather than a
    timestamp: the events are decades apart on the calendar and their absolute
    times are meaningless side by side, while the hour within the event is
    exactly what the phase-dependent failure modes are read against.
    """
    df = pd.read_csv(METRICS_FILE)
    wanted = columns(configuration)
    missing = [c for c in wanted.values() if c not in df.columns]
    if missing:
        raise KeyError(
            f"{os.path.basename(METRICS_FILE)} has no {missing}. Its "
            f"configurations are "
            f"{sorted({c[len('building_'):-len('_qwk')] for c in df.columns if c.endswith('_qwk')})}")

    if model is not None:
        df = df[df["model"] == model]
    lead = df["lead_h"].min() if lead_h is None else lead_h
    df = df[df["lead_h"] == lead]
    if df.empty:
        raise SystemExit(f"no {model} rows at lead {lead} h in {METRICS_FILE}")

    # ISO8601 rather than one format: files written before
    # write_results_simulated formatted its stamps carry midnight steps as a
    # bare date, so the column has two shapes in it and a fixed format raises
    # on the older rows.
    df = df.assign(valid_time=pd.to_datetime(df["valid_time"], format="ISO8601"))
    events = {}
    for scenario, frame in df.groupby("scenario_id"):
        frame = frame.sort_values("valid_time")
        start = frame["valid_time"].iloc[0]
        frame = frame.assign(
            hours=(frame["valid_time"] - start).dt.total_seconds() / 3600.0)
        events[int(scenario)] = frame
    return events, float(lead)


def grid(n_events):
    """Panel layout: one row while it fits, two rows beyond four events."""
    if n_events <= 4:
        return 1, n_events
    return 2, int(np.ceil(n_events / 2))


def style(ax, title, ylim, show_y, show_x, ylabel):
    ax.set_ylim(*ylim)
    ax.set_title(title, color=INK, pad=6)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=RULE, labelcolor=INK, length=3)
    ax.tick_params(axis="y", labelleft=show_y)
    ax.tick_params(axis="x", labelbottom=show_x)
    if show_y:
        ax.set_ylabel(ylabel, color=INK)


X_LABEL = "Hours from the start of the event"


def heading(fig, title, subtitle, legend_rows=0):
    """Title, italic subtitle and the single x label, placed in INCHES.

    Fractional y positions drift with the figure height, so a pair that clears
    on a one-row grid collides on a two-row one (or floats away from it). The
    offsets below are absolute, and the returned rect is what tight_layout must
    keep clear so the panels never grow into them.

    The x label is drawn ONCE for the figure rather than per panel: every panel
    shares the axis, so a label under each column is the same sentence four
    times and the reader has to check they really are the same.
    """
    height = fig.get_size_inches()[1]
    fig.suptitle(title, color=INK, fontsize=11, y=1 - 0.24 / height)
    fig.text(0.5, 1 - 0.50 / height, subtitle, ha="center", color=MUTED,
             fontsize=9, style="italic")
    label_y = (0.30 * legend_rows + 0.06) / height
    fig.text(0.5, label_y, X_LABEL, ha="center", va="bottom", color=INK,
             fontsize=11)
    return (0, label_y + 0.22 / height, 1, 1 - 0.66 / height)


def panels(events, ylim, ylabel, draw):
    """A figure of small multiples, one panel per event, with a shared y axis.

    ``draw(ax, frame)`` puts the marks in; everything else -- the grid, which
    panels wear tick labels, the axis titles -- is the same for both figures and
    lives here so the two cannot drift apart.
    """
    rows, cols = grid(len(events))
    # The title, the italic subtitle and the two-column legend are all wider
    # than one panel, so the figure has a floor independent of the panel count
    # -- at one event the panel simply stretches to fill it. Sized without a
    # floor, a single-event figure clips its own title, which is invisible in
    # the summary printout and only shows up when the PNG is opened.
    # sharex, not just sharey: the events differ in length, so with independent
    # x axes the panels whose tick labels are hidden would be read against a
    # neighbour's ticks and silently misread by up to a third of an event. One
    # shared axis means a shorter event simply stops early, which is the truth.
    fig, axes = plt.subplots(
        rows, cols, figsize=(max(2.55 * cols + 0.6, MIN_FIGURE_WIDTH),
                             2.45 * rows + 0.9),
        sharex=True, sharey=True, squeeze=False,
    )
    flat = axes.ravel()
    for index, (scenario, frame) in enumerate(sorted(events.items())):
        ax = flat[index]
        draw(ax, frame)
        style(
            ax, f"Event {scenario}", ylim,
            show_y=(index % cols == 0),
            # The bottom panel of each column, which is not the last row when
            # the grid is ragged.
            show_x=(index >= len(events) - cols),
            ylabel=ylabel,
        )
    for ax in flat[len(events):]:
        ax.set_visible(False)
    return fig, flat


def qwk_figure(events, configuration):
    """QWK against time. One series, so the title names it and no legend is drawn."""
    column = columns(configuration)["qwk"]
    values = np.concatenate([f[column].to_numpy(dtype=float) for f in events.values()])
    # Anchored at 1.0 above and 0.0 below: perfect agreement and CHANCE are both
    # fixed, meaningful marks, and keeping chance in view is what stops a run
    # where every event scores 0.85-0.95 from being drawn as though those were
    # far apart. The floor extends below 0 only if something actually scored
    # worse than chance, which is a finding and must not be cropped out.
    floor = min(0.0, np.floor(np.nanmin(values) * 10) / 10)

    def draw(ax, frame):
        ax.plot(frame["hours"], frame[column], color=DETERMINISTIC_COLOR,
                linewidth=1.8, solid_capstyle="round", zorder=3)

    fig, _ = panels(events, (floor, 1.005), "QWK", draw)
    rect = heading(
        fig, f"Hazard-class agreement per forecast step ({configuration})",
        "Ensemble-mean class vs the simulated map; 1 is exact agreement, "
        "0 is chance")
    fig.tight_layout(rect=rect)
    return fig


def rps_figure(events, configuration):
    """RPS against time, with the deterministic forecast's RPS beside it.

    Two series, so a legend is always present; the gap between them is the
    quantity the caption has to name, since it is the skill score.
    """
    names = columns(configuration)
    values = np.concatenate([
        np.concatenate([f[names["rps"]].to_numpy(dtype=float),
                        f[names["reference"]].to_numpy(dtype=float)])
        for f in events.values()])
    # Zero is the meaningful floor -- a perfect probabilistic forecast scores
    # exactly 0 -- and the top follows the data, since RPS has no ceiling worth
    # anchoring to and its magnitude changes by an order of magnitude between
    # configurations.
    top = float(np.nanmax(values)) * 1.08 or 1.0

    def draw(ax, frame):
        ax.plot(frame["hours"], frame[names["reference"]],
                color=DETERMINISTIC_COLOR, linewidth=1.6,
                linestyle=(0, (4, 1.6)), solid_capstyle="round", zorder=3,
                label="Deterministic (ensemble-mean class)")
        ax.plot(frame["hours"], frame[names["rps"]], color=ENSEMBLE_COLOR,
                linewidth=1.8, solid_capstyle="round", zorder=4,
                label="Ensemble (exceedance distribution)")

    fig, flat = panels(events, (0.0, top), "RPS", draw)
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.6, columnspacing=2.0,
               labelcolor=INK)
    rect = heading(
        fig, f"Ranked probability score per forecast step ({configuration})",
        "Lower is better; the gap between the lines is the skill of the "
        "probabilistic reading",
        legend_rows=1)
    fig.tight_layout(rect=rect)
    return fig


def summary(events, configuration):
    """Per-event means, and the POOLED skill score.

    RPSS does not average: pooling timesteps means summing the numerators and
    the denominators separately, weighted by the buildings each step scored,
    which is why the raw scores and n are carried in the file rather than the
    ratio alone. Averaging the per-step RPSS instead would let a quiet step
    with a near-zero denominator dominate the number.
    """
    names = columns(configuration)
    rows = []
    for scenario, frame in sorted(events.items()):
        weight = frame[names["n"]].to_numpy(dtype=float)
        rps = float(np.sum(frame[names["rps"]] * weight) / weight.sum())
        reference = float(np.sum(frame[names["reference"]] * weight) / weight.sum())
        rows.append({
            "event": scenario,
            "steps": len(frame),
            "QWK": frame[names["qwk"]].mean(),
            "RPS": rps,
            "RPS (det.)": reference,
            "RPSS": np.nan if reference <= 0 else 1.0 - rps / reference,
        })
    table = pd.DataFrame(rows).set_index("event")
    return table


def save(fig, stem):
    os.makedirs(PLOT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(PLOT_DIR, f"{stem}.{ext}")
        fig.savefig(out, dpi=600)
        print(f"saved {out}")


def main():
    events, lead = load()
    save(qwk_figure(events, CONFIGURATION), "fig_qwk_by_timestep")
    save(rps_figure(events, CONFIGURATION), "fig_rps_by_timestep")

    table = summary(events, CONFIGURATION)
    print(f"\n{CONFIGURATION}, lead {lead:.0f} h, {len(events)} events, "
          f"{int(table['steps'].sum())} timesteps:")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
