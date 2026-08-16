"""Calibration of the ensemble's tier probabilities against simulated truth.

The pipeline reports P(tier) as the share of ensemble members electing that
tier, and warns when an exceedance probability clears a threshold
(fuzzy_risk.final_risk). That makes the numbers ACT like probabilities, but
nothing so far has shown they ARE ones: a forecast of 0.7 is only a probability
if the event happens about 70% of the time when 0.7 is issued. This script is
that check.

The question it answers: are the ensemble's exceedance probabilities reliable,
or is the deep ensemble over-confident? Under-dispersion is the expected
pathology of both NWP and deep ensembles -- spread narrower than the error it
is meant to cover -- and it shows up here as a reliability curve flatter than
the diagonal, with the extreme bins pulled toward the middle.

Two events are scored, both taken from the exceedance axis the warning
decision actually reads:

  at_least_H2_H3   P(tier >= medium) vs. the simulated tier reaching medium
  H4_above         P(tier == severe) vs. the simulated tier being severe

Note the truth here is the reference SIMULATION's tier, not the surveyed
placards -- error_hotspots.csv pairs each building's ensemble probabilities
with the tier the reference run assigned it. So this measures whether the
surrogate ensemble is calibrated against the simulator it emulates, which is
the question a deep ensemble's spread can actually answer. It says nothing
about the simulator's own error against reality; that is the binary validation's
job (see error_depth_analysis.py).

Writes, all under out/:
  reliability_curve.csv          per-bin forecast/observed frequencies + counts
  reliability_metrics.csv        Brier, its decomposition, BSS, bootstrap CIs
  fig_reliability_diagram.png    attributes diagram + sharpness, per event
  fig_reliability_diagram.pdf

Every number quoted in the manuscript should come from these files rather than
from a transcript, so a config change regenerates the text's evidence.
"""
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = f"{PROJECT_ROOT}/out"
HOTSPOTS = f"{OUT}/error_hotspots.csv"

# Ten equal-width bins is the conventional attributes-diagram resolution, and
# the Murphy decomposition below is computed on these same bins so the plotted
# curve and the reported REL/RES are the same partition rather than two views
# that can disagree.
BINS = np.linspace(0.0, 1.0, 11)

# Bootstrap resamples CLUSTERS, not buildings. Buildings within one forecast
# share a flood field, so they are nowhere near independent; resampling them
# individually would report an interval several times too narrow. The cluster
# is one (scenario, timestamp) forecast -- the unit that is actually exchangeable.
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260816

# Categorical slots 1 and 2 of the reference data-viz palette, used unmodified.
# The palette documents this pair as clearing the CVD and normal-vision floors
# in light mode on both the adjacent and all-pairs lists.
SERIES = {"at_least_H2_H3": "#2a78d6", "H4_above": "#eb6834"}

# Ink and furniture. Text never wears the series colour -- the marks carry
# identity, the labels stay in neutral ink.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

EVENTS = {
    "at_least_H2_H3": {
        "prob": "overall_building_hazard_prob_at_least_H2_H3",
        "label": "P(tier ≥ medium)",
        "truth": lambda gt: (gt >= 2).astype(float),
    },
    "H4_above": {
        "prob": "overall_building_hazard_prob_H4_above",
        "label": "P(tier = severe)",
        "truth": lambda gt: (gt == 3).astype(float),
    },
}


def brier_decomposition(prob, obs, bins=BINS):
    """Brier score and its Murphy decomposition, BS = REL - RES + UNC.

    Reliability is the bin-weighted squared gap between what was forecast and
    what happened, so REL = 0 is perfect calibration and SMALLER IS BETTER.
    Resolution is the bin-weighted spread of the bin outcomes about the base
    rate -- how much the forecast separates wet cases from dry -- so LARGER IS
    BETTER. Uncertainty is the base rate's own variance, a property of the
    events rather than of the forecast, and it is the Brier score climatology
    scores; that is what makes UNC the denominator of the skill score.

    The decomposition is exact only up to the binning, so the returned `brier`
    is the direct mean squared error and `residual` reports how far REL-RES+UNC
    lands from it. A residual much above ~1e-3 means the bins are too coarse to
    describe this forecast.
    """
    prob = np.asarray(prob, dtype=float)
    obs = np.asarray(obs, dtype=float)
    n = prob.size
    base_rate = obs.mean()

    # np.digitize puts a forecast of exactly 1.0 in a bin of its own; folding it
    # back into the top bin keeps the partition at ten and matters here because
    # a saturated ensemble puts real mass on exactly 1.0.
    index = np.clip(np.digitize(prob, bins[1:-1], right=False), 0, len(bins) - 2)

    rel = res = 0.0
    for k in range(len(bins) - 1):
        member = index == k
        count = int(member.sum())
        if count == 0:
            continue
        mean_prob = prob[member].mean()
        mean_obs = obs[member].mean()
        rel += count * (mean_prob - mean_obs) ** 2
        res += count * (mean_obs - base_rate) ** 2

    rel /= n
    res /= n
    unc = base_rate * (1.0 - base_rate)
    brier = float(np.mean((prob - obs) ** 2))
    return {
        "brier": brier,
        "reliability": rel,
        "resolution": res,
        "uncertainty": unc,
        "bss": 1.0 - brier / unc if unc > 0 else np.nan,
        "residual": (rel - res + unc) - brier,
        "base_rate": base_rate,
        "mean_forecast": float(prob.mean()),
        "n": n,
    }


def reliability_curve(prob, obs, bins=BINS):
    """Per-bin mean forecast, observed frequency and count -- the plotted curve.

    Empty bins are kept as NaN rows rather than dropped so the CSV always has
    one row per bin and two runs stay row-comparable.
    """
    prob = np.asarray(prob, dtype=float)
    obs = np.asarray(obs, dtype=float)
    index = np.clip(np.digitize(prob, bins[1:-1], right=False), 0, len(bins) - 2)

    rows = []
    for k in range(len(bins) - 1):
        member = index == k
        count = int(member.sum())
        rows.append({
            "bin_lower": bins[k],
            "bin_upper": bins[k + 1],
            "count": count,
            "mean_forecast": prob[member].mean() if count else np.nan,
            "observed_frequency": obs[member].mean() if count else np.nan,
        })
    return pd.DataFrame(rows)


def cluster_bootstrap(frame, prob_column, truth, cluster_column,
                      n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Percentile CIs for Brier and BSS, resampling whole forecasts.

    Returns NaNs when there are too few clusters to resample meaningfully. With
    a handful of forecasts the interval is dominated by which forecasts were
    drawn, so it is reported as indicative of between-forecast variability
    rather than as a sampling interval on a large sample -- see the caveat the
    summary prints.
    """
    clusters = frame[cluster_column].unique()
    if clusters.size < 2:
        return {k: np.nan for k in
                ("brier_lo", "brier_hi", "bss_lo", "bss_hi", "n_clusters")}

    grouped = [frame[frame[cluster_column] == c] for c in clusters]
    rng = np.random.default_rng(seed)
    briers, bsss = [], []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(grouped), size=len(grouped))
        sample = pd.concat([grouped[i] for i in pick], ignore_index=True)
        prob = sample[prob_column].to_numpy(dtype=float)
        obs = truth(sample["overall_building_hazard_level_gt"].to_numpy())
        base_rate = obs.mean()
        brier = float(np.mean((prob - obs) ** 2))
        unc = base_rate * (1.0 - base_rate)
        briers.append(brier)
        bsss.append(1.0 - brier / unc if unc > 0 else np.nan)

    return {
        "brier_lo": float(np.nanpercentile(briers, 2.5)),
        "brier_hi": float(np.nanpercentile(briers, 97.5)),
        "bss_lo": float(np.nanpercentile(bsss, 2.5)),
        "bss_hi": float(np.nanpercentile(bsss, 97.5)),
        "n_clusters": int(clusters.size),
    }


def _panel(ax, curve, metrics, colour, title, label):
    """One attributes diagram: the curve against its diagonal and skill region."""
    base_rate = metrics["base_rate"]

    # The region where a point contributes POSITIVE skill: between the no-skill
    # line y = (x + base_rate) / 2 and the diagonal. Shading it is what makes
    # this an attributes diagram rather than a bare reliability plot -- a curve
    # inside the band beats climatology even where it misses the diagonal.
    edge = np.array([0.0, 1.0])
    no_skill = (edge + base_rate) / 2.0
    ax.fill_between(edge, no_skill, edge, color=colour, alpha=0.07, linewidth=0)

    ax.plot(edge, edge, color=INK_MUTED, linewidth=1.2, linestyle="--",
            zorder=2, label="perfect reliability")
    ax.plot(edge, no_skill, color=INK_MUTED, linewidth=1.0, linestyle=":",
            zorder=2, label="no skill")
    ax.axhline(base_rate, color=GRID, linewidth=1.0, zorder=1)
    ax.text(0.985, base_rate, f"climatology {base_rate:.2f}", fontsize=7.5,
            color=INK_MUTED, ha="right", va="bottom")

    drawn = curve.dropna(subset=["mean_forecast"])
    ax.plot(drawn["mean_forecast"], drawn["observed_frequency"],
            color=colour, linewidth=2.0, marker="o", markersize=8,
            markerfacecolor=colour, markeredgecolor=SURFACE,
            markeredgewidth=2.0, zorder=4, label="ensemble")

    # Label only the endpoint bins and the largest interior one. A number on
    # every point is noise; these three are the ones an argument gets made from.
    interior = drawn.iloc[1:-1]
    marked = {drawn.index[0], drawn.index[-1]}
    if not interior.empty:
        marked.add(interior["count"].idxmax())
    for i in sorted(marked):
        row = drawn.loc[i]
        ax.annotate(f"n={int(row['count']):,}",
                    (row["mean_forecast"], row["observed_frequency"]),
                    textcoords="offset points", xytext=(7, -11),
                    fontsize=7.5, color=INK_MUTED)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel(f"forecast probability, {label}", fontsize=9, color=INK)
    ax.set_ylabel("observed frequency", fontsize=9, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK, pad=24, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)

    # Scores ride above the axes as a subtitle. Inside the panel they collide
    # with whichever corner the curve happens to occupy, and which corner that
    # is changes with the data.
    ax.text(0.0, 1.015,
            f"BS {metrics['brier']:.3f}    BSS {metrics['bss']:+.3f}    "
            f"REL {metrics['reliability']:.4f}    RES {metrics['resolution']:.4f}",
            transform=ax.transAxes, fontsize=8, color=INK_MUTED,
            va="bottom", ha="left")

    # Sharpness: how often each probability is issued at all. A curve can sit on
    # the diagonal and still be useless if every forecast is the base rate, so
    # the histogram is part of the diagnostic, not decoration.
    inset = ax.inset_axes([0.60, 0.10, 0.36, 0.24])
    centres = (curve["bin_lower"] + curve["bin_upper"]) / 2.0
    inset.bar(centres, curve["count"].clip(lower=0), width=0.085,
              color=colour, alpha=0.75, linewidth=0)
    inset.set_yscale("log")
    inset.set_xlim(-0.02, 1.02)
    inset.set_title("sharpness", fontsize=7, color=INK_MUTED, pad=3)
    inset.tick_params(colors=INK_MUTED, labelsize=6)
    inset.grid(True, axis="y", color=GRID, linewidth=0.4, alpha=0.6)
    inset.set_axisbelow(True)
    for spine in ("top", "right"):
        inset.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        inset.spines[spine].set_color(GRID)


def main():
    hotspots = pd.read_csv(HOTSPOTS)
    hotspots["forecast_id"] = (hotspots["scenario_id"].astype(str) + "@"
                               + hotspots["timestamp"].astype(str))

    curves, metrics_rows = [], []
    for name, spec in EVENTS.items():
        frame = hotspots.dropna(subset=[spec["prob"],
                                        "overall_building_hazard_level_gt"])
        prob = frame[spec["prob"]].to_numpy(dtype=float)
        obs = spec["truth"](frame["overall_building_hazard_level_gt"].to_numpy())

        metrics = brier_decomposition(prob, obs)
        metrics.update(cluster_bootstrap(frame, spec["prob"], spec["truth"],
                                         "forecast_id"))
        metrics["event"] = name
        metrics_rows.append(metrics)

        curve = reliability_curve(prob, obs)
        curve.insert(0, "event", name)
        curves.append(curve)

    curve_frame = pd.concat(curves, ignore_index=True)
    metrics_frame = pd.DataFrame(metrics_rows).set_index("event")
    curve_frame.to_csv(f"{OUT}/reliability_curve.csv", index=False)
    metrics_frame.to_csv(f"{OUT}/reliability_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), facecolor=SURFACE)
    titles = {"at_least_H2_H3": "Medium hazard or worse",
              "H4_above": "Severe hazard"}
    for ax, (name, spec) in zip(axes, EVENTS.items()):
        ax.set_facecolor(SURFACE)
        _panel(ax, curve_frame[curve_frame["event"] == name].reset_index(drop=True),
               metrics_frame.loc[name], SERIES[name], titles[name], spec["label"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
               labelcolor=INK_MUTED, loc="upper right",
               bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("Reliability of the ensemble's exceedance probabilities",
                 fontsize=12.5, color=INK, x=0.045, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for extension in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_reliability_diagram.{extension}", dpi=220,
                    facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(metrics_frame[["n", "n_clusters", "base_rate", "mean_forecast",
                         "brier", "brier_lo", "brier_hi", "bss", "bss_lo",
                         "bss_hi", "reliability", "resolution", "uncertainty",
                         "residual"]].to_string())
    print(f"\nwrote {OUT}/reliability_curve.csv, {OUT}/reliability_metrics.csv,"
          f" {OUT}/fig_reliability_diagram.png|pdf")


if __name__ == "__main__":
    main()
