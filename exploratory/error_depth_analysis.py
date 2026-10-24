"""Depth analysis of the binary validation's two error populations.

Reruns Integrator.validate_exposure_method -- the surveyed July 2021 depth
raster through the same exposure and FIS -- scores every surveyed
building as hit / miss / false alarm / correct negative, and characterises the
two error classes against the depth statistics the FIS actually saw.

The question it answers: are the errors a symmetric spray around the decision
boundary (a boundary in the wrong place) or two separated populations (a
boundary in the right place, with a variable missing)? The answer sits in the
gap between the miss and false-alarm centroids.

Writes, all under out/:
  error_analysis_buildings.csv   per-building record, the underlying data
  error_analysis_summary.csv     depth distributions by outcome, long form
  error_analysis_metrics.csv     contingency table, skill scores, config record
  fig_error_depth_scatter.png    h_max vs h_mean by outcome, with the boundary
  fig_error_depth_scatter.pdf

Every number quoted in the manuscript should come from these files rather than
from a transcript, so a config change regenerates the text's evidence.
"""
import inspect
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

from modules.lib import PLACRD_FILE_PATH, SURVEYED_FLOOD_DEPTH_JUL_2021
from modules.impact_module import fuzzy_risk as fr
from modules.impact_module.risk_module import ImpactModule, WET_DEPTH_MIN
from modules.forecast_runner.forecast_runner_helper import (
    AFFECTED, AFFECTED_DAMAGE_MIN, _damage_to_affected, _hazards_to_affected,
)

OUT = f"{PROJECT_ROOT}/out"

# The exposure parameters now live as defaults on ImpactModule.buffered_flood_
# depth rather than being passed in, so they are read back off the signature
# for the config record instead of being restated here -- a restated copy would
# silently disagree with the run the moment a default moved.
_EXPOSURE_DEFAULTS = {
    name: parameter.default
    for name, parameter in inspect.signature(
        ImpactModule.buffered_flood_depth).parameters.items()
    if name in ("buffer_pct", "mean_threshold", "max_threshold")
}

# Categorical slots 1-3 of the validated palette, which is the set cleared for
# all-pairs use (scatter). Correct negatives take a recessive gray: they are the
# "nothing happened" background, not a fourth series competing for identity.
COLOURS = {"hit": "#2a78d6", "false_alarm": "#eb6834", "miss": "#1baf7a",
           "correct_neg": "#c3c2b7"}
LABELS = {"hit": "Hit", "false_alarm": "False alarm", "miss": "Miss",
          "correct_neg": "Correct negative"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#ffffff"

PERCENTILES = [.05, .25, .5, .75, .95]


def score():
    """Run the pipeline and label every surveyed building with its outcome."""
    with rio.open(SURVEYED_FLOOD_DEPTH_JUL_2021) as src:
        surveyed_depth = src.read(1).astype(float)

    impact = ImpactModule("error_analysis").get_impact_forecast(
        {"17-07-2021 00:00:00": [surveyed_depth]},
        rasterfile=SURVEYED_FLOOD_DEPTH_JUL_2021,
    )
    predicted = next(iter(impact["building_impact"].values())).copy()
    predicted = predicted[predicted["num_ring_cells"] > 0]

    placards = gpd.read_file(PLACRD_FILE_PATH)
    placards = placards[placards["DamageNb"].isin([0.0, 1.0, 2.0, 3.0])]
    placards["observed_affected"] = placards["DamageNb"].apply(_damage_to_affected)

    df = placards[["building_id", "DamageNb", "damage", "observed_affected"]].merge(
        predicted.drop(columns="geometry"), on="building_id", how="inner"
    ).dropna(subset=["building_hazard_level", "observed_affected"])

    observed = df["observed_affected"].eq(AFFECTED)
    predicted_affected = _hazards_to_affected(df[["building_hazard_level"]]).eq(AFFECTED)
    df["predicted_affected"] = predicted_affected
    df["outcome"] = np.select(
        [observed & predicted_affected, observed & ~predicted_affected,
         ~observed & predicted_affected],
        ["hit", "miss", "false_alarm"], default="correct_neg",
    )

    # Why each miss was missed, read from the memberships rather than rule
    # labels so it survives edits to the rule base.
    h_max_is_low = fr.fuzzify(df["depth_max"].values, "building").argmax(-1) == 0
    df["miss_reason"] = np.where(
        df["outcome"].ne("miss"), "",
        np.select([df["depth_max"] < WET_DEPTH_MIN, h_max_is_low],
                  ["dry", "shallow"], default=""),
    )
    return df


def contingency(df):
    """Contingency counts and the standard warning-verification scores."""
    n = df["outcome"].value_counts()
    tp, fn = int(n.get("hit", 0)), int(n.get("miss", 0))
    fp, tn = int(n.get("false_alarm", 0)), int(n.get("correct_neg", 0))
    total = tp + fn + fp + tn
    pod = tp / (tp + fn) if tp + fn else 0.0
    pofd = fp / (fp + tn) if fp + tn else 0.0
    observed_rate, predicted_rate = (tp + fn) / total, (tp + fp) / total
    expected = predicted_rate * observed_rate + (1 - predicted_rate) * (1 - observed_rate)
    accuracy = (tp + tn) / total
    return {
        "hits": tp, "misses": fn, "false_alarms": fp, "correct_negatives": tn,
        "n_scored": total, "n_affected": tp + fn, "n_not_affected": fp + tn,
        "POD": pod, "POFD": pofd,
        "FAR": fp / (tp + fp) if tp + fp else 0.0,
        "CSI": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "bias": (tp + fp) / (tp + fn) if tp + fn else np.nan,
        "PSS": pod - pofd,
        "accuracy": accuracy,
        "kappa": (accuracy - expected) / (1 - expected) if expected < 1 else np.nan,
    }


def config_record():
    """The settings the numbers came from, so a table can cite its own inputs."""
    building = fr.OBJECT_FUZZIFICATION["building"]
    record = {
        **_EXPOSURE_DEFAULTS,
        "wet_depth_min": WET_DEPTH_MIN,
        "affected_damage_min": AFFECTED_DAMAGE_MIN,
        "building_cutoffs": str(building["cutoffs"]),
        "building_transition": str(building["transition"]),
        "hazard_indicator": "h_max",
    }
    for term, points in fr.trapezoid_breakpoints("building").items():
        record[f"building_{term}_set"] = str(points)
    return record


def depth_summary(df):
    """Depth distributions per outcome, long form, ready to become a table."""
    rows = []
    for column in ("depth_max", "depth_mean", "wet_fraction"):
        if column not in df:
            continue
        described = df.groupby("outcome")[column].describe(percentiles=PERCENTILES)
        described.insert(0, "statistic", column)
        rows.append(described.reset_index())
    summary = pd.concat(rows, ignore_index=True)
    # h_mean/h_max says whether the water is spread or concentrated, which is
    # what separates the two error populations; carried as its own row.
    ratio = (df["depth_mean"] / df["depth_max"].replace(0, np.nan))
    described = ratio.groupby(df["outcome"]).describe(percentiles=PERCENTILES)
    described.insert(0, "statistic", "mean_over_max_ratio")
    return pd.concat([summary, described.reset_index()], ignore_index=True)


def separability(df):
    """How well each statistic tells false alarms from hits (AUC).

    0.5 means the two are indistinguishable on that statistic, so nothing more
    can be extracted from it; a high value means a better threshold exists.
    """
    subset = df[df["outcome"].isin(["hit", "false_alarm"])]
    truth = subset["outcome"].eq("hit").astype(int)
    return {
        f"AUC_hit_vs_false_alarm_{c}": roc_auc_score(truth, subset[c])
        for c in ("depth_max", "depth_mean", "wet_fraction") if c in subset
    }


def figure(df, scores):
    """h_max against h_mean, coloured by outcome, with the decision boundary.

    The point of the figure is the gap between the miss and false-alarm
    centroids: errors on either side of the boundary rather than scattered
    along it. Centroids are drawn as ringed markers and labelled directly, so
    identity never rests on colour alone.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    x_max, y_max = 1.8, 1.1

    # Boundary: classify across the (h_max, h_mean) plane and contour it
    # between low and medium. Read from the live cutoffs, so it cannot drift.
    # Now that h_max is the only indicator the boundary is vertical -- h_mean is
    # plotted because it still describes how the water sat in the ring, not
    # because anything decides on it.
    gx, gy = np.meshgrid(np.linspace(0, x_max, 400), np.linspace(0, y_max, 400))
    tier = fr.hazard_class(gx, "building")
    ax.contourf(gx, gy, tier > 1, levels=[0.5, 1.5], colors=["#2a78d6"], alpha=0.05)
    ax.contour(gx, gy, tier > 1, levels=[0.5], colors=[MUTED],
               linewidths=1.4, linestyles="--")

    # Background class first, errors last, so the rare marks stay visible.
    for outcome in ("correct_neg", "hit", "false_alarm", "miss"):
        sub = df[df["outcome"] == outcome]
        ax.scatter(sub["depth_max"], sub["depth_mean"], s=13,
                   c=COLOURS[outcome], alpha=0.45 if outcome == "correct_neg" else 0.6,
                   linewidths=0, zorder=2 if outcome == "correct_neg" else 3)

    for outcome in ("hit", "false_alarm", "miss"):
        sub = df[df["outcome"] == outcome]
        cx, cy = sub["depth_max"].median(), sub["depth_mean"].median()
        ax.scatter([cx], [cy], s=190, c=COLOURS[outcome], edgecolors=SURFACE,
                   linewidths=2, zorder=5)
        ax.annotate(f"{LABELS[outcome]}  n={len(sub)}\n{cx:.2f}, {cy:.2f} m",
                    (cx, cy), textcoords="offset points", xytext=(14, 10),
                    fontsize=10, color=INK, fontweight="bold", zorder=6)

    ax.set_xlabel("$h^{\\mathrm{max}}$ — deepest ring cell (m)", fontsize=12, color=INK)
    ax.set_ylabel("$h^{\\mathrm{mean}}$ — ring mean depth (m)", fontsize=12, color=INK)
    ax.set_title(
        "Predicted hazard against observed damage, by ring depth\n"
        f"POD {scores['POD']:.2f} · FAR {scores['FAR']:.2f} · "
        f"CSI {scores['CSI']:.2f} · PSS {scores['PSS']:.2f} · "
        f"{scores['n_scored']} surveyed buildings",
        fontsize=12.5, color=INK, pad=12, loc="left",
    )
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")

    handles = [Line2D([], [], marker="o", ls="", markersize=8, color=COLOURS[o],
                      label=LABELS[o]) for o in
               ("hit", "miss", "false_alarm", "correct_neg")]
    handles.append(Line2D([], [], ls="--", color=MUTED, lw=1.4,
                          label="Predicted affected"))
    # Upper left: the only quadrant with no marks in it, since h_mean cannot
    # exceed h_max and the cloud therefore hugs the diagonal.
    ax.legend(handles=handles, loc="upper left", frameon=False,
              fontsize=10, labelcolor=INK)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_error_depth_scatter.{extension}", dpi=220,
                    facecolor=SURFACE)
    plt.close(fig)


def main():
    df = score()
    scores = contingency(df)
    scores.update(separability(df))
    scores.update(config_record())

    print(f"scored {scores['n_scored']} buildings "
          f"({scores['n_affected']} affected / {scores['n_not_affected']} not)")
    print(f"  hits={scores['hits']} misses={scores['misses']} "
          f"false_alarms={scores['false_alarms']} "
          f"correct_negatives={scores['correct_negatives']}")
    print(f"  POD={scores['POD']:.2f} FAR={scores['FAR']:.2f} CSI={scores['CSI']:.2f} "
          f"bias={scores['bias']:.2f} PSS={scores['PSS']:.3f} kappa={scores['kappa']:.3f}")

    summary = depth_summary(df)
    print("\nmedian by outcome:")
    print(summary[summary["50%"].notna()]
          .pivot(index="statistic", columns="outcome", values="50%").round(3).to_string())

    misses = df[df["outcome"] == "miss"]
    print(f"\nmiss composition (n={len(misses)}): "
          f"{misses['miss_reason'].value_counts().to_dict()}")
    print(f"  h_max max {misses['depth_max'].max():.2f} m, "
          f"h_mean max {misses['depth_mean'].max():.2f} m")
    print("  by damage class:", misses["damage"].value_counts().to_dict())

    false_alarms = df[df["outcome"] == "false_alarm"]
    print(f"\nfalse alarms (n={len(false_alarms)}): "
          f"driest h_max {false_alarms['depth_max'].min():.2f} m, "
          f"{(false_alarms['wet_fraction'] >= 0.9).mean():.0%} with ring coverage >= 90%")
    print("  separability from hits: " + ", ".join(
        f"{k.split('_')[-1]} AUC {v:.3f}" for k, v in scores.items()
        if k.startswith("AUC_")))

    columns = ["building_id", "DamageNb", "damage", "observed_affected",
               "predicted_affected", "outcome", "miss_reason", "depth_max",
               "depth_mean", "wet_fraction", "num_ring_cells", "num_wet_cells",
               "building_hazard_level", "building_hazard_likelihood",
               "property_hazard_level", "street_hazard_level", "overall_building_hazard_level"]
    df[[c for c in columns if c in df]].to_csv(
        f"{OUT}/error_analysis_buildings.csv", index=False)
    summary.to_csv(f"{OUT}/error_analysis_summary.csv", index=False)
    pd.DataFrame([scores]).to_csv(f"{OUT}/error_analysis_metrics.csv", index=False)
    figure(df, scores)

    print(f"\nwrote {OUT}/error_analysis_buildings.csv ({len(df)} rows)")
    print(f"wrote {OUT}/error_analysis_summary.csv")
    print(f"wrote {OUT}/error_analysis_metrics.csv")
    print(f"wrote {OUT}/fig_error_depth_scatter.png and .pdf")


if __name__ == "__main__":
    main()
