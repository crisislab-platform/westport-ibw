"""Per-building error record for each hazard indicator configuration.

Labels every surveyed building hit / miss / false alarm / correct negative for
each configuration the validation figures report, so the hazard class
behind each error is recoverable per configuration rather than only in aggregate.

Writes, all under out/:
  error_analysis_buildings_<configuration>.csv   one file per configuration
  error_analysis_by_configuration.csv            the same rows, long form
  false_alarm_hazard_classes.csv                 H-class split of the false alarms

Two deliberate differences from error_depth_analysis.py:

* Row selection mirrors ValidationModule.test_against_placard_data rather than
  error_depth_analysis.score(): placards filtered to those with a surveyed depth
  within a 5x5 window, a LEFT merge onto the impact frame, and a dropna over the
  five columns the placard scoring requires. score() instead filters
  num_ring_cells > 0 and inner-merges, which is why it scores 2244 buildings where
  the validation scores 2247. Matching the validation is the point here -- these
  files have to reconcile with binary_validation_metrics.csv.
* The pipeline runs once and is labelled once per configuration. All of them
  are on the same impact frame -- they differ only in which objects Equation (7)
  takes the max over and which indicator it reads -- so a run each would only
  burn time.

The run ends by reconciling its own contingency counts against
binary_validation_metrics.csv and printing any configuration that disagrees.
"""
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.transform import rowcol

from modules.lib import PLACRD_FILE_PATH, SURVEYED_FLOOD_DEPTH_JUL_2021
from modules.impact_module.risk_module_helper import (
    CLASS_TERMS, configuration_prefix,
)
from modules.impact_module.risk_module import ImpactModule
from modules.forecast_runner.forecast_runner_helper import (
    AFFECTED, NOT_AFFECTED, CONFIGURATION_PREFIXES, SCORED_CONFIGURATIONS,
    _damage_to_affected, _exceedance_to_affected,
)

OUT = f"{PROJECT_ROOT}/out"

# label -> the exceedance column the scoring thresholds, and the class column
# reported for that configuration. Both come off the finder's own registry via
# forecast_runner.SCORED_CONFIGURATIONS, so this script cannot label a
# configuration the validation scored differently -- which is the one error
# these files exist to make impossible.
#
# Ordered by expanding spatial context, and within each context by
# forecast_runner.STATISTIC_ORDER, so the table reads as the indicators compared
# at three scales.
CONFIGURATIONS = {
    name: (column, f"{column[:-len('_prob_at_least_H2')]}_hazard_class")
    for column, (name, _) in SCORED_CONFIGURATIONS.items()
}

# The class reported for a configuration is its OWN elected class: the highest
# ARR class any member of the ensemble reaches on that spatial context. Unlike
# the previous engine there is no separate depth-thresholded configuration, so
# every row's class and its call come from the same quantity.
CLASS_COLUMN = {name: class_column
                for name, (_, class_column) in CONFIGURATIONS.items()}

# Columns carried into the per-building files when the impact frame has them:
# the exposure depths, every configuration's readings, and the per-object
# diagnostics that explain them.
CARRY = [
    "building_id", "DamageNb", "damage", "observed_affected",
    "max_depth", "mean_depth", "num_ring_cells", "inundated",
    "property_known", "street_known",
] + [
    f"{prefix}_{suffix}"
    for prefix in CONFIGURATION_PREFIXES
    for suffix in ("indicator_mean", "hazard_level", "hazard_class",
                   "likelihood", "likelihood_class", "prob_at_least_H2",
                   "prob_at_least_H3_above", "ensemble_mean_hazard_level")
] + [
    f"{object_type}_max_{suffix}"
    for object_type in ("property", "street")
    for suffix in ("indicator_mean", "hazard_level", "likelihood")
]

# The same dropna subset ValidationModule.test_against_placard_data uses, which is
# what makes these files reconcile with binary_validation_metrics.csv.
REQUIRED = ["observed_affected", "max_depth", "mean_depth",
            f"{configuration_prefix('BuildingOnly', 'Max')}_hazard_level",
            f"{configuration_prefix('BuildProp', 'Max')}_hazard_level"]


def placards_with_surveyed_depth(window=2):
    """Placards that have a surveyed depth within a 5x5 window of their centroid.

    Replicates Integrator.get_obs_placard_filtered. It is copied rather than called
    because constructing an Integrator would spin up the boundary-condition module
    and the flow-matching client, neither of which this script needs. If that filter
    changes upstream, this copy has to change with it.
    """
    placards = gpd.read_file(PLACRD_FILE_PATH)
    placards = placards[placards["DamageNb"].isin([0.0, 1.0, 2.0, 3.0])]

    with rio.open(SURVEYED_FLOOD_DEPTH_JUL_2021) as src:
        depth = src.read(1).astype(float)
        transform = src.transform

    keep = []
    for geom in placards.geometry.centroid:
        row, col = rowcol(transform, geom.x, geom.y)
        r0, r1 = max(0, row - window), min(depth.shape[0], row + window + 1)
        c0, c1 = max(0, col - window), min(depth.shape[1], col + window + 1)
        keep.append(bool(np.any(~np.isnan(depth[r0:r1, c0:c1]))))

    return placards.loc[keep].copy()


def scored_frame():
    """Run the FIS on the surveyed raster and assemble the frame the scoring sees."""
    with rio.open(SURVEYED_FLOOD_DEPTH_JUL_2021) as src:
        surveyed_depth = src.read(1).astype(float)

    # domain_raster defines the scored population -- see the note in
    # regenerate_binary_validation_metrics.py. Without it this script scores the
    # model grid and reconciles against a CSV built on the surveyed domain.
    impact = ImpactModule(
        "error_analysis_by_configuration",
        domain_raster=SURVEYED_FLOOD_DEPTH_JUL_2021).get_impact_forecast(
        {"17-07-2021 00:00:00": [surveyed_depth]},
        rasterfile=SURVEYED_FLOOD_DEPTH_JUL_2021,
    )
    predicted = next(iter(impact["building_impact"].values())).copy()
    predicted = predicted.rename(
        columns={"depth_max": "max_depth", "depth_mean": "mean_depth"})
    if "geometry" in predicted.columns:
        predicted = predicted.drop(columns="geometry")

    placards = placards_with_surveyed_depth()
    placards["observed_affected"] = placards["DamageNb"].apply(_damage_to_affected)

    keep = [c for c in predicted.columns if c in CARRY and c != "building_id"]
    df = placards[["building_id", "DamageNb", "damage", "observed_affected"]].merge(
        predicted[["building_id"] + keep], on="building_id", how="left")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"impact frame is missing required columns: {missing}")

    return df.dropna(subset=REQUIRED).copy()


def label(df, configuration):
    """Attach predicted_affected / outcome / hazard_class for one configuration."""
    column, _ = CONFIGURATIONS[configuration]
    # The same call the validation makes: P(h >= 0.30 m) at or above
    # forecast_runner.AFFECTED_MIN_PROBABILITY.
    predicted = _exceedance_to_affected(df[column])

    out = df.copy()
    out["configuration"] = configuration
    out["predictor"] = column
    out["predicted_affected"] = predicted

    observed = out["observed_affected"].eq(AFFECTED)
    flagged = predicted.eq(AFFECTED)
    out["outcome"] = np.select(
        [observed & flagged, observed & ~flagged, ~observed & flagged],
        ["hit", "miss", "false_alarm"], default="correct_neg",
    )

    out["hazard_class_column"] = CLASS_COLUMN[configuration]
    out["hazard_class"] = out[CLASS_COLUMN[configuration]]
    return out


def reconcile(labelled):
    """Compare the contingency counts against the published validation metrics."""
    published = pd.read_csv(f"{OUT}/binary_validation_metrics.csv")
    published = published[(published["scenario_id"] == 0)
                          & (published["model"] == "surveyed")
                          & (published["damage_threshold"] == 1)]
    by_indicator = published.set_index("indicator")

    rows = []
    for configuration, frame in labelled.items():
        counts = frame["outcome"].value_counts()
        row = {
            "configuration": configuration,
            "hits": int(counts.get("hit", 0)),
            "misses": int(counts.get("miss", 0)),
            "false_alarms": int(counts.get("false_alarm", 0)),
            "n": len(frame),
        }
        # The CSV is keyed by the PAPER name now
        # (forecast_runner.SCORED_CONFIGURATIONS), which is the same key this
        # script's CONFIGURATIONS uses -- so the join needs no translation.
        if configuration in by_indicator.index:
            reference = by_indicator.loc[configuration]
            row["published_misses"] = int(reference["misses"])
            row["published_false_alarms"] = int(reference["false_alarms"])
            row["published_n"] = int(reference["num_buildings_evaluated"])
            row["matches"] = (row["misses"] == row["published_misses"]
                              and row["false_alarms"] == row["published_false_alarms"]
                              and row["n"] == row["published_n"])
        rows.append(row)
    return pd.DataFrame(rows)


def false_alarm_classes(labelled):
    """H-class split of each configuration's false alarms, counts and percentages."""
    classes = list(CLASS_TERMS)
    rows = []
    for configuration, frame in labelled.items():
        false_alarms = frame[frame["outcome"] == "false_alarm"]
        counts = false_alarms["hazard_class"].value_counts()
        total = len(false_alarms)
        row = {"configuration": configuration,
               "hazard_class_column": CLASS_COLUMN[configuration],
               "false_alarms": total}
        for name in classes:
            count = int(counts.get(name, 0))
            row[f"n_{name}"] = count
            row[f"pct_{name}"] = round(100.0 * count / total, 1) if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def slug(configuration):
    return configuration.lower().replace("-", "_")


def main():
    df = scored_frame()
    print(f"scored {len(df)} buildings\n")

    labelled = {name: label(df, name) for name in CONFIGURATIONS}

    for configuration, frame in labelled.items():
        path = f"{OUT}/error_analysis_buildings_{slug(configuration)}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame)} rows)")

    combined = pd.concat(labelled.values(), ignore_index=True)
    combined.to_csv(f"{OUT}/error_analysis_by_configuration.csv", index=False)
    print(f"wrote {OUT}/error_analysis_by_configuration.csv ({len(combined)} rows)\n")

    check = reconcile(labelled)
    print("reconciliation against binary_validation_metrics.csv:")
    print(check.to_string(index=False), "\n")
    if "matches" in check and not check["matches"].fillna(False).all():
        print("WARNING: counts differ from the published metrics for at least one "
              "configuration -- the impact module config has moved since that CSV "
              "was written. Treat the published figures as stale, or pin the config.\n")

    split = false_alarm_classes(labelled)
    split.to_csv(f"{OUT}/false_alarm_hazard_classes.csv", index=False)
    print("false alarms by assigned hazard class:")
    print(split.to_string(index=False))
    print(f"\nwrote {OUT}/false_alarm_hazard_classes.csv")


if __name__ == "__main__":
    main()
