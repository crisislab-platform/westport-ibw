"""Did the system get each building's PEAK hazard class right, and when?

Every other score in this repo is taken a timestep at a time: QWK, RPS and the
reliability diagram all ask "at this hour, was this building in the right
class". That is the right question for calibration and the wrong one for a
warning. What a resident or a controller acts on is the WORST it is going to
get and WHEN -- evacuate before the peak, not during it -- and a forecast can
be excellent hour by hour and still miss both, by shaving the top class off the
peak or by putting the peak two hours late.

So this script collapses the time axis instead of the building axis. For each
building in each event it takes

    L_obs = max_t  observed class          t_obs = first t where L_obs is reached
    L_fc  = max_t  forecast class          t_fc  = first t where L_fc  is reached

and reports whether L_fc == L_obs (the AMPLITUDE question) and t_fc - t_obs
(the TIMING question). FIRST attainment, not last: the actionable moment is
when the worst case arrives, and a class held for six hours would otherwise be
timed by when it let go.

THE TWO OFFSETS, and why one is not enough. ``peak_offset_h`` compares the two
peaks directly, so it only means anything when the two peaks are the same
class -- a forecast peaking at H2 an hour before an observed H3+ peak is not
"one hour early", it is a different forecast. ``onset_offset_h`` closes that
gap: the first time the forecast reaches the class that was actually observed,
against the first time it was actually reached. That one is defined whenever
the forecast ever gets there, so it is the one to quote over a whole
population. Both are signed the same way -- POSITIVE IS LATE.

TWO SCORING RULES, because the pipeline's own is extreme at 100 members.
``HazardIndicatorFinder(min_class_probability=0.0)`` elects the highest class
ANY member reaches, which is deliberate -- it is what lets a 2% chance of H3+
be seen rather than averaged away -- but on a peak-over-time statistic the two
permissive steps compound: the max over 100 members is taken, and then the max
over ~50 timesteps of that. The result is a near-certain over-forecast of the
peak, which says more about the rule than about the ensemble. So both rules are
scored, and the pair IS the result:

  any_member   P(h >= tau_A) >  0.0   the operational rule (the finder's default)
  majority     P(h >= tau_A) >= 0.5   the rule the binary validation already
                                      uses (forecast_runner.AFFECTED_MIN_PROBABILITY)

Reads out/verification_pairs_buildprop_max.csv -- the per-building (forecast
probability, observed class) pairs from forecast_runner.write_verification_pairs,
which is BuildProp-Max, the configuration the paper's skill figures report.
The forecast class is re-derived from the probabilities rather than read from
the exceedance file, so the two rules come out of one code path and the row
sets cannot drift apart; at threshold 0.0 the derivation reproduces that file's
``bp_hazard_level`` exactly.

A SECOND QUESTION, on the same rows: not "was the peak class right" but "how
many buildings ever reached H2, how many ever reached H3+, how many did the
system say did, and how late was it in saying so". That one is asked of a FIXED
class rather than of the building's own peak, is counted per BUILDING PER EVENT
rather than per building-step, and is at-least rather than exactly (a building
that reached H3+ reached H2 too). It is what
``hazard_class_onset_by_scenario.csv`` holds, and its hit rate is event-wise,
so it does not and should not match the per-timestep POD in
binary_validation_metrics.csv.

NOT LEAD TIME. Every row in the pairs file is lead_h = 0 -- these are nowcasts,
issue_time == valid_time throughout. So a negative offset is not the system
warning ahead of the event; it is the surrogate ensemble reaching a class
BEFORE the reference simulation does at the same instant. Read every offset
here as a timing bias against the simulator, not as warning lead time.

Writes, all under out/:
  peak_hazard_timing_buildings.csv     one row per rule x scenario x building
  peak_hazard_timing_summary.csv       peak-class rates and timing, by
                                       rule x scenario x population
  hazard_class_onset_by_scenario.csv   per-class building counts and onset
                                       timing, by rule x scenario x class
  hazard_class_onset_table.csv         the same numbers with the classes side
                                       by side, one row per rule x scenario --
                                       the reading/paper table
"""
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUT = os.path.join(PROJECT_ROOT, "out")
PAIRS_FILE = os.path.join(OUT, "verification_pairs_buildprop_max.csv")
BUILDINGS_FILE = os.path.join(OUT, "peak_hazard_timing_buildings.csv")
SUMMARY_FILE = os.path.join(OUT, "peak_hazard_timing_summary.csv")
CLASS_FILE = os.path.join(OUT, "hazard_class_onset_by_scenario.csv")
CLASS_TABLE_FILE = os.path.join(OUT, "hazard_class_onset_table.csv")

# The class boundaries as they appear on the pairs file, severe -> safe. Read in
# this order so the first probability a building attains IS its class, which is
# the finder's "highest attained class" rule written as a search from the top.
# P(>= H1) is not consulted: tau_H1 = 0 m is the floor of the depth scale, so
# every building attains it and H1 is the fallback.
CLASS_PROBABILITY_COLUMNS = (
    (3, "prob_at_least_H3_above"),
    (2, "prob_at_least_H2"),
)

# (name, threshold, comparison). The comparisons differ on purpose and are each
# copied from the code that owns them: HazardIndicatorFinder.hazard_class tests
# ``probabilities > self.min_class_probability`` (strict, so a class no member
# reaches is not attained), while _exceedance_to_affected tests
# ``probability >= threshold`` (inclusive, so exactly half the ensemble counts).
# Reproducing each exactly is the point -- an off-by-one-member disagreement
# with the published scores would be untraceable otherwise.
RULES = (
    ("any_member", 0.0, np.greater),
    ("majority", 0.5, np.greater_equal),
)

# The class a building has to reach before "peak class" is a warning statement
# rather than a note that it stayed dry. H1 is "generally safe" and includes
# every building the flood never touched, so pooling those in would report a
# hit rate dominated by the trivially correct.
WARNING_CLASS = 2

# The classes counted and timed one at a time, as (level, column name). H1 is
# not among them for the same reason it has no probability column: every
# building attains it at the first timestep, so its count is the population and
# its timing is identically zero.
CLASS_NAMES = ((2, "H2"), (3, "H3"))

# The reading table, as (column of the long class table, name it takes there).
# One row per rule x scenario with the classes SIDE BY SIDE, which is how the
# result is read -- "how many reached H2, how many reached H3+, what did the
# system say, how late was it" is one sentence about one scenario, and the long
# form splits it across two rows. Same numbers, so the two files can be checked
# against each other; this one is the one to paste into a paper.
CLASS_TABLE_COLUMNS = (
    ("observed_reached", "observed_{name}"),
    ("forecast_reached", "forecast_{name}"),
    ("pod", "pod_{name}"),
    ("offset_mean_h", "mean_offset_{name}_h"),
    ("offset_median_h", "median_offset_{name}_h"),
    ("offset_within_1h", "within_1h_{name}"),
)

KEYS = ["scenario_id", "building_id"]


def load_pairs():
    """The verification pairs, ONE ROW PER BUILDING PER STEP, latest run only.

    The file is APPENDED to by every run, and the sweep has been run more than
    once: it currently holds two smoke passes (ensemble 4 and 1), two partial
    passes, and a full 100-member pass, with the same (scenario, valid_time,
    building) appearing in several of them. Reading it as-is double-counts most
    of the ten scenarios and mixes ensemble sizes into one score.

    A run appends its rows contiguously, so a run is a maximal block of
    consecutive rows sharing (scenario_id, ensemble_size), and the LAST block
    for each scenario is its most recent run. That is what is kept, after
    dropping every ensemble size below the production one. The choice is
    checked, not assumed: if the last block does not cover every valid_time the
    scenario has anywhere in the file, the run was partial and keeping it would
    silently shorten that event, so it raises instead.
    """
    pairs = pd.read_csv(PAIRS_FILE, dtype={
        "scenario_id": "int32", "ensemble_size": "int32", "building_id": "int64",
        "observed_level": "int8", "prob_at_least_H1": "float32",
        "prob_at_least_H2": "float32", "prob_at_least_H3_above": "float32",
    }, parse_dates=["valid_time"])

    ensemble_size = int(pairs["ensemble_size"].max())
    smaller = pairs["ensemble_size"] < ensemble_size
    if smaller.any():
        print(f"dropping {int(smaller.sum()):,} rows from smoke runs at ensemble "
              f"sizes {sorted(pairs.loc[smaller, 'ensemble_size'].unique().tolist())}; "
              f"scoring the {ensemble_size}-member runs")
    pairs = pairs[~smaller]

    run = pairs["scenario_id"].astype(str) + "|" + pairs["ensemble_size"].astype(str)
    block = (run != run.shift()).cumsum()
    latest = block.groupby(pairs["scenario_id"]).transform("max")
    kept = pairs[block == latest]

    for scenario, group in pairs.groupby("scenario_id"):
        missing = set(group["valid_time"].unique()) - set(
            kept.loc[kept["scenario_id"] == scenario, "valid_time"].unique())
        if missing:
            raise ValueError(
                f"scenario {scenario}: its last run covers {len(missing)} fewer "
                f"timesteps than the file holds for it, earliest "
                f"{min(missing)} -- that run was partial, so no single run of "
                f"this scenario is complete")

    duplicated = kept.duplicated(KEYS + ["valid_time"])
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} duplicate building-steps "
                         f"survived de-duplication")

    steps = kept.groupby("scenario_id")["valid_time"].nunique()
    print(f"{len(kept):,} building-steps: {len(steps)} scenarios, "
          f"{int(steps.sum())} timesteps, "
          f"{kept['building_id'].nunique():,} buildings, "
          f"{ensemble_size}-member ensemble")
    return kept


def forecast_level(pairs, threshold, compare):
    """The class the ensemble elects for each row, under one rule.

    The highest class whose exceedance probability clears the threshold, and H1
    where none does -- HazardIndicatorFinder.hazard_class, evaluated on the
    published probabilities instead of the member stacks that no longer exist.
    Written as a search from the severe end so the first hit wins, which is what
    "highest attained" means.
    """
    level = np.ones(len(pairs), dtype="int8")
    for class_level, column in CLASS_PROBABILITY_COLUMNS:
        attained = compare(pairs[column].to_numpy(), threshold)
        level = np.where((level == 1) & attained, class_level, level)
    return level.astype("int8")


def _peak(frame, level_column, prefix):
    """Peak class per building, when it was first reached, and how long it held.

    ``{prefix}_peak_level`` is the max over the event; ``{prefix}_peak_time`` is
    the FIRST timestep attaining it (see the module docstring); ``_peak_steps``
    counts how many timesteps sat at the peak, which is what separates a sharp
    crest from a plateau -- a one-hour timing error means something different
    against each.
    """
    peak = frame.groupby(KEYS)[level_column].transform("max")
    at_peak = frame[frame[level_column] == peak]
    grouped = at_peak.groupby(KEYS)
    return pd.DataFrame({
        f"{prefix}_peak_level": grouped[level_column].first(),
        f"{prefix}_peak_time": grouped["valid_time"].min(),
        f"{prefix}_peak_steps": grouped.size(),
    })


def _first_time_at_least(frame, level_column, level):
    """When each building FIRST reaches ``level`` or worse; NaT if it never does.

    "Reached H2" is at-least, not exactly-H2: the classes are the exceedance
    ladder P(h >= tau_A) of Equation (8), so a building that goes straight to
    H3+ has reached H2 as well. Counting them as exactly-H2 instead would make
    the H2 and H3+ columns disjoint and neither would answer "how many
    buildings got at least this bad".
    """
    reached = frame[frame[level_column].to_numpy() >= level]
    return reached.groupby(KEYS)["valid_time"].min()


def per_building(pairs, rule, threshold, compare):
    """One row per building per scenario: the peak pair and the two offsets."""
    frame = pairs[KEYS + ["valid_time", "observed_level"]].copy()
    frame["forecast_level"] = forecast_level(pairs, threshold, compare)

    peaks = _peak(frame, "observed_level", "observed").join(
        _peak(frame, "forecast_level", "forecast"))

    # The same first-attainment question asked of a FIXED class rather than of
    # the building's own peak. This is what a per-class count and a per-class
    # timing bias are built from: the peak columns above cannot give them,
    # because a building whose peak is H3+ contributes nothing to an H2 count
    # even though it passed through H2.
    for level, name in CLASS_NAMES:
        peaks[f"observed_first_{name}"] = _first_time_at_least(
            frame, "observed_level", level)
        peaks[f"forecast_first_{name}"] = _first_time_at_least(
            frame, "forecast_level", level)
        peaks[f"onset_offset_{name}_h"] = (
            (peaks[f"forecast_first_{name}"] - peaks[f"observed_first_{name}"])
            / pd.Timedelta(hours=1))

    # Lead time on the class that ACTUALLY happened: the first timestep the
    # forecast puts the building at or above its observed peak. Defined whether
    # or not the forecast's own peak matches, which is what makes it quotable
    # across the whole population rather than only the buildings already right.
    # Broadcast by reindex rather than merge: this has to stay row-aligned with
    # frame["forecast_level"] below, and a merge is free to reorder its output.
    observed_peak = peaks["observed_peak_level"].reindex(
        pd.MultiIndex.from_frame(frame[KEYS])).to_numpy()
    reached = frame[frame["forecast_level"].to_numpy() >= observed_peak]
    peaks["forecast_reached_observed_peak_time"] = (
        reached.groupby(KEYS)["valid_time"].min())

    peaks["peak_level_error"] = (peaks["forecast_peak_level"]
                                 - peaks["observed_peak_level"])
    peaks["peak_level_correct"] = peaks["peak_level_error"] == 0
    peaks["forecast_reached_observed_peak"] = (
        peaks["forecast_reached_observed_peak_time"].notna())

    hour = pd.Timedelta(hours=1)
    peaks["peak_offset_h"] = (
        (peaks["forecast_peak_time"] - peaks["observed_peak_time"]) / hour)
    peaks["onset_offset_h"] = (
        (peaks["forecast_reached_observed_peak_time"]
         - peaks["observed_peak_time"]) / hour)

    peaks = peaks.reset_index()
    peaks.insert(0, "rule", rule)
    return peaks


def _timing(offsets):
    """The distribution of one signed offset column, positive = late."""
    offsets = offsets.dropna()
    if offsets.empty:
        return {}
    return {
        "n_timed": len(offsets),
        "median_h": offsets.median(),
        "mean_h": offsets.mean(),
        "p10_h": offsets.quantile(0.10),
        "p90_h": offsets.quantile(0.90),
        "mae_h": offsets.abs().mean(),
        "within_1h": (offsets.abs() <= 1).mean(),
        "within_3h": (offsets.abs() <= 3).mean(),
        "early": (offsets < 0).mean(),
        "on_time": (offsets == 0).mean(),
        "late": (offsets > 0).mean(),
    }


def summarise(buildings):
    """Rates and timing distributions by rule x scenario x population.

    Two populations, because they answer different questions and the first one
    alone is misleading: ``all`` is every building, most of which stayed at H1
    all event and are trivially correct, while ``observed_ge_H2`` is the
    buildings the flood actually put in a warning class -- the ones a warning
    system is judged on. Scenario ``all`` pools the events.

    The peak-to-peak timing is taken over the CORRECT-PEAK subset only (see the
    module docstring on why comparing the times of two different classes is not
    a timing error), while the onset timing is taken over every building the
    forecast ever got to the observed class, which is the wider and more useful
    population.
    """
    rows = []
    populations = {
        "all": lambda frame: frame,
        "observed_ge_H2": lambda frame: frame[
            frame["observed_peak_level"] >= WARNING_CLASS],
    }
    for rule, by_rule in buildings.groupby("rule", sort=False):
        scenarios = [(scenario, group) for scenario, group
                     in by_rule.groupby("scenario_id")] + [("all", by_rule)]
        for scenario, group in scenarios:
            for population, select in populations.items():
                frame = select(group)
                if frame.empty:
                    continue
                error = frame["peak_level_error"]
                correct = frame[frame["peak_level_correct"]]
                rows.append({
                    "rule": rule,
                    "scenario_id": scenario,
                    "population": population,
                    "n_buildings": len(frame),
                    "peak_correct": frame["peak_level_correct"].mean(),
                    "peak_under": (error < 0).mean(),
                    "peak_over": (error > 0).mean(),
                    "peak_level_mae": error.abs().mean(),
                    "reached_observed_peak":
                        frame["forecast_reached_observed_peak"].mean(),
                    **{f"peak_{name}": value for name, value
                       in _timing(correct["peak_offset_h"]).items()},
                    **{f"onset_{name}": value for name, value
                       in _timing(frame["onset_offset_h"]).items()},
                })
    return pd.DataFrame(rows)


def summarise_classes(buildings):
    """Per scenario and class: how many buildings got there, how many the system
    said got there, and how late it was in saying so.

    The counts are of BUILDINGS OVER THE WHOLE EVENT, not of building-steps: a
    building is counted once for a class if it ever reached it. Crossed into a
    contingency table on that basis --

      hits          reached it, and the system said so
      misses        reached it, the system never said so
      false_alarms  never reached it, the system said it did

    -- so ``pod`` here is an EVENT-WISE hit rate for the class, not the
    per-timestep POD in binary_validation_metrics.csv, and the two will not
    agree. A building the system flags at the wrong hour is a hit here and a
    miss plus a false alarm there.

    ``offset_mean_h`` is the average time difference on the HITS: signed,
    positive = the system reached the class later than the truth did. Taken on
    hits alone because a miss has no time to difference, which also means the
    mean is conditioned on the system having got there at all -- read it beside
    ``pod``, never on its own.
    """
    rows = []
    for rule, by_rule in buildings.groupby("rule", sort=False):
        scenarios = [(scenario, group) for scenario, group
                     in by_rule.groupby("scenario_id")] + [("all", by_rule)]
        for scenario, group in scenarios:
            for _, name in CLASS_NAMES:
                observed = group[f"observed_first_{name}"].notna()
                forecast = group[f"forecast_first_{name}"].notna()
                hits = observed & forecast
                offsets = group.loc[hits, f"onset_offset_{name}_h"]
                rows.append({
                    "rule": rule,
                    "scenario_id": scenario,
                    "class": name,
                    "n_buildings": len(group),
                    "observed_reached": int(observed.sum()),
                    "forecast_reached": int(forecast.sum()),
                    "hits": int(hits.sum()),
                    "misses": int((observed & ~forecast).sum()),
                    "false_alarms": int((~observed & forecast).sum()),
                    "pod": hits.sum() / observed.sum() if observed.any() else np.nan,
                    "count_ratio": (forecast.sum() / observed.sum()
                                    if observed.any() else np.nan),
                    # n_timed is dropped: on this table it is `hits` by
                    # construction, since every hit has both times.
                    **{f"offset_{statistic}": value
                       for statistic, value in _timing(offsets).items()
                       if statistic != "n_timed"},
                })
    return pd.DataFrame(rows)


def wide_class_table(classes):
    """The long class table pivoted to one row per rule x scenario.

    Built by walking the records rather than with pivot/unstack because
    ``scenario_id`` mixes integers with the pooled "all" row, and an index of
    mixed type is exactly what those reorder or raise on. Insertion order is
    kept, so the rows come out in the order the long table already put them.
    """
    rows = {}
    for record in classes.to_dict("records"):
        row = rows.setdefault(
            (record["rule"], record["scenario_id"]),
            {"rule": record["rule"], "scenario_id": record["scenario_id"],
             "n_buildings": record["n_buildings"]})
        for source, template in CLASS_TABLE_COLUMNS:
            row[template.format(name=record["class"])] = record.get(source)
    return pd.DataFrame(list(rows.values()))


def main():
    pairs = load_pairs()
    buildings = pd.concat(
        [per_building(pairs, rule, threshold, compare)
         for rule, threshold, compare in RULES],
        ignore_index=True)
    summary = summarise(buildings)
    classes = summarise_classes(buildings)
    table = wide_class_table(classes)

    buildings.to_csv(BUILDINGS_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    classes.to_csv(CLASS_FILE, index=False)
    table.to_csv(CLASS_TABLE_FILE, index=False)
    print(f"\n{len(buildings):,} building-events -> {BUILDINGS_FILE}")
    print(f"{len(summary):,} summary rows -> {SUMMARY_FILE}")
    print(f"{len(classes):,} class rows -> {CLASS_FILE}")
    print(f"{len(table):,} table rows -> {CLASS_TABLE_FILE}\n")

    pooled = summary[summary["scenario_id"] == "all"]
    shown = ["rule", "population", "n_buildings", "peak_correct", "peak_under",
             "peak_over", "peak_level_mae", "reached_observed_peak",
             "peak_median_h", "peak_within_1h", "onset_median_h",
             "onset_early", "onset_late"]
    print("PEAK CLASS, pooled over scenarios")
    print(pooled[shown].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    shown = ["scenario_id", "class", "observed_reached", "forecast_reached",
             "hits", "misses", "false_alarms", "pod", "offset_mean_h",
             "offset_median_h", "offset_within_1h"]
    for rule, group in classes.groupby("rule", sort=False):
        print(f"\nCLASS ONSET, rule = {rule}")
        print(group[shown].to_string(index=False,
                                     float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
