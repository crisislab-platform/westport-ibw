"""Timing, merging and results-file helpers for forecast_runner.Integrator."""
from contextlib import contextmanager
import pandas as pd
import logging
import os
import shutil
import time

logger = logging.getLogger(__name__)


# One row per issue in runtime_metrics.csv, timing the framework end to end.
# Named for the components rather than the modules that implement them, since
# this is what the efficiency results are reported as:
#
#   bc_s         BC_Module.get_bc_forecast -- reading the boundary conditions
#   inundation_s FlowMatchingModelClient.get_hazard_forecast -- the surrogate
#   exposure_s   ExposureModule.get_exposure_assessment -- the building
#                depths (risk_s before 2026-10-23)
#   extent_s     ImpactModule.get_flood_extent -- the depth-class polygons
#   delete_s     clearing the PREVIOUS issue off the hosted layers
#   publish_s    writing THIS issue to them -- the framework's update cost
#   total_s      the whole issue, including anything not broken out above
#
# delete_s is kept out of publish_s deliberately. Publishing replaces a layer's
# contents, so every issue first removes the last one, but that teardown is a
# property of how the layers are kept current rather than the cost of putting a
# forecast on the map. total_s is still true wall clock and includes it, so the
# update cost of the framework is total_s - delete_s.
#
# The tuple is fixed and complete so the header stays stable across horizons
# and across run/test mode: a stage that did not run reports NaN rather than
# dropping its column and misaligning every later row in an appended CSV.
#
# One issue covers num_timesteps forecast steps (the horizon, or fewer once the
# boundary conditions run out near the end of the window), so per-timestep cost
# is stage_s / num_timesteps -- which is what makes horizons comparable.
TIMING_COLUMNS = (
    "forecast_horizon",
    "num_timesteps",
    # The ensemble the timings were produced at. M and K are recorded
    # separately as well as pooled because they cost differently: M loads and
    # runs that many distinct models, K draws that many samples from each, and
    # ensemble_size = M*K is what the decode and the downstream exposure
    # computation are linear in.
    "num_deep_ensemble",
    "samples_per_member",
    "ensemble_size",
    "bc_s",
    "inundation_s",
    "exposure_s",
    "extent_s",
    "delete_s",
    "publish_s",
    "total_s",
)


RUNTIME_METRICS_FILE = "runtime_metrics.csv"


def _stamp(value):
    """One timestamp spelling for every appended results file.

    Formatted rather than left as a Timestamp because a chunk whose stamps all
    fall on midnight writes as a bare date, so an appended file ends up with two
    time formats in one column and anything reading it back has to parse both.
    """
    return None if value is None else pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def _timed(timings, stage, accumulate=False):
    """Record the wall-clock seconds spent in the block under ``stage``.

    perf_counter is enough for the inundation surrogate despite it running on
    the GPU: a step ends at torch.from_numpy, so the device work has already
    been synced to host by the time the block exits.

    ``accumulate`` adds to whatever is already recorded instead of replacing
    it, which is what the stepped horizon needs: inundation and risk now
    interleave per step rather than running as two contiguous blocks, so each
    stage is the sum over the steps. Requires the stage to start at 0.0.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[stage] = timings[stage] + elapsed if accumulate else elapsed


def _timed_steps(timings, stage, steps):
    """Yield from ``steps``, charging only the time spent PRODUCING each item.

    The hazard forecast is a generator now, so its work happens inside
    ``next()`` rather than in one call the caller can wrap. Timing the
    iteration this way keeps inundation_s measuring the surrogate alone, with
    the consumer's own work charged to whatever stage it opens.
    """
    timings[stage] = 0.0
    iterator = iter(steps)
    while True:
        start = time.perf_counter()
        try:
            item = next(iterator)
        except StopIteration:
            timings[stage] += time.perf_counter() - start
            return
        timings[stage] += time.perf_counter() - start
        yield item


def _merge_impact(accumulated, step_impact):
    """Fold one step's impact frames into the run's accumulator.

    get_exposure_assessment returns ``{object_type: {timestamp: gdf}}``; stepping
    the horizon means merging the inner dicts rather than replacing them. What
    accumulates here is only the impact frames -- a few thousand rows per
    object type per step -- while the depth maps they were derived from are
    released as soon as the step is done with.
    """
    for key, series in step_impact.items():
        accumulated.setdefault(key, {}).update(series)
    return accumulated


def _align_appended_csv(output_file, columns):
    """Make an appended results file's header match ``columns`` before writing.

    Every results file here is written with ``mode="a"`` and the header written
    ONCE, so the columns of a later run have to line up with the header already
    on disk -- positionally, not just by name -- or the run's values land under
    the wrong names. Two cases, and they are NOT the same failure:

    * The new run only ADDS columns (a statistic added to the ablation, say).
      Nothing already in the file changes meaning, so the file is REWRITTEN with
      the new columns in place and empty on the existing rows, and the run
      appends to it. The accumulated runs survive a widened schema, and an empty
      cell says exactly what happened: that run did not compute it.
    * Anything else -- a column renamed, dropped, or one whose meaning moved --
      can silently re-point a column at a different quantity, which is what this
      guard exists to stop. The file is archived and a new one started.

    Both cases leave a timestamped ``.csv.bak`` copy, so the rewrite is
    recoverable too. Returns the path of that copy, or None if the header
    already matched.
    """
    if not os.path.exists(output_file):
        return None

    existing = pd.read_csv(output_file, nrows=0).columns.tolist()
    columns = list(columns)
    if existing == columns:
        return None

    name = os.path.basename(output_file)
    stem = os.path.splitext(output_file)[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Second-resolution stamps collide if one file is migrated twice inside the
    # same second, and os.rename would overwrite the earlier copy without a
    # word -- the one thing a backup must not do.
    archived = f"{stem}_{stamp}.csv.bak"
    for attempt in range(2, 100):
        if not os.path.exists(archived):
            break
        archived = f"{stem}_{stamp}-{attempt}.csv.bak"

    # Subset, not strict subset: a pure REORDER is also safe to migrate, since
    # the rewrite below places every existing value by name.
    if set(existing) <= set(columns):
        shutil.copy2(output_file, archived)
        added = [column for column in columns if column not in existing]
        pd.read_csv(output_file).reindex(columns=columns).to_csv(
            output_file, index=False)
        change = (
            f"gained {len(added)} column(s) ({', '.join(added[:3])}"
            f"{', ...' if len(added) > 3 else ''})" if added
            else "kept its columns in a different order")
        logger.warning(
            "%s %s; rewrote it to match, existing rows carry the new columns "
            "empty, copy at %s", name, change, os.path.basename(archived),
        )
        return archived

    # A column dropped or renamed: archive the file and let the run start a new one.
    os.rename(output_file, archived)
    dropped = [column for column in existing if column not in columns]
    logger.warning(
        "%s lost %d column(s) (%s%s); archived it as %s and started a new file",
        name, len(dropped), ", ".join(dropped[:3]), ", ..." if len(dropped) > 3 else "",
        os.path.basename(archived),
    )
    return archived
