import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from modules.exposure_module.exposure_module import is_exposed


# Binary outcome labels for the placard validation.
AFFECTED = "affected"
NOT_AFFECTED = "not_affected"

# Lowest MBIE damage class counted as "affected". 1 (Slight damage) makes the
# outcome "water got into the building", which is the inundation threshold a
# depth model actually predicts; raising it to 2 or 3 moves the test towards
# structural failure, which is force-driven and needs the velocity this model
# does not carry. Exposed as a constant so the cut can be swept.
AFFECTED_DAMAGE_MIN = 1

# The MBIE damage classes the survey uses (0 = no damage ... 3 = heavy). Fixed
# rather than read off the data because the per-level columns are appended to a
# CSV whose header is written once: a level set that shrinks when a scenario
# happens to contain no class-3 building would silently misalign later rows.
DAMAGE_LEVELS = (0, 1, 2, 3)


def crps_ensemble(members, observed):
    """CRPS of an ensemble forecast, per building.

    ``members`` is (members, buildings), ``observed`` (buildings,). The
    empirical-ensemble form E|X - y| - 1/2 E|X - X'|, with E|X - X'| taken from
    the sorted members as (2 / M^2) * sum_i (2i - M - 1) x_(i).
    """
    members = np.asarray(members, dtype=float)
    observed = np.asarray(observed, dtype=float)
    m = members.shape[0]
    accuracy = np.abs(members - observed[None, :]).mean(axis=0)
    ordered = np.sort(members, axis=0)
    weights = (2 * np.arange(1, m + 1) - m - 1)[:, None]
    spread = (2.0 / m**2) * (weights * ordered).sum(axis=0)
    return accuracy - 0.5 * spread


def has_spread(members):
    """Whether a forecast is an ensemble, ``members`` being (members, ...).

    CRPS and the Brier score are scored only when it is: for a one-member
    (deterministic) forecast CRPS is just the absolute error and the Brier
    score the 0/1 exposure miss, so they are left NaN rather than reported as
    probabilistic scores.
    """
    return np.asarray(members).shape[0] > 1


def mean_std(values, name):
    """``{name: mean, name_std: standard deviation}`` of ``values``, NaNs
    ignored; both NaN when nothing is left. np.std (population SD), as the
    westport scores report their spreads."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if not values.size:
        return {name: np.nan, f"{name}_std": np.nan}
    return {name: float(values.mean()), f"{name}_std": float(values.std())}


def brier_score(probability, outcome):
    """Mean squared difference between P(event) and the 0/1 outcome."""
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    return float(np.mean((probability - outcome) ** 2)) if len(outcome) else np.nan


def rmse(forecast, observed):
    forecast = np.asarray(forecast, dtype=float)
    observed = np.asarray(observed, dtype=float)
    return float(np.sqrt(np.mean((forecast - observed) ** 2))) if len(observed) else np.nan


def stratified_rmse(forecast, observed, prefix="rmse"):
    """RMSE over all buildings, and over the observed-wet and observed-dry ones
    (wet = observed depth exposed, is_exposed)."""
    forecast = np.asarray(forecast, dtype=float)
    observed = np.asarray(observed, dtype=float)
    wet = is_exposed(observed)
    return {
        f"{prefix}_all": rmse(forecast, observed),
        f"{prefix}_wet": rmse(forecast[wet], observed[wet]),
        f"{prefix}_dry": rmse(forecast[~wet], observed[~wet]),
    }


def pearson(x, y):
    """Pearson correlation, NaN when either side is constant."""
    return pearson_test(x, y)[0]


def pearson_test(x, y):
    """Pearson correlation and its two-sided p-value (H0: r = 0).
    (NaN, NaN) when either side is constant."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan
    result = pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def contingency(observed_event, forecast_event):
    """Hits, misses, false alarms and correct negatives, with the hit rate
    (hits / (hits + misses)) and CSI (hits / (hits + misses + false alarms))."""
    observed_event = np.asarray(observed_event, dtype=bool)
    forecast_event = np.asarray(forecast_event, dtype=bool)
    hits = int((observed_event & forecast_event).sum())
    misses = int((observed_event & ~forecast_event).sum())
    false_alarms = int((~observed_event & forecast_event).sum())
    correct_negatives = int((~observed_event & ~forecast_event).sum())

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else np.nan

    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "hit_rate": ratio(hits, hits + misses),
        "csi": ratio(hits, hits + misses + false_alarms)
    }


def onset_times(series, times):
    """First time each building is exposed (is_exposed).

    ``series`` is (times, buildings). Returns an object array of Timestamps,
    NaT where the building is never exposed. A building already exposed at the
    first step gets the first step's time: the onset is censored there.
    """
    exposed = is_exposed(series)
    first = exposed.argmax(axis=0)
    times = pd.DatetimeIndex(times)
    out = np.array(times[first], dtype="datetime64[ns]")
    out[~exposed.any(axis=0)] = np.datetime64("NaT")
    return out


def peaks(series, times):
    """Each building's peak depth and the (first) time it is reached."""
    series = np.asarray(series)
    index = series.argmax(axis=0)
    depth = series[index, np.arange(series.shape[1])]
    return depth, pd.Series(pd.DatetimeIndex(times)[index]).to_numpy()


def hours_between(later, earlier):
    """later - earlier in hours, NaN where either is NaT."""
    return (pd.to_datetime(later) - pd.to_datetime(earlier)) / pd.Timedelta(hours=1)


def _errors_by_damage_level(y_true, y_pred, damage_levels):
    """Split the misses and false alarms across the observed damage classes.

    The binary score collapses DamageNb to affected/not-affected, which hides
    whether the misses are slight-damage buildings the depth threshold sits on
    top of or heavily damaged ones the model had no business missing. This
    keeps the same contingency table and only asks which observed class each
    error came from.

    Returns flat ``*_damage_<k>`` columns so the breakdown rides along in the
    existing one-row-per-indicator CSV. ``n_damage_<k>`` is the number of
    scored buildings in that class, so hits (k >= AFFECTED_DAMAGE_MIN) and
    correct negatives (k = 0) are the complement of the error column and are
    not stored twice. Note the two error columns are mutually exclusive by
    construction: only class 0 can produce a false alarm and only the affected
    classes can produce a miss -- the zeros on the other side are kept so the
    schema does not depend on where AFFECTED_DAMAGE_MIN is cut.
    """
    if damage_levels is None:
        return {
            f"{stat}_damage_{level}": pd.NA
            for level in DAMAGE_LEVELS
            for stat in ("n", "misses", "false_alarms")
        }

    if not isinstance(damage_levels, pd.Series):
        damage_levels = pd.Series(damage_levels, index=y_true.index)
    # Align on the scored rows: the caller hands in the column off the same
    # frame y_true came from, so this only guards against a stray extra row.
    levels = damage_levels.reindex(y_true.index)

    missed = (y_true == AFFECTED) & (y_pred == NOT_AFFECTED)
    false_alarms = (y_true == NOT_AFFECTED) & (y_pred == AFFECTED)

    breakdown = {}
    for level in DAMAGE_LEVELS:
        at_level = levels == level
        breakdown[f"n_damage_{level}"] = int(at_level.sum())
        breakdown[f"misses_damage_{level}"] = int((at_level & missed).sum())
        breakdown[f"false_alarms_damage_{level}"] = int(
            (at_level & false_alarms).sum()
        )
    return breakdown


def _damage_to_affected(damage_nb):
    """Collapse an observed MBIE damage class to affected / not_affected.

    DamageNb 0 (No damage / Unaffected) is the negative case; anything at or
    above AFFECTED_DAMAGE_MIN is positive. Damaged classes below the threshold
    return NaN and drop out of the scoring rather than being counted as
    negatives, so raising the cut removes the ambiguous middle instead of
    relabelling buildings that were demonstrably affected as unaffected.
    """
    if int(damage_nb) == 0:
        return NOT_AFFECTED
    if int(damage_nb) >= AFFECTED_DAMAGE_MIN:
        return AFFECTED
    return np.nan
