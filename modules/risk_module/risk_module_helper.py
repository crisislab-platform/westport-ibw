"""Hazard indicators, context aggregation and exceedance probabilities.

Steps 2-5 of the risk module: the indicators an exposure pass has already
summarised per object (Step 1) are aggregated over the objects of a spatial
context, the aggregation is repeated over the K ensemble members, and the
resulting ensemble is read against the ARR class lower bounds to give the two
axes of the risk matrix -- a hazard class and the likelihood of reaching it.

NO INFERENCE RUNS HERE. This module replaces the fuzzy engine that used to sit
between the object depths and the reported tier: Equation (7) is a max over
objects, Equation (8) counts members over a threshold, and nothing else stands
between the depth raster and the risk matrix. The consequences of that are what
the ablation of the nine configurations measures, so the rule must stay free of
weights and tuned parameters -- see CONFIGURATIONS.

Step 1 itself stays in ``impact_module``: h_max, h_mean and h_wet_mean of the
cells intersecting each object (Equations 5 and 6) are what
``buffered_flood_depth`` and ``zonal_flood_depth`` already return, on the
``depth_max``, ``depth_mean`` and ``depth_wet_mean`` columns of one exposure
frame per member per object type.
"""

import numpy as np
import pandas as pd

# The three hazard classes of the risk matrix and the depth each one STARTS at,
# in metres: the ARR (Australian Rainfall and Runoff) Guideline 7-3 hazard
# vulnerability curves read at zero velocity, which is where a depth-only
# indicator has to read them. Closed below, so a building at exactly 0.30 m is
# H2. tau_H1 = 0 m is not a cutoff of the guideline's but the floor of the
# depth scale, which is why P(h >= tau_H1) = 1 for every building, dry ones
# included -- H1 is the "generally safe for people, vehicles and buildings"
# class and a dry building belongs in it.
CLASS_TERMS = ("H1", "H2", "H3+")
CLASS_LOWER_BOUNDS = (0.0, 0.30, 0.50)

# 1-based positions of the classes, which is how a hazard class travels in the
# impact frames and the published layers: 1 = H1, 2 = H2, 3 = H3+. Ordinal, so
# an ordered score (QWK, RPS) can be taken on them directly.
HAZARD_LEVELS = tuple(range(1, len(CLASS_TERMS) + 1))

# Position of the first class that carries a warning -- H2, the first one above
# the safe class. Indexed rather than named so a consumer selecting "the classes
# that matter" does not restate which one that is.
MIDDLE_CLASS = 1

# Column-name spellings of the terms: "+" is not usable in an identifier, and
# the same "_above" spelling the previous engine published under keeps the
# regenerated CSVs readable by the scripts that already parse those names.
CLASS_COLUMN_TERMS = ("H1", "H2", "H3_above")

# The likelihood axis: the continuous exceedance probability partitioned into
# three ordinal categories,
#
#     Unlikely  p < 10%      Likely  10% <= p < 25%      Very Likely  p >= 25%
#
# The terms are the paper's own (Step 3 of the risk module), so a layer popup,
# a table and the manuscript all name a band the same way; they were low /
# medium / high until 2026-08-27. The prose here said "low < 5% <= medium" for
# some time while the constant read 0.1 -- the CONSTANT is authoritative and the
# cuts have always been 10% and 25%.
#
# THESE BOUNDS ARE A DEMONSTRATION CHOICE, not a calibrated or consulted one.
# They are set low on purpose: a modest level of ensemble agreement should be
# enough to surface a rare but severe outcome, since the failure that matters
# in a warning system is the under-estimated one. Raising them makes the system
# quieter and more confident in equal measure.
LIKELIHOOD_TERMS = ("Unlikely", "Likely", "Very Likely")
LIKELIHOOD_BOUNDS = (0.1, 0.25)

# The three objects a building's spatial context can draw on: its own one-cell
# ring R_b, its property parcel P(b) and its street parcel S(b).
OBJECTS = ("building", "property", "street")

# The configurations of the ablation, each naming the objects whose indicators
# Equation (7) takes the max over. The contexts are NESTED, so widening one can
# only raise the aggregated indicator -- the ablation therefore trades misses
# for false alarms monotonically, and reading the rows in this order is
# reading that trade.
CONFIGURATIONS = {
    "BuildingOnly": ("building",),
    "BuildProp": ("building", "property"),
    "BuildPropStreet": ("building", "property", "street"),
}

# The statistics of Equations (5) and (6), as (paper suffix, exposure column).
# Every object of a configuration is read with the SAME statistic -- mixing them
# would make the max of Equation (7) a comparison between different quantities.
#
# The three are ONE ORDERED FAMILY, not three unrelated readings: on any object,
#
#     h_mean <= h_wet_mean <= h_max
#
# because h_wet_mean is h_mean with the dry cells taken out of the denominator
# (impact_module.WET_DEPTH_MIN). h_mean asks how much water the object's
# footprint holds, h_max how deep the worst cell touching it is, and h_wet_mean
# how deep the water is WHERE THE OBJECT IS WET -- the reading that does not
# punish an object for standing beside dry ground, and does not let one deep
# cell speak for the whole object either. Where it lands between the other two
# is exactly the wet fraction, which is why the ablation can now separate "the
# context is wetter" from "the context is bigger".
STATISTICS = {"Max": "depth_max", "Mean": "depth_mean",
              "WetMean": "depth_wet_mean"}


def class_lower_bounds():
    """``{term: tau_A}`` in metres, safe -> severe.

    The bounds of Equation (8), for a caller that needs to report or log what
    the probabilities were taken against rather than recompute them.
    """
    return dict(zip(CLASS_TERMS, CLASS_LOWER_BOUNDS.tolist()
                    if hasattr(CLASS_LOWER_BOUNDS, "tolist")
                    else CLASS_LOWER_BOUNDS))


def _name(prefix, suffix):
    """``prefix_suffix``, or bare ``suffix`` when the prefix is empty.

    An empty prefix is what a layer reporting ONE configuration under generic
    names uses -- the road and property layers, and the building layer's
    headline columns -- so the published schema does not have to be rewritten
    when the reported configuration changes.
    """
    return f"{prefix}_{suffix}" if prefix else suffix


def configuration_prefix(configuration, statistic):
    """Column-name prefix of one configuration, e.g. ``buildpropstreet_max``.

    The paper name lowercased, so a column can be traced back to a row of the
    results table by eye and the ``out/error_analysis_buildings_*.csv`` naming
    already in use is reproduced exactly.
    """
    return f"{configuration.lower()}_{statistic.lower()}"


class RiskLevelFinder:
    """Aggregates object hazard indicators and reads them against the risk matrix.
    """

    def __init__(self, class_bounds=CLASS_LOWER_BOUNDS, class_terms=CLASS_TERMS,
                 likelihood_bounds=LIKELIHOOD_BOUNDS,
                 likelihood_terms=LIKELIHOOD_TERMS,
                 min_class_probability=0.0):
        """
        Parameters
        ----------
        class_bounds : sequence of float
            The class lower bounds tau_A in metres, ascending. Must be the same
            length as ``class_terms``.
        class_terms : sequence of str
            Names of the classes, safe -> severe.
        likelihood_bounds : sequence of float
            The cuts of the likelihood axis, ascending; one fewer than
            ``likelihood_terms``.
        likelihood_terms : sequence of str
            Names of the likelihood categories, least -> most likely.
        min_class_probability : float
            How much of the ensemble has to reach a class before the building is
            placed in it. The default 0.0 is the literal reading of the risk
            matrix -- the highest class ANY member reaches, with the likelihood
            axis then saying how much of the ensemble backed it, which is what
            lets an unlikely cell report a rare severe outcome instead of
            hiding it. Raise it to elect only classes a minimum share of members
            agree on -- and note that AT OR ABOVE the first likelihood bound
            (0.1) the elected class can never be "Unlikely", which makes the
            Unlikely column of risk_module.RISK_MATRIX unreachable.
        """
        if len(class_bounds) != len(class_terms):
            raise ValueError("class_bounds and class_terms must be the same length")
        if len(likelihood_bounds) != len(likelihood_terms) - 1:
            raise ValueError("likelihood_bounds must be one shorter than "
                             "likelihood_terms")
        self.class_bounds = np.asarray(class_bounds, dtype=float)
        self.class_terms = tuple(class_terms)
        self.likelihood_bounds = np.asarray(likelihood_bounds, dtype=float)
        self.likelihood_terms = tuple(likelihood_terms)
        self.min_class_probability = float(min_class_probability)

    # -- Step 1 output: the object indicators, gathered onto the reported rows --
    @staticmethod
    def object_indicators(members, statistic="Max", index=None):
        """Stack one object type's indicator across the ensemble.

        Parameters
        ----------
        members : sequence of DataFrame
            One exposure frame per ensemble member for a single object type,
            carrying the ``depth_max``, ``depth_mean`` and ``depth_wet_mean``
            columns of Equations (5) and (6) on that object's own rows.
        statistic : key of STATISTICS
            Which indicator to read -- "Max", "Mean" or "WetMean".
        index : array-like of int, optional
            Positional row of the object belonging to each building, as the
            building link table gives it; -1 for a building with no such object,
            whose indicator becomes 0.0 (see the class docstring). Omitted, the
            object's own rows are returned unchanged -- which is what a road or
            property layer reported in its own right wants.

        Returns
        -------
        ndarray, shape (n_members, n_rows)
        """
        column = STATISTICS[statistic]
        depths = np.vstack([np.asarray(m[column], dtype=float) for m in members])
        if index is None:
            return depths

        index = np.asarray(index, dtype=int)
        missing = index < 0
        depths = depths[:, np.where(missing, 0, index)]
        depths[:, missing] = 0.0
        return depths

    # -- Step 2/3: aggregation over the objects of a spatial context --

    def aggregate(self, object_depths, configuration):
        """Equation (7): the context indicator, max over the configuration's objects.

        Parameters
        ----------
        object_depths : mapping of str to ndarray
            ``{"building": ..., "property": ..., "street": ...}``, each
            ``(n_members, n_features)`` on the same rows and read with the same
            statistic. Only the objects the configuration names are touched.
        configuration : str
            A key of CONFIGURATIONS.

        Returns
        -------
        ndarray, shape (n_members, n_features)
            The most severe indicator in the context, per member per building.
        """
        objects = CONFIGURATIONS[configuration]
        missing = [o for o in objects if o not in object_depths]
        if missing:
            raise KeyError(f"{configuration} needs {missing}, which object_depths "
                           f"does not carry")

        aggregated = np.asarray(object_depths[objects[0]], dtype=float)
        for name in objects[1:]:
            other = np.asarray(object_depths[name], dtype=float)
            if other.shape != aggregated.shape:
                raise ValueError(
                    f"{name} indicators are {other.shape}, {objects[0]} are "
                    f"{aggregated.shape}; all objects of a configuration must be "
                    f"on the same members and rows")
            # np.maximum, not np.fmax: an object whose depth could not be
            # evaluated must not be quietly ignored in favour of its neighbour.
            aggregated = np.maximum(aggregated, other)
        return aggregated

    # -- Step 4: the ensemble read against the class bounds --

    def exceedance_probabilities(self, indicator):
        """Equation (8): P_A = share of members with h >= tau_A, per class.

        Parameters
        ----------
        indicator : array-like, shape (n_members, n_features)
            The aggregated context indicator of Equation (7), in metres.

        Returns
        -------
        ndarray, shape (n_features, n_classes)
            Non-increasing across each row, since the bounds are nested; column
            0 is 1.0 wherever tau_H1 is 0 m. All-NaN if the ensemble is empty.
        """
        indicator = np.atleast_2d(np.asarray(indicator, dtype=float))
        n_members, n_features = indicator.shape
        if n_members == 0:
            return np.full((n_features, len(self.class_bounds)), np.nan)
        # Compared PER MEMBER, so no averaged depth ever meets a bound: the
        # ulp-below-the-cutoff problem that made an averaged 0.30 m digitise
        # one class low cannot arise here, and no boundary tolerance is needed.
        # The averaged indicator is reported (columns) but never thresholded.
        exceeds = (np.isnan(indicator)[..., None]
                   | (indicator[..., None] >= self.class_bounds))
        return exceeds.mean(axis=0)

    # -- Step 5: the two axes of the risk matrix --

    def hazard_class(self, probabilities):
        """The hazard axis: the highest class whose lower bound is exceeded.

        Returned as a 1-based level -- 1 = H1, 2 = H2, 3 = H3+ -- matching the
        ordering of ``class_terms`` so a level indexes them directly. Class 1 is
        always attainable while tau_H1 is 0 m, so the level is never undefined.

        ``min_class_probability`` decides what "exceeded" means; see __init__.
        """
        probabilities = np.asarray(probabilities, dtype=float)
        attained = probabilities > self.min_class_probability
        n_classes = probabilities.shape[1]
        level = n_classes - np.argmax(attained[:, ::-1], axis=1)
        return np.where(attained.any(axis=1), level, 1).astype(int)

    def hazard_class_of_depth(self, depth):
        """The class ONE depth falls in, without an ensemble behind it.

        The deterministic counterpart of ``hazard_class``: it takes a single
        (n_features,) indicator -- the ensemble mean of Equation (7), or the
        indicator read off the ensemble-mean flood map -- and compares it to the
        bounds directly. Those two collapse the ensemble BEFORE the class rather
        than after it, and are reported as references the probabilistic reading
        is scored against; neither is the risk matrix's own answer.

        Closed below, so exactly 0.30 m is H2. A NaN depth takes the most severe
        class, as everything else here does.
        """
        depth = np.asarray(depth, dtype=float)
        return np.where(
            np.isnan(depth), len(self.class_bounds),
            # bounds[0] is the floor of the scale, exceeded by every depth, so
            # only the interior boundaries decide the class.
            np.digitize(depth, self.class_bounds[1:]) + 1,
        ).astype(int)

    def likelihood(self, probabilities, level=None):
        """The likelihood axis: P_A of the class the building was placed in."""
        probabilities = np.asarray(probabilities, dtype=float)
        if level is None:
            level = self.hazard_class(probabilities)
        return np.take_along_axis(
            probabilities, np.asarray(level, dtype=int)[:, None] - 1, axis=1
        )[:, 0]

    def likelihood_category(self, probability):
        """The ordinal band a likelihood falls in, as a 1-based level.

        1 = Unlikely, 2 = Likely, 3 = Very Likely, closed below at each bound so
        exactly 10% is Likely and exactly 25% is Very Likely. A NaN likelihood
        -- an empty ensemble -- takes the top band, the same refusal to
        under-report an unevaluated forecast the rest of the module makes.
        """
        probability = np.asarray(probability, dtype=float)
        return np.where(
            np.isnan(probability), len(self.likelihood_terms),
            np.digitize(probability, self.likelihood_bounds) + 1,
        ).astype(int)

    # -- The whole of Steps 2-5 for one configuration --

    def assess(self, object_depths, configuration, statistic="Max"):
        """Run one configuration end to end and return its risk-matrix reading.

        Returns
        -------
        dict
            ``configuration``, ``statistic``, ``objects``, the aggregated
            ``indicator`` stack, its ``probabilities`` (n_features, n_classes),
            the ``level`` and ``term`` of the hazard axis, and the
            ``likelihood`` with its ``likelihood_level`` and
            ``likelihood_term``.
        """
        indicator = self.aggregate(object_depths, configuration)
        probabilities = self.exceedance_probabilities(indicator)
        
        # Select the ordinally higest class with at least 10% exceedence probability and get its likelihood
        level = self.hazard_class(probabilities)
        likelihood = self.likelihood(probabilities, level)
        
        likelihood_level = self.likelihood_category(likelihood)
        terms = np.asarray(self.class_terms)
        likelihood_terms = np.asarray(self.likelihood_terms)
        
        # risk_level = self.risk_level(likelihood_terms[likelihood_level - 1])
        # risk_term = self.risk_category(risk_level)
        return {
            "configuration": configuration,
            "statistic": statistic,
            "objects": CONFIGURATIONS[configuration],
            "indicator": indicator,
            "probabilities": probabilities,
            "level": level,
            "term": terms[level - 1],
            "likelihood": likelihood,
            "likelihood_level": likelihood_level,
            "likelihood_term": likelihood_terms[likelihood_level - 1],
            # "risk_ceatrgory":
            # "risk_term":
        }
        
    # def risk_level(self,hazard_level, likelihood_level):
    #     """The ordinal band a likelihood falls in, as a 1-based level.

    #     1 = low, 2 = medium, 3 = high, closed below at each bound so exactly 5%
    #     is medium and exactly 25% is high. A NaN likelihood -- an empty ensemble
    #     -- takes the top band, the same refusal to under-report an unevaluated
    #     forecast the rest of the module makes.
    #     """
    #     hazard_level = np.asarray(hazard_level, dtype=float)
    #     likelihood_level = np.asarray(likelihood_level, dtype=float)
       
        
    # def risk_term(self,risk_level):

        
        
    def columns(self, object_depths, configuration, statistic="Max", prefix=None):
        """One configuration's reading, flattened into DataFrame columns.

        ``{prefix}_indicator_mean`` and ``_indicator_std`` are the ensemble mean
        and spread of the aggregated indicator in metres. The mean is reported
        beside the classes rather than instead of them because a class alone
        cannot distinguish an ensemble agreeing on 0.55 m from one spread over
        0.0-1.1 m, and those are different forecasts to act on.

        ``{prefix}_hazard_level`` (1/2/3) and ``{prefix}_likelihood`` are the two
        axes of the risk matrix; ``_hazard_class`` and ``_likelihood_class`` are
        the same two as names, for a map legend or a table.

        Two REFERENCE readings travel beside them, both collapsing the ensemble
        before the class instead of after it, and neither is the risk matrix's
        answer:

        ``{prefix}_ensemble_mean_hazard_level``
            the class of the averaged aggregated indicator -- the mean of the
            members' maxima.
        ``{prefix}_mean_map_hazard_level``
            the class of the indicator aggregated from ``mean_map_depths``,
            exposure run once on the ensemble-mean flood map. It cannot be
            recovered from the member stacks: the deepest cell of the average
            flood is not the average of the members' deepest cells, and the
            members need not peak in the same cell. Omitted, the column is not
            emitted.

        An empty ``prefix`` publishes under bare names (``hazard_level``, ...),
        which is what a layer reporting a single configuration wants.
        """
        prefix = configuration_prefix(configuration, statistic) if prefix is None \
            else prefix
        assessed = self.assess(object_depths, configuration, statistic)
        indicator, probabilities = assessed["indicator"], assessed["probabilities"]

        columns = {
            _name(prefix, "indicator_mean"): np.nanmean(indicator, axis=0),
            _name(prefix, "indicator_std"): (
                np.nanstd(indicator, axis=0, ddof=1) if len(indicator) > 1
                else np.zeros(indicator.shape[1])
            ),
        }
        
        for position, term in enumerate(CLASS_COLUMN_TERMS[:len(self.class_terms)]):
            columns[_name(prefix, f"prob_at_least_{term}")] = probabilities[:, position]
        columns[_name(prefix, "hazard_level")] = assessed["level"]
        columns[_name(prefix, "hazard_class")] = assessed["term"]
        columns[_name(prefix, "likelihood")] = assessed["likelihood"]
        columns[_name(prefix, "likelihood_level")] = assessed["likelihood_level"]
        columns[_name(prefix, "likelihood_class")] = assessed["likelihood_term"]

        columns[_name(prefix, "ensemble_mean_hazard_level")] = (
            self.hazard_class_of_depth(columns[_name(prefix, "indicator_mean")]))
        return columns

    # -- Every configuration of the ablation, in one pass --
    def building_columns(self, building_members, property_members, street_members,
                         property_idx, street_idx,
                         configurations=tuple(CONFIGURATIONS),
                         statistics=tuple(STATISTICS),
                         reported=None):
        """Every configuration x statistic, from the exposure frames themselves.

        The building rows are the reported rows; the property and street frames
        are gathered onto them with ``property_idx`` / ``street_idx`` (the
        positional links, -1 where a building has neither). Exposure is not
        repeated per statistic -- every indicator is already on the frames this
        is handed, so a configuration costs one maximum over arrays in hand.

        ``mean_map_members`` is the same three frames computed once from the
        ensemble-mean flood map, as ``{"building": frame, "property": frame,
        "street": frame}``; supplied, every configuration also reports the class
        that map gives (see ``columns``).

        ``reported`` is the ``(configuration, statistic)`` whose reading is
        ALSO published under bare names, for consumers that want one hazard
        class per building without knowing which configuration produced it --
        the hosted layers and the maps. It is a duplicate of that
        configuration's own columns, not a separate computation.

        Also emitted: ``property_known`` / ``street_known``, and the property's
        and street's own readings under ``{object}_{statistic}_*``, so a row
        says what was measured on the parcel and the street beside what the
        context made of them. The footprint's own reading is BuildingOnly.

        Returns a single dict ready for ``DataFrame.assign``.
        """
        columns = {
            "property_known": np.asarray(property_idx) >= 0,
            "street_known": np.asarray(street_idx) >= 0,
        }
        for statistic in statistics:
            object_depths = {
                "building": self.object_indicators(building_members, statistic),
                "property": self.object_indicators(
                    property_members, statistic, property_idx),
                "street": self.object_indicators(
                    street_members, statistic, street_idx),
            }
            # mean_map_depths = None if mean_map_members is None else {
            #     "building": self.object_indicators(
            #         [mean_map_members["building"]], statistic),
            #     "property": self.object_indicators(
            #         [mean_map_members["property"]], statistic, property_idx),
            #     "street": self.object_indicators(
            #         [mean_map_members["street"]], statistic, street_idx),
            # }
            for configuration in configurations:
                emitted = self.columns(object_depths, configuration, statistic)
                columns.update(emitted)
                if reported == (configuration, statistic):
                    reported_prefix = configuration_prefix(configuration, statistic)
                    columns.update({
                        name[len(reported_prefix) + 1:]: values
                        for name, values in emitted.items()
                    })

            # The property and street read one at a time, on the building
            # rows: a single object IS its own context, so this is the
            # BuildingOnly rule applied to whichever object is passed. The
            # BUILDING is not repeated here -- BuildingOnly already is its own
            # reading, and emitting it twice under two names would let a
            # consumer compare one against the other and think it had learnt
            # something.
            for name in OBJECTS[1:]:
                columns.update(self.columns(
                    {"building": object_depths[name]}, "BuildingOnly", statistic,
                    prefix=f"{name}_{statistic.lower()}"))
        return columns

    def  object_columns(self, members, statistic="Max", prefix="",
                       mean_map_member=None):
        """A road parcel or property assessed in its own right, on its own rows.

        There is no context to aggregate over -- the object IS the context -- so
        this is Equation (8) applied straight to the object's own indicator. It
        gives a street parcel a class of its own rather than one only readable
        off the buildings that happen to face it. The default empty prefix
        publishes the bare names the building layer's headline columns use, so
        one legend reads all three layers.
        """
        depths = self.object_indicators(members, statistic)
        mean_map_depths = None if mean_map_member is None else {
            "building": self.object_indicators([mean_map_member], statistic)}
        return self.columns({"building": depths}, "BuildingOnly", statistic,
                            prefix=prefix)

    def risk_matrix_counts(self, columns, prefix):
        """How many buildings land in each cell of the risk matrix.

        A summary for the figure and the log, over the columns ``columns``
        emitted: hazard class down the rows, likelihood category across.
        """
        table = pd.crosstab(
            pd.Categorical(columns[f"{prefix}_hazard_class"],
                           categories=list(self.class_terms), ordered=True),
            pd.Categorical(columns[f"{prefix}_likelihood_class"],
                           categories=list(self.likelihood_terms), ordered=True),
            dropna=False,
        )
        table.index.name, table.columns.name = "hazard_class", "likelihood"
        return table
