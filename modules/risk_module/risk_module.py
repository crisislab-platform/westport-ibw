import hashlib
import rasterio

from modules.lib import BUILDING_FOOTPRINTS_PATH, ROAD_NETWORK_FILE_PATH, OUTPUT_DIR, PROPERTY_BOUNDARIES_PATH, BUILDING_LINKS_PATH, BUILDING_RINGS_DIR
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import logging
import shapely
from shapely.ops import unary_union
from shapely.geometry import shape
from rasterio import features
from rasterio.transform import Affine
from modules.utils.spatial_operators.operators import  get_sample_georeference,_get_feature_labels
from modules.utils.spatial_operators.operators import load_features
# The reductions below run wherever the depth tensor lands. check_device()
# returns CUDA when it is there and CPU otherwise, so the same code path
# serves a GPU workstation and a CPU-only laptop.
from modules.utils.utils import check_device
from modules.risk_module.risk_module_helper import (
    RiskLevelFinder, CONFIGURATIONS, STATISTICS, configuration_prefix,
)
import torch

logger = logging.getLogger(__name__)

# Probability at or above which a feature is treated as "likely" exposed.
LIKELY_PROBABILITY = 0.5

MIN_RING_CELLS = 4         # tune to your resolution / stats needs
MAX_BUFFER_PCT = 3        # cap growth so it doesn't blow up in open areas
GROWTH_STEP_PCT = 1.0

# Depth (m) at or above which a cell counts as wet. Cells below it are floored
# to zero before any statistic is taken, and it is also what splits the two
# means the zonal stats return.
#
# depth_mean divides by EVERY cell of the zone, dry ones included as zeros, so
# it is the product of two things: how deep the water is and how much of the
# zone it covers. A ring 40% covered at 0.55 m and one uniformly 0.22 m deep
# both score 0.22 m.
#
# depth_wet_mean divides by the WET CELLS ALONE, which is the other half of
# that product: how deep the water is WHERE THERE IS WATER, with coverage
# divided out. The identity is exact --
#
#     depth_mean = wet_fraction x depth_wet_mean
#
# -- so the three travel together and any two determine the third. A zone with
# no wet cell scores 0.0 on both, which keeps dry meaning dry.
#
# The consequence for a hazard indicator is that depth_wet_mean sits BETWEEN
# depth_mean and depth_max by construction, and behaves like depth_max on a
# barely-wet zone: one 0.60 m cell in a 200-cell ring reads 0.60 m, the same as
# the max, where depth_mean reads 0.003 m. It is the conditional reading -- how
# bad it is where it is wet -- not a diluted one.
WET_DEPTH_MIN = 0.05

# Depth class breaks for the published inundation extent, in metres. The first
# is the wet threshold; the rest are the ARR class limits H1/H2/H3/H4 that the
# hazard classes are built on, so the extent polygons and the object classes
# break at the same depths and a map reader can line the two up.
DEPTH_BREAKS_M = (0.05, 0.30, 0.50, 1.20, 2.00)

# Which percentile bands get vectorised: the best estimate and the reasonable
# worst case. Publishing only one would discard the spread the percentiles
# exist to carry.
FLOOD_EXTENT_BANDS = ("depth_p50", "depth_p90")

# Smallest polygon kept in the extent layer (m^2). Below roughly a tenth of a
# hectare the shapes are single-cell speckle: ~85% of the polygon count, none
# of the mapped area.
MIN_EXTENT_POLYGON_M2 = 100.0

# Cells aggregated per side for the published inundation grid, so 12 turns the
# 4.8 m model grid into 57.6 m blocks. The grid is the map of the water BETWEEN
# the modelled objects -- roads, open ground, access routes -- which the object
# layers cannot show and the dissolved extent polygons only show as a class.
#
# The org has no raster analytics licence and publishes through app credentials
# that cannot create items, so a hosted imagery layer is not available and the
# grid has to travel as features (see arcgis_module's docstring for the same
# constraint on the extent). That makes the block size a publishing budget, and
# a tighter one than it looks: the grid covers the whole domain rather than the
# wet part, and the domain is ~2.3M cells. Per timestep that is ~147k features
# at 4, ~38k at 8, ~17k at 12 and ~9.8k at 16, against 2000 features an upload
# batch. 12 buys full coverage for about what the wet-only 19.2 m grid cost,
# and still resolves which streets go under. Raise it to coarsen.
INUNDATION_BLOCK = 12

def _ensemble_mean(members, col):
    """Concatenate a column across ensemble members and return the per-row mean."""
    return pd.concat([m[col] for m in members], axis=1).mean(axis=1)


def _ensemble_std(members, col):
    """As _ensemble_mean, but the spread across members -- the forecast
    uncertainty on that depth, in the same units as the depth itself.
    """
    return pd.concat([m[col] for m in members], axis=1).std(axis=1)


def _with_columns(frame, columns):
    """``frame`` with ``columns`` added, joined in a single pass.

    ``DataFrame.assign`` sets its keyword columns ONE AT A TIME, so the 178
    columns of ``compute_risk_matrix`` arrive as 178 separate inserts and pandas
    warns about a fragmented frame once per insert past the hundredth -- ~78
    warnings per timestep, and a rebuilt block manager each time. Collecting
    them into one frame and concatenating joins them in one operation.

    Dropping first preserves assign's OVERWRITE semantics for any name the frame
    already carries; a plain concat would duplicate the column instead.
    """
    if not columns:
        return frame
    added = pd.DataFrame(columns, index=frame.index)
    overlap = [name for name in added.columns if name in frame.columns]
    if not overlap:
        return pd.concat([frame, added], axis=1)
    # A name the frame already carries has to be OVERWRITTEN, not repeated, and
    # assign overwrites it where it already sits. Dropping and re-appending
    # would move it to the end, so put the columns back in the order assign
    # would have left them: the frame's own order, then whatever is new.
    joined = pd.concat([frame.drop(columns=overlap), added], axis=1)
    return joined[list(frame.columns)
                  + [name for name in added.columns if name not in frame.columns]]


def _build_indicator_columns(
        members, depth_cols=("depth_max", "depth_mean", "depth_wet_mean")):
    """Ensemble-mean depth columns and their spread, plus the fraction of
    members flagging the feature inundated.

    Each depth is reported with a ``_std`` companion: a mean over members with
    no spread beside it cannot distinguish ten members that agree on 0.4 m from
    ten that range over 0.0-0.9 m, and those are different forecasts to act on.
    This is the depth-space uncertainty; the class-space one is the exceedance
    probability of hazard_indicator_finder.Equation (8).
    """
    cols = {}
    for depth_col in depth_cols:
        cols[depth_col] = _ensemble_mean(members, depth_col)
        cols[f"{depth_col}_std"] = _ensemble_std(members, depth_col)
    return cols


# The configuration published under BARE column names -- hazard_level,
# likelihood, prob_at_least_H2 -- beside its own prefixed ones.
#
# BuildProp-Max as of 2026-08-27, at the user's decision, replacing
# BuildPropStreet-Max. The two disagree, and while the layers published the
# street-inclusive reading, every skill number in the paper -- the exceedance
# file, the horizon sweep, the reliability diagram -- was BuildProp-Max. A map
# and a table that disagree about what "the" hazard class is cannot both be
# cited, and the paper is the harder of the two to change.
#
# The other eight are still computed and still on the frame; this only decides
# which of them a consumer that asks for "the" hazard class gets. Changing it
# changes what the hosted layers and the maps show, and nothing else.
REPORTED_CONFIGURATION = ("BuildProp", "Max")
REPORTED_PREFIX = configuration_prefix(*REPORTED_CONFIGURATION)

# BuildProp-Max, which the exceedance file and the probabilistic verification in
# forecast_runner are written for: the footprint and its parcel, on h_max.
BUILDPROP_MAX_PREFIX = configuration_prefix("BuildProp", "Max")


# The risk matrix of Step 3: the hazard class an object reaches, read against
# the likelihood band that class was elected with, giving the MetService
# Yellow(1) / Orange(2) / Red(3) tier the warning is communicated as.
#
# Rows are the hazard level (H1, H2, H3+); columns are the likelihood level
# (1 = Unlikely, p < 10%; 2 = Likely, 10-25%; 3 = Very Likely, >= 25%), the
# bands of LIKELIHOOD_BOUNDS.
#
#                     UL      L      VL
#            H3+       2      3       3
#            H2        1      2       2
#            H1        1      1       1
#
# H1 is Yellow however likely it is; H2 and H3+ are downgraded one tier when the
# ensemble only calls them Unlikely.
#
# NOTE the interaction with min_class_probability. A class is elected only when
# more than that share of the ensemble reaches it, so at the 0.1 the sweeps run
# with -- which is exactly the UL/L boundary -- an elected class can never be
# Unlikely, the UL column is unreachable, and this matrix reduces to the
# identity: risk level == hazard level on every row. The column does work only
# if classes can be elected below 10% (min_class_probability < 0.1).
RISK_MATRIX = np.array([
    # UL  L  VL
    [1, 1, 1],   # H1
    [1, 2, 2],   # H2
    [2, 3, 3],   # H3+
])


def risk_level(hazard_level, likelihood_level):
    """The risk tier of a (hazard class, likelihood band) pair.

    Both are 1-based levels, as the impact frame carries them. Returns 1
    (Yellow), 2 (Orange) or 3 (Red).
    """
    hazard_level = np.asarray(hazard_level, dtype=int)
    likelihood_level = np.asarray(likelihood_level, dtype=int)
    return RISK_MATRIX[hazard_level - 1, likelihood_level - 1]


PEAK_SOURCE_COLUMNS = ("hazard_level", "hazard_class", "likelihood",
                       "likelihood_level", "likelihood_class",
                       "prob_at_least_H2", "prob_at_least_H3_above")


def window_peak(impact_series, id_column):
    """Collapse a window's per-step frames to one row per object.

    What the warning issued for this window promises an object over the whole
    horizon, rather than what it says at any one hour.

    ``peak_hazard_level`` is the highest class the object reaches at ANY step.
    Each step's class already respects ``min_class_probability`` -- a class is
    elected only when more than that share of the ensemble reaches it -- so the
    threshold rides along and is not applied a second time here.

    ``peak_valid_time`` is FIRST attainment of that class, and the likelihood,
    its band and the two probabilities are read AT THAT STEP: the row then
    describes one identifiable hour rather than a mixture of hours. Ties on the
    hour are impossible; ties on the class are broken by taking the earliest,
    which is the conservative reading for a warning.

    ``peak_risk_level`` is the max over the window of each step's RISK_MATRIX
    cell -- the step's hazard class read against the likelihood band it was
    elected with -- so it can peak at a different step from the class.

    Parameters
    ----------
    impact_series : dict
        ``{valid_time: GeoDataFrame}`` for one object type, one window.
    id_column : str
        The object's identifier column.

    Returns
    -------
    GeoDataFrame
        One row per object, carrying geometry and the ``peak_*`` columns.
    """
    if not impact_series:
        return None

    frames = []
    for valid_time, frame in impact_series.items():
        missing = [c for c in PEAK_SOURCE_COLUMNS if c not in frame.columns]
        if missing:
            raise KeyError(
                f"cannot collapse the window: the {valid_time} frame lacks "
                f"{missing}. The peak reading is the BARE configuration's "
                f"(risk_module.REPORTED_CONFIGURATION)."
            )
        part = frame[[id_column, *PEAK_SOURCE_COLUMNS]].copy()
        part["peak_valid_time"] = pd.Timestamp(valid_time)
        frames.append(part)

    stacked = pd.concat(frames, ignore_index=True)
    stacked["risk_level"] = risk_level(
        stacked["hazard_level"], stacked["likelihood_level"])

    peak_level = stacked.groupby(id_column)["hazard_level"].transform("max")
    at_peak = stacked[stacked["hazard_level"] == peak_level]
    # Earliest step that reaches the peak class, and that step's reading.
    first = (at_peak.sort_values("peak_valid_time")
             .groupby(id_column, as_index=False).first())

    first = first.rename(columns={
        "hazard_level": "peak_hazard_level",
        "hazard_class": "peak_hazard_class",
        "likelihood": "peak_likelihood",
        "likelihood_level": "peak_likelihood_level",
        "likelihood_class": "peak_likelihood_class",
        "prob_at_least_H2": "peak_prob_at_least_H2",
        "prob_at_least_H3_above": "peak_prob_at_least_H3_above",
    }).drop(columns="risk_level")
    first["peak_risk_level"] = (
        stacked.groupby(id_column)["risk_level"].max()
        .reindex(first[id_column]).to_numpy())

    # Geometry from any step -- the object set and its footprints do not change
    # inside a window.
    reference = next(iter(impact_series.values()))
    geometry = reference[[id_column, reference.geometry.name]]
    peak = gpd.GeoDataFrame(
        geometry.merge(first, on=id_column, how="inner"),
        geometry=reference.geometry.name, crs=reference.crs,
    )
    return peak


def compute_risk_matrix(building_impact, building_members, property_members,
                        road_members, property_idx, road_idx, finder):
    """Attach the risk-matrix reading of every configuration to the building rows.

    Steps 2-5 of the risk module, all of them in
    ``HazardIndicatorFinder.building_columns``: the objects' indicators are
    aggregated over each spatial context by the max of Equation (7), the
    aggregation is repeated across the ensemble, and the members are counted
    against the ARR class lower bounds by Equation (8). NO INFERENCE RUNS -- the
    fuzzy engine this replaced weighed the three objects against each other with
    a rule base; the max does it with no parameters to tune.

    All nine configurations (three contexts x h_mean/h_wet_mean/h_max) land side
    by side on the same rows, which is what makes the ablation of Section
    "Results: exposure" a comparison rather than nine pipeline runs. Exposure is
    NOT repeated for them: all three indicators are already on the frames handed
    in.

    ``mean_map_exposure`` is the same three exposure frames computed once from
    the ensemble-mean flood map, which cannot be derived from the member frames:
    that mean is taken over the depth rasters, upstream of exposure.
    """
    return _with_columns(building_impact, finder.building_columns(
        building_members, property_members, road_members,
        property_idx, road_idx,
        reported=REPORTED_CONFIGURATION,
    ))


class RiskModule():
    def __init__(self, scenario_id, min_exceed_prob, domain_raster=None):
       self.scenario_id = scenario_id
       self.domain_raster = domain_raster
       self.buildings = load_features(BUILDING_FOOTPRINTS_PATH, domain_raster)
       self._load_building_links()
       self.building_labels = None
       self.num_buildings = None
       self._building_label_grid = None
       self._zonal_labels = {}
       self._zonal_label_grid = {}
       self._footprint_union_cache = None
       self.use_adaptive_ring=True
       self.min_exceed_prob = min_exceed_prob
       self.risk_level_finder = RiskLevelFinder(min_class_probability=self.min_exceed_prob)

    def _load_building_links(self):
        """Take the objects to model from the building link table.

        The exposure question is per building: what is the depth on ITS
        property and ITS road. So rather than measure every parcel and every
        road in the domain and index into them afterwards, only the ones some
        building actually points at are kept. That is the same computation a
        strictly per-building loop would do, minus the repetition where several
        buildings share a parcel.

        Positional indices rather than ids because the aggregation gathers
        each building's property and road rows out of the object layers'
        (n_members, n_objects) depth stacks with a single take; -1 marks a
        building with no such object, which HazardIndicatorFinder gives a depth
        of 0.0 -- under a max, an absent parcel simply leaves the remaining
        objects to decide (see its class docstring).
        """
        links = pd.read_csv(BUILDING_LINKS_PATH)

        # The DOMAIN decides which buildings are modelled; the link table only
        # supplies the optional property and road. A building absent from it is
        # kept and reindexed to a null row, which the mapping below turns into
        # the -1 the aggregation already handles -- the same treatment a
        # building with no parcel gets. Dropping them instead made
        # the link
        # table a second, silent domain filter, and since it was built against
        # the model grid it removed buildings the surveyed raster covers,
        # including two placarded at the most severe damage level.
        known = self.buildings["building_id"].isin(links["building_id"])
        if not known.all():
            logger.warning(
                f"{int((~known).sum())} of {len(self.buildings)} buildings are absent "
                f"from {BUILDING_LINKS_PATH} and are scored without a property or "
                f"road; rebuild the link table to give them one"
            )
        links = links.set_index("building_id").reindex(self.buildings["building_id"])

        # Only the parcels and road parcels the buildings refer to. Read raw
        # rather than through load_features: the link table has already decided
        # these are the right objects, and domain-filtering them again would
        # drop parcels that merely straddle the edge, handing their buildings a
        # missing input for a purely geometric reason. Zonal depth over a parcel
        # that is partly outside simply uses the cells that are inside.
        used_properties = links["property_id"].dropna().unique()
        properties = gpd.read_file(PROPERTY_BOUNDARIES_PATH).to_crs(self.buildings.crs)
        self.properties = (
            properties[properties["unit_of_property_id"].isin(used_properties)]
            .reset_index(drop=True)
        )
        used_roads = links["road_parcel_id"].dropna().unique()
        roads = gpd.read_file(ROAD_NETWORK_FILE_PATH).to_crs(self.buildings.crs)
        self.roads = roads[roads["id"].isin(used_roads)].reset_index(drop=True)

        property_row = pd.Series(
            np.arange(len(self.properties)), index=self.properties["unit_of_property_id"]
        )
        road_row = pd.Series(np.arange(len(self.roads)), index=self.roads["id"])
        self._building_property_idx = (
            links["property_id"].map(property_row).fillna(-1).to_numpy(dtype=int)
        )
        self._building_road_idx = (
            links["road_parcel_id"].map(road_row).fillna(-1).to_numpy(dtype=int)
        )
        self._building_links = links.reset_index()

        logger.info(
            f"building-centred objects: {len(self.buildings)} buildings, "
            f"{len(self.properties)} properties, {len(self.roads)} road parcels"
        )
        for label, idx in (("property", self._building_property_idx),
                           ("road parcel", self._building_road_idx)):
            missing = int((idx < 0).sum())
            if missing:
                logger.warning(
                    f"{missing} of {len(self.buildings)} buildings have no {label}; "
                    f"the max of Equation (7) simply leaves it out rather than "
                    f"scoring it dry"
                )

    def get_impact_forecast(self, hazard_forecast, rasterfile=None):
        # One GeoDataFrame per feature type; per-timestamp results are collected
        building_impact_series = {}
        road_impact_series = {}
        property_impact_series = {}
        for timestamp, flood_maps_ensemble in hazard_forecast.items():
            building_impact = self.buildings[[self.buildings.geometry.name, 'building_id']].copy()
            road_impact = self.roads[[self.roads.geometry.name, "id"]].copy()
            property_impact = self.properties[
                [self.properties.geometry.name, "unit_of_property_id"]
            ].copy()

            building_members = []
            road_members = []
            property_members = []
            for flood_map in flood_maps_ensemble:
                building_members.append(self.buffered_flood_depth(self.buildings, flood_map, raster_file=rasterfile, use_adapative_ring=self.use_adaptive_ring))
                road_members.append(self.zonal_flood_depth(self.roads, flood_map, rasterfile, cache_key="road"))
                property_members.append(self.zonal_flood_depth(self.properties, flood_map, rasterfile, cache_key="property"))
                
                
            # mean_flood_map = np.asarray(flood_maps_ensemble).mean(axis=0)
            # mean_map_exposure = {
            #     "building": self.buffered_flood_depth(
            #         self.buildings, mean_flood_map, raster_file=rasterfile,
            #         use_adapative_ring=self.use_adaptive_ring),
            #     "road": self.zonal_flood_depth(
            #         self.roads, mean_flood_map, rasterfile, cache_key="road"),
            #     "property": self.zonal_flood_depth(
            #         self.properties, mean_flood_map, rasterfile, cache_key="property"),
            # }


            building_cols = _build_indicator_columns(building_members)
            building_cols["num_ring_cells"] = pd.concat(
                [m["num_ring_cells"] for m in building_members], axis=1
            ).iloc[:, 0]
            building_impact = _with_columns(building_impact, building_cols)

            building_impact = compute_risk_matrix(
                building_impact,
                building_members,
                property_members,
                road_members,
                self._building_property_idx,
                self._building_road_idx,
                self.risk_level_finder
            )

            building_impact_series[timestamp] = building_impact
        
            for series, impact, members, object_type in (
                (road_impact_series, road_impact, road_members, "road"),
                (property_impact_series, property_impact, property_members, "property"),
            ):
                impact = _with_columns(impact, {
                    **_build_indicator_columns(members),
                    **self.risk_level_finder.object_columns(
                        members, statistic=REPORTED_CONFIGURATION[1]),
                })
                series[timestamp] = impact

        return {
            "building_impact": building_impact_series,
            "road_impact": road_impact_series,
            "property_impact": property_impact_series,
        }


    def buffered_flood_depth(
        self,
        exposed_objects,
        flood_map,
        buffer_pct: float = 2.0,
        mean_threshold: float = 0.05,
        max_threshold: float = 0.15,
        raster_file: str = None,
        use_adapative_ring:bool =False,
    ):
        """
        Compute buffered (ring) flood-depth statistics using GPU reductions.
        Rings are allowed to overlap between neighboring buildings (common in
        tightly packed blocks); the sparse rasterization path preserves each
        building's own pixels instead of losing them to whichever neighbor's
        ring rasterizes last. Labels/indices are computed once and cached.
        """

        # ---------------------------------------------------------------
        # Keep flood map on GPU if possible
        # ---------------------------------------------------------------
        if isinstance(flood_map, np.ndarray):
            depth = torch.from_numpy(flood_map).float().to(check_device())
        elif torch.is_tensor(flood_map):
            depth = flood_map.squeeze().float()
            if depth.device.type != check_device().type:
                depth = depth.to(check_device())
        else:
            raise TypeError(
                "flood_map must be either a NumPy array or a torch.Tensor"
            )

        if depth.ndim != 2:
            raise ValueError(
                f"buffered_flood_depth expects a 2D flood map, got shape {tuple(depth.shape)}"
            )

        # ---------------------------------------------------------------
        # Georeferencing
        # ---------------------------------------------------------------
        transform, raster_crs, _ = get_sample_georeference(raster_file)

        cell_size_x = abs(transform.a)
        cell_size_y = abs(transform.e)

        if not np.isclose(cell_size_x, cell_size_y, rtol=1e-3):
            raise ValueError(
                f"Non-square pixels ({cell_size_x} x {cell_size_y})"
            )

        buffer_dist = cell_size_x * buffer_pct

        # ---------------------------------------------------------------
        # Build ring indices once per grid (rebuild if the depth grid /
        # raster file changes, since the cached indices are grid-specific).
        # ---------------------------------------------------------------
        grid_key = (tuple(depth.shape), raster_file)
        if self.building_labels is None or self._building_label_grid != grid_key:
            ring_objects = exposed_objects.copy()
            if use_adapative_ring:
                # Precomputed rings if there are any for this grid. The geometry
                # is identical to what the loop would produce -- the same method
                # wrote it -- so this is a pure time saving, and a mismatched or
                # absent file falls through to building them here.
                ring_key = self._ring_cache_key(
                    tuple(depth.shape), transform, raster_crs)
                cached = self._load_ring_cache(exposed_objects, ring_key)
                if cached is None:
                    ring_geom, _, exhausted = self._build_adaptive_rings(
                        exposed_objects, tuple(depth.shape), transform,
                        raster_crs, cell_size_x
                    )
                else:
                    ring_geom, exhausted, self._ring_buffer_pct = cached
                self._exhausted_buildings = exhausted
                ring_objects["geometry"] = ring_geom.values
            else:
                 all_buildings_geom = self._footprint_union()
                 ring_objects["geometry"] = (
                     exposed_objects.geometry            
                     .buffer(buffer_dist)            
                     .difference(all_buildings_geom)            
                 )
            # allow_overlap=True: rings can legitimately overlap in packed
            # blocks; every building keeps its own uncontested pixels instead of
            # losing them to whichever neighbor's ring rasterizes last.
            self._row_idx, self._col_idx, self._feat_idx, self.num_buildings = _get_feature_labels(
                ring_objects,
                tuple(depth.shape),
                transform,
                raster_crs,
                allow_overlap=True,
            )
            self.building_labels = True  # sentinel: cache has been built
            self._building_label_grid = grid_key

        # ---------------------------------------------------------------
        # GPU zonal statistics (sparse, overlap-safe)
        # ---------------------------------------------------------------
        (
            depth_max,
            _,
            depth_sum,
            depth_mean,
            depth_wet_mean,
            counts,
            wet_counts,
        ) = self._gpu_zonal_stats_sparse(
            depth,
            self._row_idx,
            self._col_idx,
            self._feat_idx,
            self.num_buildings,
        )

        # ---------------------------------------------------------------
        # Build output
        # ---------------------------------------------------------------
        # Built as one frame and joined in a single concat. Assigning the
        # columns one at a time inserts a block per column and leaves the frame
        # fragmented (the pandas PerformanceWarning), which is worth avoiding
        # here because this runs once per ensemble member per timestep.
        depth_max_np = np.round(depth_max.cpu().numpy(), 2)
        depth_mean_np = np.round(depth_mean.cpu().numpy(), 2)
        counts_np = counts.cpu().numpy()
        wet_counts_np = wet_counts.cpu().numpy()

        stats = pd.DataFrame(
            {
                "depth_max": depth_max_np,
                "depth_mean": depth_mean_np,
                # h over the wet cells alone, beside h over all of them: the
                # same ring, read with coverage divided out (see WET_DEPTH_MIN).
                "depth_wet_mean": np.round(depth_wet_mean.cpu().numpy(), 2),
                "buffer_dist": buffer_dist,
                "ring_pixel_count": counts_np,
                "num_ring_cells": counts_np,
                # Ring coverage, the factor between the two means.
                # num_ring_cells stays the GEOMETRIC cell count, so the "ring
                # never sampled anything" filter is unaffected.
                "num_wet_cells": wet_counts_np,
                "wet_fraction": np.divide(
                    wet_counts_np, counts_np,
                    out=np.zeros(len(exposed_objects), dtype=float),
                    where=counts_np > 0,
                ),
                "inundated": (
                    (depth_max_np >= max_threshold)
                    & (depth_mean_np >= mean_threshold)
                ),
            },
            index=exposed_objects.index,
        )
        # Dropping first reproduces the overwrite semantics of result[col] = ...
        # for any of these columns the input frame already carries; a plain
        # concat would duplicate them instead.
        return pd.concat(
            [exposed_objects.drop(columns=stats.columns, errors="ignore"), stats],
            axis=1,
        )

    def zonal_flood_depth(
        self,
        exposed_objects,
        flood_map,
        raster_file=None,
        cache_key="road",
    ):
        """
        Compute zonal flood statistics using GPU reductions.
        Rasterization is cached and performed only once.

        ``cache_key`` names the object type so each set of zones (roads,
        properties) keeps its own cached label raster; passing one key for two
        different geometries would reuse the wrong labels.
        """

        mean_threshold = 0.05
        max_threshold = 0.15

        # ---------------------------------------------------------------
        # Keep the flood map on the GPU if possible
        # ---------------------------------------------------------------
        if isinstance(flood_map, np.ndarray):
            depth = torch.from_numpy(flood_map).float().to(check_device())
        elif torch.is_tensor(flood_map):
            depth = flood_map.squeeze().float()
            if depth.device.type != check_device().type:
                depth = depth.to(check_device())
        else:
            raise TypeError(
                "flood_map must be either a NumPy array or a torch.Tensor"
            )

        if depth.ndim != 2:
            raise ValueError(
                f"zonal_flood_depth expects a 2D flood map, got shape {tuple(depth.shape)}"
            )

        # ---------------------------------------------------------------
        # Georeferencing
        # ---------------------------------------------------------------
        transform, raster_crs, _ = get_sample_georeference(raster_file)

        if exposed_objects.crs != raster_crs:
            exposed_objects = exposed_objects.to_crs(raster_crs)

        # ---------------------------------------------------------------
        # Build labels once per grid (rebuild if the depth grid / raster file
        # changes, since the cached labels are grid-specific).
        # ---------------------------------------------------------------
        grid_key = (tuple(depth.shape), raster_file)
        if self._zonal_label_grid.get(cache_key) != grid_key:
            labels, n_features = _get_feature_labels(
                exposed_objects,
                tuple(depth.shape),
                transform,
                raster_crs,
            )
            self._zonal_label_grid[cache_key] = grid_key
            self._zonal_labels[cache_key] = (
                torch.from_numpy(labels).long().to(depth.device),
                n_features,
            )

        labels_gpu, n_features = self._zonal_labels[cache_key]

        # ---------------------------------------------------------------
        # Prepare flood map
        # ---------------------------------------------------------------
        depth = torch.nan_to_num(depth, nan=0.0)
        depth[depth < mean_threshold] = 0.0

        # ---------------------------------------------------------------
        # GPU zonal statistics
        # ---------------------------------------------------------------
        (
            depth_max,
            depth_min,
            depth_sum,
            depth_mean,
            depth_wet_mean,
            counts,
            wet_counts,
        ) = self._gpu_zonal_stats(
            depth,
            labels_gpu,
            n_features,
        )

        # ---------------------------------------------------------------
        # Copy only the statistics back to CPU
        # ---------------------------------------------------------------
        # One concat rather than a column-at-a-time insert; see the note in
        # zonal_flood_depth_rings.
        depth_max_np = depth_max.cpu().numpy()
        depth_mean_np = depth_mean.cpu().numpy()

        stats = pd.DataFrame(
            {
                "depth_max": depth_max_np,
                "depth_min": depth_min.cpu().numpy(),
                "depth_mean": depth_mean_np,
                "depth_wet_mean": depth_wet_mean.cpu().numpy(),
                "depth_sum": depth_sum.cpu().numpy(),
                "num_pixels": counts.cpu().numpy(),
                "num_wet_pixels": wet_counts.cpu().numpy(),
                "inundated": (
                    (depth_max_np >= max_threshold)
                    & (depth_mean_np >= mean_threshold)
                ),
            },
            index=exposed_objects.index,
        )
        return pd.concat(
            [exposed_objects.drop(columns=stats.columns, errors="ignore"), stats],
            axis=1,
        )
    
    import torch


    def _gpu_zonal_stats(self, depth, labels, n_features):
        """
        depth  : torch.FloatTensor (H,W) on GPU
        labels : torch.LongTensor  (H,W) on GPU

        Returns
        -------
        depth_max
        depth_min
        depth_sum
        depth_mean
        depth_wet_mean
        counts
        wet_counts
        """

    
        labels_flat = labels.reshape(-1)
        depth_flat = depth.reshape(-1)

        # Ignore background
        valid = (labels_flat > 0) & torch.isfinite(depth_flat)
        labels_flat = labels_flat[valid]
        depth_flat = depth_flat[valid]
        counts = torch.bincount(labels_flat, minlength=n_features + 1)

        depth_flat = torch.nan_to_num(depth_flat, nan=0.0)
        depth_flat[depth_flat < WET_DEPTH_MIN] = 0.0

        # Wet-cell tally per feature: the coverage half of depth_mean, and the
        # denominator of depth_wet_mean. Taken after the flooring so it counts
        # exactly the cells that contribute to depth_sum -- a cell in the sum
        # but not in the count would bias the wet mean downward.
        wet_counts = torch.zeros(
            n_features + 1,
            device=depth.device,
            dtype=depth.dtype,
        )
        wet_counts.scatter_add_(0, labels_flat, (depth_flat > 0).to(depth.dtype))

        # ---------------- Sum ----------------

        depth_sum = torch.zeros(
            n_features + 1,
            device=depth.device,
            dtype=depth.dtype,
        )

        depth_sum.scatter_add_(0, labels_flat, depth_flat)

        # ---------------- Max ----------------

        depth_max = torch.full(
            (n_features + 1,),
            -torch.inf,
            device=depth.device,
        )

        depth_max.scatter_reduce_(
            0,
            labels_flat,
            depth_flat,
            reduce="amax",
            include_self=True,
        )

        # ---------------- Min ----------------

        depth_min = torch.full(
            (n_features + 1,),
            torch.inf,
            device=depth.device,
        )

        depth_min.scatter_reduce_(
            0,
            labels_flat,
            depth_flat,
            reduce="amin",
            include_self=True,
        )

        depth_max[torch.isinf(depth_max)] = 0
        depth_min[torch.isinf(depth_min)] = 0

        counts = counts[1:]
        wet_counts = wet_counts[1:]
        depth_sum = depth_sum[1:]
        depth_max = depth_max[1:]
        depth_min = depth_min[1:]

        # Mean over EVERY cell of the zone, dry ones entering as zeros.
        depth_mean = torch.where(
            counts > 0,
            depth_sum / counts.float(),
            torch.zeros_like(depth_sum),
        )

        # The same sum over the WET cells alone -- depth where there is depth,
        # with coverage divided out (see WET_DEPTH_MIN). A zone with no wet cell
        # has an empty denominator and scores 0.0, not NaN: it is dry, and the
        # class rules read 0.0 as dry.
        depth_wet_mean = torch.where(
            wet_counts > 0,
            depth_sum / wet_counts,
            torch.zeros_like(depth_sum),
        )

        return (
            depth_max,
            depth_min,
            depth_sum,
            depth_mean,
            depth_wet_mean,
            counts,
            wet_counts,
        )
        
        
    def _gpu_zonal_stats_sparse(self, depth, row_idx, col_idx, feature_idx, n_features):
        """Same outputs as _gpu_zonal_stats, but driven by sparse (row, col,
        feature) triples so overlapping features are supported — a pixel can
        contribute to more than one feature_idx.
        """
        if row_idx.size == 0:
            z = torch.zeros(n_features, device=depth.device, dtype=depth.dtype)
            c = torch.zeros(n_features, device=depth.device, dtype=torch.long)
            return (z.clone(), z.clone(), z.clone(), z.clone(), z.clone(), c,
                    z.clone())

        row_t = torch.from_numpy(row_idx).long().to(depth.device)
        col_t = torch.from_numpy(col_idx).long().to(depth.device)
        feat_t = torch.from_numpy(feature_idx).long().to(depth.device)

        vals = depth[row_t, col_t]
        vals = torch.nan_to_num(vals, nan=0.0)
        vals = torch.where(vals < WET_DEPTH_MIN, torch.zeros_like(vals), vals)

        counts = torch.bincount(feat_t, minlength=n_features)

        depth_sum = torch.zeros(n_features, device=depth.device, dtype=depth.dtype)
        depth_sum.scatter_add_(0, feat_t, vals)

        # Denominator of depth_wet_mean, and the coverage half of depth_mean:
        # wet cells only, counted after the flooring so it matches exactly what
        # went into depth_sum.
        wet_counts = torch.zeros(n_features, device=depth.device, dtype=depth.dtype)
        wet_counts.scatter_add_(0, feat_t, (vals > 0).to(depth.dtype))

        depth_max = torch.full((n_features,), -torch.inf, device=depth.device)
        depth_max.scatter_reduce_(0, feat_t, vals, reduce="amax", include_self=True)

        depth_min = torch.full((n_features,), torch.inf, device=depth.device)
        depth_min.scatter_reduce_(0, feat_t, vals, reduce="amin", include_self=True)

        depth_max[torch.isinf(depth_max)] = 0
        depth_min[torch.isinf(depth_min)] = 0

        # Mean over EVERY ring cell, dry ones entering as zeros.
        depth_mean = torch.where(
            counts > 0, depth_sum / counts.float(), torch.zeros_like(depth_sum)
        )

        # The wet-cell mean: the same sum over the wet cells alone, 0.0 where
        # the ring never got wet (see WET_DEPTH_MIN).
        depth_wet_mean = torch.where(
            wet_counts > 0, depth_sum / wet_counts, torch.zeros_like(depth_sum)
        )

        return (depth_max, depth_min, depth_sum, depth_mean, depth_wet_mean,
                counts, wet_counts)
        

    def _footprint_union(self):
        """Every building footprint in the layer, dissolved, in the modelled CRS.

        The geometry subtracted from the buffers so a ring is the ground around
        a building and never a roof. It has to come from the FULL layer rather
        than from self.buildings: load_features keeps only footprints sitting on
        a finite depth cell, so a building just outside the model domain is not
        in the modelled set, yet is still physically there next to one that is.
        Subtracting only the modelled set leaves those roofs inside a
        neighbour's ring, where their cells are sampled as if they were ground.

        Read raw rather than through load_features for exactly that reason --
        the domain filter is what has to be bypassed here -- and cached, since
        the union over the whole layer is the same for every call.
        """
        if self._footprint_union_cache is None:
            footprints = gpd.read_file(BUILDING_FOOTPRINTS_PATH).to_crs(
                self.buildings.crs
            )
            self._footprint_union_cache = unary_union(footprints.geometry.values)
        return self._footprint_union_cache

    def _ring_cache_key(self, out_shape, transform, raster_crs,
                        min_cells=MIN_RING_CELLS, max_buffer_pct=MAX_BUFFER_PCT,
                        step_pct=GROWTH_STEP_PCT):
        """Everything the ring geometry depends on, as one comparable string.

        Deliberately NOT the raster file: the growth loop counts cells
        geometrically and never reads a depth value, so rings are a property of
        the grid and the footprints, not of the scenario. One cached file
        therefore serves every scenario and timestep sharing a grid, which is
        the whole point of caching it. What does invalidate it -- a different
        grid, a different CRS, retuned growth parameters, a changed building
        population -- is exactly what this key carries.

        The grid matters for more than the labels: a building outside a raster's
        extent clips no cell at any width, so it grows to the cap there and not
        on a grid that covers it. The surveyed July 2021 raster is smaller than
        the model grid and 86 buildings fall outside it, which is why each grid
        gets its own file rather than sharing one.
        """
        return "|".join([
            f"{out_shape[0]}x{out_shape[1]}",
            ",".join(f"{v:.6f}" for v in (transform.a, transform.b, transform.c,
                                          transform.d, transform.e, transform.f)),
            str(raster_crs),
            f"min_cells={min_cells:g},max_buffer={max_buffer_pct:g},step={step_pct:g}",
            f"buildings={len(self.buildings)}",
        ])

    def _ring_cache_path(self, key, directory=BUILDING_RINGS_DIR):
        """Where the rings for one grid live.

        Named from the key so every grid gets its own file and neither run has
        to be told which one it wants: the grid it is working on picks the path.
        The shape is kept legible in the name and the digest carries the rest of
        the key, which is too long to read but has to be distinguished.
        """
        digest = hashlib.sha1(key.encode()).hexdigest()[:10]
        return os.path.join(
            directory, f"building_rings_{key.split('|')[0]}_{digest}.geojson"
        )

    def build_ring_cache(self, out_shape, transform, raster_crs, cell_size,
                         path=None, **growth_kwargs):
        """Compute the rings once and write them where a run can load them.

        Called by scripts/build_building_rings.py rather than from a forecast:
        writing during a run would have several processes racing for the same
        file, and the rebuild is the cost the cache exists to avoid.
        """
        key = self._ring_cache_key(out_shape, transform, raster_crs,
                                   **growth_kwargs)
        path = path or self._ring_cache_path(key)

        ring_geom, _, exhausted = self._build_adaptive_rings(
            self.buildings, out_shape, transform, raster_crs, cell_size,
            **growth_kwargs,
        )

        rings = gpd.GeoDataFrame(
            {
                "building_id": self.buildings["building_id"].values,
                "ring_buffer_pct": self._ring_buffer_pct,
                "exhausted": exhausted,
                # Stamped on every feature because GeoJSON has nowhere else to
                # put it, and a cache without provenance is worse than none.
                "ring_key": key,
            },
            geometry=ring_geom.values,
            crs=self.buildings.crs,
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Written in lon/lat, which is what GeoJSON is defined in. Handing the
        # driver a projected frame instead writes metre coordinates with no crs
        # member, and every reader -- including this module's own loader -- then
        # takes them for degrees. The reprojection round trip is exact to about
        # a nanometre, well under the rasterization's reach.
        rings.to_crs(epsg=4326).to_file(path, driver="GeoJSON")
        logger.info(f"wrote {len(rings)} rings to {path}")
        return rings

    def _load_ring_cache(self, exposed_objects, key, path=None):
        """The cached rings if they match this grid and population, else None.

        Every rejection is logged with its reason and falls back to rebuilding,
        so a stale file costs a slow run rather than wrong geometry.
        """
        path = path or self._ring_cache_path(key)
        if not os.path.exists(path):
            logger.info(f"no ring cache at {path}; building the rings in-process")
            return None

        rings = gpd.read_file(path)
        cached_key = rings["ring_key"].iloc[0] if len(rings) else ""
        if cached_key != key:
            logger.warning(
                f"ring cache at {path} was built for a different setup and is "
                f"ignored\n  cached: {cached_key}\n  wanted: {key}\n"
                f"  rebuild it with scripts/build_building_rings.py"
            )
            return None

        # Matched on building_id rather than row order: the cache is keyed to
        # buildings, and a reordered layer would otherwise hand every building
        # its neighbour's ring.
        rings = rings.set_index("building_id").reindex(
            exposed_objects["building_id"].values
        )
        if rings.geometry.isna().any():
            logger.warning(
                f"ring cache at {path} is missing "
                f"{int(rings.geometry.isna().sum())} of the modelled buildings "
                f"and is ignored; rebuild it with scripts/build_building_rings.py"
            )
            return None

        if rings.crs != exposed_objects.crs:
            rings = rings.to_crs(exposed_objects.crs)

        logger.info(f"loaded {len(rings)} rings from {path}")
        return (
            gpd.GeoSeries(rings.geometry.values, crs=exposed_objects.crs),
            rings["exhausted"].to_numpy(dtype=bool),
            rings["ring_buffer_pct"].to_numpy(dtype=float),
        )

    def _build_adaptive_rings(self, exposed_objects, out_shape, transform,
                              raster_crs, cell_size, min_cells=MIN_RING_CELLS,max_buffer_pct=MAX_BUFFER_PCT, step_pct=GROWTH_STEP_PCT):
        """Grow each building's ring until it holds min_cells, capped.

        Takes the grid's SHAPE rather than the depth field, because since the
        counting fix nothing here reads a depth value -- the loop rasterizes
        candidate rings and counts cells. That is what makes the result cacheable
        across scenarios; see _ring_cache_key.
        """
        all_buildings_geom = self._footprint_union()
        n = len(exposed_objects)
        buffer_pct = np.full(n, 1, dtype=float)
        needs_growth = np.ones(n, dtype=bool)

        final_geom = None
        final_counts = None
        final_buffer_pct = buffer_pct.copy()

        while needs_growth.any():
            dist = buffer_pct * cell_size
            candidate_geom = exposed_objects.geometry.buffer(dist).difference(all_buildings_geom)

            # Counted through the overlap-preserving path, which is what
            # _gpu_zonal_stats_sparse uses downstream, so the loop tests the
            # sample size the model will actually get. Rings SHOULD overlap in
            # packed blocks -- a cell of water between two buildings is real
            # water for both of them, and neither has a claim to it over the
            # other. The dense path cannot express that: one pixel carries one
            # label, so a shared cell goes to whichever ring rasterizes last and
            # counts for nobody else. Under it a building hemmed in by later
            # neighbours reads as empty however wide it grows, and the loop runs
            # to the cap chasing a number set by row order in the layer.
            _, _, feat_idx, n_feat = _get_feature_labels(
                gpd.GeoDataFrame(geometry=candidate_geom, crs=exposed_objects.crs),
                out_shape, transform, raster_crs, allow_overlap=True,
            )
            counts_np = np.bincount(feat_idx, minlength=n_feat)

            if final_geom is None:
                final_geom, final_counts = candidate_geom, counts_np
            else:
                update = needs_growth
                final_geom = final_geom.where(~update, candidate_geom)
                final_counts = np.where(update, counts_np, final_counts)

            final_buffer_pct = np.where(needs_growth, buffer_pct, final_buffer_pct)

            maxed_out = buffer_pct >= max_buffer_pct
            needs_growth = (final_counts < min_cells) & (~maxed_out)
            buffer_pct = np.where(needs_growth, buffer_pct + step_pct, buffer_pct)

        # The per-building buffer the loop settled on, in cells. Kept on the
        self._ring_buffer_pct = final_buffer_pct

        # these need the block-dissolve fallback, not a bigger ring.
        exhausted = (final_counts < min_cells) & (final_buffer_pct >= max_buffer_pct)
        return final_geom, final_counts, exhausted
    
                
