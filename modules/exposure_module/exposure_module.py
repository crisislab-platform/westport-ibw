from modules.lib import BUILDING_FOOTPRINTS_PATH, FOOTPRINT_INDEX_DIR
import hashlib
import os
import pandas as pd
import numpy as np
import logging
from modules.utils.spatial_operators.operators import  get_sample_georeference,_get_feature_labels
from modules.utils.spatial_operators.operators import load_features
from modules.utils.utils import check_device

import torch

logger = logging.getLogger(__name__)

# A building is exposed when its depth reaches this. Footprint cells shallower
# than it are floored to 0 before the spatial summary is taken, so a film of
# water never counts as depth.
EXPOSURE_DEPTH_M = 0.05
# How a building's depth is read off the depth grid: the max, mean, median or
# 90th percentile of the cells its footprint touches, or the single cell holding
# its centroid. Cells below EXPOSURE_DEPTH_M enter every summary as 0, so the
# mean counts dry cells. The choice is calibrated against the July 2021 placards and surveyed
# depths (July2021Validation.calibrate_spatial_summarisation_method);
# SPATIAL_SUMMARY is the one used.
SPATIAL_SUMMARIES = ("max", "mean", "median", "p90", "centroid")
# Chosen 2026-10-23 against the July 2021 placards: the highest Pearson r of
# building depth with damage class (mean 0.686; median 0.684, p90 0.672,
# centroid 0.662, max 0.657).
SPATIAL_SUMMARY = "mean"
_FOOTPRINT_QUANTILES = {"median": 0.5, "p90": 0.9}
# Ensemble quantiles published beside the median, as the forecast's spread.
ENSEMBLE_LOW_Q, ENSEMBLE_HIGH_Q = 0.1, 0.9


def is_exposed(depth):
    """Whether a depth counts as exposed: strictly above EXPOSURE_DEPTH_M, the
    same test the westport flood-map scores use per cell (depth > 0.05 m)."""
    return np.asarray(depth) > EXPOSURE_DEPTH_M


def summarise_members(depths):
    """Per-building forecast columns from member depths ``(members, buildings)``.

    depth_median is the point forecast; depth_p10 / depth_p90 its ensemble
    spread; prob_exposed the share of members exposed (is_exposed).
    """
    return {
        "depth_median": np.median(depths, axis=0),
        "depth_p10": np.quantile(depths, ENSEMBLE_LOW_Q, axis=0),
        "depth_p90": np.quantile(depths, ENSEMBLE_HIGH_Q, axis=0),
        "prob_exposed": is_exposed(depths).mean(axis=0),
    }


class ExposureModule():
    def __init__(self, scenario_id, domain_raster=None, spatial_summary=SPATIAL_SUMMARY):
       if spatial_summary not in SPATIAL_SUMMARIES:
           raise ValueError(f"spatial_summary must be one of {list(SPATIAL_SUMMARIES)}")
       self.scenario_id = scenario_id
       self.domain_raster = domain_raster
       self.spatial_summary = spatial_summary
       self.buildings = load_features(BUILDING_FOOTPRINTS_PATH, domain_raster)
       self._cell_grid = None
       self._cell_ids = None

    def get_exposure_assessment(self, inundation_forecast, rasterfile=None):
        """``{"building_impact": {timestamp: GeoDataFrame}}``, one frame per
        step, carrying the summarise_members columns and num_cells."""
        building_exposure_timeseries = {}
        for timestamp, flood_maps_ensemble in inundation_forecast.items():
            depths, num_cells = self.building_depths(flood_maps_ensemble, rasterfile)
            columns = {"num_cells": num_cells, **summarise_members(depths)}
            building_exposure = self.buildings[[self.buildings.geometry.name, 'building_id']].copy()
            building_exposure_timeseries[timestamp] = pd.concat(
                [building_exposure, pd.DataFrame(columns, index=building_exposure.index)],
                axis=1,
            )

        return {
            "building_impact": building_exposure_timeseries,
        }

    def building_depths(self, flood_maps, raster_file=None, statistic=None):
        """Each member's depth for every building in self.buildings: the
        spatial summary of the cells its footprint touches.

        ``flood_maps`` is one step's ensemble: a (members, H, W) array or
        tensor, or a list of (H, W) maps. Returns ``(depths, num_cells)``:
        depths is a (members, buildings) float array in self.buildings order,
        num_cells the cells each footprint touches (0 for a footprint off the
        grid, whose depth is then 0) -- for "centroid", 1 if the centroid is on
        the grid, else 0.

        Footprint cells are read directly because the DEM (Topo.tif) carries no
        buildings: the hydraulic model's water flows through footprints, so
        their cells hold modelled depth. Neighbouring footprints can touch the
        same cell; the sparse rasterization path gives it to each of them.
        """
        statistic = statistic or self.spatial_summary
        depth = self._as_ensemble_tensor(flood_maps)
        self._build_footprint_index(tuple(depth.shape[1:]), raster_file)
        rows = self._cell_ids.get_indexer(self.buildings["building_id"].to_numpy())

        if statistic == "centroid":
            vals = self._floored(depth[:, self._centroid_row_t, self._centroid_col_t])
            stat = torch.where(self._centroid_on_grid_t, vals, torch.zeros_like(vals))
            return stat.cpu().numpy()[:, rows], self._centroid_on_grid[rows].astype(int)

        vals = self._floored(depth[:, self._row_t, self._col_t])
        if statistic == "max":
            stat = self._segment_max(vals)
        elif statistic == "mean":
            stat = self._segment_mean(vals)
        else:
            stat = self._segment_quantile(vals, _FOOTPRINT_QUANTILES[statistic])

        depths = stat.cpu().numpy()[:, rows]
        return depths, self._counts[rows]

    @staticmethod
    def _floored(vals):
        vals = torch.nan_to_num(vals, nan=0.0)
        return torch.where(vals < EXPOSURE_DEPTH_M, torch.zeros_like(vals), vals)

    def _as_ensemble_tensor(self, flood_maps):
        if isinstance(flood_maps, (list, tuple)):
            flood_maps = np.stack([
                m.squeeze().cpu().numpy() if torch.is_tensor(m) else np.asarray(m).squeeze()
                for m in flood_maps
            ])
        if isinstance(flood_maps, np.ndarray):
            depth = torch.from_numpy(flood_maps.astype(np.float32))
        elif torch.is_tensor(flood_maps):
            depth = flood_maps.float()
        else:
            raise TypeError("flood maps must be a NumPy array, a torch.Tensor or a list of them")
        if depth.ndim == 2:
            depth = depth[None]
        if depth.ndim != 3:
            raise ValueError(f"expected (members, H, W) flood maps, got shape {tuple(depth.shape)}")
        return depth.to(check_device())

    def _build_footprint_index(self, out_shape, raster_file):
        """The cells each footprint touches, and the cell its centroid falls in,
        once per grid.

        Keyed by building_id: the cached feature order is self._cell_ids, and
        building_depths looks its buildings up there. The (row, col, feature)
        triples are kept sorted by feature, so each building's cells are one
        contiguous segment -- what the quantile statistics read.

        Kept in memory for the life of the module and on disk in
        FOOTPRINT_INDEX_DIR, so a later run on the same grid and buildings
        loads it instead of rasterizing again.
        """
        grid_key = (out_shape, raster_file)
        building_ids = self.buildings["building_id"].to_numpy()
        if self._cell_grid == grid_key and (self._cell_ids.get_indexer(building_ids) >= 0).all():
            return

        cell_ids = pd.Index(building_ids)
        if not cell_ids.is_unique:
            raise ValueError("building_id must be unique to index the footprints")

        transform, raster_crs, _ = get_sample_georeference(raster_file)
        path = os.path.join(
            FOOTPRINT_INDEX_DIR,
            f"footprint_index_{self._footprint_index_key(out_shape, transform, raster_crs)}.npz")
        if os.path.exists(path):
            with np.load(path) as saved:
                index = {name: saved[name] for name in saved.files}
            logger.info("loaded the footprint index from %s", path)
        else:
            index = self._compute_footprint_index(out_shape, transform, raster_crs)
            os.makedirs(FOOTPRINT_INDEX_DIR, exist_ok=True)
            # Written under a temporary name and moved into place, so a run
            # started meanwhile never loads a half-written file.
            partial = f"{path}.{os.getpid()}.partial.npz"
            np.savez(partial, **index)
            os.replace(partial, path)
            logger.info("saved the footprint index to %s", path)

        device = check_device()
        self._row_t = torch.from_numpy(index["row"]).long().to(device)
        self._col_t = torch.from_numpy(index["col"]).long().to(device)
        self._feat_t = torch.from_numpy(index["feat"]).long().to(device)
        self._counts = index["counts"]
        self._starts = index["starts"]
        self._n_features = len(index["counts"])
        self._centroid_on_grid = index["centroid_on_grid"]
        self._centroid_on_grid_t = torch.from_numpy(index["centroid_on_grid"]).to(device)
        self._centroid_row_t = torch.from_numpy(index["centroid_row"]).long().to(device)
        self._centroid_col_t = torch.from_numpy(index["centroid_col"]).long().to(device)
        self._cell_ids = cell_ids
        self._cell_grid = grid_key

    def _footprint_index_key(self, out_shape, transform, raster_crs):
        """A hash of everything the index depends on: the grid (shape,
        transform, CRS) and the buildings (ids in order, CRS, geometry)."""
        digest = hashlib.sha1()
        digest.update(repr((tuple(out_shape), tuple(transform)[:6], str(raster_crs))).encode())
        digest.update(str(self.buildings.crs).encode())
        digest.update(self.buildings["building_id"].to_numpy().tobytes())
        for wkb in self.buildings.geometry.to_wkb():
            digest.update(wkb)
        return digest.hexdigest()[:16]

    def _compute_footprint_index(self, out_shape, transform, raster_crs):
        # allow_overlap=True: a cell touched by two footprints counts for both.
        row_idx, col_idx, feat_idx, n_features = _get_feature_labels(
            self.buildings, out_shape, transform, raster_crs, allow_overlap=True,
        )
        order = np.argsort(feat_idx, kind="stable")
        counts = np.bincount(feat_idx, minlength=n_features)

        # The cell holding each centroid, in the same feature order. A centroid
        # off the grid has no cell; its building reads 0 with num_cells 0.
        footprints = self.buildings if self.buildings.crs == raster_crs else self.buildings.to_crs(raster_crs)
        centroids = footprints.geometry.centroid
        col_f, row_f = ~transform * (centroids.x.to_numpy(), centroids.y.to_numpy())
        rows_c, cols_c = np.floor(row_f).astype(int), np.floor(col_f).astype(int)
        on_grid = (rows_c >= 0) & (rows_c < out_shape[0]) & (cols_c >= 0) & (cols_c < out_shape[1])
        return {
            "row": row_idx[order],
            "col": col_idx[order],
            "feat": feat_idx[order],
            "counts": counts,
            "starts": np.concatenate([[0], np.cumsum(counts)[:-1]]),
            "centroid_row": np.clip(rows_c, 0, out_shape[0] - 1),
            "centroid_col": np.clip(cols_c, 0, out_shape[1] - 1),
            "centroid_on_grid": on_grid,
        }

    def _segment_mean(self, vals):
        """Mean over each building's cells, per member: (members, features).
        0 for a building with no cells."""
        total = torch.zeros((vals.shape[0], self._n_features), device=vals.device, dtype=torch.float64)
        total.scatter_add_(1, self._feat_t.expand(vals.shape[0], -1), vals.double())
        counts = torch.from_numpy(self._counts).to(vals.device).double()
        return torch.where(counts > 0, total / counts.clamp(min=1), torch.zeros_like(total)).float()

    def _segment_max(self, vals):
        """Max over each building's cells, per member: (members, features)."""
        out = torch.zeros((vals.shape[0], self._n_features), device=vals.device, dtype=vals.dtype)
        out.scatter_reduce_(1, self._feat_t.expand(vals.shape[0], -1), vals,
                            reduce="amax", include_self=True)
        return out

    def _segment_quantile(self, vals, q):
        """The q-quantile of each building's cells, per member, interpolated
        linearly between order statistics as numpy.quantile does.

        Each building's cells are already one contiguous segment; adding the
        feature index times an offset larger than any depth and sorting puts
        every segment in value order without mixing segments. Float64 keeps the
        depths exact under that offset.
        """
        offset = 1e3
        feat = self._feat_t.double()
        ordered = torch.sort(vals.double() + feat * offset, dim=1).values - feat * offset

        counts = torch.from_numpy(self._counts).to(vals.device)
        starts = torch.from_numpy(self._starts).to(vals.device)
        pos = q * (counts - 1).clamp(min=0).double()
        lo = pos.floor().long()
        hi = pos.ceil().long()
        n_cells = ordered.shape[1]
        idx_lo = (starts + lo).clamp(max=max(n_cells - 1, 0))
        idx_hi = (starts + hi).clamp(max=max(n_cells - 1, 0))
        v_lo = ordered[:, idx_lo]
        v_hi = ordered[:, idx_hi]
        out = v_lo + (v_hi - v_lo) * (pos - lo.double())
        out[:, counts == 0] = 0.0
        return out.float()
