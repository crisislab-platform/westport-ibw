import rasterio

from modules.lib import BUILDING_FOOTPRINTS_PATH, ROAD_NETWORK_FILE_PATH, ROAD_CENTERLINES_FILE_PATH, OUTPUT_DIR
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import os
import re
from shapely.ops import unary_union
from modules.utils.spatial_operators.operators import zonal_flood_depth, get_sample_georeference, buffered_flood_depth
from modules.utils.spatial_operators.operators import load_features

# Flood-depth thresholds (metres) at which to report exposure probability.
EXPOSURE_THRESHOLDS = (0.1, 0.3, 0.5, 1.0)

# Probability at or above which a feature is treated as "likely" exposed.
LIKELY_PROBABILITY = 0.5

# Flood depth (m) at or above which a road is treated as impassable.
PASSABILITY_THRESHOLD = 0.1

# Buffer (m) applied to centerlines so the flood raster is sampled along them.
CENTERLINE_BUFFER_M = 5.0

# Snapping tolerance (m) for merging near-coincident centerline endpoints into
# shared graph nodes.
NODE_SNAP_M = 1.0

class ImpactModule():
    def __init__(self, scenario_id):
       self.scenario_id = scenario_id
       self.buildings = load_features(BUILDING_FOOTPRINTS_PATH)
       self.roads = load_features(ROAD_NETWORK_FILE_PATH)
       self.roads_centerlines = load_features(ROAD_CENTERLINES_FILE_PATH)
       self._build_access_network()
       

    def _build_access_network(self):
        """Precompute a routable road graph from the centerlines plus each
        building's nearest road segment.

        Nodes are centerline endpoints (snapped together within ``NODE_SNAP_M``);
        edges are the centerline segments. The graph is built once (geometry is
        constant) and reused for every timestep's access test. A buffered copy of
        the centerlines is kept so the flood raster can be sampled along each
        segment.
        """
        centerlines = self.roads_centerlines.to_crs(self.buildings.crs).reset_index(drop=True)

        # The raw centerlines are not noded (roads cross without shared
        # endpoints), so intersections would not become graph nodes. Union the
        # lines to split them at every intersection, then rebuild segments.
        noded = unary_union(centerlines.geometry.values)
        seg_geoms = [noded] if noded.geom_type == "LineString" else list(noded.geoms)
        centerlines = gpd.GeoDataFrame(geometry=seg_geoms, crs=centerlines.crs).reset_index(drop=True)
        self._centerlines = centerlines

        # Buffered centerlines for raster sampling (thin lines miss pixel centres).
        buffered = centerlines[[centerlines.geometry.name]].copy()
        buffered["geometry"] = centerlines.geometry.buffer(CENTERLINE_BUFFER_M)
        self._centerlines_buffered = buffered

        def node_key(pt):
            return (round(pt[0] / NODE_SNAP_M), round(pt[1] / NODE_SNAP_M))

        # One graph edge per segment; nodes are snapped endpoints.
        graph = nx.Graph()
        edge_nodes = []
        for idx, geom in enumerate(centerlines.geometry):
            coords = list(geom.coords)
            u, v = node_key(coords[0]), node_key(coords[-1])
            edge_nodes.append((u, v))
            if u != v:
                graph.add_edge(u, v)
            else:
                graph.add_node(u)
        self._road_graph = graph
        self._edge_nodes = edge_nodes

        # Centerlines form several disconnected clusters, so access is judged
        # relative to each segment's own dry component (dry baseline = 0 cutoffs).
        self._dry_components = [set(c) for c in nx.connected_components(graph)]

        # Nearest centerline segment for each building (positional index).
        nearest = gpd.sjoin_nearest(
            self.buildings[[self.buildings.geometry.name]],
            centerlines[[centerlines.geometry.name]],
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")].sort_index()
        self._building_nearest_seg = nearest["index_right"].to_numpy()

    def _access_cutoff(self, segment_mean_depth, threshold=PASSABILITY_THRESHOLD):
        """Return a boolean array (per building) of whether access is cut off.

        A road segment is passable when its ensemble-mean depth is below
        ``threshold``. Access is assessed per dry cluster: within each cluster
        the largest still-passable connected node set is the retained network. A
        building is cut off if its nearest segment is impassable, or if neither
        endpoint of that segment remains in the retained network. With no
        flooding this yields zero cutoffs.

        Parameters:
        segment_mean_depth (ndarray): Ensemble-mean depth per centerline segment,
            aligned by position to ``self._centerlines``.
        threshold (float): Passability depth (m).

        Returns:
        ndarray: Boolean array of length ``len(self.buildings)``.
        """
        passable_seg = segment_mean_depth < threshold

        # Passable sub-graph: keep only edges whose segment is passable.
        passable_graph = nx.Graph()
        passable_graph.add_nodes_from(self._road_graph.nodes)
        for idx, (u, v) in enumerate(self._edge_nodes):
            if passable_seg[idx] and u != v:
                passable_graph.add_edge(u, v)

        # Retained network = largest passable connected node set within each
        # dry cluster.
        connected_nodes = set()
        for component in self._dry_components:
            sub = passable_graph.subgraph(component)
            sub_components = list(nx.connected_components(sub))
            if sub_components:
                connected_nodes |= max(sub_components, key=len)

        cutoff = np.empty(len(self.buildings), dtype=bool)
        for i, idx in enumerate(self._building_nearest_seg):
            u, v = self._edge_nodes[idx]
            reachable = passable_seg[idx] and (u in connected_nodes or v in connected_nodes)
            cutoff[i] = not reachable
        return cutoff

    def get_impact_forecast(self, hazard_forecast):
        # One GeoDataFrame per feature type; per-timestamp results are collected
        # as columns and added in a single assign to avoid fragmenting the frame.
        building_impact = self.buildings[[self.buildings.geometry.name, 'building_id']].copy()
        road_impact = self.roads[[self.roads.geometry.name]].copy()

        building_cols = {}
        road_cols = {}
        cutoff_history = []   # (timestamp, cutoff_bool_array) per step, in order
        for timestamp, flood_maps_ensemble in hazard_forecast.items():
            # Sample each ensemble member; rows stay aligned to self.buildings/self.roads.
            building_members = []
            road_members = []
            for flood_map in flood_maps_ensemble:
                building_members.append(buffered_flood_depth(self.buildings, flood_map, scenario_id=self.scenario_id))
                road_members.append(zonal_flood_depth(self.roads, flood_map))

            # Collect this timestamp's exposure probabilities as suffixed columns.
            ts_str = pd.to_datetime(timestamp).strftime("%Y%m%dT%H%M%S")
            building_cols.update(self._exposure_columns(building_members, ts_str))
            building_cols["inundated"] = (
                pd.concat(
                    [m["inundated"] for m in building_members],
                    axis=1
                ).mean(axis=1) >= 0.5
            )
            road_cols["inundated"] = (
                pd.concat(
                    [m["inundated"] for m in road_members],
                    axis=1
                ).mean(axis=1) >= 0.5
            )
            road_cols.update(self._exposure_columns(road_members, ts_str))

            # Access cutoff: isolation from the road network using the
            # ensemble-mean depth sampled along the centerline segments.
            segment_members = [
                zonal_flood_depth(self._centerlines_buffered, flood_map)
                for flood_map in flood_maps_ensemble
            ]
            segment_mean_depth = np.vstack(
                [m["depth_max"].to_numpy() for m in segment_members]
            ).mean(axis=0)
            cutoff = self._access_cutoff(segment_mean_depth)
            building_cols[f"access_cutoff_{ts_str}"] = cutoff
            cutoff_history.append((pd.to_datetime(timestamp), cutoff))

        # First timestamp at which each building loses access (NaT if never).
        building_cols["access_cutoff_time"] = self._first_cutoff_time(cutoff_history)

        building_impact = building_impact.assign(**building_cols)
        road_impact = road_impact.assign(**road_cols)

        return {
            "building_impact": building_impact,
            "road_impact": road_impact,
        }

    def _first_cutoff_time(self, cutoff_history):
        """Return the first timestamp each building loses access (NaT if never).

        Parameters:
        cutoff_history (list[tuple]): ``(timestamp, cutoff_bool_array)`` for each
            forecast step, in chronological order.

        Returns:
        ndarray: datetime64 array of length ``len(self.buildings)``; entries are
            NaT for buildings that keep access throughout the horizon.
        """
        first = np.full(len(self.buildings), np.datetime64("NaT"), dtype="datetime64[ns]")
        for timestamp, cutoff in cutoff_history:
            newly = cutoff & np.isnat(first)
            first[newly] = np.datetime64(timestamp)
        return first


    def _exposure_columns(self, members, ts_str,
                          thresholds=EXPOSURE_THRESHOLDS, depth_col="depth_max"):
        """
        Compute an ensemble's exposure results for a single timestamp as a dict
        of column name -> array (rows aligned by position to the feature set).

        For each feature and threshold the probability is the fraction of
        ensemble members whose ``depth_col`` is at or above that threshold, i.e.
        an empirical estimate of P(depth >= threshold). Column names are suffixed
        with ``ts_str`` so multiple timestamps coexist on a single geometry.

        Parameters:
        members (list[GeoDataFrame]): One depth result per ensemble member, all
            row-aligned to the feature set.
        ts_str (str): Timestamp label used as the column suffix.
        thresholds (iterable of float): Depth thresholds (metres).
        depth_col (str): Column holding the per-feature depth to test.

        Returns:
        dict: Mapping of column name to per-feature value array.
        """
        # Shape: (n_members, n_features)
        depths = np.vstack([m[depth_col].to_numpy() for m in members])

        columns = {f"expected_depth_{ts_str}": depths.mean(axis=0)}
        for t in thresholds:
            columns[f"p_ge_{t:g}m_{ts_str}"] = (depths >= t).mean(axis=0)
        return columns

    def _aggregate_metrics(self, building_impact, road_impact,
                           thresholds=EXPOSURE_THRESHOLDS, likely_prob=LIKELY_PROBABILITY):
        """
        Summarise per-feature exposure probabilities into scalar metrics for a
        single timestamp.

        For each threshold two views are reported:
        - expected value: the sum of per-feature probabilities, which by
          linearity of expectation is the expected number of buildings (or
          expected road length) exposed at that depth;
        - likely value: a count/length restricted to features whose probability
          is at least ``likely_prob``.

        Parameters:
        building_impact (GeoDataFrame): Output of :meth:`_exposure_probability`
            for buildings.
        road_impact (GeoDataFrame): Output of :meth:`_exposure_probability`
            for roads.
        thresholds (iterable of float): Depth thresholds (metres).
        likely_prob (float): Probability cutoff for the "likely" metrics.

        Returns:
        dict: Scalar metrics keyed by name (counts of buildings, km of road).
        """
        # Road length in CRS units; assumes a projected CRS (metres).
        road_length_m = road_impact.geometry.length

        metrics = {}
        for t in thresholds:
            col = f"p_exposed_ge_{t:g}m"
            b_prob = building_impact[col]
            r_prob = road_impact[col]

            metrics[f"expected_buildings_ge_{t:g}m"] = float(b_prob.sum())
            metrics[f"likely_buildings_ge_{t:g}m"] = int((b_prob >= likely_prob).sum())
            metrics[f"expected_road_km_ge_{t:g}m"] = float((r_prob * road_length_m).sum() / 1000.0)
            metrics[f"likely_road_km_ge_{t:g}m"] = float(
                road_length_m[r_prob >= likely_prob].sum() / 1000.0
            )
        return metrics

    def save_impact_forecast(self, impact_forecast, flood_maps, issue_time, output_dir=OUTPUT_DIR):
        """
        Persist an impact forecast to disk under ``output_dir/<issue_time>/``.

        Per-feature exposure layers are written to a single GeoPackage (one
        layer per timestamp and feature type) and the per-timestamp aggregate
        metrics are written to a CSV time series.

        Parameters:
        impact_forecast (dict): Output of :meth:`get_impact_forecast`.
        issue_time: Time the forecast was issued (used as the run folder name).
        output_dir (str): Root directory for forecast outputs.

        Returns:
        str: Path to the created run directory.
        """
        issue_str = pd.to_datetime(issue_time).strftime("%Y%m%dT%H%M%S")
        run_dir = os.path.join(output_dir, issue_str)
        os.makedirs(run_dir, exist_ok=True)

        gpkg_path = os.path.join(run_dir, f"impact_{issue_str}.gpkg")
        # One layer per feature type; every timestamp is a set of columns.
        impact_forecast["building_impact"].to_file(gpkg_path, layer="buildings", driver="GPKG")
        impact_forecast["road_impact"].to_file(gpkg_path, layer="roads", driver="GPKG")

        # Also export WGS84 GeoJSON per feature type for web/GIS tools
        # (e.g. ArcGIS Online / Scene Viewer), which do not read GeoPackage.
        self._write_geojson(impact_forecast["building_impact"],
                            os.path.join(run_dir, "buildings.geojson"))
        self._write_geojson(impact_forecast["road_impact"],
                            os.path.join(run_dir, "roads.geojson"))

        # Long-format (one row per feature per timestamp) with a real datetime
        # field so the layers can be time-enabled in ArcGIS Online / Map Viewer.
        self._write_timeseries_geojson(
            impact_forecast["building_impact"],
            os.path.join(run_dir, "buildings_timeseries.geojson"),
            id_feild='building_id',
            use_centroids=False
        )
        # self._write_timeseries_geojson(
        #     impact_forecast["road_impact"],
        #     os.path.join(run_dir, "roads_timeseries.geojson"),
        #     use_centroids=False,
        #     id_feild='road_id'
        # )

        # Compute the ensemble mean and standard deviation flood map per
        # timestamp and save as GeoTIFFs, georeferenced the same way as the zonal
        # sampling so they align with the building/road layers. The std map
        # conveys forecast uncertainty to users.
        transform, raster_crs, nodata = get_sample_georeference()
        for timestamp, flood_maps_ensemble in flood_maps.items():
            # Coerce (possibly torch, possibly extra dims) to a 3D float32 stack
            # of (n_members, H, W).
            ensemble = flood_maps_ensemble
            if hasattr(ensemble, "detach"):
                ensemble = ensemble.detach().cpu().numpy()
            ensemble = np.squeeze(np.asarray(ensemble, dtype="float32"))
            if ensemble.ndim != 3:
                raise ValueError(
                    f"flood map ensemble must be (n_members, H, W), got shape {ensemble.shape}"
                )

            mean_flood_map = ensemble.mean(axis=0)
            std_flood_map = ensemble.std(axis=0)

            timestamp_str = pd.to_datetime(timestamp).strftime("%Y%m%dT%H%M%S")
            self._write_geotiff(
                os.path.join(run_dir, f"mean_flood_map_{timestamp_str}.tif"),
                mean_flood_map, transform, raster_crs, nodata,
            )
            self._write_geotiff(
                os.path.join(run_dir, f"std_flood_map_{timestamp_str}.tif"),
                std_flood_map, transform, raster_crs, nodata,
            )
        return run_dir

    def _write_geojson(self, gdf, path):
        """Write a GeoDataFrame to WGS84 GeoJSON for web/GIS consumption.

        Reprojects to EPSG:4326 and stringifies datetime columns (e.g.
        ``access_cutoff_time`` with NaT) so they serialise cleanly.
        """
        out = gdf.to_crs(epsg=4326).copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].astype(str)
        out.to_file(path, driver="GeoJSON")

    def _write_timeseries_geojson(self, impact_gdf, path, use_centroids=True, id_feild='None'):
        """Write a long-format WGS84 GeoJSON with a ``valid_time`` datetime field.

        The wide per-timestamp columns (``expected_depth_<ts>``,
        ``p_ge_<t>m_<ts>``, ``access_cutoff_<ts>``) are melted so each feature
        appears once per timestamp with a real datetime, letting ArcGIS Online
        time-enable the layer. Centroids can be used to keep the (heavily
        duplicated) geometry small.

        Parameters:
        impact_gdf (GeoDataFrame): Wide per-feature impact output (buildings or roads).
        path (str): Destination GeoJSON path.
        use_centroids (bool): Emit centroids (points) instead of polygons.
        """
        geom_name = impact_gdf.geometry.name
        timestamps = sorted(
            c[len("expected_depth_"):] for c in impact_gdf.columns
            if c.startswith("expected_depth_")
        )

        base = impact_gdf[[geom_name, id_feild]].copy()
        if use_centroids:
            base[geom_name] = base.geometry.centroid
        base = base.reset_index(names="feature_id")

        frames = []
        for ts in timestamps:
            step = base.copy()
            step["valid_time"] = pd.to_datetime(ts, format="%Y%m%dT%H%M%S").strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
            step["expected_depth"] = impact_gdf[f"expected_depth_{ts}"].to_numpy()
            if f"access_cutoff_{ts}" in impact_gdf.columns:
                step["access_cutoff"] = impact_gdf[f"access_cutoff_{ts}"].to_numpy()
            for col in impact_gdf.columns:
                m = re.match(rf"(p_ge_[0-9.]+m)_{ts}$", col)
                if m:
                    step[m.group(1)] = impact_gdf[col].to_numpy()
            frames.append(step)

        long = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True), geometry=geom_name,
            crs=impact_gdf.crs,
        ).to_crs(epsg=4326)
        long.to_file(path, driver="GeoJSON")

    def _write_geotiff(self, path, array, transform, crs, nodata):
        """Write a single-band 2D float array to a GeoTIFF at ``path``."""
        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=array.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(array, 1)