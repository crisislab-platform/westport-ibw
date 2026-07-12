from dotenv import load_dotenv
import os
import requests
import logging

from arcgis.gis import GIS
from arcgis.features import FeatureLayer
from modules.utils.spatial_operators.operators import gdf_to_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArcGISModule():
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self.arcgis_client_id = os.getenv("ARCGIS_CLIENT_ID")  # Get the API key from environment variables
        self.arcgis_client_secret = os.getenv("ARCGIS_CLIENT_SECRET")  # Get the client secret from environment
        self.authenticate()
        # self.test_update_feature_layer()

    def authenticate(self):

        url = "https://www.arcgis.com/sharing/rest/oauth2/token"
        params = {
            "client_id": self.arcgis_client_id,
            "client_secret": self.arcgis_client_secret,
            "grant_type": "client_credentials"
        }

        response = requests.post(url, data=params)
        self.access_token = response.json()["access_token"]
        logger.info("Successfully authenticated with ArcGIS API. token: %s", self.access_token)

    def test_update_feature_layer(self):
        url = "https://services6.arcgis.com/ZVM1rEuVZjtC1Wwk/arcgis/rest/services/buildings_timeseries_new_1/FeatureServer/0"
        gis = GIS("https://www.arcgis.com", client_id=self.arcgis_client_id, client_secret=self.arcgis_client_secret)
        logger.info("Successfully connected to ArcGIS GIS. gis: %s", gis._client_id)
        feature_layer = FeatureLayer(url, gis=gis)

        # print properties of the feature layer
        result = feature_layer.query(
            where="1=1",
            out_fields="*",
            return_all_records=False,
            result_record_count=10  # or whatever your layer's max is
        )
        for f in result.features:
            logger.info("Feature attributes: %s", f.attributes)

    def push_impacts_to_spatial_db(self, impact_forecast, start_time, end_time):
        """
        Upserts features for the given time window: existing features (matched
        on building_id + valid_time) are updated in place, new ones are added.
        Nothing is deleted.
        """
        building_impact_gdf = impact_forecast["building_impact"]

        # NOTE: FeatureLayer must point at a specific layer index, not the service root
        url = "https://services6.arcgis.com/ZVM1rEuVZjtC1Wwk/arcgis/rest/services/buildings_timeseries_new_1/FeatureServer/0"
        feature_layer = self.get_feature_layer(url)

        self.upsert_features(
            feature_layer=feature_layer,
            gdf=building_impact_gdf,
            id_field='building_id',
            time_field='valid_time'
        )

    def upsert_features(self, feature_layer: FeatureLayer, gdf, id_field, time_field):
        """
        Updates existing features (matched by id_field + time_field) in place,
        and adds new ones for rows that don't already exist. No deletes.
        """
        existing_map = self._get_existing_key_map(feature_layer, id_field, time_field)

        updates, adds = self._split_updates_and_adds(gdf, id_field, time_field, existing_map)

        if updates:
            self._push_updates(feature_layer, updates)
        if adds:
            self._push_adds(feature_layer, adds)

        logger.info("Upsert complete: %d updated, %d added", len(updates), len(adds))

    def _get_existing_key_map(self, feature_layer:FeatureLayer, id_field, time_field):
        """
        Returns { (building_id_value, time_value_ms): OBJECTID } for every existing feature.
        """
        result = feature_layer.query(
            where="1=1",
            out_fields=f"feature_id,{id_field},{time_field}",
            return_geometry=False,
            return_all_records=True
        )

        key_map = {}
        for f in result.features:
            attrs = f.attributes
            key = (attrs[id_field], attrs[time_field])
            key_map[key] = attrs["OBJECTID"]

        logger.info("Fetched %d existing feature keys from layer", len(key_map))
        return key_map

    def _split_updates_and_adds(self, gdf, id_field, time_field, existing_map):
        from arcgis.features import Feature

        updates = []
        adds = []

        for _, row in gdf.iterrows():
            attributes = row.drop(labels="geometry").to_dict()
            attributes = self._sanitize_attributes(attributes)

            geom = row.geometry.__geo_interface__
            esri_geom = self._geojson_geom_to_esri(geom)

            # time_field value must already be epoch-ms (via _sanitize_attributes)
            # to match the format returned by the server in existing_map
            key = (attributes[id_field], attributes[time_field])

            if key in existing_map:
                attributes["OBJECTID"] = existing_map[key]
                updates.append(Feature(geometry=esri_geom, attributes=attributes))
            else:
                adds.append(Feature(geometry=esri_geom, attributes=attributes))

        return updates, adds

    def _push_updates(self, feature_layer, updates):
        max_batch = feature_layer.properties.maxRecordCount or 1000
        for i in range(0, len(updates), max_batch):
            batch = updates[i:i + max_batch]
            result = feature_layer.edit_features(updates=batch)
            failures = [r for r in result["updateResults"] if not r["success"]]
            if failures:
                logger.warning("Update failures in batch %d: %s", i, failures)
            else:
                logger.info("Updated batch %d: %d features", i, len(batch))

    def _push_adds(self, feature_layer, adds):
        max_batch = feature_layer.properties.maxRecordCount or 1000
        for i in range(0, len(adds), max_batch):
            batch = adds[i:i + max_batch]
            result = feature_layer.edit_features(adds=batch)
            failures = [r for r in result["addResults"] if not r["success"]]
            if failures:
                logger.warning("Add failures in batch %d: %s", i, failures)
            else:
                logger.info("Added batch %d: %d features", i, len(batch))

    def _sanitize_attributes(self, attributes: dict) -> dict:
        import pandas as pd

        clean = {}
        for k, v in attributes.items():
            if pd.isna(v):
                clean[k] = None
            elif isinstance(v, pd.Timestamp):
                # Esri date fields expect epoch milliseconds (UTC)
                clean[k] = int(v.timestamp() * 1000)
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        return clean

    def _geojson_geom_to_esri(self, geom):
        geom_type = geom["type"]
        coords = geom["coordinates"]
        sr = {"wkid": 4326}

        if geom_type == "Point":
            x, y = coords
            return {"x": x, "y": y, "spatialReference": sr}
        elif geom_type == "MultiPoint":
            return {"points": coords, "spatialReference": sr}
        elif geom_type == "LineString":
            return {"paths": [coords], "spatialReference": sr}
        elif geom_type == "MultiLineString":
            return {"paths": coords, "spatialReference": sr}
        elif geom_type == "Polygon":
            return {"rings": coords, "spatialReference": sr}
        elif geom_type == "MultiPolygon":
            rings = []
            for polygon in coords:
                rings.extend(polygon)
            return {"rings": rings, "spatialReference": sr}
        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")

    def get_feature_layer(self, url):

        gis = GIS("https://www.arcgis.com", client_id=self.arcgis_client_id, client_secret=self.arcgis_client_secret)
        logger.info("Successfully connected to ArcGIS GIS. gis: %s", gis._client_id)
        feature_layer = FeatureLayer(url, gis=gis)
        return feature_layer