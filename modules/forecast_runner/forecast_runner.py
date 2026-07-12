from modules.bc_forecast_module.bc_module import BC_Module
from modules.hazard_forecast_module.model_client import FlowMatchingModelClient
from modules.impact_module.impact_module import ImpactModule
from modules.arcgis_module.arcgis_module import ArcGISModule
from modules.lib import OUTPUT_DIR, PLACRD_FILE_PATH, SURVEYED_FLOOD_DEPTH_JUL_2021, EXPOSURE_STUDY_RESULTS
import pandas as pd
import geopandas as gpd
import rasterio as rio
import torch
import numpy as np
import logging
from sklearn.metrics import confusion_matrix
import os

logger = logging.getLogger(__name__)

class Integrator():
    def __init__(self, scenario_id, forecast_horizon, push_to_arcgis):
        self.scenario_id = scenario_id
        self.forecast_horizon = forecast_horizon
        self.push_to_arcgis = push_to_arcgis
        self.bc_module = BC_Module(scenario_id)
        self.hazard_module = FlowMatchingModelClient()
        self.impact_module = ImpactModule(scenario_id)
        if self.push_to_arcgis:
            self.arcgis_module = ArcGISModule()

    def run_hourly_forecast(self,  start_time, end_time):
        # Immitate the hourly forecast by running the forecast hourly starting from the start_time
        current_time = pd.to_datetime(start_time)
        last_available_time = max(self.bc_module.bc_dict.keys())
        end_time = min(pd.to_datetime(end_time), last_available_time)
        while current_time <= end_time:
            bc_forecast = self.bc_module.get_bc_forecast(current_time, self.forecast_horizon)
            #bc_forecast is a dictionary with keys as timestamps and values as boundary condition data
            #check if bc_forecast is empty
            if not bc_forecast:
                print(f"No boundary condition data available for the forecast period starting at {current_time}. Skipping this hour.")
                current_time += pd.Timedelta(hours=1)
                continue
            
            hazard_forecast = self.hazard_module.get_hazard_forecast(bc_forecast)
            #convert hazard forecast to tensor to numpy array for saving as a layer
            #Get the impact forecast for the forecast horizon using the hazard forecast
            impact_forecast = self.impact_module.get_impact_forecast(hazard_forecast)

            #Save as layers in a spatial format (e.g., GeoPackage) plus a metrics CSV.
            self.impact_module.save_impact_forecast(impact_forecast, hazard_forecast, current_time)
            
            #Update feature layer on spatial db
            if self.push_to_arcgis:
                self.arcgis_module.push_impacts_to_spatial_db(impact_forecast, start_time, end_time)
                
            current_time += pd.Timedelta(hours=1)
            
    def validate_forecast(self):
        
        if self.scenario_id == 0:
            # First validate the exposure calcaulation method
            self.validate_exposure_method()
            boundary_conditions = self.bc_module.get_bc_forecast()
            hazard_forecast = self.hazard_module.get_hazard_forecast(boundary_conditions)
            # Select the timestep with the highest inundation
            impact_forecast = self.impact_module.get_impact_forecast(hazard_forecast)
            self.test_against_placard_data()
            
        else: 
            pass
        
    def validate_exposure_method(self):
        #Load the srveyed flood depth
        #surveyed_flood_depthi
        with rio.open(SURVEYED_FLOOD_DEPTH_JUL_2021) as src:
            obs_depth = src.read(1).astype(float)
            transform = src.transform

        hazard_forecast = {
            "17-07-2021 00:00:00": [obs_depth]
        }
        impact_forecast = self.impact_module.get_impact_forecast(hazard_forecast)
        self.test_against_placard_data(impact_forecast)
        
    def test_against_placard_data(self, impact_forecast):
        building_impact = impact_forecast['building_impact']
        #load placard data
        obs_placard = gpd.read_file(PLACRD_FILE_PATH)
        obs_placard_affected = obs_placard[obs_placard['DamageNb'].isin([1.0, 2.0, 3.0])]
        
        building_ids_affected = obs_placard_affected['building_id'].values
        building_impact = building_impact.copy()

        # Ground truth: affected if the building ID is in the observations
        building_impact["observed"] = building_impact["building_id"].isin(building_ids_affected)

        # Prediction
        building_impact["predicted"] = building_impact["inundated"]

        tn, fp, fn, tp = confusion_matrix(
            building_impact["observed"],
            building_impact["inundated"]
        ).ravel()
        
        hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Critical Success Index (CSI / Threat Score).
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        metrics = {
            "scenario_id":self.scenario_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "hit_rate": hit_rate,
            "csi": csi,
            "num_buildings":len(building_impact.predicted)
        }
        
        metrics_df = pd.DataFrame([metrics])

        output_dir = EXPOSURE_STUDY_RESULTS# Change to your desired output directory
        os.makedirs(output_dir, exist_ok=True)

        metrics_df.to_csv(
            os.path.join(output_dir, "exposure_validation_metrics.csv"),
            index=False,
        )

        print("Metrics saved successfully.")
        #Save the metrics to a file 


        