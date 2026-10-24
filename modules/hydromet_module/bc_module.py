"""
This module is responsible for fetching the boundary condition data and processing it to be used in the hazard model.
"""
import pandas as pd
from modules.lib import JULY_2021_BC_FILE_PATH, LATENT_FLOW_MATCHING, MODEL_CHECKPOINTS_DIR, BOUNDARY_CONDITIONS_DIR, FLOOD_MAPS_DIR
import os
import joblib
import numpy as np
import pandas as pd

# Flood scenarios hydrographs
import os
import glob
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# July 2021 is forecast from here: the flood began on the 16th. The lags of the
# first hours still come from the record before it.
JULY_2021_START = pd.Timestamp("2021-07-16 00:00:00")


class HydroMetModule():
    def __init__(self, scenario_id):
        self.lag = 7
        self.scenario_id = scenario_id
        self.create_indices()
        self.event_code = self.event_index.get(self.scenario_id)
        self.event_name = self.event_name_map.get(self.event_code)
        self.fetch_boundary_conditions()
        
    def create_indices(self):
        self.event_index = {
            0: "July2021",
            1: "Base0",
            2: "Base1",
            3: "Base2",
            4: "Base3",
            5: "Base4",
            6: "Base5",
            7: "Base6",
            8: "Base7",
            9: "Base8", 
        }
        for i in range(1,98):
            self.event_index[9+i] = f"S{i:02d}"
            
        self.event_name_map = {
            "July2021": "July_2021",
            "Base0": "Base0_Buller20yrARI_v2",
            "Base1": "Base1_Buller50yrARI_v2",
            "Base2": "Base2_Buller100yrARI",
            "Base3": "Base3_Buller50yrARI_RCP4pt5",
            "Base4": "Base4_Buller50yrARI_RCP6pt0",
            "Base5": "Base5_Buller50yrARI_RCP8pt5",
            "Base6": "Base6_Buller100yrARI_RCP4pt5",
            "Base7": "Base7_Buller100yrARI_RCP6pt0",
            "Base8": "Base8_Buller100yrARI_RCP8pt5"
        }
        
        for i in range(1,98):
            self.event_name_map[f"S{i:02d}"] = f"S{i:02d}"
        
    def get_bc_forecast(self, start_time=None, forecast_horizon=None):
        if start_time == None and forecast_horizon == None:
            return forecast_data
        forecast_end_time = start_time + pd.Timedelta(hours=forecast_horizon-1)
        forecast_data = {k: v for k, v in self.bc_dict.items() if start_time <= k <= forecast_end_time}
        return forecast_data
    
    def extract_timestep_from_filename(self, filename):
        # Timestep format is depth_at_2000-01-02_00-00-00.tif, so we need to extract the part after 'depth_at_' and before '.tif'
        timestep_str = filename.split("depth_at_")[1].split(".tif")[0]
        # Convert to datetime object
        timestep_dt = pd.to_datetime(timestep_str, format="%Y-%m-%d_%H-%M-%S")
        return timestep_dt
    
    def fetch_boundary_conditions(self):
        bc_file_path = os.path.join(BOUNDARY_CONDITIONS_DIR, f"{self.event_code}.csv")
        bcs = pd.read_csv(bc_file_path)
        self.preprocess_boundary_conditions(bcs)
        
    def _to_hourly_grid(self, inflow_data):
        """Keep the on-the-hour rows, so every event is on one hourly grid.

        37 of the 107 boundary-condition files are recorded half-hourly and the
        other 70 hourly, but everything downstream assumes hourly and breaks
        three separate ways on the half-hourly ones:

        * the lags below are POSITIONAL shifts, so shift(1..7) spans 3.5 h of
          history instead of 7 h -- and the scalers were fitted on the hourly
          window, so the model is handed a shorter history normalised with the
          wrong statistics;
        * get_bc_forecast returns TWO steps for a one-hour horizon, doubling
          num_timesteps and the work per issue;
        * run_hourly_forecast advances an hour at a time while
          validate_impact_forecast takes the first forecast step, so the
          comparison becomes a 30-minute-ahead forecast against the nowcast
          map -- a different lead time from the hourly events it is averaged
          with.

        Selecting rather than resampling is exact: every file is aligned to the
        hour (minutes are {0} for the hourly files and {0, 30} for the rest),
        so no interpolation is involved and this is a no-op for the 70 already
        hourly. The half-past flood maps go unused, which costs those events
        half their validation timesteps and buys a lead time that means the
        same thing across the set.
        """
        on_the_hour = inflow_data['Date_Time'].dt.minute == 0
        if not on_the_hour.all():
            logger.info(
                "%s: boundary conditions are sub-hourly; keeping %d of %d rows "
                "to put the event on an hourly grid",
                self.event_code, int(on_the_hour.sum()), len(inflow_data),
            )
            inflow_data = inflow_data[on_the_hour].copy()

        # The lags are positional, so a gap here silently means the wrong
        # history rather than a missing value. Worth saying out loud.
        gaps = inflow_data['Date_Time'].diff().dropna()
        if len(gaps) and (gaps != pd.Timedelta(hours=1)).any():
            logger.warning(
                "%s: boundary conditions are not contiguous hourly after "
                "gridding (gaps: %s); the lagged conditioning spans clock time "
                "unevenly", self.event_code,
                sorted({str(g) for g in gaps[gaps != pd.Timedelta(hours=1)]}),
            )
        return inflow_data

    def preprocess_boundary_conditions(self, inflow_data):
        #load the scalers first
        flow_scaler, sea_level_scaler = self.load_scalers()
        self.bc_dict = {}
        self.flood_map_dict = {}
        inflow_data['Date_Time'] = pd.to_datetime(inflow_data['Date_Time'])
        inflow_data = inflow_data.sort_values(by="Date_Time").reset_index(drop=True)
        
        date_time = inflow_data["Date_Time"]
        timestep= (date_time - date_time.min()).dt.total_seconds() / 3600.0 
        inflow_data['timestep'] = timestep.values.reshape(-1, 1)  # Reshape for proper 
        
        lagged_columns = []
        #convert the river flow and sea level columns to float32 to save memory and ensure compatibility with PyTorch
        inflow_data['River_flow'] = inflow_data['River_flow'].astype(np.float32)
        inflow_data['Sea_level'] = inflow_data['Sea_level'].astype(np.float32)
        
        if self.scenario_id != 0:
            flood_map_files = glob.glob(f"{FLOOD_MAPS_DIR}/{self.event_name}_depth_snapshots/depth_at_*.tif")
            flood_map_files.sort()
            
            # Extract the timestamp from the flood map file names and store in a list
            flood_map_timestamps = [self.extract_timestep_from_filename(file) for file in flood_map_files]
            
            inflow_data['Flood_Map_Available'] = inflow_data['Date_Time'].apply(lambda x: True if x in flood_map_timestamps else False)
            inflow_data['flood_map_file'] = inflow_data['Date_Time'].apply(lambda x: next((file for file in flood_map_files if self.extract_timestep_from_filename(file) == x), None))
            inflow_data = inflow_data[inflow_data['Flood_Map_Available']]
            inflow_data = inflow_data[inflow_data['Flood_Map_Available']]
            inflow_data.drop(columns=['Flood_Map_Available'], inplace=True)

        inflow_data = self._to_hourly_grid(inflow_data)

        for i in range(1, self.lag + 1):
            inflow_data[f'River_flow-{i}'] = inflow_data['River_flow'].shift(i)
            inflow_data[f'Sea_level-{i}'] = inflow_data['Sea_level'].shift(i)

        flow_window_cols = ['River_flow'] + [f'River_flow-{i}' for i in range(1, self.lag + 1)]
        flow_columns = flow_window_cols
        sea_level_columns = ['Sea_level'] + [f'Sea_level-{i}' for i in range(1, self.lag + 1)]
  
        lagged_flow_cols = [f'River_flow-{i}' for i in range(1, self.lag + 1)]
        lagged_sea_cols = [f'Sea_level-{i}' for i in range(1, self.lag + 1)]
        lagged_columns = lagged_flow_cols + lagged_sea_cols

        inflow_data[lagged_columns] = (
            inflow_data[lagged_columns]
            .interpolate(method='linear', limit_direction='both')
            .bfill()
            .ffill()
        )
        if self.scenario_id == 0:
            # Trimmed after the lags, so the first hours' history is the
            # recorded one rather than filled in.
            inflow_data = inflow_data[inflow_data['Date_Time'] >= JULY_2021_START]
    
        scaled_flow = flow_scaler.transform(inflow_data[flow_columns].values)
        scaled_flow = scaled_flow.astype(np.float32)

        scaled_sea_level = sea_level_scaler.transform(inflow_data[sea_level_columns].values)
        scaled_sea_level = scaled_sea_level.astype(np.float32)
        
        physical_cond_channels = 2
        lagged_flow_sea_level = np.hstack((scaled_flow, scaled_sea_level)).reshape(-1, physical_cond_channels, self.lag + 1)
        
        #Assign each timestep's boundary condition data to a dictionary for easy access
        for i, row in enumerate(lagged_flow_sea_level):
            self.bc_dict[inflow_data['Date_Time'].iloc[i]] = row
            if self.scenario_id != 0:
                self.flood_map_dict[inflow_data['Date_Time'].iloc[i]] = inflow_data['flood_map_file'].iloc[i]

    @staticmethod
    def load_scalers():
        scaler_file_path = os.path.join(MODEL_CHECKPOINTS_DIR, LATENT_FLOW_MATCHING, "scalers", "bc_scalers.joblib")
        
        """Load previously saved boundary-condition scalers.

        Returns a tuple ``(flow_scaler, sea_level_scaler)``.
        """
        scalers = joblib.load(scaler_file_path)
        return scalers["flow_scaler"], scalers["sea_level_scaler"]
    
    def get_flood_map_file(self, timestamp):
        timestamp = pd.to_datetime(timestamp)
        return self.flood_map_dict.get(timestamp)
