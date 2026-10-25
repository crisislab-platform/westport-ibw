import logging
import os

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from modules.inundation_module.ensemble_client import _warn_on_cpu_fallback, select_providers
from modules.hydromet_module.bc_module import simulated_map_files
from modules.lib import BOUNDARY_CONDITIONS_DIR, CNN1D_CHECKPOINT_DIR

logger = logging.getLogger(__name__)


class Cnn1dInundation():
    """Hazard forecasts from the westport 1D-CNN benchmark, run with onnxruntime
    so it is scored by exactly the code that scores the surrogate.

    The network, its input scalers and the cells it predicts come from
    scripts/export_cnn1d_onnx.py. It maps one hour's boundary conditions --
    river flow and sea level now and at the LAG previous rows, and the hours
    since the event's first simulated map, each standardised -- to the depth of
    every in-domain cell. Deterministic, so every hour is a one-member
    ensemble: (B, 1, H, W).
    """

    def __init__(self, hydromet_module):
        scalers = joblib.load(os.path.join(CNN1D_CHECKPOINT_DIR, "scalers", "bc_scalers.joblib"))
        self.inputs = self.event_inputs(hydromet_module.event_code, hydromet_module.event_name, scalers,
                                        forecast_times=hydromet_module.bc_dict.keys())
        self.domain_mask = np.load(os.path.join(CNN1D_CHECKPOINT_DIR, "domain_mask.npy"))
        providers = select_providers()
        self.session = ort.InferenceSession(os.path.join(CNN1D_CHECKPOINT_DIR, "model.onnx"), providers=providers)
        _warn_on_cpu_fallback(providers, self.session)

    @staticmethod
    def event_inputs(event_code, event_name, scalers, forecast_times=()):
        """``{valid_time: (features,)}`` for every row of the event's boundary
        conditions that has a simulated map, as westport's
        CNN1DDataManager.preprocess_boundary_conditions builds them.

        Its rules, kept as they are because the network was trained on them:
        only rows with a simulated map are kept; missing lags at the start are
        interpolated and back/forward filled; the time input is hours since the
        first kept row. One departure: only on-the-hour rows are kept, as
        HydroMetModule._to_hourly_grid does for the surrogate, so both models
        get hourly boundary conditions and a lag is always an hour (westport
        shifted over the half-hourly rows of 37 events, a half-hour lag there).

        An event with no simulated maps (July 2021) builds the lags over its
        whole record, so they are recorded history, and then keeps the rows at
        ``forecast_times``; its time input counts from the first of those, as a
        test event's counts from its first scored hour.
        """
        lag = scalers["lag"]
        # Named columns: the test events' files lead with an unnamed index
        # column, which westport drops; July 2021's file has none.
        bcs = pd.read_csv(os.path.join(BOUNDARY_CONDITIONS_DIR, f"{event_code}.csv"),
                          usecols=["Date_Time", "River_flow", "Sea_level"])
        bcs["Date_Time"] = pd.to_datetime(bcs["Date_Time"])
        map_times = set(simulated_map_files(event_name))
        on_the_hour = bcs["Date_Time"].dt.minute == 0
        bcs = bcs[(bcs["Date_Time"].isin(map_times) & on_the_hour) if map_times else on_the_hour]
        bcs = bcs.sort_values(by="Date_Time").reset_index(drop=True)

        bcs["River_flow"] = bcs["River_flow"].astype(np.float32)
        bcs["Sea_level"] = bcs["Sea_level"].astype(np.float32)
        for i in range(1, lag + 1):
            bcs[f"River_flow-{i}"] = bcs["River_flow"].shift(i)
            bcs[f"Sea_level-{i}"] = bcs["Sea_level"].shift(i)
        lagged = [f"River_flow-{i}" for i in range(1, lag + 1)] + [f"Sea_level-{i}" for i in range(1, lag + 1)]
        bcs[lagged] = bcs[lagged].interpolate(method="linear", limit_direction="both").bfill().ffill()
        if not map_times:
            bcs = bcs[bcs["Date_Time"].isin(set(forecast_times))].reset_index(drop=True)
        bcs["timestep"] = (bcs["Date_Time"] - bcs["Date_Time"].min()).dt.total_seconds() / 3600.0

        features = np.hstack([
            scalers["flow_scaler"].transform(bcs[scalers["flow_columns"]].values),
            scalers["sea_level_scaler"].transform(bcs[scalers["sea_level_columns"]].values),
            scalers["time_step_scaler"].transform(bcs["timestep"].values.reshape(-1, 1)),
        ]).astype(np.float32)
        return dict(zip(bcs["Date_Time"], features))

    def iter_hazard_forecast_batches(self, bc_forecast, hours_per_batch=1):
        """As InundationModule.iter_hazard_forecast_batches: ``(timestamps,
        (B, 1, H, W))`` per batch. Only the boundary conditions' timestamps are
        used; the CNN builds its own inputs for them. Depths are clamped at 0
        and cells outside the domain are 0, as westport writes them."""
        timestamps = [t for t in bc_forecast if t in self.inputs]
        missing = [t for t in bc_forecast if t not in self.inputs]
        if missing:
            logger.warning("%d hours have no CNN input and are not scored, first %s", len(missing), missing[0])
        for start in range(0, len(timestamps), hours_per_batch):
            batch = timestamps[start:start + hours_per_batch]
            bc = np.stack([self.inputs[t] for t in batch])[:, None, :]
            depth = np.maximum(self.session.run(None, {"bc": bc})[0], 0.0)
            maps = np.zeros((len(batch), *self.domain_mask.shape), dtype=np.float32)
            maps[:, self.domain_mask] = depth
            yield batch, torch.from_numpy(maps)[:, None]
