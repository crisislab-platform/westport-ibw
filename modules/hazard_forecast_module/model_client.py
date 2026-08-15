import logging
import os

import numpy as np
import torch

from modules.lib import (
    HAZARD_MODEL_PATH,
    LATENT_FLOW_MATCHING_ENSEMBLE,
    AUTO_ENCODER,
    AUTO_ENCODER_DECODER,
)
from modules.hazard_forecast_module.ensemble_client import LatentFlowEnsemble
from modules.hazard_forecast_module.spatial_sampler.latent_canvas import LatentCanvas
from modules.utils.utils import check_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FlowMatchingModelClient")


class FlowMatchingModelClient():
    """Hazard forecasts from the deep ensemble of latent flow-matching models.

    Every pooled sample is propagated downstream rather than a collapsed mean,
    so the exposure computation stays probabilistic. The cost of that is linear
    in ``M*K``: the ensemble holds ``M*K*B`` depth maps at ~24 MB each, and the
    impact module runs its exposure computation once per member, so ``K`` is
    the knob to turn if a run is too slow or too large.
    """

    # M = None uses every exported member. K and N default to the values the
    # research repo's ensemble_test was run at.
    def __init__(self, num_members=10, samples_per_member=1, num_steps=10, seed=None):
        self.latent_canvas = LatentCanvas()
        self.device = check_device()
        self.ensemble = self.load_model(num_members, samples_per_member, num_steps, seed)

    def load_model(self, num_members, samples_per_member, num_steps, seed):
        ensemble_dir = os.path.join(HAZARD_MODEL_PATH, LATENT_FLOW_MATCHING_ENSEMBLE)
        decoder_path = os.path.join(
            HAZARD_MODEL_PATH, AUTO_ENCODER, AUTO_ENCODER_DECODER + ".onnx"
        )
        return LatentFlowEnsemble(
            ensemble_dir=ensemble_dir,
            decoder_path=decoder_path,
            latent_canvas=self.latent_canvas,
            num_members=num_members,
            samples_per_member=samples_per_member,
            num_steps=num_steps,
            seed=seed,
        )

    def iter_hazard_forecast(self, bc_forecast):
        """Stream the hazard forecast, one step at a time.

        Yields ``(timestamp, (M*K, H, W))`` -- every pooled ensemble member as
        a depth map, NaN outside the computational domain -- decoding each step
        only when it is reached.

        This is the memory-bounded way to run a horizon. The depth maps are
        ~24 MB each, so a step costs ``M*K`` of them and the whole horizon
        costs ``M*K*B``: at M*K=100 over 12 steps that is ~28 GB held at once,
        against ~2.4 GB if each step is scored and dropped before the next is
        decoded. The peak stops depending on the horizon entirely.

        Because the tensors are views onto the step's decoded block, a consumer
        that keeps them keeps that block alive too. Use ``get_hazard_forecast``
        if the whole horizon really is needed at once.
        """
        timestamps = list(bc_forecast.keys())
        for idx, result in self._forecast_steps(bc_forecast):
            yield timestamps[idx], torch.from_numpy(result.members[:, 0])

    def get_hazard_forecast(self, bc_forecast):
        """Hazard forecast for a boundary-condition forecast, all steps at once.

        Returns ``{timestamp: (M*K, H, W)}``. Every step is held simultaneously,
        so prefer ``iter_hazard_forecast`` for anything but a single step.
        """
        return dict(self.iter_hazard_forecast(bc_forecast))

    def get_hazard_forecast_with_uncertainty(self, bc_forecast):
        """As ``get_hazard_forecast``, keeping the uncertainty decomposition.

        Returns ``{timestamp: EnsembleForecast}``, each holding that step's
        members alongside the pixel-space mean, the per-cell std over the
        pooled samples, and the aleatoric/epistemic split.
        """
        timestamps = list(bc_forecast.keys())
        return {
            timestamps[idx]: result
            for idx, result in self._forecast_steps(bc_forecast)
        }

    def _forecast_steps(self, bc_forecast):
        """``(step_index, EnsembleForecast)`` for each step of ``bc_forecast``."""
        # np.array first: building a tensor straight from a list of arrays
        # copies them one at a time and torch warns about it.
        cond = torch.tensor(
            np.array(list(bc_forecast.values())), dtype=torch.float32
        )
        # The DEM latent is static; the condition carries one row per forecast
        # step, so the whole horizon is still SAMPLED as one batch -- only the
        # decode is stepped.
        return self.ensemble.forecast_steps(cond, self.latent_canvas.z_dem)


def _select_step(forecast, idx):
    """Slice one forecast step out of a batched EnsembleForecast.

    The step axis is kept (length 1) so ``members`` stays (M*K, 1, H, W) and
    the reductions stay (1, H, W), matching the batched layout.

    No longer used on the forecast path -- LatentFlowEnsemble.forecast_steps
    reduces each step on its own instead, which is what makes the peak
    independent of the horizon. Kept because the slices it produces are the
    layout contract the stepped path reproduces, and because it is still the
    way to take one step out of a forecast that was batched deliberately.
    """
    return type(forecast)(
        members=forecast.members[:, idx:idx + 1],
        mean=forecast.mean[idx:idx + 1],
        std=forecast.std[idx:idx + 1],
        aleatoric=forecast.aleatoric[idx:idx + 1],
        epistemic=forecast.epistemic[idx:idx + 1],
        num_members=forecast.num_members,
        samples_per_member=forecast.samples_per_member,
    )
