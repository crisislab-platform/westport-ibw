import os
import logging
import onnx
import torch
from modules.lib import HAZARD_MODEL_PATH, LATENT_FLOW_MATCHING, AUTO_ENCODER, AUTO_ENCODER_DECODER
from modules.utils.utils import check_device
import rasterio as rio
import numpy as np

import onnxruntime as ort
from modules.hazard_forecast_module.spatial_sampler.latent_canvas import LatentCanvas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FlowMatchingModelClient")

class FlowMatchingModelClient():
    def __init__(self,):
        self.flood_map_shape = None
        self.load_model() 
        self.latent_canvas = LatentCanvas()
        self.device = check_device()
        self.number_of_inference_steps = 20
        self.sampling_method = "euler"
        self.use_dem = True
        
    def load_model(self) -> bool:
        try:
            
            # Load the ONNX model for flow matching
            flow_matching_model_path = os.path.join(HAZARD_MODEL_PATH, LATENT_FLOW_MATCHING, LATENT_FLOW_MATCHING + ".onnx")
            flow_matching_model = onnx.load(flow_matching_model_path)
            onnx.checker.check_model(flow_matching_model)
            
            # Load the ONNX model for autoencoder
            autoencoder_model_path = os.path.join(HAZARD_MODEL_PATH, AUTO_ENCODER, AUTO_ENCODER_DECODER + ".onnx")
            autoencoder_model = onnx.load(autoencoder_model_path)
            onnx.checker.check_model(autoencoder_model)
            
            # Create ONNX Runtime sessions for inference
            self.flow_matching_session = ort.InferenceSession(flow_matching_model_path, providers=['CUDAExecutionProvider'])
            self.autoencoder_session = ort.InferenceSession(autoencoder_model_path, providers=['CUDAExecutionProvider'])
            
        except Exception as e:
            logger.error(f"Error loading hazard model: {e}")
            return False
        
    def get_hazard_forecast(self, bc_forecast):
        """
        Get the hazard forecast for a given a boundary condition forecast. This method takes in a boundary condition forecast and returns the corresponding hazard forecast.
        """
        
        #bc_forecast is a dictionary with keys as timestamps and values as boundary condition data. We need to convert this into a tensor that can be fed into the model.
        bc_forecast_keys = list(bc_forecast.keys())
        bc_forecast_values = torch.tensor(list(bc_forecast.values()), dtype=torch.float32).to(self.device)
        # Perform inference using the flow matching model to get the latent flood map
        mean, members = self.inference_ensemble(q_k=bc_forecast_values, ensemble_size=2)
        
        #Need to assign nan to out_of_domain values
        
        
        hazard_forecast = {}
        for idx, timestamp in enumerate(bc_forecast_keys):
            #get all the members for this timestep. 
            #now the shape is (ensemble_size, timsteps, C, H, W).
            hazard_forecast[timestamp] = members[:, idx, :, :].detach().cpu()
        return hazard_forecast

    def inference_ensemble(self, q_k, ensemble_size=5):
        """
        Returns
        -------
        mean : torch.Tensor
            Ensemble mean latent (B, C, H, W) - used for the deterministic metrics.
        members : torch.Tensor
            Stacked latent samples (N, B, C, H, W), N = ``ensemble_size``.
        """
        z_dem = self.latent_canvas.z_dem.clone().to(self.device)
        batch_size = q_k.shape[0]

        samples = self.sample(q_k, z_dem, num_samples=ensemble_size)
        members = samples.view(ensemble_size, batch_size, *samples.shape[1:]).detach()
        mean = members.mean(dim=0)
        return mean, members
    
    def predict_velocity(self, t, x, q_k):
        inputs = {
            "x": x.detach().cpu().numpy(),
            "t": t.detach().cpu().numpy(),
            "cond": q_k.detach().cpu().numpy(),
        }
        u_t = self.flow_matching_session.run(["velocity"], inputs)[0]
        return torch.from_numpy(u_t).to(x.device)


    def sample(self, condition: torch.Tensor, z_dem: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        with torch.no_grad():
            # The condition carries one entry per forecast step (batch dim), while the
            # DEM latent is shared across all steps. Repeat the DEM so its batch matches
            # the condition batch before running the model.
            cond_batch = condition.shape[0]
            if z_dem.shape[0] != cond_batch:
                z_dem = z_dem.repeat(cond_batch, *([1] * (z_dem.dim() - 1)))

            if num_samples > 1:
                condition = condition.repeat(num_samples, *([1] * (condition.dim() - 1)))
                z_dem = z_dem.repeat(num_samples, *([1] * (z_dem.dim() - 1)))
                z_t = torch.randn_like(z_dem).to(self.device)
            else:
                z_t = torch.randn_like(z_dem).to(self.device)
 
            for idx in range(self.number_of_inference_steps):
                t_schedule = idx / self.number_of_inference_steps
                t = torch.full(
                    (z_t.shape[0],),
                    fill_value=t_schedule,
                    device=z_t.device, 
                    dtype=z_t.dtype
                )
                if self.sampling_method == 'euler':
                    z_t = self.euler_step(z_t, t, condition, z_dem)
                elif self.sampling_method == 'heun':
                    z_t = self.heun_step(z_t, t, condition, z_dem) 
                elif self.sampling_method == 'rk4':
                    z_t = self.rk4_step(z_t, t, condition, z_dem)
                elif self.sampling_method == 'euler_maruyama':
                    z_t = self.euler_maruyama(z_t, t, condition, z_dem)
                elif self.sampling_method == 'heun_sde':
                    z_t = self.sde_heun(z_t, t, condition, z_dem)
            reconstructed_flood_map = self.reconstruct_flood_map(z_t)
            return reconstructed_flood_map

    def _build_input(self, z_t, z_dem):
        parts = [z_t]
        if self.use_dem:
            parts.append(z_dem)
        return torch.cat(parts, dim=1)

    def rk4_step(self, z_t, t, condition, z_dem):
        dt = 1.0 / self.number_of_inference_steps
        input_batch_k1 = self._build_input(z_t, z_dem)
        k1 = self.predict_velocity(t, input_batch_k1, condition)
        input_batch_k2 = self._build_input(z_t + 0.5*dt*k1, z_dem)
        k2 = self.predict_velocity(t + 0.5*dt, input_batch_k2, condition)
        input_batch_k3 = self._build_input(z_t + 0.5*dt*k2, z_dem)
        k3 = self.predict_velocity(t + 0.5*dt, input_batch_k3, condition)
        input_batch_k4 = self._build_input(z_t + dt*k3, z_dem)
        k4 = self.predict_velocity(t + dt, input_batch_k4, condition)
        return z_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def euler_step(self, z_t, t, condition, z_dem):
        input_batch = self._build_input(z_t, z_dem)
        u_t = self.predict_velocity(t, input_batch, condition)
        dt = torch.tensor(1.0 / self.number_of_inference_steps, device=z_t.device, dtype=z_t.dtype)
        z_t_next = z_t + u_t * dt
        return z_t_next
    
    def heun_step(self, z_t, t, condition, z_dem):
        input_batch = self._build_input(z_t, z_dem)
        u_t = self.predict_velocity(t, input_batch, condition)
        dt = torch.tensor(1.0 / self.number_of_inference_steps, device=z_t.device, dtype=z_t.dtype)
        z_t_euler = z_t + u_t * dt
        
        # Compute velocity at the Euler step
        input_batch_euler = self._build_input(z_t_euler, z_dem)
        u_t_euler = self.predict_velocity(t + dt, input_batch_euler, condition)
        
        # Heun's method update
        z_t_next = z_t + (u_t + u_t_euler) * (dt / 2)
        return z_t_next

    def euler_maruyama(self, z_t, t, condition, z_dem):
        input_batch = self._build_input(z_t, z_dem)
        u_t = self.predict_velocity(t, input_batch, condition)
        dt = 1.0 / self.number_of_inference_steps
        g = float(getattr(self.config, "churn", 0.0))

        # t is filled with a single schedule value across the batch in sample().
        t_val = float(t.reshape(-1)[0])
        if g > 0.0 and self.config.gaussian_source and (1.0 - t_val) > dt:
            t_col = t.view((-1,) + (1,) * (z_t.dim() - 1))
            score = (t_col * u_t - z_t) / (1.0 - t_col)
            noise = torch.randn_like(z_t)
            return z_t + (u_t + 0.5 * g * g * score) * dt + g * (dt ** 0.5) * noise
        return z_t + u_t * dt

    def sde_heun(self, z_t, t, condition, z_dem):
        dt = 1.0 / self.number_of_inference_steps
        g = float(getattr(self.config, "churn", 0.0))
        t_val = float(t.reshape(-1)[0])

        input_batch = self._build_input(z_t, z_dem)
        u_t = self.predict_velocity(t, input_batch, condition)

        # The corrector evaluates the score at t+dt, where 1-(t+dt) can hit 0 on the final
        # interior step; guard both the current and next score against the 1/(1-t) blow-up.
        stochastic = (g > 0.0 and self.config.gaussian_source
                      and (1.0 - t_val) > dt and (1.0 - (t_val + dt)) > dt)
        if not stochastic:
            # Deterministic Heun (drift only).
            z_t_euler = z_t + u_t * dt
            input_batch_euler = self._build_input(z_t_euler, z_dem)
            u_t_euler = self.predict_velocity(t + dt, input_batch_euler, condition)
            return z_t + (u_t + u_t_euler) * (dt / 2)

        t_col = t.view((-1,) + (1,) * (z_t.dim() - 1))
        score = (t_col * u_t - z_t) / (1.0 - t_col)
        drift = u_t + 0.5 * g * g * score

        # Single shared noise increment dW = sqrt(dt) * xi, reused by predictor + corrector.
        noise = g * (dt ** 0.5) * torch.randn_like(z_t)
        z_pred = z_t + drift * dt + noise

        t_next = t + dt
        t_next_col = t_next.view((-1,) + (1,) * (z_t.dim() - 1))
        input_batch_pred = self._build_input(z_pred, z_dem)
        u_pred = self.predict_velocity(t_next, input_batch_pred, condition)
        score_pred = (t_next_col * u_pred - z_pred) / (1.0 - t_next_col)
        drift_pred = u_pred + 0.5 * g * g * score_pred

        return z_t + 0.5 * (drift + drift_pred) * dt + noise
    
    def reconstruct_flood_map(self, tiles):
        canvas = self.latent_canvas.inverse_normalise_latent_flood_map(tiles)
        if canvas.dim() == 3:
            canvas = canvas.unsqueeze(0)

        images = []
        for b in range(canvas.shape[0]):
            latent_tiles = self.latent_canvas.canvas_to_latent_tiles(canvas[b]).to(self.device)
            pixel_tiles = self.autoencoder_session.run(
                ["reconstructed"],
                {"latent": latent_tiles.detach().cpu().numpy()},
            )[0]
            pixel_tiles = torch.from_numpy(pixel_tiles)
            image = self.latent_canvas.pixel_tiles_to_image(pixel_tiles)
            images.append(image)
        return torch.stack(images, dim=0)
    


