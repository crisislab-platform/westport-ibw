"""Latent Canvas with spatially-coherent layout.

The canvas is a direct spatial analogue of the full flood map, downsampled
by ``vae_downsample`` (D). Each latent tile is placed at the same (row, col)
position it would occupy in pixel space, just scaled down by D.

    Pixel domain:   (H,  W)   -> padded -> (Hp,  Wp)
    Latent canvas:  (C, Hp//D, Wp//D)

Crucially, the canvas geometry is read *directly* from the same
``FloodMapSampler`` that produced the pixel tiles (just divided by D). It uses
the exact same overlapping-layer layout, the exact same per-layer tile filter,
and the exact same row/col origins. This guarantees the latent tiles that come
out of the VAE (one per *filtered* pixel tile) line up 1:1 with the canvas
placement — there is no second, independently-filtered sampler that can drift
out of sync.

Reconstruction mirrors ``FloodMapSampler`` exactly: overlapping layers are
summed and divided by the per-cell overlap count, so neighbouring latent cells
correspond to neighbouring pixels and the spatial orientation of the original
flood map is preserved.
"""

import glob
import os

import torch
import logging
from modules.hazard_forecast_module.spatial_sampler.pixel_space_sampler import PixelSpaceSampler
import rasterio as rio
import numpy as np

from modules.lib import FLOOD_MAPS_DIR

logger = logging.getLogger(__name__)


class LatentCanvas:
    """Spatially-coherent latent canvas backed by FloodMapSampler geometry.

    Only a single :class:`FloodMapSampler` is used — the one that tiled the
    pixel-space flood map. The latent canvas geometry is obtained by dividing
    that sampler's pixel geometry by ``vae_downsample``.
    """

    def __init__(self):
        self.sampling_dist = 256
        self.tile_resolution = 256
        self.downsample_factor  = 16
        self.lt = self.tile_resolution // self.downsample_factor
        self.sample_flood_map = self.find_sample_flood_map()
        self.pixel_sampler = PixelSpaceSampler(self.sampling_dist, self.tile_resolution, self.sample_flood_map)
        self._domain_mask = self.get_domain_mask()
        
        #Load the static data from the latent distribution stats.
        self.z_dem = self.get_latent_dem_tiles()
        self.z_mean = torch.load("/data/westport/data/latent_floodmaps/latent_flood_map_stats.pt")["z_mean"]
        self.z_std = torch.load("/data/westport/data/latent_floodmaps/latent_flood_map_stats.pt")["z_std"]

        if self.tile_resolution % self.downsample_factor != 0:
            raise ValueError(
                f"tile_resolution ({self.tile_resolution}) must be divisible by "
                f"vae_downsample ({self.downsample_factor})"
            )

        # Make sure the pixel sampler's reconstruction geometry is available.
        # (FloodMapSampler builds this in __init__, but call it defensively so
        # we never read stale / missing attributes.)
        if not getattr(self.pixel_sampler, "layers_seperation_idxs", None):
            self.pixel_sampler.reconstruction_init()

        # Padded canvas dimensions, scaled down from the pixel domain.
        pad_h, pad_w, orig_h, orig_w = self.pixel_sampler.get_padding_height_width()
        padded_h = orig_h + pad_h
        padded_w = orig_w + pad_w
        if padded_h % self.downsample_factor != 0 or padded_w % self.downsample_factor != 0:
            raise ValueError(
                f"Padded pixel domain ({padded_h}, {padded_w}) must be divisible "
                f"by vae_downsample ({self.downsample_factor})"
            )
        self.orig_h = orig_h
        self.orig_w = orig_w
        self.Hc = padded_h // self.downsample_factor   # latent canvas height
        self.Wc = padded_w // self.downsample_factor   # latent canvas width

        # Layer geometry is read from the SAME sampler, just divided by D.
        self._layer_meta = self._build_layer_meta()

        self._num_tiles = sum(
            int(torch.sum(f).item())
            for f in self.pixel_sampler.tile_filter_list
        )

        # The latent canvas is only LOSSLESS when tiles do not overlap. Overlap
        # (sampling_dist < tile_resolution => more than one layer) forces the
        # forward/inverse pass to average independently-encoded latents, which
        # the VAE never saw and cannot decode back exactly. Warn loudly so the
        # canvas is not silently lossy.
        self._overlapping = len(self._layer_meta) > 1
        if self._overlapping:
            logger.warning(
                "LatentCanvas: %d overlapping layers detected "
                "(sampling_dist=%d < tile_resolution=%d). The canvas round-trip "
                "will be LOSSY because overlapping latent tiles are averaged. "
                "Use sampling_dist == tile_resolution for a lossless canvas.",
                len(self._layer_meta), self.pixel_sampler.sampling_dist, self.tile_resolution,
            )

        logger.info(
            f"LatentCanvas initialised — "
            f"pixel tile: {self.tile_resolution}px, latent tile: {self.lt}px, "
            f"canvas: ({self.Hc}, {self.Wc}), "
            f"layers: {len(self._layer_meta)}, "
            f"total_filtered_tiles: {self._num_tiles}"
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _build_layer_meta(self):
        """Read geometry from the pixel_sampler and divide spatial sizes by D."""
        meta = []
        ps = self.pixel_sampler

        for lyr_i in range(len(ps.layers_seperation_idxs)):
            row_origin_px, col_origin_px = ps.map_origins[lyr_i]
            row_size_px,   col_size_px   = ps.layer_sizes[lyr_i]
            start_idx,     end_idx       = ps.layers_seperation_idxs[lyr_i]
            filter_arr                   = ps.tile_filter_list[lyr_i]

            if row_origin_px % self.downsample_factor != 0 or col_origin_px % self.downsample_factor != 0:
                raise ValueError(
                    f"Layer {lyr_i} origin ({row_origin_px}, {col_origin_px}) "
                    f"is not divisible by vae_downsample ({self.downsample_factor}); the "
                    f"sampling_dist must be a multiple of D so latent tiles "
                    f"align to the canvas grid."
                )

            ax0_n = row_size_px // self.tile_resolution
            ax1_n = col_size_px // self.tile_resolution

            meta.append(dict(
                row_origin = row_origin_px // self.downsample_factor,
                col_origin = col_origin_px // self.downsample_factor,
                row_size   = ax0_n * self.lt,
                col_size   = ax1_n * self.lt,
                ax0_n      = ax0_n,
                ax1_n      = ax1_n,
                start_idx  = start_idx,
                end_idx    = end_idx,
                # keep the filter on CPU; moved to the working device on use
                filter_arr = filter_arr.detach().cpu().bool(),
            ))
        return meta

    def _unfilter_layer(self, flat_tiles, filter_arr, ax0_n, ax1_n, C):
        """Scatter filtered latent tiles back into a full (ax0_n*ax1_n) grid.

        Out-of-domain tile positions stay zero — the latent-space analogue of
        the flood-map sampler leaving out-of-domain pixels at zero.
        """
        full = torch.zeros(ax0_n * ax1_n, C, self.lt, self.lt,
                           dtype=flat_tiles.dtype, device=flat_tiles.device)
        full[filter_arr.to(flat_tiles.device)] = flat_tiles
        return full

    def _grid_to_patch(self, grid_tiles, ax0_n, ax1_n, C):
        tiles = grid_tiles.reshape(ax0_n, ax1_n, C, self.lt, self.lt)
        patch = tiles.permute(2, 0, 3, 1, 4).contiguous()
        patch = patch.reshape(C, ax0_n * self.lt, ax1_n * self.lt)
        return patch

    def _patch_to_grid(self, patch, ax0_n, ax1_n):
        C = patch.shape[0]
        tiles = patch.reshape(C, ax0_n, self.lt, ax1_n, self.lt)
        tiles = tiles.permute(1, 3, 0, 2, 4).contiguous()
        tiles = tiles.reshape(ax0_n * ax1_n, C, self.lt, self.lt)
        return tiles

    # ------------------------------------------------------------------
    # Pixel-tile helpers (delegate to the pixel sampler so the tiling matches
    # exactly what the autoencoder was trained / evaluated on)
    # ------------------------------------------------------------------
    def image_to_tiles(self, image: torch.Tensor) -> torch.Tensor:
        """(H, W) or (1, 1, H, W) -> (N_filtered, 1, tr, tr) pixel tiles."""
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        pixel_tiles = self.pixel_sampler.tiles_prep_func_gpu(image)  # (N, tr, tr)
        return pixel_tiles.unsqueeze(1)                              # (N, 1, tr, tr)

    def pixel_tiles_to_image(self, pixel_tiles: torch.Tensor) -> torch.Tensor:
        """(N_filtered, 1, tr, tr) or (N_filtered, tr, tr) -> (H, W)."""
        if pixel_tiles.dim() == 4:
            pixel_tiles = pixel_tiles.squeeze(1)
        image = self.pixel_sampler.reconstruct_full_map(
            pixel_tiles.to(self.pixel_sampler.device)
        )
        #apply the domain mask to the reconstructed image to ensure that areas outside the computational domain are set to NaN
        domain_mask = self.get_domain_mask().to(image.device)
        image = image * domain_mask
        #convert the image to a numpy array and set the areas outside the computational domain to NaN
        image = image.cpu().numpy()
        image[domain_mask.cpu().numpy() == 0] = np.nan
        image = torch.from_numpy(image)
        return image

    # ------------------------------------------------------------------
    # Forward: latent tiles -> canvas
    # ------------------------------------------------------------------
    def latent_tiles_to_canvas(self, latent_tiles: torch.Tensor) -> torch.Tensor:
        """(N_filtered, C, lt, lt) -> (C, Hc, Wc).

        Overlapping layers are summed and divided by the per-cell overlap count,
        exactly mirroring ``FloodMapSampler.reconstruct_full_map``.
        """
        N, C, lt, _ = latent_tiles.shape
        assert lt == self.lt,         f"Expected latent tile {self.lt}, got {lt}"
        assert N  == self._num_tiles, f"Expected {self._num_tiles} tiles, got {N}"

        device = latent_tiles.device
        canvas = torch.zeros(C, self.Hc, self.Wc,
                             dtype=latent_tiles.dtype, device=device)
        counts = torch.zeros(1, self.Hc, self.Wc,
                             dtype=latent_tiles.dtype, device=device)

        for meta in self._layer_meta:
            start, end   = meta['start_idx'],  meta['end_idx']
            ax0_n, ax1_n = meta['ax0_n'],      meta['ax1_n']
            r0, c0       = meta['row_origin'],  meta['col_origin']
            rs, cs       = meta['row_size'],    meta['col_size']
            filter_arr   = meta['filter_arr']

            layer_tiles = self._unfilter_layer(
                latent_tiles[start:end], filter_arr, ax0_n, ax1_n, C
            )
            patch = self._grid_to_patch(layer_tiles, ax0_n, ax1_n, C)

            canvas[:, r0:r0 + rs, c0:c0 + cs] += patch
            counts[:, r0:r0 + rs, c0:c0 + cs] += 1.0

        return canvas / counts.clamp_min(1.0)

    # Name used by latent_space_generator.create_latent_canvas_dataset
    overlapping_latent_tiles_to_canvas = latent_tiles_to_canvas

    def image_to_canvas(self, image: torch.Tensor, vae_encode_fn) -> torch.Tensor:
        pixel_tiles  = self.image_to_tiles(image)
        latent_tiles = vae_encode_fn(pixel_tiles)
        return self.latent_tiles_to_canvas(latent_tiles)

    # ------------------------------------------------------------------
    # Inverse: canvas -> latent tiles
    # ------------------------------------------------------------------
    def canvas_to_latent_tiles(self, canvas: torch.Tensor) -> torch.Tensor:
        """(C, Hc, Wc) -> (N_filtered, C, lt, lt)."""
        if canvas.dim() == 4:
            canvas = canvas.squeeze(0)
        C, Hc, Wc = canvas.shape
        assert (Hc, Wc) == (self.Hc, self.Wc), \
            f"Expected canvas ({self.Hc}, {self.Wc}), got ({Hc}, {Wc})"

        all_tiles = []
        for meta in self._layer_meta:
            ax0_n, ax1_n = meta['ax0_n'],      meta['ax1_n']
            r0, c0       = meta['row_origin'],  meta['col_origin']
            rs, cs       = meta['row_size'],    meta['col_size']
            filter_arr   = meta['filter_arr']

            patch      = canvas[:, r0:r0 + rs, c0:c0 + cs]
            grid_tiles = self._patch_to_grid(patch, ax0_n, ax1_n)
            all_tiles.append(grid_tiles[filter_arr.to(grid_tiles.device)])

        return torch.cat(all_tiles, dim=0)

    def canvas_to_image(self, canvas: torch.Tensor, vae_decode_fn) -> torch.Tensor:
        latent_tiles = self.canvas_to_latent_tiles(canvas)
        pixel_tiles  = vae_decode_fn(latent_tiles).squeeze(1)
        return self.pixel_sampler.reconstruct_full_map(pixel_tiles)
    
    def inverse_normalise_latent_flood_map(self, normalized_latent_flood_map):
        latent_flood_map = normalized_latent_flood_map * self.z_std[None, :, None, None] + self.z_mean[None, :, None, None]
        return latent_flood_map
    
    def get_latent_dem_tiles(self):
        latent_dem_file = "/data/westport/data/latent_floodmaps/DEM/dem_latent.pt"
        if not os.path.exists(latent_dem_file):
            logger.warning(f"No latent DEM file found at {latent_dem_file}")
            return None

        latent_dem_tensor = torch.load(latent_dem_file)
        latent_z_mean = latent_dem_tensor.mean(dim=[0, 2, 3], keepdim=True)
        latent_z_std = latent_dem_tensor.std(dim=[0, 2, 3], keepdim=True)
        latent_dem_tensor = (latent_dem_tensor - latent_z_mean) / latent_z_std
        return latent_dem_tensor

    @property
    def num_tiles(self) -> int:
        return self._num_tiles

    @property
    def canvas_shape(self):
        return (self.Hc, self.Wc)
    
    def get_domain_mask(self):
        """Static computational-domain mask at full flood-map resolution.

        1.0 = inside the computational domain, 0.0 = outside (NaN / nodata).
        Read from the RAW sample raster because process_inundation_file() zeroes
        NaN. The mask is static across events/timesteps, so it is built once and
        cached. Returned on CPU to match ``pixel_tiles_to_image`` output.
        """
        if getattr(self, "_domain_mask", None) is None:
            if self.sample_flood_map is None:
                self.sample_flood_map = self.find_sample_flood_map()
            with rio.open(self.sample_flood_map) as src:
                data = src.read(1).astype(np.float32)
                nodata_value = src.nodata
            m = ~np.isnan(data)
            if nodata_value is not None:
                m &= (data != nodata_value)
            self._domain_mask = torch.from_numpy(m.astype(np.float32))
        return self._domain_mask
    
    def find_sample_flood_map(self):
        if hasattr(self, 'sample_flood_map') and self.sample_flood_map is not None:
            return self.sample_flood_map

        test_event_name = "Base0_Buller20yrARI_v2"
        
        inundation_files = glob.glob(f"{FLOOD_MAPS_DIR}/{test_event_name}_depth_snapshots/depth_at_*.tif")
        inundation_files.sort()
        sample_file = inundation_files[0] if inundation_files else None
        if sample_file is None:
            raise ValueError(f"No inundation files found for event {test_event_name}")
        return sample_file
    
    
    
    