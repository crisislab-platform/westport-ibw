import numpy as np
import torch
import logging
import rasterio
from modules.lib import DEM_FILE_PATH, FLOOD_MAPS_DIR
import glob
from functools import partial

from modules.utils.utils import check_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PixelSpaceSampler")

class PixelSpaceSampler():
    
    def __init__(self, sampling_dist, tile_resolution, sample_flood_map):
        self.sampling_dist = sampling_dist
        self.tile_resolution = tile_resolution
        self.sampling_distance = sampling_dist
        self.sample_flood_map = sample_flood_map
        self.flood_map_shape = self.find_flood_map_shape()
        self.domain_mask = self._load_domain_mask()
        self.device = check_device()
        logger.info("Device for PixelSpaceSampler: {}".format(self.device))
        self.dem_file = DEM_FILE_PATH
        self.number_of_tiles_per_timestep = None
        self.num_of_nodes_per_axes = round(self.tile_resolution / self.sampling_distance)  
        self.build_reshaping_func_seq(self.num_of_nodes_per_axes)
        self.prepare_dem_tiles()
        logger.info("Preparation of reshaping functions completed")
        logger.info("PixelSpaceSampler initialized successfully")
        
    def find_flood_map_shape(self):
        with rasterio.open(self.sample_flood_map) as src:
            return src.shape
        
    
    def prepare_dem_tiles(self):
        with rasterio.open(self.dem_file) as src:
            no_data_value = src.nodata
            dem_data = src.read(1).astype(np.float32)
        dem_tensor = torch.from_numpy(dem_data).float().to(self.device)
        dem_tensor = dem_tensor.unsqueeze(0).unsqueeze(0) 
        self.dem_template_tiles = self.tiles_prep_func_gpu(dem_tensor)
   
        
    def calculate_batch_size_and_number_of_tiles(self, indices_per_timestep=10):
        # Calculate the number of tiles per time step based on the sample resolution and sampling distance, and then calculate the batch size based on the desired number of batches per time step.
        num_of_tiles = self.dem_template_tiles.shape[0]
        self.number_of_tiles_per_timestep = num_of_tiles

        if num_of_tiles % indices_per_timestep != 0:
            logger.warning(f"The total number of tiles per time step ({num_of_tiles}) is not perfectly divisible by the desired number of batches per time step ({indices_per_timestep}).")
            raise ValueError("Please adjust the sampling distance or the desired number of batches per time step to ensure that the total number of tiles is divisible by the number of batches.")
        
        tiles_per_index = num_of_tiles // indices_per_timestep
        logger.info(f"Total number of tiles per time step: {num_of_tiles}, Tiles per batch index: {tiles_per_index}")
        return  tiles_per_index, num_of_tiles
        

    def build_reshaping_func_seq(self, num_of_sub_maps_per_axes):
        # Function sequence to create reshaped maps(tiles) used for training. 
        # Multiple functions are created for creating different tiles from channels with different starting points to achieve overlapping tiles.
        self.reshaping_func_seq = []
        for r_i in range(num_of_sub_maps_per_axes):
            self.reshaping_func_seq.append(self.build_tiling_func(r_i * self.sampling_distance, r_i * self.sampling_distance))
            
    def tiles_prep_func_gpu(self, x):
            with torch.no_grad():
                tiles_maps =  [tiling_func(x) for tiling_func in self.reshaping_func_seq]
                tiles_maps = torch.cat(tiles_maps, dim=0)
                return tiles_maps

    def _load_domain_mask(self):
        # Static computational-domain mask derived from the sample flood map.
        # 1.0 = inside the computational domain (valid data),
        # 0.0 = outside (NaN / nodata: sea, channels, mountains).
        # Read the RAW raster here because process_inundation_file() zeroes NaN.
        with rasterio.open(self.sample_flood_map) as src:
            data = src.read(1).astype(np.float32)
            nodata_value = src.nodata
        mask = ~np.isnan(data)
        if nodata_value is not None:
            mask &= (data != nodata_value)
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return mask

    def get_domain_mask_tiles(self):
        # Cached, tiled, DEM-filtered domain mask aligned 1:1 with flood-map tiles.
        # The mask is static, so it is tiled once and reused for every timestep.
        if getattr(self, "_domain_mask_tiles", None) is None:
            domain_mask = self._load_domain_mask()
            self._domain_mask_tiles = self.tiles_prep_func_gpu(domain_mask)
        return self._domain_mask_tiles
    
    def get_padding_height_width(self):
        # Calculate padding needed for height and width to be divisible by map_size
        orig_height, orig_width = self.flood_map_shape
        pad_height = (self.tile_resolution - (orig_height % self.tile_resolution)) % self.tile_resolution
        pad_width = (self.tile_resolution - (orig_width % self.tile_resolution)) % self.tile_resolution
        return pad_height, pad_width, orig_height, orig_width
    
    def build_tiling_func(self, cross_tile_size_r, cross_tile_size_c, training_mode=True):
        pad_height, pad_width, orig_height, orig_width = self.get_padding_height_width()
        sample_flood_map = self.process_inundation_file(self.sample_flood_map)
        sample_flood_map = torch.from_numpy(sample_flood_map)
        # Load the sample flood map and convert to tensor
        sample_flood_map = sample_flood_map.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, H, W)

        # Create padding function using partial instead of nested function
        padding_func = partial(self._padding_func, pad_height=pad_height, pad_width=pad_width,
                               cross_tile_size_r=cross_tile_size_r, cross_tile_size_c=cross_tile_size_c)
        
        new_shape = padding_func(sample_flood_map).shape
        ax0_n = new_shape[2] // self.tile_resolution
        ax1_n = new_shape[3] // self.tile_resolution
        logger.info(f"New shape after padding: {new_shape}, maps: {ax0_n}x{ax1_n}")
        
        # Create a reshaping function to get the maps(tiles) for this specific starting point
        reshaping_func = partial(self.reshaping_func_def, padding_func, ax0_n=ax0_n, ax1_n=ax1_n)
        
        domain_mask_tiles = reshaping_func(self.domain_mask)
        #if any tile has 0% valid data, exclude it from training by creating a filter array
        valid_data_per_tile = domain_mask_tiles.view(domain_mask_tiles.shape[0], -1).mean(dim=1)
        tile_filter_arr = valid_data_per_tile > 0

        if training_mode:
            tiling_func = partial(self._apply_tile_filter, reshaping_func, tile_filter_arr)
            logger.info(f"Created tiling function successfully")
            return tiling_func
        else:
            return reshaping_func, tile_filter_arr, ax0_n, ax1_n, (cross_tile_size_r, cross_tile_size_c)

    def reshaping_func_def(self, slicing_func, x, ax0_n=None, ax1_n=None):
        # First reshape the map to the desired size (a multiple of the map size)
        new_map_x = slicing_func(x)[:, :, :self.tile_resolution * ax0_n, :self.tile_resolution * ax1_n]
        del x  # Free memory after slicing
        new_map_x = new_map_x.to(self.device)
        # Shape of the new map is (1, 1, sample_resolution * ax0_n, sample_resolution * ax1_n). Remove batch and channel dims.
        new_map_x = new_map_x.squeeze(0).squeeze(0)  # (1, 1, H, W) -> (H, W)
        new_map_x = new_map_x.reshape((-1, ax1_n, self.tile_resolution)) 
        new_map_x = new_map_x.swapaxes(0, 1)
        new_map_x = new_map_x.reshape((ax1_n, -1, self.tile_resolution, self.tile_resolution))
        new_map_x = new_map_x.swapaxes(0, 1)
        new_map_x = new_map_x.reshape((-1, self.tile_resolution, self.tile_resolution))
        return new_map_x.detach().cpu()
    
    def _apply_tile_filter(self, reshaping_func, tile_filter_arr, x):
        """Apply tile filter - pickle-compatible helper for tiling_func"""
        return reshaping_func(x)[tile_filter_arr, :, :]
    
    def _back_transform(self, ax1_n, x):
        """Back transformation helper - pickle-compatible alternative to lambda"""
        x = x.to(self.device)
        x = x.reshape(-1, ax1_n, self.tile_resolution, self.tile_resolution)
        x = x.swapaxes(0, 1)
        x = x.reshape(ax1_n, -1, self.tile_resolution)
        x = x.swapaxes(0, 1)
        x = x.reshape(-1, ax1_n * self.tile_resolution)
        return x.detach().cpu()
    
    def _padding_func(self, x, pad_height=None, pad_width=None, cross_tile_size_r=None, cross_tile_size_c=None):
        """Padding helper - pickle-compatible alternative to nested function"""
        # x is expected to be of shape (1, 1, H, W)
        # Use constant padding (0) instead of reflect to avoid boundary artifacts that misalign during reconstruction
        x = x.to(self.device)
        padded = torch.nn.functional.pad(x, (0, pad_width, 0, pad_height), mode='constant', value=0)
        sliced = padded[:, :, cross_tile_size_r:, cross_tile_size_c:]  # Slice to create the tiled version with desired offset
        del padded  # Free memory after slicing
        del x  # Free memory after padding
        return sliced.detach().cpu()

    def reconstruction_init(self):
        logger.info("Initializing reconstruction")
        self.output_maps_list = []
        self.return_template_list = []
        self.tile_filter_list = []
        self.layers_seperation_idxs = []
        self.map_origins = []
        self.layer_sizes = []
        self.back_transformation_functions = []
        start_idx = 0  
        
        with torch.no_grad():
            for r_i in range(self.num_of_nodes_per_axes):
                c_i = r_i
                reshaping_func, filter_arr, ax0_n, ax1_n, map_origin = \
                    self.build_tiling_func(r_i * self.sampling_distance, 
                                           c_i * self.sampling_distance, training_mode=False)

                self.layers_seperation_idxs.append((start_idx, start_idx + torch.sum(filter_arr).item()))
                start_idx = self.layers_seperation_idxs[-1][1]
                pad_height, pad_width, original_height, original_width = self.get_padding_height_width()
                output_map = torch.zeros((original_height + pad_height, original_width + pad_width), dtype=torch.float32).to(self.device)
                self.output_maps_list.append(output_map)
                output_map = output_map.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, H, W)
                self.return_template_list.append(reshaping_func(output_map).to(self.device))
                self.tile_filter_list.append(filter_arr.bool().to(self.device))
                # Store origin in the UNPADDED coordinate space for proper reconstruction placement
                # This ensures that overlapping tiles align correctly regardless of padding artifacts
                self.map_origins.append(map_origin)
                self.layer_sizes.append((ax0_n*self.tile_resolution, ax1_n*self.tile_resolution))
                self.back_transformation_functions.append(self.build_back_func_lambda(ax1_n))
                
            # Check if the reconstruction output covers the entire map
            temp_tensor = torch.ones(self.dem_template_tiles.shape).to(self.device)
            self.final_map = self.reconstruct_full_map_return_and_sum(temp_tensor).to(self.device)
            self.final_map[self.final_map == 0] = 1.0
            del temp_tensor
            logger.info("Reconstruction initialization completed successfully")
        return 0

    def reconstruct_full_map_return_and_sum(self, x): 
        for lyr_i in range(len(self.layers_seperation_idxs)):
            start_idx, end_idx = self.layers_seperation_idxs[lyr_i]
            self.return_template_list[lyr_i][self.tile_filter_list[lyr_i]] = x[start_idx:end_idx].squeeze(1)
            row_size, col_size = self.layer_sizes[lyr_i]
            row_origin, col_origin = self.map_origins[lyr_i]
            output_map_for_layer = self.back_transformation_functions[lyr_i](self.return_template_list[lyr_i]) 
            self.output_maps_list[lyr_i][row_origin:row_origin + row_size, col_origin:col_origin + col_size] = \
                output_map_for_layer
        output_sum = torch.sum(torch.stack(self.output_maps_list), dim=0)
        
        #clear the output maps for the next reconstruction
        for i in range(len(self.output_maps_list)):
            self.output_maps_list[i] = torch.zeros_like(self.output_maps_list[i]).to(self.device)
        return output_sum
    
    def reconstruct_full_map(self, x):
        with torch.no_grad():
            output_final = torch.div(self.reconstruct_full_map_return_and_sum(x), self.final_map)
            pad_height, pad_width, original_height, original_width = self.get_padding_height_width()
            output_final = output_final[:original_height, :original_width]
        return output_final
    
    def build_back_func_lambda(self, ax1_n):
        return partial(self._back_transform, ax1_n)
    
    def process_inundation_file(self, file_path):
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                nodata_value = src.nodata
                
                # Handle NoData values by replacing them with 0 (no flooding)
                # This is appropriate for flood depth data where NoData typically means
                # areas outside the study domain or dry areas
                
                # Replace negative, nan and nodata values with 0
                data = np.where(np.isnan(data), 0, data)
                data = np.where(data == nodata_value, 0, data)
                
                return data
        except Exception as e:
            logger.error(f"Error processing inundation file {file_path}: {e}")
            return None  