import torch
import logging
import rasterio
from rasterio.crs import CRS
import numpy as np
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_device():
    """
    Check if CUDA is available and return the appropriate device.
    Returns:
        torch.device: The device to be used for computations.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
    return device

def gdal_asarray(rasterfile):
    """Read raster as numpy array"""
    with rasterio.open(rasterfile) as src:
        return src.read(1)


def gdal_transform(rasterfile):
    """Get the geotransform from raster file"""
    with rasterio.open(rasterfile) as src:
        # Return affine transform directly for compatibility with rc2coords
        # but also maintain the GDAL style tuple format for backwards compatibility
        affine = src.transform
        return (affine.c, affine.a, affine.b, affine.f, affine.e, affine.d)


def gdal_writetiff(arr_data, outfile, ras_temp=None, target_transform=None, target_projection=None):
    """Write array to GeoTIFF file"""
    height, width = arr_data.shape
    # Set up metadata from template or provided values
    if ras_temp is not None:
        with rasterio.open(ras_temp) as src:
            transform = src.transform
            crs = src.crs
    elif all([target_transform, target_projection]):
        transform = rasterio.Affine(
            target_transform[1], target_transform[2], target_transform[0],
            target_transform[4], target_transform[5], target_transform[3]
        )
        crs = CRS.from_wkt(target_projection)
    elif target_transform:
        transform = rasterio.Affine(
            target_transform[1], target_transform[2], target_transform[0],
            target_transform[4], target_transform[5], target_transform[3]
        )
        crs = CRS.from_wkt(gdal_read_prj())
    else:
        raise ValueError('Please provide a raster template or target transform (/&projection)!')
    
    # Create the new raster file
    with rasterio.open(
        outfile,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=arr_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr_data, 1)

    return 0

def compute_metrics(pred, gt, id_col):
    # Match prediction and ground truth by feature ID
    df = pred[[id_col, "inundated", "expected_depth"]].merge(
        gt[[id_col, "inundated", "expected_depth"]],
        on=id_col,
        suffixes=("_pred", "_gt")
    )

    pred_mask = df["inundated_pred"].astype(bool)
    gt_mask = df["inundated_gt"].astype(bool)

    TP = np.sum(pred_mask & gt_mask)
    FP = np.sum(pred_mask & ~gt_mask)
    FN = np.sum(~pred_mask & gt_mask)
    TN = np.sum(~pred_mask & ~gt_mask)

    hit_rate = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    csi = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else np.nan

    # ---------- Depth residuals ----------
    residual = df["expected_depth_pred"] - df["expected_depth_gt"]

    me = residual.mean()
    mae = residual.abs().mean()
    rmse = np.sqrt((residual**2).mean())

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "HitRate": hit_rate,
        "CSI": csi,
        "ME": me,
        "MAE": mae,
        "RMSE": rmse,
    }
    
def append_metrics(metrics, timestamp, scenario_id, csv_path):
    row = {
        "timestamp": pd.to_datetime(timestamp),
        "scenario_id": scenario_id,
        **metrics,
    }

    df = pd.DataFrame([row])

    df.to_csv(
        csv_path,
        mode="a",
        header=not os.path.exists(csv_path),
        index=False,
    )
    
    