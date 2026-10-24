import torch
import logging
import rasterio
from rasterio.crs import CRS
import numpy as np
import pandas as pd
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_gpu_memory_mib(pid=None):
    """GPU memory this process holds, in MiB, as the NVIDIA driver reports it
    (nvidia-smi), summed over GPUs; NaN if it cannot be read.

    Read from the driver rather than torch.cuda because the flood-map surrogate
    runs in ONNX Runtime, whose allocations torch never sees. The figure counts
    everything the process holds -- ONNX Runtime, torch and the ~0.6 GB CUDA
    context -- and allocators keep what they reserved, so a reading taken after
    some work is the most the process has needed so far.
    """
    import subprocess
    pid = os.getpid() if pid is None else pid
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        used = [float(mem) for proc, mem in (line.split(",") for line in out.strip().splitlines() if line.strip())
                if int(proc) == pid]
        return float(sum(used)) if used else np.nan
    except (OSError, ValueError, subprocess.SubprocessError):
        return np.nan


class GpuMemoryMonitor:
    """Peak GPU memory of this process WHILE work runs, not after it.

    A background thread streams the process's memory from the NVIDIA driver
    every ``interval_ms`` (nvidia-smi -lms), so it sees allocations that are
    released before the work returns -- which a reading taken afterwards
    misses (low-memory mode releases each member and the decoder's pool as
    soon as they are done). The figure is everything the process holds:
    ONNX Runtime, PyTorch and the CUDA context.

        monitor = GpuMemoryMonitor().start()
        monitor.reset()
        ... work ...
        peak_mib = monitor.peak()

    A spike shorter than ``interval_ms`` can fall between samples.
    """

    def __init__(self, interval_ms=10, pid=None):
        import threading
        self.interval_ms = int(interval_ms)
        self.pid = os.getpid() if pid is None else pid
        self._lock = threading.Lock()
        self._current = self._peak = np.nan
        self._process = None

    def start(self):
        import subprocess
        import threading
        try:
            self._process = subprocess.Popen(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return self   # no nvidia-smi: peaks stay NaN
        threading.Thread(target=self._read, daemon=True).start()
        return self

    def _read(self):
        for line in self._process.stdout:
            try:
                proc, mem = line.split(",")
                if int(proc) != self.pid:
                    continue
                mem = float(mem)
            except ValueError:
                continue
            with self._lock:
                self._current = mem
                self._peak = mem if np.isnan(self._peak) else max(self._peak, mem)

    def reset(self):
        """Start a new window, from what the process holds right now."""
        with self._lock:
            self._peak = self._current

    def peak(self, settle_s=0.03):
        """Highest reading since reset(), in MiB (NaN if none). Waits a few
        sampling intervals first so the last moments of the work are in."""
        time.sleep(settle_s)
        with self._lock:
            return float(self._peak)

    def stop(self):
        if self._process is not None:
            self._process.terminate()
            self._process = None


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
    
    