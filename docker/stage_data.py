"""Assemble the data bundle the container image is built from.

Run this ON THE WORKSTATION, where /data/westport is mounted. It writes
``docker/data-bundle/`` -- a miniature of the /data/westport tree holding only
what one forecast reads, which the Dockerfile then COPYs into the image. The
full tree is 711 GB; the bundle is a few hundred MB, and almost all of that is
the ensemble weights.

Two kinds of file go in:

REAL, because they set the size of the computation and are small anyway --
the ONNX ensemble and decoder (the actual cost), the exposure vectors and the
precomputed rings (how many buildings, roads and properties get reduced over),
and the boundary-condition series (the forecast's input signal).

SYNTHETIC, because they are large and the timings do not depend on their
values -- the sample flood map, the latent statistics and the latent DEM. What
those files contribute is geometry: raster shape, transform, CRS, and which
cells are inside the computational domain. Those are copied exactly from the
real files, so every array the run allocates has the shape it would have on
real data. Only the numbers inside are made up, and no arithmetic in the
forecast branches on them.

The synthetic sample flood map is what the code calls the "sample" raster: it
is opened for its georeferencing and its NaN footprint, never for its depths.
The domain footprint IS copied from the real raster (as a bitmask, not as
depths), because the fraction of the grid inside the domain decides how many
tiles the sampler keeps and how many buildings the impact module scores -- get
that wrong and the timings stop meaning anything.
"""

import argparse
import os
import shutil

import numpy as np
import rasterio
import torch

SRC_DATA_ROOT = os.environ.get("WESTPORT_DATA_ROOT", "/data/westport")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The sample raster the code reaches for; see LatentCanvas.find_sample_flood_map.
SAMPLE_EVENT = "Base0_Buller20yrARI_v2"
SAMPLE_RASTER_NAME = "depth_at_2000-01-02_00-00-00.tif"

# Copied verbatim. Paths are relative to the data root on both sides.
REAL_FILES = [
    "out/model_checkpoints/AUTO_ENCODER/AUTO_ENCODER.decoder.onnx",
    # The fitted flow / sea-level scalers BC_Module normalises its input with.
    # Real, and a kilobyte: synthesising them would put the boundary conditions
    # outside the range the ensemble was trained on.
    "out/model_checkpoints/LATENT_FLOW_MATCHING_V1/scalers/bc_scalers.joblib",
    "data/dem/Topo.tif",
    "data/exposure/buildings.gpkg",
    "data/exposure/road_network.gpkg",
    "data/exposure/road_centerlines.gpkg",
    "data/exposure/property_boundaries.gpkg",
    "data/exposure/building_property.csv",
    "data/exposure/building_nearest_road_parcel.csv",
    "data/Impact-based_data/obs_placard.gpkg",
]
REAL_DIRS = [
    "data/flood-mapping-with-ml-main-DATA/DATA/boundary_conditions",
]

# Repo-side artefacts, staged separately because they live under the checkout
# rather than the data root: the ring cache (rebuilding it per run is exactly
# what the cache exists to avoid) and the building link table.
REPO_FILES = [
    "out/building_links.csv",
]
REPO_DIRS = [
    "out/rings",
]


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return os.path.getsize(dst)


def _human(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


def stage_members(bundle, num_members):
    """Copy the first ``num_members`` ensemble members in canonical order.

    Canonical order is the numeric index in the filename, the same order
    LatentFlowEnsemble._discover_members caps on, so a bundle built with M=3
    holds exactly the members an M=3 run would have loaded out of the full ten.
    """
    rel_dir = "out/model_checkpoints/LATENT_FLOW_MATCHING_V1/ensemble"
    src_dir = os.path.join(SRC_DATA_ROOT, rel_dir)
    members = sorted(
        name for name in os.listdir(src_dir)
        if name.startswith("member_") and name.endswith(".onnx")
    )
    if not members:
        raise FileNotFoundError(f"no ensemble members under {src_dir}")
    members = members[:num_members]

    total = 0
    for name in members:
        total += _copy(os.path.join(src_dir, name),
                       os.path.join(bundle, rel_dir, name))
    print(f"  ensemble        {len(members)} members  {_human(total)}")
    return len(members)


def stage_sample_raster(bundle):
    """Write a synthetic sample raster with the real grid and real domain mask.

    Depths are drawn from a smooth field rather than noise so anything that
    looks at the map (the extent classifier, the ring reductions) sees
    contiguous wet regions of a plausible size instead of salt-and-pepper.
    Out-of-domain cells are NaN, exactly as in the real raster -- that is the
    part the code actually reads.
    """
    rel = f"data/flood_maps/{SAMPLE_EVENT}_depth_snapshots/{SAMPLE_RASTER_NAME}"
    src = os.path.join(SRC_DATA_ROOT, rel)
    with rasterio.open(src) as handle:
        profile = handle.profile
        real = handle.read(1)
        nodata = handle.nodata

    inside = np.isfinite(real)
    if nodata is not None and not np.isnan(nodata):
        inside &= real != nodata
    del real  # the depths themselves are deliberately not carried over

    height, width = inside.shape
    rows = np.linspace(0, 6 * np.pi, height, dtype=np.float32)[:, None]
    cols = np.linspace(0, 6 * np.pi, width, dtype=np.float32)[None, :]
    field = 1.2 * (np.sin(rows) * np.cos(cols)) + 1.2
    depths = np.where(inside, np.clip(field, 0.0, None), np.nan).astype("float32")

    # Tiled + deflate rather than the source's striped layout: it is the same
    # grid either way, and it takes the bundle from 23 MB to under 1 MB.
    profile.update(dtype="float32", count=1, nodata=np.nan,
                   compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    dst = os.path.join(bundle, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with rasterio.open(dst, "w", **profile) as out:
        out.write(depths, 1)

    print(f"  sample raster   {width}x{height}, {inside.mean():.1%} in domain, "
          f"{_human(os.path.getsize(dst))} (synthetic depths, real geometry)")
    return width, height


def stage_latents(bundle):
    """Synthesise the latent statistics and the latent DEM at the real shapes.

    Shapes are read from the real files rather than hardcoded, so a re-export
    that changes the latent channel count cannot silently produce a bundle the
    decoder rejects. The stats are saved on CPU: the real ones were saved from
    CUDA tensors, which is a load-time error on a machine with no GPU.
    """
    stats_rel = "data/latent_floodmaps/latent_flood_map_stats.pt"
    dem_rel = "data/latent_floodmaps/DEM/dem_latent.pt"

    real_stats = torch.load(os.path.join(SRC_DATA_ROOT, stats_rel),
                            map_location="cpu", weights_only=False)
    channels = real_stats["z_mean"].shape[0]
    generator = torch.Generator().manual_seed(0)
    stats = {
        "z_mean": torch.zeros(channels),
        # Unit scale: the canvas multiplies by z_std and adds z_mean, so 1/0
        # keeps decoded latents in the range the decoder was trained on without
        # reusing the fitted values.
        "z_std": torch.ones(channels),
    }
    dst_stats = os.path.join(bundle, stats_rel)
    os.makedirs(os.path.dirname(dst_stats), exist_ok=True)
    torch.save(stats, dst_stats)

    real_dem = torch.load(os.path.join(SRC_DATA_ROOT, dem_rel),
                          map_location="cpu", weights_only=False)
    dem = torch.randn(real_dem.shape, generator=generator)
    dst_dem = os.path.join(bundle, dem_rel)
    os.makedirs(os.path.dirname(dst_dem), exist_ok=True)
    torch.save(dem, dst_dem)

    print(f"  latents         stats C={channels}, dem {tuple(real_dem.shape)} "
          f"(synthetic values, real shapes)")


def stage_real(bundle):
    total = 0
    for rel in REAL_FILES:
        src = os.path.join(SRC_DATA_ROOT, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        total += _copy(src, os.path.join(bundle, rel))
    for rel in REAL_DIRS:
        src = os.path.join(SRC_DATA_ROOT, rel)
        dst = os.path.join(bundle, rel)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        total += sum(os.path.getsize(os.path.join(root, name))
                     for root, _, names in os.walk(dst) for name in names)
    print(f"  exposure + BC   {_human(total)}")


def stage_repo(bundle):
    """Stage the repo-side artefacts under ``repo/`` inside the bundle.

    They are kept apart from the data tree because they are addressed by
    WESTPORT_REPO_ROOT rather than WESTPORT_DATA_ROOT; the image drops them
    into the checkout's own out/ directory.
    """
    total = 0
    for rel in REPO_FILES:
        src = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        total += _copy(src, os.path.join(bundle, "repo", rel))
    for rel in REPO_DIRS:
        src = os.path.join(REPO_ROOT, rel)
        dst = os.path.join(bundle, "repo", rel)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        total += sum(os.path.getsize(os.path.join(root, name))
                     for root, _, names in os.walk(dst) for name in names)
    print(f"  rings + links   {_human(total)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "docker", "data-bundle"),
                        help="where to write the bundle (the Docker build context)")
    parser.add_argument("--num-members", type=int, default=10,
                        help="ensemble members to stage; caps the M a run can ask for")
    args = parser.parse_args()

    bundle = os.path.abspath(args.out)
    if os.path.exists(bundle):
        shutil.rmtree(bundle)
    os.makedirs(bundle)

    print(f"staging from {SRC_DATA_ROOT} -> {bundle}")
    stage_members(bundle, args.num_members)
    stage_real(bundle)
    stage_sample_raster(bundle)
    stage_latents(bundle)
    stage_repo(bundle)
    os.makedirs(os.path.join(bundle, "out", "forecasts"), exist_ok=True)

    total = sum(os.path.getsize(os.path.join(root, name))
                for root, _, names in os.walk(bundle) for name in names)
    print(f"bundle total    {_human(total)}")


if __name__ == "__main__":
    main()
