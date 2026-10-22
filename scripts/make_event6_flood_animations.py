#!/usr/bin/env python3
"""Generate and animate the real event-6 flood-map ensemble.

The script has two deliberately separate stages. ``simulate`` runs the hazard
model and writes one compressed archive per valid time, so an interrupted run
can be resumed without repeating completed hours. ``animate`` reads those
archives and the reference GeoTIFFs listed in ``out/event_6.csv`` and produces
presentation-ready animations with a cursor moving over the boundary
conditions.

Examples
--------
Run the ten-model x ten-sample ensemble (100 maps per hour), then animate it::

    python scripts/make_event6_flood_animations.py all

Only render again after changing presentation options::

    python scripts/make_event6_flood_animations.py animate --format mp4

The reference animation shows completed solver maps through event time.

``--pacing equal`` (the default for the combined layout) runs the two models in
phase: one frame per event timestep in each panel, both panels on the same valid
time, both finishing together. It compares the maps, and makes no runtime claim.
``--pacing staggered`` restores the older runtime comparison, in which the solver
reaches only ``--reference-steps-by-flow-finish`` timesteps before the AI model
has finished and then continues alone. That version is time-compressed, not a
benchmark; its label states this explicitly. Change ``--reference-slowdown`` to
the measured ratio if a defensible solver-to-surrogate benchmark is available.

Every map is drawn over a Westport basemap fetched once from a tile service and
cached in ``<output-dir>/basemap_cache``, so the flood depth is read against the
real coastline, river and street layout. ``--basemap`` picks the layer
(``imagery``, ``streets``, ``topo``, ``light`` or ``none``) and ``--depth-alpha``
controls how much of it shows through the flooded area. An unreachable tile
service downgrades to no basemap rather than losing the render.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageSequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import rasterio


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.data_module.bc_module import BC_Module
from modules.hazard_module.model_client import FlowMatchingModelClient


LOG = logging.getLogger("event6-animation")
DEFAULT_CSV = REPO_ROOT / "out" / "event_6.csv"
DEFAULT_OUTPUT = REPO_ROOT / "out" / "event_6_presentation"
DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "flood_depth", ["#e8f5ff", "#70c1ff", "#1677c8", "#08306b"]
)
# The two models as the audience should read them, in every title and legend.
AI_LABEL = "Our AI model"
PHYSICS_LABEL = "Physics-based model"
# Tile layers offered under the flood depth, as xyzservices provider paths.
# All four serve real tiles without an API key. OpenStreetMap.Mapnik and the
# CartoDB layers do not: they answer with "Access blocked" and "API key
# required" placeholder images under HTTP 200, which look like a successful
# fetch and get cached. Verify any layer added here actually renders Westport.
BASEMAP_SOURCES = {
    "imagery": ("Esri", "WorldImagery"),
    "streets": ("Esri", "WorldStreetMap"),
    "topo": ("Esri", "WorldTopoMap"),
    "light": ("Esri", "WorldGrayCanvas"),
}
# Web-Mercator metres per tile pixel at zoom 0, used to pick a tile zoom.
MERCATOR_PIXEL_SPAN = 2 * 20037508.342789244 / 256
MAX_BASEMAP_ZOOM = 16
# Some providers ship a credit line longer than a map panel is wide, which
# otherwise runs off the panel and into its neighbour.
ATTRIBUTION_LIMIT = 92


class MapView(NamedTuple):
    """The fixed geographic view every frame is drawn into."""

    extent: tuple[float, float, float, float]
    image: np.ndarray | None
    image_extent: tuple[float, float, float, float] | None
    attribution: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("simulate", "animate", "all"))
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--members", type=int, default=10,
                        help="Deep-ensemble models M (default: 10)")
    parser.add_argument("--samples-per-member", type=int, default=10,
                        help="Noise samples K per model (default: 10)")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--start", help="First timestamp to include")
    parser.add_argument("--end", help="Last timestamp to include")
    parser.add_argument("--step-hours", type=int, default=1,
                        help="Animate every Nth hour of the event instead of every "
                             "hour (default: 1). Applies to animate only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--format", choices=("gif", "mp4", "both"),
                        default="both")
    parser.add_argument("--layout", choices=("separate", "flow", "combined", "both"),
                        default="combined",
                        help="Render one figure holding both models (combined), a video "
                             "per model (separate), the AI model alone (flow), or both")
    parser.add_argument("--pacing", choices=("equal", "staggered"),
                        default="equal",
                        help="Combined layout: run both models in phase over the same "
                             "timesteps (equal), or let the solver lag behind (staggered)")
    parser.add_argument("--seconds-per-step", type=float, default=1.5,
                        help="Seconds each timestep is held in the equally phased "
                             "combined animation (default: 1.5)")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Flow animation frames per second")
    parser.add_argument("--reference-slowdown", type=float, default=100.0,
                        help="Runtime ratio annotated on the time-compressed reference animation")
    parser.add_argument("--reference-seconds-per-step", type=float, default=8.0,
                        help="Seconds each physics snapshot is held after flow matching finishes (default: 8)")
    parser.add_argument("--reference-steps-by-flow-finish",
                        "--reference-steps-during-flow",
                        dest="reference_steps_by_flow_finish", type=int, default=2,
                        help="Physics timesteps completed when flow matching finishes; physics then continues to completion (default: 2)")
    parser.add_argument("--samples-shown", type=int, default=100,
                        help="Generated members shown at each event timestep in the separate flow animation")
    parser.add_argument("--combined-flow-maps", type=int, default=20,
                        help="Snapshots shown per model in the combined animation (default: 20)")
    parser.add_argument("--basemap",
                        choices=("imagery", "topo", "streets", "light", "none"),
                        default="imagery",
                        help="Tile layer drawn under the flood depth (default: imagery)")
    parser.add_argument("--basemap-zoom", type=int, default=None,
                        help="Tile zoom level; the default matches --render-width")
    parser.add_argument("--basemap-provider", default=None,
                        help="Dotted xyzservices provider path (e.g. Esri.NatGeoWorldMap) "
                             "used instead of --basemap")
    parser.add_argument("--depth-alpha", type=float, default=0.82,
                        help="Opacity of the flood depth over the basemap (default: 0.82)")
    parser.add_argument("--gif-colors", type=int, default=None,
                        help="GIF palette size; fewer colours produce smaller files "
                             "(default: 128 with a basemap, 64 without)")
    parser.add_argument("--render-width", type=int, default=900,
                        help="Downsample map to at most this many columns")
    parser.add_argument("--max-depth", type=float, default=3.0,
                        help="Fixed colour-scale maximum in metres")
    return parser.parse_args()


def read_event(path: Path, start: str | None, end: str | None,
               step_hours: int = 1) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Date_Time", "River_flow", "Sea_level", "flood_map_file"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["Date_Time"] = pd.to_datetime(frame["Date_Time"])
    frame = frame.sort_values("Date_Time").drop_duplicates("Date_Time")
    frame = frame.loc[frame["Date_Time"].dt.minute.eq(0)].copy()
    if start:
        frame = frame.loc[frame["Date_Time"] >= pd.Timestamp(start)]
    if end:
        frame = frame.loc[frame["Date_Time"] <= pd.Timestamp(end)]
    if frame.empty:
        raise ValueError("No event rows remain after applying the time window")
    if step_hours > 1:
        # Counted from the first retained hour, so the window start is always
        # shown whatever the event happens to begin on.
        frame = frame.iloc[::step_hours]
    missing_maps = [p for p in frame["flood_map_file"] if not Path(p).is_file()]
    if missing_maps:
        raise FileNotFoundError(f"Reference map does not exist: {missing_maps[0]}")
    return frame.reset_index(drop=True)


def make_conditions(frame: pd.DataFrame) -> OrderedDict[pd.Timestamp, np.ndarray]:
    """Apply the same lagging and saved scalers as BC_Module to out/event_6.csv."""
    lag = 7
    work = frame[["Date_Time", "River_flow", "Sea_level"]].copy()
    for column in ("River_flow", "Sea_level"):
        work[column] = work[column].astype(np.float32)
        for i in range(1, lag + 1):
            work[f"{column}-{i}"] = work[column].shift(i)
    lagged = [f"{name}-{i}" for name in ("River_flow", "Sea_level")
              for i in range(1, lag + 1)]
    work[lagged] = work[lagged].interpolate(limit_direction="both").bfill().ffill()
    flow_scaler, sea_scaler = BC_Module.load_scalers()
    flow_cols = ["River_flow"] + [f"River_flow-{i}" for i in range(1, lag + 1)]
    sea_cols = ["Sea_level"] + [f"Sea_level-{i}" for i in range(1, lag + 1)]
    flow = flow_scaler.transform(work[flow_cols].values).astype(np.float32)
    sea = sea_scaler.transform(work[sea_cols].values).astype(np.float32)
    cond = np.hstack((flow, sea)).reshape(-1, 2, lag + 1)
    return OrderedDict((pd.Timestamp(ts), row) for ts, row in
                       zip(work["Date_Time"], cond))


def stamp(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%dT%H%M%S")


def simulate(args: argparse.Namespace, event: pd.DataFrame) -> None:
    ensemble_dir = args.output_dir / "flow_maps"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    conditions = make_conditions(event)
    expected = args.members * args.samples_per_member

    pending = OrderedDict()
    for ts, cond in conditions.items():
        path = ensemble_dir / f"flow_{stamp(ts)}.npz"
        if args.overwrite or not path.exists():
            pending[ts] = cond
    if not pending:
        LOG.info("All %d timestep archives already exist", len(conditions))
        return

    LOG.info("Loading hazard model: M=%d, K=%d (%d maps/timestep)",
             args.members, args.samples_per_member, expected)
    model = FlowMatchingModelClient(
        num_members=args.members,
        samples_per_member=args.samples_per_member,
        seed=args.seed,
    )
    timings = []
    # A one-step call keeps peak memory independent of event length and permits
    # true resume. It also makes each saved duration an honest per-hour timing.
    for index, (ts, cond) in enumerate(pending.items(), 1):
        target = ensemble_dir / f"flow_{stamp(ts)}.npz"
        started = time.perf_counter()
        generated_ts, tensor = next(model.iter_hazard_forecast(
            OrderedDict([(ts, cond)])))
        maps = tensor.numpy().astype(np.float32, copy=False)
        elapsed = time.perf_counter() - started
        if maps.shape[0] != expected:
            raise RuntimeError(f"Expected {expected} maps, received {maps.shape[0]}")
        # savez_compressed preserves float32 values exactly and works without
        # adding a zarr dependency. Write through /tmp then replace atomically.
        temporary = target.with_suffix(".npz.partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, maps=maps)
        os.replace(temporary, target)
        timings.append({"Date_Time": generated_ts, "seconds": elapsed,
                        "num_maps": maps.shape[0], "archive": str(target)})
        LOG.info("[%d/%d] %s: %d maps in %.2f s -> %s",
                 index, len(pending), ts, maps.shape[0], elapsed, target)
        del tensor, maps

    timing_path = args.output_dir / "generation_timings.csv"
    pd.DataFrame(timings).to_csv(
        timing_path, mode="a", header=not timing_path.exists(), index=False)
    write_manifest(args, event)


def write_manifest(args: argparse.Namespace, event: pd.DataFrame) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(event.iloc[0]["flood_map_file"]) as src:
        raster = {"height": src.height, "width": src.width,
                  "crs": str(src.crs), "transform": list(src.transform)[:6]}
    payload = {
        "scenario_id": 6,
        "event": "Base5_Buller50yrARI_RCP8pt5",
        "boundary_conditions": str(args.csv.resolve()),
        "num_members": args.members,
        "samples_per_member": args.samples_per_member,
        "ensemble_size": args.members * args.samples_per_member,
        "seed": args.seed,
        "first_timestep": str(event["Date_Time"].min()),
        "last_timestep": str(event["Date_Time"].max()),
        "num_timesteps": len(event),
        "raster": raster,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def domain_crop(reference_path: str) -> tuple[slice, slice]:
    with rasterio.open(reference_path) as src:
        valid = src.read_masks(1) > 0
        data = src.read(1, masked=True)
        valid &= ~np.ma.getmaskarray(data)
    rows, cols = np.where(valid)
    if not len(rows):
        return slice(None), slice(None)
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


def render_stride(crop: tuple[slice, slice], max_width: int) -> int:
    width = crop[1].stop - crop[1].start
    return max(1, int(np.ceil(width / max_width)))


def prepare_depth(array: np.ndarray, crop: tuple[slice, slice], stride: int):
    shown = np.asarray(array[crop[0], crop[1]][::stride, ::stride], dtype=float)
    return np.ma.masked_where(~np.isfinite(shown) | (shown <= 0.01), shown)


def map_extent(reference_path: str, crop: tuple[slice, slice], stride: int):
    """Map coordinates spanned by the array ``prepare_depth`` returns.

    Each displayed cell stands for ``stride`` source pixels, so the extent runs
    to the far edge of the last displayed cell rather than to the crop bound.
    """
    with rasterio.open(reference_path) as src:
        transform, crs = src.transform, src.crs
        height, width = src.height, src.width
    row0 = crop[0].start or 0
    col0 = crop[1].start or 0
    row1 = height if crop[0].stop is None else crop[0].stop
    col1 = width if crop[1].stop is None else crop[1].stop
    rows = int(np.ceil((row1 - row0) / stride))
    cols = int(np.ceil((col1 - col0) / stride))
    left, top = transform * (col0, row0)
    right, bottom = transform * (col0 + cols * stride, row0 + rows * stride)
    return (left, right, bottom, top), crs


def basemap_zoom(west: float, east: float, target_width: int) -> int:
    """Lowest tile zoom whose pixels are finer than the rendered map."""
    span = east - west
    for zoom in range(1, MAX_BASEMAP_ZOOM):
        if span / (MERCATOR_PIXEL_SPAN / 2 ** zoom) >= target_width:
            return zoom
    return MAX_BASEMAP_ZOOM


def load_basemap(kind: str, extent, crs, cache_dir: Path, zoom: int | None,
                 target_width: int, provider: str | None = None):
    """Fetch, reproject and cache the tiles that sit under the flood depth.

    The view never moves, so one warped image serves every frame of every
    animation. It is cached on disk because re-rendering is the common case,
    and a tile service that cannot be reached downgrades to no basemap rather
    than losing a long render.
    """
    if kind == "none" and not provider:
        return None, None, None
    left, right, bottom, top = extent
    parts = provider.split(".") if provider else BASEMAP_SOURCES[kind]
    label = provider or kind
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (f"{label.replace('.', '-')}_z{zoom or 'auto'}_"
                         f"{left:.0f}_{right:.0f}_{bottom:.0f}_{top:.0f}.npz")
    if cache.exists():
        with np.load(cache) as archive:
            LOG.info("Using cached basemap %s", cache.name)
            return (archive["image"], tuple(archive["extent"].tolist()),
                    archive["attribution"].item())
    try:
        import contextily
        from pyproj import Transformer
    except ImportError as exc:
        LOG.warning("Rendering without a basemap: %s", exc)
        return None, None, None

    source = contextily.providers
    try:
        for part in parts:
            source = source[part]
    except KeyError:
        LOG.warning("Rendering without a basemap: unknown provider %r", provider)
        return None, None, None
    to_mercator = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
    xs, ys = to_mercator.transform([left, right, left, right],
                                   [bottom, bottom, top, top])
    west, east, south, north = min(xs), max(xs), min(ys), max(ys)
    level = zoom if zoom is not None else basemap_zoom(west, east, target_width)
    LOG.info("Fetching %s basemap tiles at zoom %d", label, level)
    try:
        tiles, tile_extent = contextily.bounds2img(
            west, south, east, north, zoom=level, source=source, ll=False)
        image, image_extent = contextily.warp_tiles(
            tiles, tile_extent, t_crs=str(crs))
    except Exception as exc:  # unreachable tile service, throttling, bad tile
        LOG.warning("Rendering without a basemap (%s: %s)",
                    type(exc).__name__, exc)
        return None, None, None
    attribution = source.get("attribution", "")
    np.savez_compressed(cache, image=image,
                        extent=np.asarray(image_extent, dtype=float),
                        attribution=np.asarray(attribution))
    return image, tuple(image_extent), attribution


def setup_map_axes(ax, view: MapView) -> None:
    """Draw the basemap and pin the axes to the flood domain.

    Limits are set once, with autoscaling off, so the depth image added
    afterwards cannot pull the view out to the larger basemap extent.
    """
    if view.image is not None:
        ax.imshow(view.image, extent=view.image_extent, origin="upper",
                  interpolation="bilinear", zorder=0)
        if view.attribution:
            credit = view.attribution
            if len(credit) > ATTRIBUTION_LIMIT:
                credit = credit[:ATTRIBUTION_LIMIT - 1].rstrip(" ,;") + "…"
            ax.text(0.995, 0.008, credit, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=4.5, color="white",
                    zorder=5, bbox=dict(boxstyle="square,pad=0.25",
                                        facecolor="#000000a0", edgecolor="none"))
    ax.set_xlim(view.extent[0], view.extent[1])
    ax.set_ylim(view.extent[2], view.extent[3])
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.set_axis_off()


def depth_image(ax, initial, args, view: MapView):
    """The flood-depth layer, georeferenced and drawn over the basemap."""
    return ax.imshow(initial, cmap=DEPTH_CMAP, vmin=0, vmax=args.max_depth,
                     extent=view.extent, origin="upper", zorder=2,
                     alpha=args.depth_alpha, interpolation="nearest")


def add_bc_axes(fig, event: pd.DataFrame):
    ax = fig.add_axes([0.09, 0.08, 0.82, 0.22])
    sea_ax = ax.twinx()
    dates = event["Date_Time"]
    ax.plot(dates, event["River_flow"], color="#254b87", lw=2.4,
            label="River flow")
    sea_ax.plot(dates, event["Sea_level"], color="#e07a24", lw=1.8,
                label="Sea level")
    ax.set_ylabel("River flow (m³/s)", color="#254b87")
    sea_ax.set_ylabel("Sea level (m)", color="#e07a24")
    date_locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(date_locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y\n%H:%M"))
    ax.set_xlabel("Date and time")
    ax.tick_params(axis="x", labelsize=9, pad=4)
    ax.grid(alpha=0.2)
    cursor = ax.axvline(dates.iloc[0], color="#d62728", lw=2.5, zorder=10)
    return ax, cursor


def writer_for(fmt: str, fps: float):
    if fmt == "gif":
        return PillowWriter(fps=min(fps, 8))
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("ffmpeg is required for MP4 output; use --format gif")
    return FFMpegWriter(fps=fps, codec="h264", bitrate=3500,
                        extra_args=["-pix_fmt", "yuv420p"])


def optimize_gif(path: Path, colors: int) -> None:
    """Reduce GIF palette and enable Pillow frame-delta optimization."""
    temporary = path.with_name(f"{path.stem}.optimized.gif")
    with Image.open(path) as source:
        default_duration = source.info.get("duration", 125)
        loop = source.info.get("loop", 0)
        frames = []
        durations = []
        for frame in ImageSequence.Iterator(source):
            # Pillow may already have coalesced identical frames. Preserve each
            # resulting frame duration instead of replacing long holds with the
            # default single-frame duration.
            durations.append(frame.info.get("duration", default_duration))
            frames.append(frame.convert("RGB").convert(
                "P", palette=Image.Palette.ADAPTIVE, colors=colors))
    frames[0].save(temporary, save_all=True, append_images=frames[1:],
                   duration=durations, loop=loop, optimize=True, disposal=1)
    os.replace(temporary, path)


def save_flow_animation(args, event, fmt: str, crop, stride, view: MapView) -> None:
    fps = args.fps
    output = args.output_dir / f"event6_flow_matching.{fmt}"
    fig = plt.figure(figsize=(10, 8), dpi=120)
    map_ax = fig.add_axes([0.08, 0.35, 0.78, 0.57])
    bc_ax, flow_cursor = add_bc_axes(fig, event)
    flow_cursor.set_color("#16a085")
    flow_cursor.set_label(AI_LABEL)
    reference_cursor = bc_ax.axvline(
        event.iloc[0].Date_Time, color="#9b59b6", lw=2.5,
        linestyle="--", zorder=10, label=PHYSICS_LABEL)
    bc_ax.legend(handles=[flow_cursor, reference_cursor], loc="upper left")
    setup_map_axes(map_ax, view)
    first_archive = args.output_dir / "flow_maps" / f"flow_{stamp(event.iloc[0].Date_Time)}.npz"
    with np.load(first_archive) as archive:
        initial = prepare_depth(archive["maps"][0], crop, stride)
    image = depth_image(map_ax, initial, args, view)
    color_ax = fig.add_axes([0.88, 0.38, 0.025, 0.50])
    fig.colorbar(image, cax=color_ax, label="Flood depth (m)")
    title = map_ax.set_title("", fontsize=15, loc="left")
    samples = max(1, min(args.samples_shown,
                         args.members * args.samples_per_member))
    writer = writer_for(fmt, fps)
    with writer.saving(fig, str(output),
                       dpi=72 if fmt == "gif" else 120):
        for step, row in event.iterrows():
            path = args.output_dir / "flow_maps" / f"flow_{stamp(row.Date_Time)}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing generated archive {path}; run simulate first")
            with np.load(path) as archive:
                maps = archive["maps"]
                if samples > len(maps):
                    raise ValueError(f"{path} contains only {len(maps)} maps")
                # Evenly span the pooled stack when fewer than all are shown.
                indices = np.linspace(0, len(maps) - 1, samples, dtype=int)
                for shown, member in enumerate(indices, 1):
                    image.set_data(prepare_depth(maps[member], crop, stride))
                    flow_cursor.set_xdata([row.Date_Time, row.Date_Time])
                    title.set_text(
                        f"{AI_LABEL} — event 6\n{row.Date_Time:%d %b %Y %H:%M}"
                        f"   generated map {shown:03d}/{samples:03d}"
                    )
                    fig.canvas.draw()
                    writer.grab_frame()
    plt.close(fig)
    if fmt == "gif":
        optimize_gif(output, max(2, min(args.gif_colors, 256)))
    LOG.info("Wrote %s", output)


def save_reference_animation(args, event, fmt: str, crop, stride, view: MapView) -> None:
    fps = args.fps if fmt == "mp4" else min(args.fps, 50)
    repeats = max(1, round(args.reference_seconds_per_step * fps))
    output = args.output_dir / f"event6_reference_{args.reference_slowdown:g}x_slow.{fmt}"
    fig = plt.figure(figsize=(10, 8), dpi=120)
    map_ax = fig.add_axes([0.08, 0.35, 0.78, 0.57])
    bc_ax, flow_cursor = add_bc_axes(fig, event)
    flow_cursor.set_color("#16a085")
    flow_cursor.set_label(AI_LABEL)
    reference_cursor = bc_ax.axvline(
        event.iloc[0].Date_Time, color="#9b59b6", lw=2.5,
        linestyle="--", zorder=10, label=PHYSICS_LABEL)
    bc_ax.legend(handles=[flow_cursor, reference_cursor], loc="upper left")
    setup_map_axes(map_ax, view)
    with rasterio.open(event.iloc[0].flood_map_file) as src:
        initial = prepare_depth(src.read(1), crop, stride)
    image = depth_image(map_ax, initial, args, view)
    color_ax = fig.add_axes([0.88, 0.38, 0.025, 0.50])
    fig.colorbar(image, cax=color_ax, label="Flood depth (m)")
    title = map_ax.set_title("", fontsize=15, loc="left")
    writer = writer_for(fmt, fps)
    with writer.saving(fig, str(output),
                       dpi=72 if fmt == "gif" else 120):
        for _, row in event.iterrows():
            with rasterio.open(row.flood_map_file) as src:
                image.set_data(prepare_depth(src.read(1), crop, stride))
            reference_cursor.set_xdata([row.Date_Time, row.Date_Time])
            title.set_text(
                f"{PHYSICS_LABEL} — event 6\n"
                f"{row.Date_Time:%d %b %Y %H:%M}"
            )
            fig.canvas.draw()
            for _ in range(repeats):
                writer.grab_frame()
    plt.close(fig)
    if fmt == "gif":
        optimize_gif(output, max(2, min(args.gif_colors, 256)))
    LOG.info("Wrote %s", output)


def save_combined_animation(args, event, fmt: str, crop, stride, view: MapView) -> None:
    """Both models side by side, in synchronized panels.

    With ``--pacing equal`` the two runs are in phase: each frame holds the same
    valid time in both panels for the same dwell, and they finish together, so
    what the animation compares is the maps. With ``--pacing staggered`` the
    solver instead falls behind by construction and finishes alone, which is the
    older runtime illustration.
    """
    equal = args.pacing == "equal"
    fps = args.fps
    if equal:
        output = args.output_dir / f"event6_ai_vs_physics_equal_phase.{fmt}"
    else:
        output = args.output_dir / f"event6_flow_vs_reference_{args.reference_slowdown:g}x.{fmt}"
    fig = plt.figure(figsize=(14, 8), dpi=120)
    flow_ax = fig.add_axes([0.035, 0.35, 0.43, 0.55])
    reference_ax = fig.add_axes([0.49, 0.35, 0.43, 0.55])
    bc_ax, flow_cursor = add_bc_axes(fig, event)
    if equal:
        # Both panels sit on the same valid time every frame, so two cursors
        # would draw on top of each other and read as a rendering fault.
        flow_cursor.set_color("#d62728")
        flow_cursor.set_label("Valid time — both models")
        reference_cursor = None
        bc_ax.legend(handles=[flow_cursor], loc="upper left")
    else:
        flow_cursor.set_color("#16a085")
        flow_cursor.set_label(AI_LABEL)
        reference_cursor = bc_ax.axvline(
            event.iloc[0].Date_Time, color="#9b59b6", lw=2.5,
            linestyle="--", zorder=10, label=PHYSICS_LABEL)
        bc_ax.legend(handles=[flow_cursor, reference_cursor], loc="upper left")
    for ax in (flow_ax, reference_ax):
        setup_map_axes(ax, view)

    first_archive = (args.output_dir / "flow_maps" /
                     f"flow_{stamp(event.iloc[0].Date_Time)}.npz")
    if not first_archive.exists():
        raise FileNotFoundError(
            f"Missing generated archive {first_archive}; run simulate first")
    with np.load(first_archive) as archive:
        flow_initial = prepare_depth(archive["maps"][0], crop, stride)
    with rasterio.open(event.iloc[0].flood_map_file) as src:
        reference_initial = prepare_depth(src.read(1), crop, stride)

    flow_image = depth_image(flow_ax, flow_initial, args, view)
    reference_image = depth_image(reference_ax, reference_initial, args, view)
    flow_title = flow_ax.set_title(AI_LABEL, fontsize=15, loc="left")
    reference_title = reference_ax.set_title(PHYSICS_LABEL, fontsize=15, loc="left")
    color_ax = fig.add_axes([0.94, 0.38, 0.018, 0.48])
    fig.colorbar(flow_image, cax=color_ax, label="Flood depth (m)")

    flow_map_count = max(1, min(args.combined_flow_maps, len(event)))
    flow_indices = np.linspace(0, len(event) - 1, flow_map_count, dtype=int)
    flow_event = event.iloc[flow_indices]
    reference_indices = flow_indices
    # A GIF writer caps its own frame rate, so a dwell in seconds has to be
    # converted at the rate the file is actually written at.
    output_fps = min(fps, 8) if fmt == "gif" else fps
    writer = writer_for(fmt, fps)
    with writer.saving(fig, str(output),
                       dpi=72 if fmt == "gif" else 120):
        if equal:
            frames_per_step = max(1, round(args.seconds_per_step * output_fps))
            for event_step in flow_indices:
                row = event.iloc[event_step]
                path = (args.output_dir / "flow_maps" /
                        f"flow_{stamp(row.Date_Time)}.npz")
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing generated archive {path}; run simulate first")
                with np.load(path) as archive:
                    flow_image.set_data(
                        prepare_depth(archive["maps"][0], crop, stride))
                with rasterio.open(row.flood_map_file) as src:
                    reference_image.set_data(
                        prepare_depth(src.read(1), crop, stride))
                flow_cursor.set_xdata([row.Date_Time, row.Date_Time])
                position = (f"{row.Date_Time:%d %b %Y %H:%M}"
                            f"  •  step {event_step + 1}/{len(event)}")
                flow_title.set_text(f"{AI_LABEL}\n{position}")
                reference_title.set_text(f"{PHYSICS_LABEL}\n{position}")
                fig.canvas.draw()
                for _ in range(frames_per_step):
                    writer.grab_frame()

        else:
            physics_steps = max(1, min(args.reference_steps_by_flow_finish,
                                       len(reference_indices)))
            for flow_step, (_, row) in enumerate(flow_event.iterrows()):
                path = (args.output_dir / "flow_maps" /
                        f"flow_{stamp(row.Date_Time)}.npz")
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing generated archive {path}; run simulate first")
                # Spread the small number of physics updates across the complete
                # surrogate run. Unlike a fixed lag, this makes the much slower
                # solver reach only ``physics_steps`` timesteps by the time the AI
                # model finishes.
                if len(flow_event) == 1 or physics_steps == 1:
                    reference_step = 0
                else:
                    reference_step = int(np.floor(
                        flow_step * (physics_steps - 1) / (len(flow_event) - 1)))
                reference_event_step = reference_indices[reference_step]
                reference_row = event.iloc[reference_event_step]
                with rasterio.open(reference_row.flood_map_file) as src:
                    reference_image.set_data(
                        prepare_depth(src.read(1), crop, stride))
                flow_cursor.set_xdata([row.Date_Time, row.Date_Time])
                reference_cursor.set_xdata(
                    [reference_row.Date_Time, reference_row.Date_Time])
                flow_title.set_text(
                    f"{AI_LABEL}\n{row.Date_Time:%d %b %Y %H:%M}")
                reference_title.set_text(
                    f"{PHYSICS_LABEL}\n{reference_row.Date_Time:%d %b %Y %H:%M}"
                    f"  •  step {reference_event_step + 1}/{len(event)}"
                    f"  •  ≈{args.reference_slowdown:g}× slower")

                with np.load(path) as archive:
                    maps = archive["maps"]
                    # One representative ensemble member per selected event time:
                    # at most 20 flow maps for the entire combined animation.
                    for member in [0]:
                        flow_image.set_data(
                            prepare_depth(maps[member], crop, stride))
                        fig.canvas.draw()
                        writer.grab_frame()

            # The AI model is now complete. Hold its final map while the physics
            # simulation continues at a deliberately slow, independently controlled
            # cadence until it finishes.
            if physics_steps < len(reference_indices):
                flow_title.set_text(
                    f"{AI_LABEL} — complete\n{event.iloc[-1].Date_Time:%d %b %Y %H:%M}")
                flow_cursor.set_xdata(
                    [event.iloc[-1].Date_Time, event.iloc[-1].Date_Time])
                frames_per_physics_step = max(
                    1, round(args.reference_seconds_per_step * output_fps))
                for reference_step in range(physics_steps, len(reference_indices)):
                    reference_event_step = reference_indices[reference_step]
                    reference_row = event.iloc[reference_event_step]
                    with rasterio.open(reference_row.flood_map_file) as src:
                        reference_image.set_data(
                            prepare_depth(src.read(1), crop, stride))
                    reference_cursor.set_xdata(
                        [reference_row.Date_Time, reference_row.Date_Time])
                    reference_title.set_text(
                        f"{PHYSICS_LABEL}\n"
                        f"{reference_row.Date_Time:%d %b %Y %H:%M}"
                        f"  •  step {reference_event_step + 1}/{len(event)}"
                        f"  •  ≈{args.reference_slowdown:g}× slower")
                    fig.canvas.draw()
                    for _ in range(frames_per_physics_step):
                        writer.grab_frame()
    plt.close(fig)
    if fmt == "gif":
        optimize_gif(output, max(2, min(args.gif_colors, 256)))
    LOG.info("Wrote %s", output)


def animate(args: argparse.Namespace, event: pd.DataFrame) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args, event)
    crop = domain_crop(event.iloc[0].flood_map_file)
    stride = render_stride(crop, args.render_width)
    extent, crs = map_extent(event.iloc[0].flood_map_file, crop, stride)
    view = MapView(extent, *load_basemap(
        args.basemap, extent, crs, args.output_dir / "basemap_cache",
        args.basemap_zoom, args.render_width, args.basemap_provider))
    if args.gif_colors is None:
        # Satellite and street tiles posterize badly at the bare-raster palette.
        args.gif_colors = 128 if view.image is not None else 64
    formats = ("gif", "mp4") if args.format == "both" else (args.format,)
    for fmt in formats:
        if args.layout in ("separate", "flow", "both"):
            save_flow_animation(args, event, fmt, crop, stride, view)
        if args.layout in ("separate", "both"):
            save_reference_animation(args, event, fmt, crop, stride, view)
        if args.layout in ("combined", "both"):
            save_combined_animation(args, event, fmt, crop, stride, view)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.step_hours < 1:
        raise SystemExit("--step-hours must be at least 1")
    if args.step_hours > 1 and args.command in ("simulate", "all"):
        # make_conditions builds the model's boundary-condition lags by shifting
        # rows, so a subsampled frame would feed it 4-hourly lags where the
        # scalers and the network expect hourly ones.
        raise SystemExit("--step-hours only applies to animate; simulate needs "
                         "every hour to build the boundary-condition lags")
    event = read_event(args.csv, args.start, args.end, args.step_hours)
    LOG.info("Event 6: %d timesteps every %d h, %s to %s", len(event),
             args.step_hours, event.iloc[0].Date_Time, event.iloc[-1].Date_Time)
    if args.command in ("simulate", "all"):
        simulate(args, event)
    if args.command in ("animate", "all"):
        animate(args, event)


if __name__ == "__main__":
    main()
