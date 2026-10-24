"""Deep-ensemble sampler for the latent flow-matching flood model.

Each ONNX file is ONLY the velocity network ``u(x, t, cond) -> velocity``.
Nothing about the ensemble lives inside the graph -- the ensemble is two plain
loops around that network:

``K`` (samples per member)
    The BATCH dimension of a single session call: K i.i.d. Gaussian initial
    states integrated together. This is the ALEATORIC (generative-noise)
    spread, and it is nearly free -- one batched call instead of K calls.

``M`` (trained members)
    One session per file, sampled in turn and concatenated. The members hold
    different weights, so this cannot be folded into one graph. This is the
    EPISTEMIC (model-disagreement) spread, and it costs a full extra rollout
    per member.

The pooled set is ``M*K`` samples, held member-major: rows ``[m*K:(m+1)*K]``
belong to member ``m``. The uncertainty split depends on that layout, so it is
preserved end to end -- neither loop is collapsed.

Uncertainty is decomposed in PIXEL space, after decoding every member. Latent
spread does not transfer: the decoder is nonlinear and contracts differences,
so a spread computed on latents overstates the spread you actually get on
depths.
"""

import glob
import logging
import os
import re
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import torch

logger = logging.getLogger(__name__)

# Members are named member_<index>_seed_<seed>.onnx. The numeric index is the
# canonical order (ascending seed, fixed at export time); capping the ensemble
# takes the FIRST num_members in that order so an M-sweep is nested and stays
# comparable to the research repo's published numbers.
MEMBER_FILE_PATTERN = re.compile(r"member_(\d+)_seed_(\d+)\.onnx$")


# Apple silicon: set WESTPORT_ORT_COREML=1 to try onnxruntime's CoreML provider
# (the M-series GPU / Neural Engine) before the CPU. Off by default: CoreML may
# compute in reduced precision, so its depths can differ from the CPU's (as
# TF32 does below), and nodes it cannot take still run on the CPU.
USE_COREML = os.environ.get("WESTPORT_ORT_COREML", "0") == "1"


def select_providers(preferred=None, use_tf32=False):
    """Keep only providers this onnxruntime build actually has.

    Passing an unavailable provider does not raise -- onnxruntime logs a
    warning and silently falls back to CPU, which is easy to miss and looks
    like the GPU is being used when it is not. Filtering up front makes the
    choice explicit and lets the caller see it in the logs.

    Being listed here is necessary but not sufficient: get_available_providers()
    reports CUDAExecutionProvider whenever the GPU build is installed, even when
    its .so cannot be loaded. Only the session's own get_providers() says what
    actually ran -- see _warn_on_cpu_fallback.

    TF32 is off by default. onnxruntime enables it for convolutions on Ampere
    and later, and its 10-bit mantissa moves decoded depths by up to ~6 cm
    against the same graph on CPU -- enough to cross the depth thresholds the
    exposure tiers are cut on, so GPU and CPU runs would not be comparable.
    Disabling it costs ~60% on the decoder and still leaves it ~15x faster than
    CPU. Pass use_tf32=True to trade that agreement back for the speed.
    """
    if preferred is None:
        preferred = ("CUDAExecutionProvider",
                     *(("CoreMLExecutionProvider",) if USE_COREML else ()),
                     "CPUExecutionProvider")
    available = ort.get_available_providers()
    providers = []
    for name in preferred:
        if name not in available:
            continue
        if name == "CUDAExecutionProvider":
            # Grow the GPU memory pool by what each allocation asks for, not by
            # doubling (onnxruntime's default), so the memory a run holds is
            # what the models need rather than the next power of two above it.
            options = {"arena_extend_strategy": "kSameAsRequested"}
            if not use_tf32:
                options["use_tf32"] = 0
            providers.append((name, options))
        else:
            providers.append(name)
    return providers or ["CPUExecutionProvider"]


def _provider_names(providers):
    """Provider names alone -- entries may be (name, options) pairs."""
    return [p[0] if isinstance(p, tuple) else p for p in providers]


def _warn_on_cpu_fallback(requested, *sessions):
    """Say so loudly when a requested GPU provider did not survive session init.

    A forecast is ~100 velocity calls plus M*K*B canvas decodes, so the gap
    between the two paths is minutes per forecast step, not percent.
    onnxruntime reports the failure once, buried in its own log stream, then
    runs anyway -- which is how a run ends up on CPU with an idle GPU beside it.
    """
    if "CUDAExecutionProvider" not in _provider_names(requested):
        return
    for session in sessions:
        if "CUDAExecutionProvider" not in session.get_providers():
            logger.warning(
                f"CUDAExecutionProvider was requested but the session runs on "
                f"{session.get_providers()} -- install onnxruntime-gpu (not "
                f"onnxruntime); everything runs on CPU until then."
            )
            return


@dataclass
class EnsembleForecast:
    """Pooled ensemble output for one batch of forecast steps.

    ``members`` is every pooled sample, not a summary -- the whole point is to
    propagate the full ensemble through the downstream exposure computation
    rather than a single collapsed map.
    """

    members: np.ndarray     # (M*K, B, H, W) pooled samples, member-major
    mean: np.ndarray        # (B, H, W) pixel-space mean over the pooled samples
    std: np.ndarray         # (B, H, W) per-cell std over the pooled samples
    aleatoric: np.ndarray   # (B, H, W) within-member (generative noise) spread
    epistemic: np.ndarray   # (B, H, W) across-member (model disagreement) spread
    num_members: int        # M
    samples_per_member: int # K


class LatentFlowEnsemble:
    """The M velocity sessions plus the decoder.

    By default every member's session is opened once and held for the run.
    Each session keeps its own GPU memory pool (~2 GB per member here), so
    holding all M costs M pools although only one member ever runs at a time.
    With ``hold_members=False`` (low-memory mode) a member's session is
    opened when it is sampled and released straight after, and the decoder
    hands its memory pool back after every call instead of keeping its largest
    allocation. The GPU then holds one member and one decoder call at a time:
    about one member's memory instead of M, for the cost of re-opening each
    member per batch. The forecasts are identical either way.
    """
    def __init__(
        self,
        ensemble_dir,
        decoder_path,
        latent_canvas,
        num_members=None,
        samples_per_member=1,
        num_steps=10,
        decode_batch=4,
        providers=None,
        seed=None,
        hold_members=True,
    ):
        self.latent_canvas = latent_canvas
        self.samples_per_member = int(samples_per_member)
        self.num_steps = int(num_steps)
        self.decode_batch = int(decode_batch)

        providers = providers or select_providers()
        self.member_paths = self._discover_members(ensemble_dir, num_members)
        # One random stream per member, keyed by (seed, the member's index in
        # its file name). Each member draws different noise, a run reproduces
        # exactly, and a member's noise does not depend on how many other
        # members are loaded or in what order they are sampled. seed=None
        # gives every member fresh entropy instead.
        self._member_rngs = [
            np.random.default_rng(
                None if seed is None
                else [int(seed), int(MEMBER_FILE_PATTERN.search(os.path.basename(path)).group(1))])
            for path in self.member_paths
        ]
        self._providers = providers
        self.hold_members = bool(hold_members)
        # Low-memory mode: return the decoder's pool to the GPU after each call.
        self._decoder_run_options = None
        if not self.hold_members:
            self._decoder_run_options = ort.RunOptions()
            self._decoder_run_options.add_run_config_entry(
                "memory.enable_memory_arena_shrinkage", "gpu:0")
        first = ort.InferenceSession(self.member_paths[0], providers=providers)
        self.member_sessions = (
            [first] + [ort.InferenceSession(path, providers=providers)
                       for path in self.member_paths[1:]]
            if self.hold_members else None
        )
        self.decoder_session = ort.InferenceSession(decoder_path, providers=providers)
        _warn_on_cpu_fallback(providers, first, self.decoder_session)

        # Dual-conditioning exports take an extra `index_distances` input. Read
        self._velocity_inputs = {i.name for i in first.get_inputs()}
        member_providers = first.get_providers()
        del first
        if "index_distances" in self._velocity_inputs:
            raise NotImplementedError(
                "This export is dual-conditioned (it takes `index_distances`), "
                "which this client does not supply."
            )

        logger.info(
            f"LatentFlowEnsemble: M={self.num_members} members, "
            f"K={self.samples_per_member} samples/member -> "
            f"{self.num_members * self.samples_per_member} pooled samples; "
            f"N={self.num_steps} Euler steps; providers="
            f"{member_providers}; members "
            f"{'held for the run' if self.hold_members else 'opened one at a time'}"
        )

    @property
    def num_members(self):
        return len(self.member_paths)

    def _member_sessions(self):
        """(index, session) per member, in member order. Held sessions are
        reused; otherwise each is opened here and released once the caller
        moves on, so only one member's session is alive at a time."""
        for index, path in enumerate(self.member_paths):
            if self.hold_members:
                yield index, self.member_sessions[index]
            else:
                session = ort.InferenceSession(path, providers=self._providers)
                yield index, session
                del session

    # ------------------------------------------------------------------
    # Members
    # ------------------------------------------------------------------
    def _discover_members(self, ensemble_dir, num_members):
        found = []
        for path in glob.glob(os.path.join(ensemble_dir, "member_*.onnx")):
            match = MEMBER_FILE_PATTERN.search(os.path.basename(path))
            if match:
                found.append((int(match.group(1)), path))
        if not found:
            raise FileNotFoundError(
                f"No member_*_seed_*.onnx files found in {ensemble_dir}"
            )

        found.sort(key=lambda item: item[0])
        paths = [path for _, path in found]
        if num_members is not None:
            if num_members < 1 or num_members > len(paths):
                raise ValueError(
                    f"num_members must be in [1, {len(paths)}], got {num_members}"
                )
            paths = paths[:num_members]
        return paths

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(self, cond, z_dem, z_init=None):
        """Integrate every member and return the pooled latent stack.

        Parameters
        ----------
        cond : array (B, 2, lag)
            Scaled boundary conditions, one row per forecast step.
        z_dem : array (1, 4, h, w) or (B, 4, h, w)
            DEM latent, shared across forecast steps.
        z_init : array (K*B, 4, h, w), optional
            Initial state, shared by all members. Only for reproducing a run
            or for common random numbers; leave unset for the i.i.d. draws the
            aleatoric term is defined on.

        Returns
        -------
        ndarray (M*K, B, 4, h, w), member-major.
        """
        cond = np.ascontiguousarray(_to_numpy(cond), dtype=np.float32)
        z_dem = np.ascontiguousarray(_to_numpy(z_dem), dtype=np.float32)

        batch = cond.shape[0]
        if z_dem.shape[0] == 1:
            z_dem = np.repeat(z_dem, batch, axis=0)
        elif z_dem.shape[0] != batch:
            raise ValueError(
                f"z_dem batch {z_dem.shape[0]} matches neither 1 nor the "
                f"condition batch {batch}"
            )

        # Tile the shared conditioning K times: rows [k*B:(k+1)*B] are draw k.
        samples = self.samples_per_member
        cond_tiled = np.tile(cond, (samples, 1, 1))
        z_dem_tiled = np.tile(z_dem, (samples, 1, 1, 1))

        members = []
        for index, session in self._member_sessions():
            latent = self._rollout(session, cond_tiled, z_dem_tiled, z_init,
                                   rng=self._member_rngs[index])
            del session
            members.append(latent.reshape(samples, batch, *latent.shape[1:]))
            logger.debug(f"sampled member {index} ({os.path.basename(self.member_paths[index])})")

        # Member-major: rows [m*K:(m+1)*K] are member m.
        return np.concatenate(members, axis=0)

    def _rollout(self, session, cond_tiled, z_dem_tiled, z_init=None, rng=None):
        """Euler-integrate the probability-flow ODE for one member."""
        if z_init is None:
            # A fresh independent draw per ROW -- this is the only source of
            # within-member spread. Reusing one noise vector across rows would
            # silently collapse the aleatoric term to zero.
            z_t = rng.standard_normal(z_dem_tiled.shape).astype(np.float32)
        else:
            z_t = np.ascontiguousarray(_to_numpy(z_init), dtype=np.float32).copy()

        rows = z_t.shape[0]
        dt = np.float32(1.0 / self.num_steps)
        for idx in range(self.num_steps):
            # t = idx/N, NOT idx/(N-1): the latter evaluates the velocity at
            # t=1 on the last step and integrates one dt past the data
            # manifold, overshooting every sample.
            t = np.full((rows,), idx / self.num_steps, dtype=np.float32)
            x = np.concatenate([z_t, z_dem_tiled], axis=1)
            velocity = session.run(
                ["velocity"], {"x": x, "t": t, "cond": cond_tiled}
            )[0]
            z_t = z_t + velocity * dt
        return z_t

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, canvases):
        """Decode normalised latent canvases to depth maps.

        Parameters
        ----------
        canvases : array (n, 4, Hc, Wc) in normalised latent space.

        Returns
        -------
        ndarray (n, H, W), NaN outside the computational domain.
        """
        # The latent statistics were saved on whatever device trained them, so
        # meet them there rather than assuming CPU.
        stats_device = self.latent_canvas.z_std.device
        canvas = torch.as_tensor(np.asarray(canvases, dtype=np.float32)).to(stats_device)
        canvas = self.latent_canvas.inverse_normalise_latent_flood_map(canvas)

        images = np.empty(
            (canvas.shape[0], self.latent_canvas.orig_h, self.latent_canvas.orig_w),
            dtype=np.float32,
        )
        for start in range(0, canvas.shape[0], self.decode_batch):
            chunk = canvas[start:start + self.decode_batch]
            # One decoder call for the whole chunk rather than one per canvas.
            tiles = torch.stack([
                self.latent_canvas.canvas_to_latent_tiles(c) for c in chunk
            ])
            n_canvases, tiles_per_canvas = tiles.shape[0], tiles.shape[1]
            flat = tiles.reshape(-1, *tiles.shape[2:]).detach().cpu().numpy()

            pixel_tiles = self.decoder_session.run(
                ["reconstructed"], {"latent": flat}, self._decoder_run_options
            )[0]
            pixel_tiles = torch.from_numpy(pixel_tiles).reshape(
                n_canvases, tiles_per_canvas, *pixel_tiles.shape[1:]
            )
            for i in range(n_canvases):
                images[start + i] = self.latent_canvas.pixel_tiles_to_image(
                    pixel_tiles[i]
                ).numpy()
        return images

    # ------------------------------------------------------------------
    # Forecast + reductions
    # ------------------------------------------------------------------
    def forecast(self, cond, z_dem, z_init=None):
        """Sample every member, decode all of them, and reduce in pixel space.

        The reductions are layered on top of the pooled stack rather than
        folded into the sampling, so ``members`` stays available for
        downstream propagation.
        """
        latents = self.sample(cond, z_dem, z_init=z_init)
        pooled, batch = latents.shape[0], latents.shape[1]

        canvas_h, canvas_w = self.latent_canvas.orig_h, self.latent_canvas.orig_w
        footprint_gb = pooled * batch * canvas_h * canvas_w * 4 / 1e9
        logger.info(
            f"decoding {pooled * batch} canvases "
            f"({pooled} pooled samples x {batch} steps), "
            f"~{footprint_gb:.1f} GB of depth maps retained"
        )

        flat = latents.reshape(pooled * batch, *latents.shape[2:])
        maps = self.decode(flat).reshape(pooled, batch, canvas_h, canvas_w)
        return self._reduce(maps)

    def forecast_steps(self, cond, z_dem, z_init=None):
        """As ``forecast``, but yielding ``(step_index, EnsembleForecast)`` one
        forecast step at a time, decoding each step only when it is asked for.

        Sampling stays batched over the whole horizon -- the latents are two
        orders of magnitude smaller than the depth maps they decode to, and one
        integration over the batch is what the ODE rollout is efficient at.
        Only the DECODE is stepped, because that is where the size is: a step
        is ``M*K`` full-resolution maps, and holding the whole horizon at once
        is what puts a 12-hour, 100-member forecast at ~28 GB.

        The caller must not retain the yielded forecasts: each one keeps its
        own decoded block alive, so holding them all defeats the point.

        Results are identical to ``forecast``, not merely close. Every
        reduction below contracts the pooled and member axes and leaves the
        step axis alone, so a step reduced on its own sees exactly the same
        elements in the same order as it would inside the batch.
        """
        latents = self.sample(cond, z_dem, z_init=z_init)
        pooled, batch = latents.shape[0], latents.shape[1]

        canvas_h, canvas_w = self.latent_canvas.orig_h, self.latent_canvas.orig_w
        per_step_gb = pooled * canvas_h * canvas_w * 4 / 1e9
        logger.info(
            f"decoding {pooled * batch} canvases one step at a time "
            f"({pooled} pooled samples x {batch} steps), "
            f"~{per_step_gb:.1f} GB retained per step"
        )

        for idx in range(batch):
            # Keep the step axis at length 1 so the reductions and the
            # EnsembleForecast layout match the batched path exactly.
            step = latents[:, idx:idx + 1]
            flat = step.reshape(pooled, *step.shape[2:])
            maps = self.decode(flat).reshape(pooled, 1, canvas_h, canvas_w)
            yield idx, self._reduce(maps)

    def _reduce(self, maps):
        """Pixel-space reductions over decoded ``(pooled, B, H, W)`` maps.

        Shared by the batched and stepped paths so the two cannot drift: the
        uncertainty split is the part where a subtle difference would be
        invisible in the output but wrong in the paper.
        """
        batch = maps.shape[1]
        # (M, K, B, H, W) -- the split below is exactly what the member-major
        # layout buys.
        per_member = maps.reshape(
            self.num_members, self.samples_per_member, batch, *maps.shape[2:]
        )
        # Within-member spread, averaged over members.
        aleatoric = np.sqrt(per_member.var(axis=1).mean(axis=0))
        # Spread of the member means. Because each member draws its own noise,
        # this carries a residual +aleatoric^2/K of Monte-Carlo noise, so it is
        # a mild over-estimate at small K.
        epistemic = np.sqrt(per_member.mean(axis=1).var(axis=0))

        return EnsembleForecast(
            members=maps,
            mean=maps.mean(axis=0),
            std=maps.std(axis=0),
            aleatoric=aleatoric,
            epistemic=epistemic,
            num_members=self.num_members,
            samples_per_member=self.samples_per_member,
        )


def _to_numpy(array):
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)
