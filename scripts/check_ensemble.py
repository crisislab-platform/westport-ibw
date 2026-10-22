"""Acceptance checks for the deep-ensemble inference path.

Run with:  python -m scripts.check_ensemble

The checks are structural -- they confirm that neither ensemble loop has been
collapsed and that the sessions really hold distinct weights. They use a random
condition vector, since none of what is asserted depends on the boundary
conditions being physical.
"""

import logging
import os
import warnings

import numpy as np
import onnxruntime as ort
import torch

from modules.lib import (
    HAZARD_MODEL_PATH,
    LATENT_FLOW_MATCHING_ENSEMBLE,
    AUTO_ENCODER,
    AUTO_ENCODER_DECODER,
)
from modules.hazard_module.ensemble_client import (
    LatentFlowEnsemble,
    select_providers,
)
from modules.hazard_module.spatial_sampler.latent_canvas import LatentCanvas

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

ENSEMBLE_DIR = os.path.join(HAZARD_MODEL_PATH, LATENT_FLOW_MATCHING_ENSEMBLE)
DECODER_PATH = os.path.join(HAZARD_MODEL_PATH, AUTO_ENCODER, AUTO_ENCODER_DECODER + ".onnx")

PASSED = []
FAILED = []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def build(canvas, num_members, samples_per_member, num_steps=10, seed=0):
    return LatentFlowEnsemble(
        ensemble_dir=ENSEMBLE_DIR,
        decoder_path=DECODER_PATH,
        latent_canvas=canvas,
        num_members=num_members,
        samples_per_member=samples_per_member,
        num_steps=num_steps,
        seed=seed,
    )


def main():
    canvas = LatentCanvas()
    lat_h, lat_w = canvas.canvas_shape
    z_dem = canvas.z_dem.numpy().astype(np.float32)
    rng = np.random.default_rng(0)
    cond = rng.standard_normal((1, 2, 7)).astype(np.float32)

    # 1. Distinct weights ------------------------------------------------
    print("\n1. Members hold distinct weights")
    ens = build(canvas, num_members=2, samples_per_member=1)
    x = rng.standard_normal((2, 8, lat_h, lat_w)).astype(np.float32)
    t = np.zeros(2, dtype=np.float32)
    c2 = np.tile(cond, (2, 1, 1))
    v0 = ens.member_sessions[0].run(["velocity"], {"x": x, "t": t, "cond": c2})[0]
    v1 = ens.member_sessions[1].run(["velocity"], {"x": x, "t": t, "cond": c2})[0]
    delta = float(np.abs(v0 - v1).max())
    check("two members give different velocities on identical input",
          delta > 1e-3, f"max|dv| = {delta:.4f}")
    check("member order is the filename index",
          [os.path.basename(p) for p in ens.member_paths]
          == ["member_00_seed_7.onnx", "member_01_seed_41.onnx"],
          ", ".join(os.path.basename(p) for p in ens.member_paths))

    # 4. Dynamic batch axis ---------------------------------------------
    print("\n4. Batch axis is dynamic (K needs no re-export)")
    x3 = rng.standard_normal((3, 8, lat_h, lat_w)).astype(np.float32)
    v3 = ens.member_sessions[0].run(
        ["velocity"],
        {"x": x3, "t": np.zeros(3, dtype=np.float32), "cond": np.tile(cond, (3, 1, 1))},
    )[0]
    check("B=3 returns (3, 4, h, w)", v3.shape == (3, 4, lat_h, lat_w), str(v3.shape))

    # 2. M=1, K=1 reproduces the single-model path -----------------------
    print("\n2. M=1, K=1 reproduces the single-model rollout")
    single = build(canvas, num_members=1, samples_per_member=1)
    z_init = rng.standard_normal((1, 4, lat_h, lat_w)).astype(np.float32)
    got = single.sample(cond, z_dem, z_init=z_init)

    # Reference: the repo's original Euler loop, written out longhand.
    # select_providers(), not single.member_sessions[0].get_providers(): the
    # latter returns provider NAMES only, dropping the options the client set
    # (use_tf32=0). A reference session built from it would run the same graph
    # with TF32 on and miss this comparison by ~1e-2 -- a difference in the
    # reference, not in the client.
    session = ort.InferenceSession(
        os.path.join(ENSEMBLE_DIR, "member_00_seed_7.onnx"),
        providers=select_providers(),
    )
    steps = single.num_steps
    z_t = z_init.copy()
    dt = np.float32(1.0 / steps)
    for idx in range(steps):
        t_step = np.full((1,), idx / steps, dtype=np.float32)
        x_in = np.concatenate([z_t, z_dem], axis=1)
        z_t = z_t + session.run(["velocity"], {"x": x_in, "t": t_step, "cond": cond})[0] * dt

    check("shape is (1, 1, 4, h, w)", got.shape == (1, 1, 4, lat_h, lat_w), str(got.shape))
    check("latents match the reference rollout bit for bit",
          np.array_equal(got[0], z_t), f"max|d| = {float(np.abs(got[0] - z_t).max()):.2e}")

    single_forecast = single.forecast(cond, z_dem, z_init=z_init)
    check("M=1,K=1 collapses both uncertainty terms to exactly 0",
          float(np.nanmax(single_forecast.aleatoric)) == 0.0
          and float(np.nanmax(single_forecast.epistemic)) == 0.0)

    # Decoding in chunks must not change the result.
    single.decode_batch = 1
    unbatched = single.decode(got.reshape(1, 4, lat_h, lat_w))
    check("chunked decode == unchunked decode",
          np.array_equal(np.nan_to_num(unbatched), np.nan_to_num(single_forecast.members[:, 0])))

    # 3. Both spreads are non-zero ---------------------------------------
    print("\n3. Neither loop is collapsed (M=2, K=2)")
    pooled = build(canvas, num_members=2, samples_per_member=2)
    result = pooled.forecast(cond, z_dem)
    check("members is (M*K, B, H, W)",
          result.members.shape == (4, 1, canvas.orig_h, canvas.orig_w),
          str(result.members.shape))

    aleatoric = float(np.nanmax(result.aleatoric))
    epistemic = float(np.nanmax(result.epistemic))
    check("aleatoric > 0 with K>1", aleatoric > 0, f"max = {aleatoric:.4f} m")
    check("epistemic > 0 with M>1", epistemic > 0, f"max = {epistemic:.4f} m")

    # The pooled mean is a pixel-space reduction over every decoded member.
    pixel_mean = np.nanmean(result.members, axis=0)
    check("mean is the pixel-space mean over pooled members",
          np.allclose(np.nan_to_num(result.mean), np.nan_to_num(pixel_mean), atol=1e-6))

    # Member-major layout: rows [m*K:(m+1)*K] belong to member m.
    member_0 = result.members[0:2]
    member_1 = result.members[2:4]
    check("member-major layout (members differ across the M blocks)",
          not np.allclose(np.nan_to_num(member_0.mean(axis=0)),
                          np.nan_to_num(member_1.mean(axis=0))))

    # 5. The K*B tiling reshapes back to the right axes --------------------
    # If the cond tiling and the (K, B) reshape disagreed, forecast steps and
    # draws would be silently transposed. Two steps run together must equal the
    # same two steps run separately, given the same initial state.
    print("\n5. Batch axis maps to forecast steps, not to draws")
    pair = build(canvas, num_members=1, samples_per_member=1)
    cond2 = rng.standard_normal((2, 2, 7)).astype(np.float32)
    init2 = rng.standard_normal((2, 4, lat_h, lat_w)).astype(np.float32)
    together = pair.sample(cond2, z_dem, z_init=init2)
    apart = np.stack([
        pair.sample(cond2[i:i + 1], z_dem, z_init=init2[i:i + 1])[0, 0]
        for i in range(2)
    ])
    # Tolerance, not bit equality: cuDNN selects a different convolution kernel
    # per batch shape, so B=2 and B=1 agree to ~1e-3 on the GPU rather than
    # exactly (on CPU this was exact). A transposed axis would put the wrong
    # cond row on the wrong step and miss by O(1), so 1e-2 still catches the bug
    # this check exists for.
    batch_shape_tol = 1e-2
    check("batched steps == the same steps run one at a time",
          np.allclose(together[0], apart, rtol=0, atol=batch_shape_tol),
          f"max|d| = {float(np.abs(together[0] - apart).max()):.2e} "
          f"(atol {batch_shape_tol:g})")

    # With K>1 the draws stack ahead of the steps: rows [k*B:(k+1)*B] are draw k.
    multi = build(canvas, num_members=1, samples_per_member=2)
    init_shared = np.tile(init2, (2, 1, 1, 1))
    stacked = multi.sample(cond2, z_dem, z_init=init_shared)
    check("identical draws collapse to identical rows (K, B) ordering",
          np.array_equal(stacked[0], stacked[1]))

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
