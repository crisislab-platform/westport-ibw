# Feasibility image

A self-contained CPU image that runs the Westport impact-based forecast end to
end, so the per-forecast-step cost can be measured on an ordinary laptop rather
than argued about. No GPU, no network, no volumes, no credentials.

```bash
docker run --rm westport-ibw:cpu --members 1 --samples 1 --horizon 1   # ~1 min
docker run --rm westport-ibw:cpu                                       # the sweep
```

## What is real and what is not

The image ships the **real** ensemble (ten ONNX velocity networks and the VAE
decoder), the **real** exposure vectors (5,556 buildings, 2,671 properties, 453
road parcels), the real precomputed ring cache and the real boundary-condition
series and scalers. Those are what set the size of the computation.

The sample flood map, the latent statistics and the latent DEM are
**synthetic** — the flood maps are 663 GB and the timings do not depend on the
numbers inside them. What the code reads out of those files is geometry, and
that is copied exactly: raster shape (2416 × 2448), affine transform, CRS
(EPSG:2193), and the NaN footprint of the computational domain (38.9 % of the
grid). Every array the run allocates therefore has the shape it would have on
real data, and the ring cache built against the real grid loads without a
rebuild — which is the check that the geometry really did survive.

What this measures is **cost, not skill**. Depths coming out of the run are not
forecasts of anything; do not read the impact tables as results.

## Building it

The bundle is staged on a machine that has `/data/westport` mounted, then baked
into the image. Build from the repo root:

```bash
python docker/stage_data.py                              # -> docker/data-bundle (~270 MB)
docker build -f docker/Dockerfile -t westport-ibw:cpu .  # -> ~3.7 GB image
```

`stage_data.py --num-members 3` stages a smaller ensemble if the image needs to
be smaller; it caps the `M` a run can then ask for.

To move the image to a machine that cannot reach the workstation:

```bash
docker save westport-ibw:cpu | zstd -o westport-ibw-cpu.tar.zst
# on the laptop:
zstd -dc westport-ibw-cpu.tar.zst | docker load
```

## Options

```
--members "1 5 10"   ensemble members M to sweep      (default: "1 5 10")
--samples K          samples per member               (default: 10)
--horizon H          forecast timesteps per issue     (default: 4)
--scenario ID        flood scenario                   (default: 0)
--start "TS"         first issue time                 (default: 2021-07-15 12:00:00)
--end "TS"           last issue time                  (default: same as --start)
--threads N          CPU threads per run              (default: all visible cores)
--shell              drop into a shell instead
```

The pooled sample count is `M*K` and the hazard cost is linear in it, so sweep
one axis at a time. `--threads` matters more than it looks: torch and
onnxruntime each size a thread pool from the visible core count, and left alone
they claim every core and then contend with each other. Pinning both is what
makes one laptop's number comparable to another's.

Timings land in `/app/results/runtime_metrics.csv` — one row per issue, with
the seconds spent in each component:

```bash
docker run --rm -v "$PWD/results:/app/results" westport-ibw:cpu
```

## Measured on 8 cores

`docker run --cpus=8 --memory=...` standing in for a laptop, on a machine whose
own hardware (16 cores, an H100) is not what is being measured. Horizon 1,
`K=10`, so the numbers are per forecast step:

| M  | M·K | s/step | peak RSS |
|----|-----|--------|----------|
| 1  | 10  | 15.5   | ~4 GB    |
| 5  | 50  | 66.6   | 12.0 GB  |
| 10 | 100 | 137.6  | 13.9 GB  |

Time is linear in `M·K` — about 1.3 s per pooled sample per step. The same
config on the workstation's H100 runs at ~10 s/step, so CPU costs roughly 13×,
which matches the ~15× the ONNX client's docstring quotes for the decoder.

**Memory is the real constraint, not time.** The run holds every pooled sample
at full raster resolution, so `M·K=100` peaks near 14 GB and is OOM-killed
under `--memory=8g`. A 16 GB laptop will not have 14 GB free for it in
practice; treat `M=10, K=10` as wanting 32 GB, and `M=5, K=10` as the
comfortable 16 GB configuration. Under an 8 GB cap `M=5` still completes, but
at 87 s/step against 67 s unconstrained — memory pressure, paid as time.

If a run has to shrink, drop `K` before `M`: `K` is generative noise within a
member, `M` is disagreement between members, and the second is the thing the
deep ensemble exists to capture.

## Publishing is off

`--push_to_arcgis` is not reachable from this image: the `arcgis` package is
not installed (it is a second GDAL stack, and only ever used for publishing),
and `ArcGISModule()` raises a clear error if a run asks for it. On the
workstation, publishing is over half the wall clock of an issue and none of it
is computation, so leaving it out is also what makes these timings mean
something.
