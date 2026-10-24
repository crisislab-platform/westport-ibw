# Set memory limit of 2GB for each process
ROOT_DIR="/home/91/23016891/projects/westport-ibw"
ENV_DIR="/data/miniconda3/envs/westport-ibw-env"

# Add --push_to_arcgis to publish each issue time's impacts to the hosted
# feature layers. It needs ARCGIS_CLIENT_ID/SECRET and the layer URLs in .env,
# with editing enabled for Add AND Delete: each issue replaces a layer's
# contents, so the layers only ever hold the latest forecast and re-running a
# published window overwrites it. --forecast_horizon therefore sets how many
# timesteps are visible at once.
#
# This run records no timings; run_scripts/runtime.sh times the same issues.
#
# parallel takes the CARTESIAN PRODUCT of the ::: lists, so the horizon sweep
# and an ensemble sweep multiply: 4 horizons x 3 values of M is 12 runs, each
# over the full 48-hour window. Sweep one axis at a time -- pin the horizon to
# a single value while varying M and K -- unless the interaction is the point.
# M is the number of ensemble members loaded and run, K the samples drawn from
# each; the inundation cost is linear in their product.
parallel --line-buffer -j 1 \
  systemd-run --scope -p MemoryMax=40G  \
  $ENV_DIR/bin/python3.11 main.py run\
  --scenario_id {1} \
  --forecast_horizon {2} \
  --start_time {3} \
  --end_time {4} \
  --samples_per_member {5}\
  --num_deep_ensemble {6}\
  --push_to_arcgis \
  ::: 0 \
  ::: 12 \
  ::: "2021-07-16 12:00:00"\
  ::: "2021-07-16 13:00:00" \
  ::: 10 \
  ::: 10


