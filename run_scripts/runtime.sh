# Runtime test: one row per issue in
# /data/westport/out/exposure/validation/runtime_metrics.csv with the seconds
# spent in each component; divide by num_timesteps to compare horizons per
# forecast step.
ROOT_DIR="/home/91/23016891/projects/westport-ibw"
ENV_DIR="/data/miniconda3/envs/westport-ibw-env"

# Swept one run at a time (-j 1) so the timings are not contending for the GPU
# or for AGOL. Add --push_to_arcgis to time the publish cost as well (it needs
# the credentials and layer URLs in .env, see run_forecast.sh).
#
# parallel takes the CARTESIAN PRODUCT of the ::: lists, so the horizon sweep
# and an ensemble sweep multiply: 4 horizons x 3 values of M is 12 runs, each
# over the full window. Sweep one axis at a time -- pin the horizon to a single
# value while varying M and K -- unless the interaction is the point. M is the
# number of ensemble members loaded and run, K the samples drawn from each; the
# inundation cost is linear in their product.
parallel --line-buffer -j 1 \
  systemd-run --scope -p MemoryMax=40G  \
  $ENV_DIR/bin/python3.11 main.py runtime\
  --scenario_id {1} \
  --forecast_horizon {2} \
  --start_time {3} \
  --end_time {4} \
  --samples_per_member {5}\
  --num_deep_ensemble {6}\
  ::: 0 \
  ::: 12 \
  ::: "2021-07-16 12:00:00"\
  ::: "2021-07-16 13:00:00" \
  ::: 10 \
  ::: 10
