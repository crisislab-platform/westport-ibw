# Calibrate the spatial summary (max / median / p90 of footprint cells) against
# the July 2021 surveyed depths and placards -> spatial_summary_calibration.csv.
# Memory limit of 20GB for the process.
ROOT_DIR="/home/91/23016891/projects/westport-ibw"
ENV_DIR="/data/miniconda3/envs/westport-ibw-env"

cd "$ROOT_DIR" && \
systemd-run --scope -p MemoryMax=20G  \
  $ENV_DIR/bin/python3.11 main.py calibrate
