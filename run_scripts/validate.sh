# Set memory limit of 2GB for each process
TEST_EVENTS="3 13 17 37 43 54 57 64 83 91 93 106"

ROOT_DIR="/home/91/23016891/projects/westport-ibw"
ENV_DIR="/data/miniconda3/envs/westport-ibw-env"

parallel --line-buffer -j 1 \
  systemd-run --scope -p MemoryMax=20G  \
  $ENV_DIR/bin/python3.11 main.py test\
  --scenario_id {1} \
  --max_threshold {2} \
  --mean_threshold {3} \
  --buffer_multiplier {4} \
  ::: 0 \
  ::: 0.15 \
  ::: 0.05 \
  ::: 2
