# Set memory limit of 2GB for each process
TEST_EVENTS="43 54 57 64 83 91 93 106"

ROOT_DIR="/home/91/23016891/projects/westport-ibw"
ENV_DIR="/data/miniconda3/envs/westport-ibw-env"

parallel --line-buffer -j 1 \
  systemd-run --scope -p MemoryMax=20G  \
  $ENV_DIR/bin/python3.11 main.py test\
  --scenario_id {1} \
  --forecast_horizon {2} \
  --samples_per_member {3} \
  --num_deep_ensemble {4} \
  --exceedence_prob {5}\
  ::: 3 \
  ::: 12 \
  ::: 10 \
  ::: 10 \
  ::: 0.1

# 3 13 17 37 43 54 57 64 83 91 93 106 