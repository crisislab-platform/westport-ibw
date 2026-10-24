# Set memory limit of 2GB for each process
TEST_EVENTS="3 13 17 37 43 54 57 64 83 91 93 106"

ROOT_DIR="/home/91/23016891/projects/westport-ibw"
ENV_DIR="/data/miniconda3/envs/westport-ibw-env"

# Which forecast to score: "flow_matching" (the surrogate) or "cnn1d" (the
# westport 1D-CNN benchmark, exported once by scripts/export_cnn1d_onnx.py).
# Each model's results go to their own folder under VALIDATION_RESULTS_DIR.
MODEL="flow_matching"

case "$MODEL" in
  flow_matching|cnn1d) ;;
  *) echo "MODEL must be flow_matching or cnn1d, not '$MODEL'" >&2; exit 1 ;;
esac

if [ "$MODEL" = "cnn1d" ]; then
  # Deterministic: one map per hour, so no ensemble or decoder settings.
  parallel --line-buffer -j 1 \
    systemd-run --scope -p MemoryMax=20G  \
    $ENV_DIR/bin/python3.11 main.py validate\
    --model cnn1d \
    --scenario_id {1} \
    --hours_per_batch {2} \
    ::: $TEST_EVENTS \
    ::: 12
else
  parallel --line-buffer -j 1 \
    systemd-run --scope -p MemoryMax=20G  \
    $ENV_DIR/bin/python3.11 main.py validate\
    --model flow_matching \
    --scenario_id {1} \
    --samples_per_member {2} \
    --num_deep_ensemble {3} \
    --hours_per_batch {4} \
    --decode_batch {5} \
    ::: $TEST_EVENTS \
    ::: 1 \
    ::: 10 \
    ::: 12 \
    ::: 5
fi

# 3 13 17 37 43 54 57 64 83 91 93 106