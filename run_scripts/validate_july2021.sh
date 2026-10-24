ROOT_DIR="/home/91/23016891/projects/westport-ibw"
ENV_DIR="/data/miniconda3/envs/westport-ibw-env"

# Which forecast to score against the July 2021 surveyed flood map (scenario 0,
# at the hour of peak discharge): "flow_matching" (the surrogate) or "cnn1d"
# (the westport 1D-CNN benchmark). Settings match validate_simulations.sh, so
# each model's July 2021 results sit in the same folder as its test events.
MODEL="cnn1d"

case "$MODEL" in
  flow_matching|cnn1d) ;;
  *) echo "MODEL must be flow_matching or cnn1d, not '$MODEL'" >&2; exit 1 ;;
esac

if [ "$MODEL" = "cnn1d" ]; then
  parallel --line-buffer -j 1 \
    systemd-run --scope -p MemoryMax=20G  \
    $ENV_DIR/bin/python3.11 main.py validate \
    --model cnn1d \
    --scenario_id {1} \
    --hours_per_batch {2} \
    ::: 0 \
    ::: 12
else
  parallel --line-buffer -j 1 \
    systemd-run --scope -p MemoryMax=20G  \
    $ENV_DIR/bin/python3.11 main.py validate \
    --model flow_matching \
    --scenario_id {1} \
    --samples_per_member {2} \
    --num_deep_ensemble {3} \
    --hours_per_batch {4} \
    --decode_batch {5} \
    ::: 0 \
    ::: 1 \
    ::: 10 \
    ::: 12 \
    ::: 5
fi
