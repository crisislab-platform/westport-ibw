# Time and memory of running each test scenario ONCE (main.py runtime, no
# forecast horizon: the whole event as one forecast), with no validation: no
# simulated maps are read and no metrics computed. Each event appends a row to
# runtime_metrics.csv in the run's folder under
# $WESTPORT_DATA_ROOT/out/exposure/validation.
#
# Plain loop rather than parallel + systemd-run (validate_simulations.sh), so it
# runs on macOS too. Data from the bundle staged at
# /data/westport/data/laptop_runtime_bundle; run from the repo root:
#   bash run_scripts/runtime_laptop.sh
# Apple silicon: WESTPORT_ORT_COREML=1 bash run_scripts/runtime_laptop.sh tries
# CoreML; move the CPU run's folder aside first, the folder name is the same.

export WESTPORT_DATA_ROOT="${WESTPORT_DATA_ROOT:-$HOME/laptop_runtime_bundle}"

TEST_EVENTS="3 13 17 37 43 54 57 64 83 91 93 106"

# "flow_matching", "cnn1d" or both.
MODELS="cnn1d flow_matching"

# As validate_simulations.sh; each can be set from the command line instead,
#   LOW_MEMORY=--low_memory HOURS_PER_BATCH=1 DECODE_BATCH=1 bash run_scripts/runtime_laptop.sh
# LOW_MEMORY=--low_memory loads the flow-matching members one at a time and
# releases the decoder's memory after each call (same forecasts, less memory).
HOURS_PER_BATCH="${HOURS_PER_BATCH:1}"
DECODE_BATCH="${DECODE_BATCH:1}"
LOW_MEMORY="${LOW_MEMORY:--low_memoery}"

for MODEL in $MODELS; do
  for EVENT in $TEST_EVENTS; do
    echo "== $MODEL, scenario $EVENT"
    if [ "$MODEL" = "cnn1d" ]; then
      python main.py runtime --model cnn1d --scenario_id "$EVENT" \
        --hours_per_batch "$HOURS_PER_BATCH"
    else
      python main.py runtime --model flow_matching --scenario_id "$EVENT" \
        --num_deep_ensemble 10 --samples_per_member 1 \
        --hours_per_batch "$HOURS_PER_BATCH" --decode_batch "$DECODE_BATCH" $LOW_MEMORY
    fi
  done
done
