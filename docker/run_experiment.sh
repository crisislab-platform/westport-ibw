#!/usr/bin/env bash
# Computational-feasibility run: how long one forecast issue takes on this
# machine, and how that scales with the ensemble size.
#
# This is the container's entrypoint. With no arguments it runs a small sweep
# over the ensemble size at a fixed horizon -- one axis at a time, because
# main.py's cost is linear in M*K and a two-axis sweep just multiplies the
# wall time without telling you more.
#
# Publishing is never enabled here. --push_to_arcgis needs credentials and a
# network round trip per layer per issue, which on the workstation is over half
# the wall clock and none of it is computation.
set -euo pipefail

HORIZON=4
MEMBERS=""
SAMPLES=10
SCENARIO=0
START="2021-07-15 12:00:00"
END="2021-07-15 12:00:00"
THREADS=""
RESULTS_DIR=/app/results

usage() {
    cat <<'EOF'
Usage: docker run --rm westport-ibw:cpu [options]

  --members "1 5 10"   ensemble members M to sweep      (default: "1 5 10")
  --samples K          samples per member               (default: 10)
  --horizon H          forecast timesteps per issue     (default: 4)
  --scenario ID        flood scenario                   (default: 0)
  --start "TS"         first issue time                 (default: 2021-07-15 12:00:00)
  --end "TS"           last issue time                  (default: same as --start)
  --threads N          CPU threads per run              (default: all cores)
  --shell              drop into a shell instead of running

The pooled sample count is M*K, and the inundation cost is linear in it. A
single M=1 K=1 run is the fastest way to check the image works at all:

  docker run --rm westport-ibw:cpu --members 1 --samples 1 --horizon 1
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --members)  MEMBERS="$2"; shift 2 ;;
        --samples)  SAMPLES="$2"; shift 2 ;;
        --horizon)  HORIZON="$2"; shift 2 ;;
        --scenario) SCENARIO="$2"; shift 2 ;;
        --start)    START="$2"; shift 2 ;;
        --end)      END="$2"; shift 2 ;;
        --threads)  THREADS="$2"; shift 2 ;;
        --shell)    exec /bin/bash ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

MEMBERS="${MEMBERS:-1 5 10}"
[[ "$END" == "" ]] && END="$START"

# Both runtimes size their own thread pool from the visible core count, so
# leaving them unset means torch and onnxruntime each claim every core and then
# contend. Pinning both to the same number is what makes a laptop timing
# comparable to another laptop's.
if [[ -n "$THREADS" ]]; then
    export OMP_NUM_THREADS="$THREADS"
    export MKL_NUM_THREADS="$THREADS"
else
    export OMP_NUM_THREADS="$(nproc)"
    export MKL_NUM_THREADS="$(nproc)"
fi

mkdir -p "$RESULTS_DIR"
METRICS=/app/out/runtime_metrics.csv
# Start from an empty ledger so the summary below describes this run and not
# whatever the image was last used for.
rm -f "$METRICS"

echo "=============================================================="
echo " Westport IBW -- computational feasibility run"
echo "=============================================================="
echo " cores visible   : $(nproc)"
echo " threads per run : $OMP_NUM_THREADS"
echo " memory          : $(awk '/MemTotal/ {printf "%.1f GB", $2/1048576}' /proc/meminfo)"
echo " device          : $(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo " providers       : $(python -c 'import onnxruntime as ort; print(",".join(ort.get_available_providers()))')"
echo " sweep over M    : $MEMBERS   (K=$SAMPLES, horizon=$HORIZON)"
echo " issue times     : $START .. $END"
echo "=============================================================="

for M in $MEMBERS; do
    echo
    echo ">>> M=$M members, K=$SAMPLES samples/member -> $((M * SAMPLES)) pooled samples"
    python main.py run \
        --scenario_id "$SCENARIO" \
        --forecast_horizon "$HORIZON" \
        --start_time "$START" \
        --end_time "$END" \
        --samples_per_member "$SAMPLES" \
        --num_deep_ensemble "$M"
done

echo
echo "=============================================================="
echo " Per-forecast-step cost"
echo "=============================================================="
python - "$METRICS" <<'PY'
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    sys.exit("no rows in runtime_metrics.csv -- the sweep produced no timings")

def f(row, key):
    value = row.get(key) or 0
    return float(value) if value else 0.0

groups = {}
for row in rows:
    key = (int(row["num_deep_ensemble"]), int(row["samples_per_member"]),
           int(row["num_timesteps"]))
    groups.setdefault(key, []).append(row)

print(f"{'M':>3} {'K':>3} {'M*K':>5} {'steps':>6} {'issues':>7} "
      f"{'hazard s/step':>14} {'impact s/step':>14} {'total s/step':>13}")
for (m, k, steps), group in sorted(groups.items()):
    n = len(group)
    # Per forecast STEP, not per issue: an issue covers `steps` timesteps, so
    # dividing is what makes horizons comparable to each other.
    hazard = sum(f(r, "inundation_s") for r in group) / n / steps
    impact = sum(f(r, "risk_s") for r in group) / n / steps
    total = sum(f(r, "total_s") for r in group) / n / steps
    print(f"{m:>3} {k:>3} {m*k:>5} {steps:>6} {n:>7} "
          f"{hazard:>14.2f} {impact:>14.2f} {total:>13.2f}")
PY

cp "$METRICS" "$RESULTS_DIR/runtime_metrics.csv"
echo
echo "timings written to $RESULTS_DIR/runtime_metrics.csv"
echo "(mount it out with:  -v \"\$PWD/results:$RESULTS_DIR\")"
