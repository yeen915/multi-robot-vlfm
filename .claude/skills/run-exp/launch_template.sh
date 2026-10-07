#!/bin/bash
# Launch script for one or more eval runs, written by /run-exp to outputs/eval/run_<name>.sh.
# Runs detached (setsid nohup), stops the model servers at the end, and ends its log with
# "ALL DONE". Fill in RUNS: one line per run, "<name> <method> <split> [extra args and overrides]".
# The duplicate check is fixed at 20%. Only a diagnostic run may add "--dup-fraction 0" to its
# extra args (the last value given wins).
source ~/yeen/vlfm_env.sh >/dev/null 2>&1
cd ~/yeen/vlfm || exit 1
# Started with setsid, so this script leads its own process group: stopping the run is
# "kill -- -$(cat outputs/eval/run_<name>.pgid)", which reaches the evaluation and the
# episode processes but nothing outside the group. The model servers are stopped separately.
echo $$ > "${0%.sh}.pgid"
RUNS=(
    "NAME METHOD SPLIT"
)
for spec in "${RUNS[@]}"; do
    read -r name method split extra <<< "$spec"
    s=$(date +%s)
    $VLFM_PYTHON scripts/eval_multi_robot.py --method "$method" --split "$split" --name "$name" \
        --dup-fraction 0.2 --keep-servers $extra > "outputs/eval/$name.log" 2>&1
    echo "$(date '+%F %T') $name exit=$? mins=$(( ($(date +%s) - s) / 60 ))"
    tail -2 "outputs/eval/$name.log"
done
bash ~/yeen/stop_vlm.sh >/dev/null 2>&1
sleep 3
echo "ports after stop: $(ss -ltn | grep -cE ':1218[1-4] ')"
echo "ALL DONE"
