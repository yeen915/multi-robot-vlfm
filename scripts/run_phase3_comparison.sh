#!/bin/bash
# Phase 3 comparison on the dev split (100 episodes, 3 map sharing modes), then a
# rerun of 5 randomly chosen episodes in every mode (data/multi_robot/phase3_recheck_split.json,
# drawn from dev with seed 20261005) to check that reruns give identical results.
# Takes 10+ hours, so start it detached from the repo root:
#     mkdir -p outputs/eval
#     setsid nohup scripts/run_phase3_comparison.sh > outputs/eval/phase3_comparison.log 2>&1 &
# The last line of the log is "ALL DONE" when it has finished. Then:
#     python scripts/summarize_multi_robot.py independent=outputs/eval/p3cmp_independent/main \
#         shared_maps_only=outputs/eval/p3cmp_shared_maps_only/main shared=outputs/eval/p3cmp_shared_vlfm/main
source ~/yeen/vlfm_env.sh >/dev/null 2>&1
cd "$(dirname "$0")/.." || exit 1
for m in independent shared_maps_only shared_vlfm; do
    s=$(date +%s)
    $VLFM_PYTHON scripts/eval_multi_robot.py --method $m --split dev --dup-fraction 0 \
        --name p3cmp_$m --keep-servers > outputs/eval/p3cmp_$m.log 2>&1
    echo "$(date '+%F %T') $m exit=$? mins=$(( ($(date +%s) - s) / 60 ))"
    tail -2 outputs/eval/p3cmp_$m.log
done
for m in independent shared_maps_only shared_vlfm; do
    $VLFM_PYTHON scripts/eval_multi_robot.py --method $m --split-file data/multi_robot/phase3_recheck_split.json \
        --split recheck --dup-fraction 0 --name p3recheck_$m --keep-servers > outputs/eval/p3recheck_$m.log 2>&1
    echo "$(date '+%F %T') recheck $m exit=$?"
    tail -1 outputs/eval/p3recheck_$m.log
done
bash ~/yeen/stop_vlm.sh >/dev/null 2>&1
sleep 3
echo "ports after stop: $(ss -ltn | grep -c -E ':1218[1-4]')"
echo "ALL DONE"
