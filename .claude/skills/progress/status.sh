#!/bin/bash
# Status of experiments, model servers, GPU and the server's health, for /progress.
# Read-only: it starts and stops nothing. Run from the repo root.
cd "$(git rev-parse --show-toplevel)" || exit 1
source ~/yeen/vlfm_env.sh >/dev/null 2>&1
PY=${VLFM_PYTHON:-~/anaconda3/envs/vlfm/bin/python}
echo "== $(date '+%F %T')"

echo; echo "== Running experiments"
# Only real python processes, not shells whose command line mentions the script
running=$(ps -eo args= | grep -E '^[^ ]*python[^ ]* scripts/eval_multi_robot.py' | sed -E 's/^[^ ]* //' | cut -c1-160)
echo "${running:-none}"
ps -eo args= | grep -qE '^[^ ]*python[^ ]* -m vlfm.run ' && echo "(habitat episode process running)"

echo; echo "== Runs under outputs/eval (newest first): episodes done in main/ and dup/, integrity"
for d in $(ls -dt outputs/eval/*/ 2>/dev/null | head -${RUNS:-8}); do
    n=$(find "$d/main" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    k=$(find "$d/dup" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    integ=""
    if [ -f "$d/integrity.json" ]; then
        integ=$($PY - "$d/integrity.json" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))
print(f"of {g.get('num_episodes')} | hw_corruption {len(g.get('hw_corruption', []))}"
      f" | no result {len(g.get('missing_after_run', []))}"
      f" | dup mismatches {len(g.get('duplicate_mismatches', []))}/{len(g.get('duplicated_episodes', []))}"
      f" | unfinished {len(g.get('unfinished_main', [])) + len(g.get('unfinished_dup', []))}")
PY
)
    fi
    echo "$(basename $d): main $n, dup $k $integ  (updated $(date -r $d '+%m-%d %H:%M'))"
done

echo; echo "== Errors in the newest run's logs (last 5)"
latest=$(ls -dt outputs/eval/*/ 2>/dev/null | head -1)
[ -n "$latest" ] && grep -hE "Traceback|Error:|Corrupted index|out of bounds" "$latest"/*/logs/*.log 2>/dev/null | grep -v Metadata | tail -5

echo; echo "== Model servers (ports 12181-12184 listening)"
echo "$(ss -ltn | grep -cE ':1218[1-4] ') of 4"

echo; echo "== GPU"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

echo; echo "== Memory faults in the kernel log since boot (segfault / general protection)"
journalctl -k -b 2>/dev/null | grep -ciE "segfault|general protection" || dmesg 2>/dev/null | grep -ciE "segfault|general protection"

echo; echo "== Tailscale"
[ -x ~/yeen/tailscale/status.sh ] && ~/yeen/tailscale/status.sh 2>&1 | head -3

echo; echo "== Git"
git status --short | head -10
echo "unpushed commits: $(git rev-list --count backup/multi-robot..HEAD 2>/dev/null)"
git log --oneline -3
