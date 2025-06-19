#!/usr/bin/env bash
export PS4='+${BASH_SOURCE}:${LINENO}: '
trap 'echo "⚠️ ERROR at $0:$LINENO (exit code $?)"' ERR
set -euo pipefail
set -x

echo "✨ SCRIPT STARTED — line $LINENO"

# ------------------------------------------------------------------
# 0) Check that OPENAI_API_KEY is set
# ------------------------------------------------------------------
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "❗ ERROR: you must export OPENAI_API_KEY before running."
  echo "   export OPENAI_API_KEY=\"sk-...\""
  exit 1
fi

# ------------------------------------------------------------------
# 1) Define your Schedulers & Allocators (codes only)
# ------------------------------------------------------------------
SCHEDULERS=(monolithic big_dag dag)
ALLOCATORS=(lp llm)

# Mapping functions (code → pretty name)
get_sched_name() {
  case "$1" in
    monolithic)  printf 'Monolithic Prompt' ;;
    big_dag)     printf 'Big DAG' ;;
    dag)         printf 'DAG' ;;
    *)           printf '%s' "$1" ;;
  esac
}
get_alloc_name() {
  case "$1" in
    lp)  printf 'Linear Programming' ;;
    llm) printf 'LLM' ;;
    *)   printf '%s' "$1" ;;
  esac
}

# ------------------------------------------------------------------
# 2) Define your 2 scenarios (as "Label:GoalCount:yaml=cnt,…") 
#    and their human-readable goal descriptions
# ------------------------------------------------------------------
SCENARIOS=(
  "3 Robots / 5 Goals:5:robot_fleet/robots/examples/moma/moma.yaml=1,robot_fleet/robots/examples/nav/nav.yaml=1,robot_fleet/robots/examples/pick_place/pick_place.yaml=1"
  "5 Robots / 10 Goals:10:robot_fleet/robots/examples/moma/moma.yaml=1,robot_fleet/robots/examples/nav/nav.yaml=2,robot_fleet/robots/examples/pick_place/pick_place.yaml=2"
)

# For each scenario, a comma-separated list of real task descriptions.
# Replace these with whatever goals you actually want.
SCENARIO_GOAL_LABELS=(
  "make breakfast,set the table,wash dishes,brew coffee,feed the cat"
  "vacuum living room,do laundry,water plants,check email,charge phone,call mom,organize desk,plan week,meditate,stretch"
)

# ------------------------------------------------------------------
# 3) Run each scenario
# ------------------------------------------------------------------
scen_idx=0
for scen in "${SCENARIOS[@]}"; do
  scen_idx=$((scen_idx + 1))

  # split scen into label / goal_count / dist_str
  label="${scen%%:*}"
  rest="${scen#*:}"
  goal_count="${rest%%:*}"
  dist_str="${rest#*:}"

  echo
  echo "=============================================="
  echo "▶ SCENARIO #$scen_idx: $label"
  echo "----------------------------------------------"

  # Register robots
  echo "→ Registering robots for ${label}..."
  IFS=, read -ra parts <<< "$dist_str"
  for p in "${parts[@]}"; do
    yaml="${p%%=*}"
    cnt="${p#*=}"
    echo "   • robotctl register $yaml -n $cnt"
    robotctl register "$yaml" -n "$cnt"
  done

  # Add human‑readable goals
  echo "→ Adding $goal_count goals…"
  GOALS=()
  labels="${SCENARIO_GOAL_LABELS[$((scen_idx-1))]}"
  IFS=, read -ra DESCS <<< "$labels"
  for desc in "${DESCS[@]}"; do
    out=$(robotctl goal add "$desc")
    id=$(printf '%s' "$out" | grep -oE 'Goal ID: [0-9]+' | awk '{print $3}')
    echo "   • added Goal ID $id for \"$desc\""
    GOALS+=("$id")
  done

  # For each scheduler × allocator
  for sched in "${SCHEDULERS[@]}"; do
    for alloc in "${ALLOCATORS[@]}"; do
      echo
      echo "  → [$(get_sched_name "$sched") / $(get_alloc_name "$alloc")]"

      goal_ids="$(IFS=,; echo "${GOALS[*]}")"
      create_out=$(robotctl plan create "$sched" "$alloc" "$goal_ids")
      echo "     • plan create →"
      echo "$create_out" | sed 's/^/       /'

      plan_id=$(printf '%s' "$create_out" | grep -oE 'Plan: [0-9]+' | awk '{print $2}')
      echo "     • got Plan ID = $plan_id"

      analysis=$(robotctl plan get "$plan_id" --analyze-idle)
      echo "     • plan get --analyze-idle →"
      echo "$analysis" | sed 's/^/       /'

      avg_idle=$(printf '%s' "$analysis" \
        | grep -oE 'Average idle time across all robots: [0-9]+\.[0-9]+%' \
        | awk '{print $7}' | tr -d '%')
      echo "     • Average idle = $avg_idle%"

      # store into var named result_<sched>_<alloc>_<scen_idx>
      var="result_${sched}_${alloc}_${scen_idx}"
      printf -v "$var" '%s' "$avg_idle"
    done
  done
done

# ------------------------------------------------------------------
# 4) Emit CSV
# ------------------------------------------------------------------
OUT="multi_robot_results.csv"
echo
echo "Writing CSV → $OUT"
{
  # header
  printf "Scheduler,Allocator"
  for scen in "${SCENARIOS[@]}"; do
    lbl="${scen%%:*}"
    printf ",Idle Time @ %s" "$lbl"
  done
  printf "\n"

  # rows
  for sched in "${SCHEDULERS[@]}"; do
    for alloc in "${ALLOCATORS[@]}"; do
      printf "%s,%s" "$(get_sched_name "$sched")" "$(get_alloc_name "$alloc")"
      for idx in $(seq 1 "${#SCENARIOS[@]}"); do
        var="result_${sched}_${alloc}_${idx}"
        val="${!var:-N/A}"
        printf ",%s" "$val"
      done
      printf "\n"
    done
  done
} > "$OUT"

echo
echo "✅ Done! Open '${OUT:-multi_robot_results.csv}' in your spreadsheet tool."
