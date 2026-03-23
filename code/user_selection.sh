#!/usr/bin/env bash
set -euo pipefail

# 用法示例：
#
#   # 1) 按条件筛选用户
#   ./user_selection.sh select \
#       --input-csv MARS_result/data/output/2025-06-15/user_profiles.csv \
#       --output-file selected.txt \
#       min_followers_count=100
#
#   # 2) 生成 oasis agent init CSV（暂无 relationship，following 全为 []）
#   ./user_selection.sh convert_profiles \
#       --profile-csv MARS_result/data/output/2025-06-15/user_profiles.csv \
#       --out-csv data/oasis/oasis_agent_init.csv \
#       --user-list selected.txt
#
#   # 3) 从平台 DB 筛选帖子并生成 oasis DB（支持多个目录）
#   ./user_selection.sh convert_posts \
#       --posts-input-dir MARS_result/data/output/2025-06-15 \
#       --oasis-db data/oasis/oasis_database.db \
#       --user-list selected.txt \
#       --calibration-end "2025-06-15 16:00:00" \
#       --ground-truth-end "2025-06-16 00:00:00"
#
#   # 4) 一键执行全部：select -> convert_profiles -> convert_posts
#   ./user_selection.sh all \
#       --input-csv MARS_result/data/output/2025-06-15/user_profiles.csv \
#       --posts-input-dir MARS_result/data/output/2025-06-15 \
#       --oasis-out-dir data/oasis \
#       --calibration-end "2025-06-15 16:00:00" \
#       --ground-truth-end "2025-06-16 00:00:00"

MODE="${1:-help}"; shift || true

PROJECT_ROOT="/remote-home/JuelinW/oasis_project"
CONDA_ACTIVATE="/remote-home/JuelinW/anaconda3/bin/activate"
CONDA_ENV="oasis"

cd "$PROJECT_ROOT"

# 尝试激活 conda（若不可用会忽略）
if [ -f "$CONDA_ACTIVATE" ]; then
  source "$CONDA_ACTIVATE" "$CONDA_ENV" 2>/dev/null || true
fi

PYTHON="$(command -v python || echo python)"

# 支持通过环境变量预设时间参数（ISO 格式），例如：
#   export CALIBRATION_END="2025-06-15 16:00:00"
#   export GROUND_TRUTH_END="2025-06-16 00:00:00"
CAL_END="${CALIBRATION_END:-}"
GT_END="${GROUND_TRUTH_END:-}"

# 将剩余参数传递给模块
ARGS=( "$MODE" "$@" )

# 若环境变量提供，则追加对应参数（避免重复）
if [ -n "$CAL_END" ]; then
  skip=false
  for a in "${ARGS[@]}"; do
    if [ "$a" = "--calibration-end" ]; then skip=true; break; fi
  done
  if [ "$skip" = false ]; then
    ARGS+=( --calibration-end "$CAL_END" )
  fi
fi

if [ -n "$GT_END" ]; then
  skip=false
  for a in "${ARGS[@]}"; do
    if [ "$a" = "--ground-truth-end" ]; then skip=true; break; fi
  done
  if [ "$skip" = false ]; then
    ARGS+=( --ground-truth-end "$GT_END" )
  fi
fi

echo "运行: python -m MARS.code.user_selection.selection_process ${ARGS[*]}"
exec "$PYTHON" -m MARS.code.user_selection.selection_process "${ARGS[@]}"
