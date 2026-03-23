#!/bin/bash
# 用法: ./process_date_range.sh [开始日期 YYYY-MM-DD] [结束日期 YYYY-MM-DD]
# 例如: ./process_date_range.sh 2025-06-15 2025-06-26
#
# 此脚本会顺序处理指定时间段内的每一天。

# --- 配置 ---

# 项目和 Conda 路径 (保持与原脚本一致)
PROJECT_ROOT="/remote-home/JuelinW/oasis_project"
CONDA_ACTIVATE="/remote-home/JuelinW/anaconda3/bin/activate"
CONDA_ENV="oasis"

# --- 脚本主体 ---

# 1. 检查参数
if [ $# -ne 2 ]; then
  echo "错误: 需要提供开始日期和结束日期。" >&2
  echo "用法: $0 [开始日期 YYYY-MM-DD] [结束日期 YYYY-MM-DD]" >&2
  exit 1
fi

START_DATE="$1"
END_DATE="$2"

# 验证日期格式 (简单检查)
if ! date -d "$START_DATE" +%Y-%m-%d > /dev/null 2>&1 || ! date -d "$END_DATE" +%Y-%m-%d > /dev/null 2>&1; then
    echo "错误: 日期格式无效。请使用 YYYY-MM-DD 格式。" >&2
    exit 1
fi

echo "准备顺序处理从 $START_DATE 到 $END_DATE 的数据..."
echo "---"

# 2. 激活 Conda 环境 (顺序执行只需在主进程激活一次即可)
if [ -f "$CONDA_ACTIVATE" ]; then
    source "$CONDA_ACTIVATE" "$CONDA_ENV"
    echo "[系统] Conda 环境 $CONDA_ENV 已激活。"
else
    echo "[系统] 警告: 未找到 conda activate 脚本。依赖外部激活的环境。" >&2
fi

# 进入项目根目录
cd "$PROJECT_ROOT" || { echo "错误: 无法进入目录 $PROJECT_ROOT"; exit 1; }

# 3. 日期循环处理逻辑
current_date="$START_DATE"

# 当 current_date <= END_DATE 时循环
# 注意：在 Bash 中直接比较 YYYY-MM-DD 字符串的字典序等同于日期比较
while [[ "$current_date" < "$END_DATE" ]] || [[ "$current_date" == "$END_DATE" ]]; do
    
    # 定义当前日期的路径
    RAW_DIR="$PROJECT_ROOT/data/raw/$current_date"
    OUT_DIR="$PROJECT_ROOT/MARS_result/data/output/$current_date"

    echo "-------------------------------------------"
    echo "[日期: $current_date] 开始处理..."
    echo "[日期: $current_date] 输入: $RAW_DIR"
    echo "[日期: $current_date] 输出: $OUT_DIR"

    # 检查输入目录
    if [ ! -d "$RAW_DIR" ]; then
        echo "[日期: $current_date] 错误: 输入目录不存在，跳过此日期。" >&2
    else
        # 创建输出目录
        mkdir -p "$OUT_DIR"

        # 运行 Python 脚本
        echo "[日期: $current_date] 运行: python MARS/code/data_process/data_process.py all --input \"$RAW_DIR\" --out \"$OUT_DIR\""
        
        python MARS/code/data_process/data_process.py all --input "$RAW_DIR" --out "$OUT_DIR"
        
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "[日期: $current_date] 处理成功。"
        else
            echo "[日期: $current_date] 失败 (退出码: $exit_code)。"
            # 如果某一天失败，你可以选择 exit 1 停止整个脚本，或者继续
            # exit 1 
        fi
    fi

    # 4. 计算下一天的日期 (使用 GNU date)
    current_date=$(date -d "$current_date + 1 day" +%Y-%m-%d)
done

echo "---"
echo "指定时间段内 ($START_DATE 至 $END_DATE) 的所有任务已处理完毕。"