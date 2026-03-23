import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

WORKER_SCRIPT_NAME = "run_benchmark.py"
PYTHON_EXEC = sys.executable
SUPPORTED_PLATFORMS = ["weibo", "toutiao", "xiaohongshu", "zhihu", "douban"]


def generate_date_range(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    delta = end - start
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]


def main():
    parser = argparse.ArgumentParser(description="Run benchmark batches for date range and platform list")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--platform", type=str, choices=SUPPORTED_PLATFORMS, help="单平台运行")
    parser.add_argument("--platforms", type=str, nargs="*", choices=SUPPORTED_PLATFORMS, help="多平台运行")
    parser.add_argument("--db-base-dir", type=str, default="MARS_result/data/output")
    parser.add_argument("--tmp-db-base-dir", type=str, default="/root/sztemp")
    parser.add_argument("--bench-output-dir", type=str, default="MARS_result/data/benchmark/results")
    parser.add_argument("--daily-profiles-dir", type=str, default="MARS_result/data/benchmark/data/platform/{platform}/candidate_user/daily_profiles")
    parser.add_argument("--stopwords-path", type=str, default="StreamProfile/benchmark/resources/stop_words.txt")
    parser.add_argument("--blacklist-base-dir", type=str, default="MARS_result/data/output")
    parser.add_argument("--profile-bench-ids", type=str, default="MARS_result/data/benchmark/data/platform/weibo/profile_bench_ids.json")
    
    # 【新增】接收复用数据库的 flag 参数
    parser.add_argument("--reuse-temp-db-if-valid", action="store_true", help="复用临时数据库")

    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    script_path = current_dir / WORKER_SCRIPT_NAME
    if not script_path.exists():
        raise FileNotFoundError(f"worker script not found: {script_path}")

    dates_to_run = generate_date_range(args.start_date, args.end_date)
    platform_list = args.platforms if args.platforms else ([args.platform] if args.platform else SUPPORTED_PLATFORMS)

    print(f"📋 计划运行 {len(dates_to_run) * len(platform_list)} 个任务")

    task_index = 0
    total_tasks = len(dates_to_run) * len(platform_list)
    for target_date in dates_to_run:
        for platform in platform_list:
            task_index += 1
            print(f"\n🚀 [{task_index}/{total_tasks}] 开始处理日期: {target_date} | 平台: {platform}")
            try:
                # 【修改】将命令组装成列表，方便动态追加参数
                cmd = [
                    PYTHON_EXEC,
                    str(script_path),
                    "--date", target_date,
                    "--platform", platform,
                    "--db-base-dir", args.db_base_dir,
                    "--tmp-db-base-dir", args.tmp_db_base_dir,
                    "--bench-output-dir", args.bench_output_dir,
                    "--daily-profiles-dir", args.daily_profiles_dir,
                    "--stopwords-path", args.stopwords_path,
                    "--blacklist-base-dir", args.blacklist_base_dir,
                    "--profile-bench-ids", args.profile_bench_ids,
                ]

                # 【新增】如果总控脚本接收到了这个参数，就往下传给子脚本
                if args.reuse_temp_db_if_valid:
                    cmd.append("--reuse-temp-db-if-valid")

                subprocess.run(
                    cmd,
                    check=True
                )
                print(f"✅ 日期 {target_date} 平台 {platform} 处理完成。")
            except subprocess.CalledProcessError as e:
                print(f"❌ 日期 {target_date} 平台 {platform} 运行出错 (Exit Code: {e.returncode})")
                return
            except KeyboardInterrupt:
                print("\n🛑 用户手动停止任务。")
                return

    print("\n🎉 所有计划任务已结束。")


if __name__ == "__main__":
    main()