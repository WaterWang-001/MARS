"""
终端入口：用户筛选 -> 提取子集 -> 转换为 oasis 模拟输入格式

流水线：
  1) select:            从 profile CSV 按条件筛选用户 -> selected.txt
  2) convert_profiles:  profiles -> oasis_agent_init.csv (relationship 暂跳过，following 为空)
  3) convert_posts:     从 data_process 产出的平台 DB 筛选 + 时间切分 -> oasis_database.db

快捷 all 命令一键执行以上全部步骤。

用法示例：
  # 1) 筛选用户
  python -m MARS.code.user_selection.selection_process select \
      --input-csv MARS_result/data/output/2025-06-15/user_profiles.csv \
      --output-file selected.txt \
      min_followers_count=100

  # 2) 生成 oasis agent init CSV（暂无 relationship，following 全为 []）
  python -m MARS.code.user_selection.selection_process convert_profiles \
      --profile-csv MARS_result/data/output/2025-06-15/user_profiles.csv \
      --out-csv data/oasis/oasis_agent_init.csv

  # 3) 从平台 DB 筛选帖子并生成 oasis DB
  python -m MARS.code.user_selection.selection_process convert_posts \
      --posts-input-dir MARS_result/data/output/2025-06-15 \
      --oasis-db data/oasis/oasis_database.db \
      --calibration-end "2025-06-15 16:00:00" \
      --ground-truth-end "2025-06-16 00:00:00"

  # 4) 一键执行全部
  python -m MARS.code.user_selection.selection_process all \
      --input-csv MARS_result/data/output/2025-06-15/user_profiles.csv \
      --posts-input-dir MARS_result/data/output/2025-06-15 \
      --oasis-out-dir data/oasis \
      --calibration-end "2025-06-15 16:00:00" \
      --ground-truth-end "2025-06-16 00:00:00"
"""
import sys
from pathlib import Path
import argparse
from typing import List, Optional

# 确保项目根在 sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MARS.code.user_selection.user_selection import UserSelector
from MARS.code.user_selection.oasis_user import OasisUserBuilder
from MARS.code.user_selection.oasis_post_from_processed import ProcessedPostCollector



# ========== 各步骤实现 ==========

def run_select(input_csv: str, output_file: str, filters: List[str]) -> List[str]:
    """从 profile CSV 按条件筛选用户，输出 user_id 列表文件。"""
    criteria = {}
    for f in filters:
        if '=' not in f:
            print(f"忽略无效 filter: {f}")
            continue
        k, v = f.split('=', 1)
        if v.lower() in ("true", "false"):
            vv = v.lower() == "true"
        else:
            try:
                vv = int(v)
            except Exception:
                try:
                    vv = float(v)
                except Exception:
                    vv = v
        criteria[k] = vv

    selector = UserSelector(profiles_csv_path=input_csv)
    user_ids = selector.select_users(**criteria)
    selector.save_list(user_ids, output_file)
    print(f"select -> saved {len(user_ids)} ids to {output_file}")
    return user_ids


def run_convert_profiles(profile_csv: str, out_csv: str,
                         relation_csv: Optional[str] = None,
                         user_list_file: Optional[str] = None):
    """
    从 profiles CSV 生成 oasis_agent_init.csv。
    如果有 user_list_file，先按列表筛选 profiles。
    relationship 暂跳过（following_agentid_list 全为 []）。
    """
    import pandas as pd

    # 如果指定了用户列表，先筛选 profiles
    if user_list_file:
        with open(user_list_file, 'r', encoding='utf-8') as f:
            user_ids = {line.strip() for line in f if line.strip()}
        df = pd.read_csv(profile_csv, dtype={'user_id': str})
        df = df[df['user_id'].isin(user_ids)].copy()
        # 写临时文件给 OasisUserBuilder
        filtered_csv = str(Path(out_csv).parent / "_filtered_profiles.csv")
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filtered_csv, index=False)
        profile_csv = filtered_csv

    if relation_csv and Path(relation_csv).exists():
        builder = OasisUserBuilder(profile_csv=profile_csv, relation_csv=relation_csv, output_csv=out_csv)
    else:
        # 无 relationship：生成空的 relation CSV
        import tempfile
        import pandas as pd
        empty_rel = Path(out_csv).parent / "_empty_relations.csv"
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["user_id", "all_followed_user_ids"]).to_csv(
            str(empty_rel), index=False)
        builder = OasisUserBuilder(profile_csv=profile_csv, relation_csv=str(empty_rel), output_csv=out_csv)

    builder.run()
    print(f"convert_profiles -> {out_csv}")


def run_convert_posts(posts_input_dirs: List[str], oasis_db: str,
                      user_list_file: Optional[str] = None,
                      calibration_end: Optional[str] = None,
                      ground_truth_end: Optional[str] = None):
    """
    从 data_process 产出的平台 DB 筛选帖子，写入 oasis DB 并按时间切分。
    """
    user_list = None
    if user_list_file:
        with open(user_list_file, 'r', encoding='utf-8') as f:
            user_list = [line.strip() for line in f if line.strip()]

    collector = ProcessedPostCollector(
        input_dirs=posts_input_dirs,
        oasis_db=oasis_db,
        user_list=user_list,
        calibration_end=calibration_end,
        ground_truth_end=ground_truth_end,
    )
    collector.run()
    print(f"convert_posts -> {oasis_db}")


# ========== CLI ==========

def cli():
    p = argparse.ArgumentParser(description="用户筛选 + 转换到 oasis 模拟输入格式")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- select ---
    sel = sub.add_parser("select", help="从 profile CSV 筛选用户并保存 user list")
    sel.add_argument("--input-csv", required=True, help="user_profiles.csv 路径")
    sel.add_argument("--output-file", required=True, help="输出的 user_id 列表文件")
    sel.add_argument("filters", nargs="*", help="过滤条件 key=value (如 min_followers_count=100)")

    # --- convert_profiles ---
    cp = sub.add_parser("convert_profiles", help="生成 oasis_agent_init.csv")
    cp.add_argument("--profile-csv", required=True, help="user_profiles.csv 路径")
    cp.add_argument("--out-csv", required=True, help="输出 oasis_agent_init.csv 路径")
    cp.add_argument("--relation-csv", default=None, help="follow_list.csv（可选，暂无则跳过）")
    cp.add_argument("--user-list", default=None, help="user_id 列表文件（可选，用于筛选 profiles）")

    # --- convert_posts ---
    cpp = sub.add_parser("convert_posts", help="从平台 DB 筛选帖子并生成 oasis DB")
    cpp.add_argument("--posts-input-dir", required=True, nargs="+",
                     help="data_process 输出目录（含 posts_*.db），可指定多个")
    cpp.add_argument("--oasis-db", required=True, help="输出 oasis DB 路径")
    cpp.add_argument("--user-list", default=None, help="user_id 列表文件（可选）")
    cpp.add_argument("--calibration-end", default=None,
                     help="calibration 截止时间 (如 '2025-06-15 16:00:00')")
    cpp.add_argument("--ground-truth-end", default=None,
                     help="ground truth 截止时间 (如 '2025-06-16 00:00:00')")

    # --- all ---
    allp = sub.add_parser("all", help="select -> convert_profiles -> convert_posts 一键执行")
    allp.add_argument("--input-csv", required=True, help="user_profiles.csv 路径")
    allp.add_argument("--posts-input-dir", required=True, nargs="+",
                      help="data_process 输出目录（含 posts_*.db），可指定多个")
    allp.add_argument("--oasis-out-dir", default=str(REPO_ROOT / "data" / "oasis"),
                      help="oasis 输出目录")
    allp.add_argument("--output-file", default=None,
                      help="select 输出的 user_id 列表文件（默认: oasis-out-dir/selected.txt）")
    allp.add_argument("--calibration-end", default=None,
                      help="calibration 截止时间")
    allp.add_argument("--ground-truth-end", default=None,
                      help="ground truth 截止时间")
    allp.add_argument("filters", nargs="*", help="select 过滤条件 key=value")

    args = p.parse_args()

    if args.cmd == "select":
        run_select(args.input_csv, args.output_file, args.filters)

    elif args.cmd == "convert_profiles":
        run_convert_profiles(args.profile_csv, args.out_csv,
                             relation_csv=args.relation_csv,
                             user_list_file=args.user_list)

    elif args.cmd == "convert_posts":
        run_convert_posts(args.posts_input_dir, args.oasis_db,
                          user_list_file=args.user_list,
                          calibration_end=args.calibration_end,
                          ground_truth_end=args.ground_truth_end)

    elif args.cmd == "all":
        oasis_out = Path(args.oasis_out_dir)
        oasis_out.mkdir(parents=True, exist_ok=True)

        # 1. select
        output_file = args.output_file or str(oasis_out / "selected.txt")
        user_ids = run_select(args.input_csv, output_file, args.filters or [])

        # 2. convert_profiles (无 relationship，following 全为 [])
        out_csv = str(oasis_out / "oasis_agent_init.csv")
        run_convert_profiles(args.input_csv, out_csv,
                             user_list_file=output_file)

        # 3. convert_posts
        oasis_db = str(oasis_out / "oasis_database.db")
        run_convert_posts(args.posts_input_dir, oasis_db,
                          user_list_file=output_file,
                          calibration_end=args.calibration_end,
                          ground_truth_end=args.ground_truth_end)

        print(f"\n=== 全部完成 ===")
        print(f"  agent init:  {out_csv}")
        print(f"  oasis db:    {oasis_db}")

    else:
        print("未知命令", args.cmd)


if __name__ == "__main__":
    cli()
