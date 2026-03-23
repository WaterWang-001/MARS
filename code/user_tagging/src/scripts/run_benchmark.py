import argparse
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent          # src/scripts/
PACKAGE_ROOT = CURRENT_DIR.parent.parent               # user_tagging/
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.core.api_client import APIClient
from src.core.db_client import DBClient
from src.core.io_helpers import load_config, load_csv, load_processed_ids
from src.core.prompt_manager import PromptManager
from src.scripts.run_blacklist import generate_blacklist, generate_tag_stats
from src.platforms import DoubanFilter, ToutiaoFilter, WeiboFilter, XiaohongshuFilter, ZhihuFilter
from src.tasks.benchmark import BenchGenerationService, BenchStateManager

SUPPORTED_PLATFORMS = ["weibo", "toutiao", "xiaohongshu", "zhihu", "douban"]

write_lock = threading.Lock()
log_lock = threading.Lock()
print_lock = threading.Lock()


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KB"
    if size < 1024**3:
        return f"{size / (1024**2):.1f} MB"
    return f"{size / (1024**3):.1f} GB"


def _copy_file_with_progress(source: Path, target: Path, chunk_size: int = 8 * 1024 * 1024, log_interval: float = 2.0):
    total_size = source.stat().st_size
    copied = 0
    start_time = time.time()
    last_log_time = 0.0

    print(
        f"📦 Copying DB to temp: {source} -> {target} | Total: {_format_bytes(total_size)}",
        flush=True,
    )

    with source.open("rb") as src, target.open("wb") as dst:
        while True:
            block = src.read(chunk_size)
            if not block:
                break
            dst.write(block)
            copied += len(block)

            now = time.time()
            if (now - last_log_time) >= log_interval or copied >= total_size:
                elapsed = max(now - start_time, 1e-6)
                speed = copied / elapsed
                progress = (copied / total_size * 100) if total_size > 0 else 100.0
                print(
                    f"   ↳ Copy Progress: {progress:6.2f}% | {_format_bytes(copied)}/{_format_bytes(total_size)} | {_format_bytes(int(speed))}/s",
                    flush=True,
                )
                last_log_time = now

    shutil.copystat(source, target)
    print("✅ Temp DB copy completed.", flush=True)


def get_platform_filter(platform: str, platform_config: dict, blacklist_file: str | None = None):
    mapping = {
        "weibo": WeiboFilter(config=platform_config, blacklist_file=blacklist_file),
        "toutiao": ToutiaoFilter(config=platform_config, blacklist_file=blacklist_file),
        "xiaohongshu": XiaohongshuFilter(config=platform_config, blacklist_file=blacklist_file),
        "zhihu": ZhihuFilter(config=platform_config, blacklist_file=blacklist_file),
        "douban": DoubanFilter(config=platform_config, blacklist_file=blacklist_file),
    }
    return mapping[platform]


def process_single_user_bench(service, user_data, batch_date, resume_log_path, user_posts):
    user_id = str(user_data.get("user_id"))
    try:
        result = service.process_user_incremental(user_data, user_posts, batch_date)
        status = result.get("status", "error")

        with print_lock:
            if status in {"success", "success_skipped_llm"}:
                category = result.get("category", "Unknown")
                suffix = " (Skip LLM)" if status == "success_skipped_llm" else ""
                print(f"✅ BENCH SAVED{suffix} | Category: {category}")
            elif status == "evicted_density_drop":
                print(f"📉 [Evicted] User {user_id}: Density dropped below threshold. Buffer cleared.")
            elif status == "evicted_behavior":
                print(f"🚫 [Evicted] User {user_id}: Failed behavior risk check.")
            elif status == "buffered_qualified":
                count = result.get("count", 0)
                print(f"💧 [Buffer] User {user_id}: Qualified density. Count: {count}")
            elif status == "error_llm":
                print(f"❌ [Error] User {user_id}: LLM Inference Failed.")

        if status != "error":
            with log_lock:
                with open(resume_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{user_id}\n")

        return status
    except Exception as e:
        with print_lock:
            print(f"❌ [User: {user_id}] Critical Error: {e}")
            print(traceback.format_exc())
        return "error"


def _prepare_temp_db(
    target_date: str,
    platform: str,
    db_base_dir: Path,
    tmp_db_base: Path,
    reuse_if_valid: bool = False,
) -> Path:
    date_dir = db_base_dir / target_date
    preferred_db = date_dir / f"posts_{platform}.db"
    legacy_db = date_dir / f"user_post_{platform}.db"
    source_db = preferred_db if preferred_db.exists() else legacy_db
    if not source_db.exists():
        raise FileNotFoundError(
            f"Source DB not found. Tried: {preferred_db} and {legacy_db}"
        )

    tmp_db_dir = tmp_db_base / platform / target_date
    tmp_db_dir.mkdir(parents=True, exist_ok=True)
    tmp_db = tmp_db_dir / f"user_post_{platform}.db"

    if reuse_if_valid and tmp_db.exists():
        source_size = source_db.stat().st_size
        temp_size = tmp_db.stat().st_size
        if temp_size == source_size:
            print(
                f"♻️ Reusing existing temp DB (size matched): {tmp_db} | {_format_bytes(temp_size)}",
                flush=True,
            )
            return tmp_db
        print(
            f"⚠️ Existing temp DB size mismatch, recopying. temp={_format_bytes(temp_size)} vs source={_format_bytes(source_size)}",
            flush=True,
        )

    _copy_file_with_progress(source_db, tmp_db)
    return tmp_db


def _cleanup_temp_db(tmp_db_path: Path):
    try:
        if tmp_db_path.exists():
            tmp_db_path.unlink()
            tmp_dir = tmp_db_path.parent
            if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                tmp_dir.rmdir()
    except Exception:
        pass


def main():
    base_dir = CURRENT_DIR

    parser = argparse.ArgumentParser(description="Run Bench Generation for a specific date")
    parser.add_argument("--date", type=str, required=True, help="Target date in YYYY-MM-DD format")
    parser.add_argument("--platform", type=str, required=True, choices=SUPPORTED_PLATFORMS, help="目标平台")
    parser.add_argument("--db-base-dir", type=str, default="MARS_result/data/output", help="源数据库根目录")
    parser.add_argument("--tmp-db-base-dir", type=str, default="/root/sztemp", help="临时数据库根目录")
    parser.add_argument(
        "--reuse-temp-db-if-valid",
        action="store_true",
        help="若临时 DB 已存在且大小与源 DB 一致，则跳过复制并复用",
    )
    parser.add_argument("--bench-output-dir", type=str, default="MARS_result/data/benchmark/results", help="Bench 输出目录")
    parser.add_argument(
        "--daily-profiles-dir",
        type=str,
        default="MARS_result/data/benchmark/data/platform/{platform}/candidate_user/daily_profiles",
        help="每日 bench 用户 CSV 目录，支持 {platform} 模板",
    )
    parser.add_argument(
        "--stopwords-path",
        type=str,
        default="StreamProfile/benchmark/resources/stop_words.txt",
        help="停用词文件路径",
    )
    parser.add_argument("--blacklist-base-dir", type=str, default="MARS_result/data/output", help="黑名单目录根路径")
    parser.add_argument(
        "--profile-bench-ids",
        type=str,
        default="MARS_result/data/benchmark/data/platform/weibo/profile_bench_ids.json",
        help="微博 profile bench 白名单路径（支持 .json 或 .csv）",
    )
    parser.add_argument("--profile-bench-csv", type=str, default="", help="[兼容旧参数] 微博 profile bench 白名单 CSV 路径")
    args = parser.parse_args()

    target_date = args.date
    platform = args.platform

    platform_config_path = base_dir / "configs" / f"{platform}_bench.yaml"
    if not platform_config_path.exists():
        raise FileNotFoundError(f"platform config not found: {platform_config_path}")
    platform_config = load_config(str(platform_config_path))

    db_base_dir = Path(args.db_base_dir)
    tmp_db_base_dir = Path(args.tmp_db_base_dir)
    daily_profiles_dir = Path(args.daily_profiles_dir.format(platform=platform))
    blacklist_base_dir = Path(args.blacklist_base_dir)

    if platform == "weibo":
        if args.profile_bench_ids:
            platform_config["profile_bench_ids_path"] = args.profile_bench_ids
        elif args.profile_bench_csv:
            platform_config["profile_bench_ids_path"] = args.profile_bench_csv
        else:
            default_profile_json = Path("MARS_result/data/benchmark/data/platform/weibo/profile_bench_ids.json")
            if default_profile_json.exists():
                platform_config["profile_bench_ids_path"] = str(default_profile_json)

    ds_config = platform_config.get("data_source", {})
    filename_template = ds_config.get("user_pool_filename", f"{{date}}_{platform}_bench_users.csv")
    actual_filename = filename_template.format(date=target_date)
    user_csv_path = daily_profiles_dir / actual_filename

    fetch_cfg = platform_config.get("fetch_limits", {})
    max_count = int(fetch_cfg.get("max_count", 10))

    bench_output_dir = str(Path(args.bench_output_dir))
    platform_output_dir = os.path.join(bench_output_dir, "platform", platform)
    os.makedirs(platform_output_dir, exist_ok=True)

    state_db_path = os.path.join(platform_output_dir, f"bench_state_buffer_{platform}.db")
    log_dir = os.path.join(platform_output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    resume_log_path = os.path.join(log_dir, f"processed_bench_users_{platform}_{target_date}.log")

    print(f"📅 Current Bench Batch Date: {target_date}")
    print(f"🌐 Platform: {platform}")
    print(f"📄 User Pool File: {user_csv_path}")
    print(f"📌 Fetch Limit max_count: {max_count}")

    temp_db_path = _prepare_temp_db(
        target_date,
        platform,
        db_base_dir,
        tmp_db_base_dir,
        reuse_if_valid=args.reuse_temp_db_if_valid,
    )
    post_db_path = str(temp_db_path)

    blacklist_tags_path = None
    blacklist_cfg = platform_config.get("blacklist", {})
    blacklist_output_dir = blacklist_base_dir / target_date / "blacklist" / platform

    default_hashtag_patterns = {
        "weibo": r"#([^#\n]+)#",
        "xiaohongshu": r"#([^\s#]+)",
        "toutiao": r"#([^\s#]+)",
        "zhihu": r"#([^\s#]+)",
        "douban": r"#([^\s#]+)",
    }
    hashtag_pattern_str = blacklist_cfg.get("hashtag_pattern") or default_hashtag_patterns.get(platform, r"#([^\s#]+)")
    sample_size = int(blacklist_cfg.get("sample_size", 500000))

    stats_platform_filter = get_platform_filter(platform, platform_config, blacklist_file=None)

    print(f"📊 平台 [{platform}] 生成每日标签抽样统计...")
    generate_tag_stats(
        db_path=post_db_path,
        target_date=target_date,
        output_dir=str(blacklist_output_dir),
        hashtag_pattern_str=hashtag_pattern_str,
        sample_size=sample_size,
        platform_filter=stats_platform_filter,
    )

    if blacklist_cfg.get("enable", False):
        print(f"🛡️ 平台 [{platform}] 开启黑名单过滤...")
        blacklist_txt_file = blacklist_output_dir / f"blacklist_{target_date}.txt"

        if blacklist_txt_file.exists():
            blacklist_tags_path = str(blacklist_txt_file)
            print(f"♻️ 复用已有黑名单文件: {blacklist_tags_path}")
        else:
            txt_file, _ = generate_blacklist(
                db_path=post_db_path,
                target_date=target_date,
                output_dir=str(blacklist_output_dir),
                hashtag_pattern_str=hashtag_pattern_str,
                sample_size=sample_size,
                blacklist_threshold_percent=blacklist_cfg.get("threshold_percent", 0.02),
            )
            blacklist_tags_path = txt_file
    else:
        print(f"⏩ 平台 [{platform}] 未启用黑名单过滤，仅保留抽样统计。")

    platform_filter = get_platform_filter(platform, platform_config, blacklist_file=blacklist_tags_path)

    llm_cfg = platform_config["llm_service"]
    max_workers = llm_cfg.get("max_workers", 10)

    db_client = DBClient(post_db_path)
    state_manager = BenchStateManager(state_db_path)

    prompt_manager = PromptManager(str(CURRENT_DIR / "prompts"))
    api_client = APIClient(
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg["base_url"],
        model_name=llm_cfg["model_name"],
        embedding_model_path=llm_cfg["embedding_model_path"],
        mode=llm_cfg.get("mode", "remote"),
        timeout=llm_cfg.get("timeout", 120),
    )

    service = BenchGenerationService(
        state_manager=state_manager,
        prompt_manager=prompt_manager,
        api_client=api_client,
        stopwords_path=str(Path(args.stopwords_path)),
        output_dir=platform_output_dir,
        config=platform_config,
        platform_filter=platform_filter,
        blacklist_tags_path=blacklist_tags_path,
    )

    print(f"=== Starting Benchmark Generation | Workers: {max_workers} ===")

    print(f"[{time.strftime('%X')}] 📂 Loading user CSV...", end=" ", flush=True)
    df_users = load_csv(str(user_csv_path))
    print(f"Done! Loaded {len(df_users)} users.", flush=True)

    if df_users.empty:
        print("⚠️ CSV is empty. Exit.")
        _cleanup_temp_db(Path(temp_db_path))
        return

    all_users_raw = df_users.to_dict("records")
    processed_ids = load_processed_ids(resume_log_path)
    pending_users = [u for u in all_users_raw if str(u.get("user_id", "")).strip() not in processed_ids]

    print(f"🚀 Remaining tasks: {len(pending_users)}")
    if not pending_users:
        print(f"🧹 Performing Anchor Statistics Maintenance for [{platform}]...")
        anchor_stats_dir = os.path.join(platform_output_dir, "daily_anchors")
        anchor_stats_csv_path = os.path.join(anchor_stats_dir, f"daily_anchors_{platform}_{target_date}.csv")
        state_manager.reset_daily_anchor_stats(export_csv_path=anchor_stats_csv_path)
        print(f"🗑️ Cleaning up temporary DB for [{platform}]: {temp_db_path}")
        _cleanup_temp_db(Path(temp_db_path))
        return

    macro_batch_size = 500
    stats = {
        "success": 0,
        "success_skipped_llm": 0,
        "rejected_by_selector": 0,
        "evicted_behavior": 0,
        "evicted_density_drop": 0,
        "skipped_low_density": 0,
        "buffered_qualified": 0,
        "error_llm": 0,
        "error": 0,
    }
    total_processed = 0

    for i in range(0, len(pending_users), macro_batch_size):
        macro_start = time.perf_counter()
        current_macro_batch = pending_users[i : i + macro_batch_size]
        print(f"\n[{time.strftime('%X')}] 🔄 Macro Batch {i // macro_batch_size + 1} ({len(current_macro_batch)} users)...")

        target_uids = [str(u["user_id"]) for u in current_macro_batch]
        batch_posts_map = {}
        print("   ⚡️ Fetching posts...", end=" ", flush=True)
        fetch_start = time.perf_counter()
        db_chunk_size = 50
        for j in range(0, len(target_uids), db_chunk_size):
            chunk = target_uids[j : j + db_chunk_size]
            partial_map = db_client.get_batch_posts(chunk, max_count=max_count)
            batch_posts_map.update(partial_map)
        fetch_elapsed = time.perf_counter() - fetch_start
        print(f"Done. ({fetch_elapsed:.2f}s)", flush=True)

        process_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for u in current_macro_batch:
                uid = str(u["user_id"])
                u_posts = batch_posts_map.get(uid, [])
                ft = executor.submit(process_single_user_bench, service, u, target_date, resume_log_path, u_posts)
                futures[ft] = uid

            batch_done = 0
            for f in as_completed(futures):
                res = f.result()
                batch_done += 1
                total_processed += 1
                stats[res] = stats.get(res, 0) + 1
                if total_processed % 100 == 0:
                    saved_count = stats.get("success", 0) + stats.get("success_skipped_llm", 0)
                    print(
                        f"   Progress: {batch_done}/{len(current_macro_batch)} | Saved: {saved_count} | Evicted: {stats.get('evicted_density_drop', 0)}",
                        flush=True,
                    )
        process_elapsed = time.perf_counter() - process_start
        macro_elapsed = time.perf_counter() - macro_start

        del batch_posts_map
        print(
            f"   ⏱️ Timing | Fetch: {fetch_elapsed:.2f}s | Process: {process_elapsed:.2f}s | Total: {macro_elapsed:.2f}s",
            flush=True,
        )
        print("   ✅ Macro Batch Completed.")

    total_saved = stats.get("success", 0) + stats.get("success_skipped_llm", 0)
    print(f"\n{'=' * 30}")
    print(f"📊 Benchmark Generation Summary ({target_date} | {platform})")
    print(f"✅ Bench Saved:       {total_saved} (LLM Skipped: {stats.get('success_skipped_llm', 0)})")
    print(f"🚫 Evicted (Behavior): {stats.get('evicted_behavior', 0)}")
    print(f"📉 Evicted (Purged):  {stats.get('evicted_density_drop', 0)}")
    print(f"💧 Buffered (Wait):   {stats.get('buffered_qualified', 0)}")
    print(f"⏩ Skipped (Density): {stats.get('skipped_low_density', 0)}")
    print(f"🚫 Errors:            {stats.get('error', 0) + stats.get('error_llm', 0)}")

    print(f"🧹 Performing Anchor Statistics Maintenance for [{platform}]...")
    anchor_stats_dir = os.path.join(platform_output_dir, "daily_anchors")
    anchor_stats_csv_path = os.path.join(anchor_stats_dir, f"daily_anchors_{platform}_{target_date}.csv")
    state_manager.reset_daily_anchor_stats(export_csv_path=anchor_stats_csv_path)

    print(f"🗑️ Cleaning up temporary DB for [{platform}]: {temp_db_path}")
    _cleanup_temp_db(Path(temp_db_path))


if __name__ == "__main__":
    main()
