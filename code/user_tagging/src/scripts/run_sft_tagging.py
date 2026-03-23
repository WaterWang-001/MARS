import argparse
import os
import shutil
import sys
import sqlite3
import json
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
from src.platforms import DoubanFilter, ToutiaoFilter, WeiboFilter, XiaohongshuFilter, ZhihuFilter
from src.tasks.sft import TaggingService
from src.tasks.sft.state_manager import StateManager

SUPPORTED_PLATFORMS = ["weibo", "toutiao", "xiaohongshu", "zhihu", "douban"]

write_lock = threading.Lock()
log_lock = threading.Lock()
print_lock = threading.Lock()


def get_platform_filter(platform: str, platform_config: dict, blacklist_file: str | None = None):
    mapping = {
        "weibo": WeiboFilter(config=platform_config, blacklist_file=blacklist_file),
        "toutiao": ToutiaoFilter(config=platform_config, blacklist_file=blacklist_file),
        "xiaohongshu": XiaohongshuFilter(config=platform_config, blacklist_file=blacklist_file),
        "zhihu": ZhihuFilter(config=platform_config, blacklist_file=blacklist_file),
        "douban": DoubanFilter(config=platform_config, blacklist_file=blacklist_file),
    }
    return mapping[platform]


def _copy_file_with_progress(source: Path, target: Path, chunk_size: int = 8 * 1024 * 1024):
    total_size = source.stat().st_size
    copied = 0
    start_time = time.time()
    print(f"📦 Copying DB to temp: {source} -> {target} | Total: {total_size / (1024**2):.1f} MB", flush=True)

    with source.open("rb") as src, target.open("wb") as dst:
        while True:
            block = src.read(chunk_size)
            if not block:
                break
            dst.write(block)
            copied += len(block)

    shutil.copystat(source, target)
    elapsed = time.time() - start_time
    print(f"✅ Temp DB copy completed. ({elapsed:.1f}s)", flush=True)


def _prepare_temp_db(target_date, platform, db_base_dir, tmp_db_base, reuse_if_valid=False):
    date_dir = db_base_dir / target_date
    preferred_db = date_dir / f"posts_{platform}.db"
    legacy_db = date_dir / f"user_post_{platform}.db"
    source_db = preferred_db if preferred_db.exists() else legacy_db
    if not source_db.exists():
        raise FileNotFoundError(f"Source DB not found. Tried: {preferred_db} and {legacy_db}")

    tmp_db_dir = tmp_db_base / platform / target_date
    tmp_db_dir.mkdir(parents=True, exist_ok=True)
    tmp_db = tmp_db_dir / f"user_post_{platform}.db"

    if reuse_if_valid and tmp_db.exists():
        if tmp_db.stat().st_size == source_db.stat().st_size:
            print(f"♻️ Reusing existing temp DB: {tmp_db}", flush=True)
            return tmp_db

    _copy_file_with_progress(source_db, tmp_db)
    return tmp_db


def _cleanup_temp_db(tmp_db_path: Path):
    try:
        if tmp_db_path.exists():
            tmp_db_path.unlink()
    except Exception:
        pass


def process_single_user_with_logging(service, user_data, batch_date, resume_log_path, user_posts):
    user_id = str(user_data.get('user_id'))
    try:
        result = service.process_user_incremental(user_data, batch_date, prefetched_posts=user_posts)

        status = "error"
        if result:
            status = result.get("status", "error")

        with print_lock:
            if status == "updated":
                user_type = result.get("user_type", "Unknown")
                print(f"✅ [User: {user_id}] UPDATED | Type: {user_type}")
            elif status == "buffered":
                count = result.get("count", 0)
                print(f"💧 [Buffer] User {user_id}: {count} posts buffered")
            elif status == "no_valid_posts":
                print(f"⚠️ [Skip] User {user_id}: No valid posts found")
            elif status == "error":
                print(f"❌ [User: {user_id}] Failed.")
            else:
                reason = result.get("reason", "N/A") if result else "N/A"
                print(f"⚠️ [Skip] User {user_id}: {status} | Reason: {reason}")

        if status != "error":
            with log_lock:
                with open(resume_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{user_id}\n")

        return status

    except Exception as e:
        with print_lock:
            print(f"❌ [User: {user_id}] Critical Error: {e}")
            print(traceback.format_exc())
        return "error"


def export_master_files(state_db_path, interest_out_path, profile_out_path):
    print("\n📦 Exporting Master Profiles from DB...")

    conn = sqlite3.connect(state_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT user_id, last_cursor_time, interest_tree_snapshot, profile_snapshot
        FROM user_full_state
    """)

    count = 0
    with open(interest_out_path, 'w', encoding='utf-8') as f_int, \
         open(profile_out_path, 'w', encoding='utf-8') as f_pro:

        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break

            for row in rows:
                user_id = row['user_id']
                update_date = row['last_cursor_time']

                if row['interest_tree_snapshot']:
                    try:
                        tree_data = json.loads(row['interest_tree_snapshot'])
                        if tree_data:
                            f_int.write(json.dumps({
                                "user_id": user_id,
                                "last_update": update_date,
                                "interest_tree": tree_data
                            }, ensure_ascii=False) + '\n')
                    except: pass

                if row['profile_snapshot']:
                    try:
                        profile_data = json.loads(row['profile_snapshot'])
                        if profile_data:
                            f_pro.write(json.dumps({
                                "user_id": user_id,
                                "last_update": update_date,
                                "user_type": profile_data.get("user_type", "Unknown"),
                                "profile": profile_data
                            }, ensure_ascii=False) + '\n')
                    except: pass

            count += len(rows)
            print(f"   Exported {count} users...", end='\r')

    conn.close()
    print(f"\n✅ Export Complete! Master files updated.")


def main():
    parser = argparse.ArgumentParser(description="Run SFT/Tagging pipeline for a specific date and platform")
    parser.add_argument("--date", type=str, required=True, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--platform", type=str, required=True, choices=SUPPORTED_PLATFORMS, help="Target platform")
    parser.add_argument("--db-base-dir", type=str, default="MARS_result/data/output", help="Source DB root directory")
    parser.add_argument("--tmp-db-base-dir", type=str, default="/root/sztemp", help="Temp DB root directory")
    parser.add_argument("--reuse-temp-db-if-valid", action="store_true", help="Reuse temp DB if size matches")
    parser.add_argument("--output-dir", type=str, default="MARS_result/data/online_cold_start", help="Output directory")
    parser.add_argument("--user-csv", type=str, default="", help="User CSV path (auto-detected if empty)")
    parser.add_argument(
        "--daily-profiles-dir", type=str,
        default="MARS_result/data/output/{date}",
        help="Directory containing user CSV, supports {date} and {platform} templates"
    )
    parser.add_argument("--stopwords-path", type=str, default="StreamProfile/benchmark/resources/stop_words.txt")
    args = parser.parse_args()

    target_date = args.date
    platform = args.platform

    # Load platform-specific config
    config_dir = PACKAGE_ROOT / "configs"
    platform_config_path = config_dir / f"{platform}_tagging.yaml"
    if not platform_config_path.exists():
        raise FileNotFoundError(f"Platform config not found: {platform_config_path}")
    platform_config = load_config(str(platform_config_path))

    # Paths
    db_base_dir = Path(args.db_base_dir)
    tmp_db_base_dir = Path(args.tmp_db_base_dir)

    platform_output_dir = os.path.join(args.output_dir, platform)
    os.makedirs(platform_output_dir, exist_ok=True)
    state_db_path = os.path.join(platform_output_dir, f"user_state_{platform}.db")

    log_dir = os.path.join(platform_output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    resume_log_path = os.path.join(log_dir, f"processed_users_{platform}_{target_date}.log")

    master_interest_path = os.path.join(platform_output_dir, f"master_interest_{platform}_{target_date}.jsonl")
    master_profile_path = os.path.join(platform_output_dir, f"master_profile_{platform}_{target_date}.jsonl")

    # User CSV
    if args.user_csv:
        user_csv_path = args.user_csv
    else:
        ds_config = platform_config.get("data_source", {})
        filename_template = ds_config.get("user_pool_filename", f"{{date}}_users.csv")
        actual_filename = filename_template.format(date=target_date)
        daily_profiles_dir = Path(args.daily_profiles_dir.format(date=target_date, platform=platform))
        user_csv_path = str(daily_profiles_dir / actual_filename)

    print(f"📅 Batch Date: {target_date}")
    print(f"🌐 Platform: {platform}")
    print(f"📄 User CSV: {user_csv_path}")

    # Prepare temp DB
    temp_db_path = _prepare_temp_db(
        target_date, platform, db_base_dir, tmp_db_base_dir,
        reuse_if_valid=args.reuse_temp_db_if_valid
    )
    post_db_path = str(temp_db_path)

    # Initialize components
    platform_filter = get_platform_filter(platform, platform_config)

    llm_cfg = platform_config["llm_service"]
    max_workers = llm_cfg.get("max_workers", 10)

    db_client = DBClient(post_db_path)
    state_manager = StateManager(state_db_path)
    prompt_manager = PromptManager(str(CURRENT_DIR.parent / "prompts"))

    api_client = APIClient(
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg["base_url"],
        model_name=llm_cfg["model_name"],
        embedding_model_path=llm_cfg["embedding_model_path"],
        mode=llm_cfg.get("mode", "remote"),
        timeout=llm_cfg.get("timeout", 120)
    )

    service = TaggingService(
        db_client=db_client,
        prompt_manager=prompt_manager,
        api_client=api_client,
        stopwords_path=str(Path(args.stopwords_path)),
        state_manager=state_manager,
        output_dir=platform_output_dir,
        config=platform_config,
        platform_filter=platform_filter
    )

    print(f"=== Starting SFT/Tagging Pipeline | Platform: {platform} | Workers: {max_workers} ===")

    # Load users
    print(f"[{time.strftime('%X')}] 📂 Loading user CSV...", end=" ", flush=True)
    df_users = load_csv(user_csv_path)
    print(f"Done! Loaded {len(df_users)} users.", flush=True)

    if df_users.empty:
        print("⚠️ CSV is empty. Exit.")
        _cleanup_temp_db(Path(temp_db_path))
        return

    all_users_raw = df_users.to_dict("records")
    processed_ids = load_processed_ids(resume_log_path)
    pending_users = [u for u in all_users_raw if str(u.get("user_id", "")).strip() not in processed_ids]

    skipped_count = len(all_users_raw) - len(pending_users)
    print(f"⚠️ Skipped {skipped_count} users (Already processed).")
    print(f"🚀 Remaining tasks: {len(pending_users)}")

    if not pending_users:
        print("🎉 All users in this batch are already processed!")
        _cleanup_temp_db(Path(temp_db_path))
        return

    # Process in macro batches
    fetch_cfg = platform_config.get("fetch_limits", {})
    max_count = int(fetch_cfg.get("max_count", 10))

    macro_batch_size = 100
    stats = {"updated": 0, "buffered": 0, "error": 0}
    total_processed = 0

    for i in range(0, len(pending_users), macro_batch_size):
        macro_start = time.perf_counter()
        current_macro_batch = pending_users[i: i + macro_batch_size]
        print(f"\n[{time.strftime('%X')}] 🔄 Macro Batch {i // macro_batch_size + 1} ({len(current_macro_batch)} users)...")

        target_uids = [str(u["user_id"]) for u in current_macro_batch]
        batch_posts_map = {}

        print("   ⚡️ Fetching posts...", end=" ", flush=True)
        fetch_start = time.perf_counter()
        db_chunk_size = 50
        for j in range(0, len(target_uids), db_chunk_size):
            chunk = target_uids[j: j + db_chunk_size]
            partial_map = db_client.get_batch_posts(chunk, max_count=max_count)
            batch_posts_map.update(partial_map)
        fetch_elapsed = time.perf_counter() - fetch_start
        print(f"Done. ({fetch_elapsed:.2f}s)", flush=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for u in current_macro_batch:
                uid = str(u["user_id"])
                u_posts = batch_posts_map.get(uid, [])
                ft = executor.submit(process_single_user_with_logging, service, u, target_date, resume_log_path, u_posts)
                futures[ft] = uid

            batch_done = 0
            for f in as_completed(futures):
                res = f.result()
                batch_done += 1
                total_processed += 1
                stats[res] = stats.get(res, 0) + 1
                if total_processed % 50 == 0:
                    print(f"   Progress: {batch_done}/{len(current_macro_batch)} | Total: {total_processed}", flush=True)

        del batch_posts_map
        macro_elapsed = time.perf_counter() - macro_start
        print(f"   ⏱️ Macro Batch ({macro_elapsed:.2f}s) ✅")

    # Summary & export
    print(f"\n{'=' * 30}")
    print(f"📊 SFT/Tagging Summary ({target_date} | {platform})")
    print(f"✅ Updated: {stats.get('updated', 0)}")
    print(f"💧 Buffered: {stats.get('buffered', 0)}")
    print(f"❌ Errors: {stats.get('error', 0)}")

    export_master_files(state_db_path, master_interest_path, master_profile_path)

    print(f"🗑️ Cleaning up temporary DB: {temp_db_path}")
    _cleanup_temp_db(Path(temp_db_path))


if __name__ == "__main__":
    main()
