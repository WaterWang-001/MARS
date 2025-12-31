import sys
import os
import re
import yaml
import threading
from tqdm import tqdm
import sqlite3
import json
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.db_client import DBClient
from core.api_client import APIClient
from core.prompt_manager import PromptManager
from core.state_manager import StateManager
from tagging_service import TaggingService
from core.io_helpers import load_csv, append_jsonl


# ================= 配置 =================
MAX_WORKERS = 10
# =======================================

write_lock = threading.Lock()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def extract_date_from_path(path_str):
    match = re.search(r'(\d{4}-\d{2}-\d{2})', path_str)
    return match.group(1) if match else None

def process_single_user_incremental(service, user_data, batch_date):
    """
    增量处理单个用户
    只负责更新 DB 状态，不再直接写入 JSONL 文件
    """
    try:
        # 调用核心 Service (更新 SQLite 中的 Interest 和 Profile)
        result = service.process_user_incremental(user_data, batch_date)
        
        if not result: return "error"
        
        # 直接返回状态，用于统计
        return result.get("status") 

    except Exception as e:
        print(f"Error {user_data.get('user_id')}: {e}")
        return "error"

# === [修改点 2] 新增导出函数：从 DB 生成 Master 文件 ===
def export_master_files(state_db_path, interest_out_path, profile_out_path):
    print("\n📦 Exporting Master Profiles from DB...")
    
    conn = sqlite3.connect(state_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 查询所有有数据的用户
    cursor.execute("""
        SELECT user_id, last_cursor_time, interest_tree_snapshot, profile_snapshot 
        FROM user_full_state
    """)
    
    count = 0
    # 同时打开两个文件进行写入
    with open(interest_out_path, 'w', encoding='utf-8') as f_int, \
         open(profile_out_path, 'w', encoding='utf-8') as f_pro:
        
        while True:
            rows = cursor.fetchmany(1000) # 批次读取，节省内存
            if not rows:
                break
                
            for row in rows:
                user_id = row['user_id']
                update_date = row['last_cursor_time']
                
                # 1. 导出 Interest
                if row['interest_tree_snapshot']:
                    try:
                        tree_data = json.loads(row['interest_tree_snapshot'])
                        if tree_data: # 只有非空才导出
                            interest_record = {
                                "user_id": user_id,
                                "last_update": update_date,
                                "interest_tree": tree_data
                            }
                            f_int.write(json.dumps(interest_record, ensure_ascii=False) + '\n')
                    except: pass
                
                # 2. 导出 Profile
                if row['profile_snapshot']:
                    try:
                        profile_data = json.loads(row['profile_snapshot'])
                        if profile_data: # 只有非空才导出
                            profile_record = {
                                "user_id": user_id,
                                "last_update": update_date,
                                "user_type": profile_data.get("user_type", "Unknown"),
                                "profile": profile_data
                            }
                            f_pro.write(json.dumps(profile_record, ensure_ascii=False) + '\n')
                    except: pass
            
            count += len(rows)
            print(f"   Exported {count} users...", end='\r')
            
    conn.close()
    print(f"\n✅ Export Complete! Master files updated.")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_config(os.path.join(base_dir, 'configs', 'settings.yaml'))
    
    # === 1. 路径配置 ===
    target_date = "2025-06-14"  # 你的数据批次
    
    post_db_path = f'remote-home/JuelinW/oasis_project/MARS/data/output/{target_date}/user_post_database.db'
    state_db_path = 'remote-home/JuelinW/oasis_project/MARS/data/output/tagging_result/processing_state.db'

    user_csv_path = f'remote-home/JuelinW/oasis_project/MARS/data/output/{target_date}/user_profiles.csv'
    stopwords_path ='remote-home/JuelinW/oasis_project/MARS/code/user_tagging/src/stop_words.txt'
    
    # [Output] Master 数据的存放位置
    master_dir = 'remote-home/JuelinW/oasis_project/MARS/data/output/tagging_result/'
    os.makedirs(master_dir, exist_ok=True)
    
    master_interest_path = os.path.join(master_dir, f'master_interest_{target_date}.jsonl')
    master_profile_path = os.path.join(master_dir, f'master_profile_{target_date}.jsonl')

    # [核心] 提取 Batch Date
    current_batch_date = extract_date_from_path(post_db_path)
    if not current_batch_date:
        print("❌ Error: Could not extract date from path!")
        return
    print(f"📅 Current Batch Logical Date: {current_batch_date}")

    # === 2. 初始化组件 ===
    llm_cfg = config['llm_service']
    pipeline_cfg = config['pipeline']
    
    MAX_WORKERS = llm_cfg.get('max_workers', 10)
    
    db_client = DBClient(post_db_path)
    state_manager = StateManager(state_db_path)
    
    prompt_manager = PromptManager(os.path.join(base_dir, 'src', 'prompts'))

    api_client = APIClient(
        api_key=llm_cfg['api_key'], 
        base_url=llm_cfg['base_url'], 
        model_name=llm_cfg['model_name'], 
        mode=llm_cfg.get('mode', 'remote'),
        timeout=llm_cfg.get('timeout', 120) 
    )
    
    service = TaggingService(
        db_client, 
        prompt_manager, 
        api_client, 
        stopwords_path, 
        state_manager,
        pipeline_cfg 
    )
    
    print(f"=== Starting Incremental Task | Workers: {MAX_WORKERS} ===")

    # === [Debug 1] 显式打印数据加载状态 ===
    print(f"[{time.strftime('%X')}] 📂 Loading user CSV from: {user_csv_path} ...", end=" ", flush=True)
    try:
        df_users = load_csv(user_csv_path)
        print(f"Done! Loaded {len(df_users)} users.", flush=True)
    except Exception as e:
        print(f"\n❌ Failed to load CSV: {e}")
        return

    if df_users.empty: 
        print("⚠️ CSV is empty, exiting.")
        return
        
    all_users = df_users.to_dict('records')

    # === [Debug 2] 增加任务提交进度的可见性 ===
    print(f"[{time.strftime('%X')}] 🚀 Submitting {len(all_users)} tasks to ThreadPool...", flush=True)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        # 如果用户量巨大 (>10万)，建议稍微分批或者加个简单的计数打印，防止这里看起来像卡死
        submit_start = time.time()
        
        for i, u in enumerate(all_users):
            ft = executor.submit(
                process_single_user_incremental, 
                service, 
                u, 
                current_batch_date
            )
            futures[ft] = u['user_id']
            
            # 每提交 5000 个任务打印一次点，证明活着
            if (i + 1) % 5000 == 0:
                print(f".", end="", flush=True)
        
        print(f"\n[{time.strftime('%X')}] ✅ Submission complete ({time.time()-submit_start:.2f}s). Waiting for results...", flush=True)
        
        stats = {"updated": 0, "buffered": 0, "skipped_quality": 0, "error": 0}
        
        # 这里的 tqdm 只有在上面的 for 循环全部跑完后才会出现
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing", mininterval=1.0):
            res = f.result()
            if res in stats: stats[res] += 1
            
    # === 5. 总结与导出 ===
    print(f"\nBatch {current_batch_date} Processing Summary:")
    print(f"✅ DB Updated: {stats['updated']} users (Ready for export)")
    print(f"💧 Buffered: {stats['buffered']} users")
    print(f"🚮 Skipped: {stats['skipped_quality']} users")
    print(f"❌ Errors: {stats['error']} users")
    
    # [修改点 4] 执行全量导出
    export_master_files(state_db_path, master_interest_path, master_profile_path)

if __name__ == "__main__":
    main()