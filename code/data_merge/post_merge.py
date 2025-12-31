import sqlite3
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm  # 导入进度条库

# --- ⚙️ 配置区域 ---
SOURCE_DB_ROOT = 'MARS/data/output/' 
TARGET_DB_FILE = 'MARS/data/merged/user_post_merged.db'
START_DATE_STR = "2025-06-14" 
END_DATE_STR   = "2025-06-15"
DB_FILENAME = "user_post_database.db"
# --------------------

def merge_fast_immutable():
    # 1. 准备路径
    source_root = Path(SOURCE_DB_ROOT)
    target_db = Path(TARGET_DB_FILE)
    start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d").date()
    end_date = datetime.strptime(END_DATE_STR, "%Y-%m-%d").date()
    
    # 2. 扫描文件
    print(f"🔍 正在扫描文件 ({start_date} -> {end_date})...")
    files_to_merge = []
    if source_root.exists():
        # 使用 tqdm 扫描目录（如果目录很多的话）
        for folder in source_root.iterdir():
            if folder.is_dir():
                try:
                    folder_dt = datetime.strptime(folder.name, "%Y-%m-%d").date()
                    if start_date <= folder_dt <= end_date:
                        db_path = folder / DB_FILENAME
                        if db_path.exists():
                            files_to_merge.append((folder_dt, db_path))
                except ValueError: continue
    
    files_to_merge.sort(key=lambda x: x[0])
    if not files_to_merge:
        print("⚠️ 没找到符合条件的文件。")
        return

    # 3. 连接目标数据库
    target_db.parent.mkdir(parents=True, exist_ok=True)
    
    dst_conn = sqlite3.connect(TARGET_DB_FILE)
    dst_cur = dst_conn.cursor()
    
    # 极速 IO 配置
    dst_cur.execute("PRAGMA journal_mode = OFF;") 
    dst_cur.execute("PRAGMA synchronous = OFF;")
    dst_cur.execute("PRAGMA cache_size = -4000000;") # 4GB
    dst_cur.execute("PRAGMA temp_store = MEMORY;")
    
    # 创建表
    dst_cur.execute("""
        CREATE TABLE IF NOT EXISTS post (
            post_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT, 
            original_post_id INTEGER,
            content TEXT,
            quote_content TEXT,
            created_at DATETIME,
            temp_sjcjId TEXT, 
            temp_original_sjcjId TEXT
        );
    """)

    # ----------------------------------------------------
    # 4. 批量导入 (INSERT 阶段 - 带进度条)
    # ----------------------------------------------------
    print(f"🚀 开始合并 {len(files_to_merge)} 个数据库文件...")
    
    # ✨ 使用 tqdm 创建进度条
    # unit='db' 显示为 xx db/s
    # ncols=100 设置进度条宽度
    pbar = tqdm(files_to_merge, unit="db", ncols=100, desc="Processing")
    
    for f_date, f_path in pbar:
        # 在进度条右侧显示当前正在处理的文件名
        pbar.set_postfix(current_file=f_path.name)
        
        try:
            dst_cur.execute("BEGIN TRANSACTION;")
            dst_cur.execute(f"ATTACH DATABASE '{f_path}' AS source_db")
            
            dst_cur.execute("""
                INSERT INTO main.post (
                    user_id, original_post_id, content, quote_content, 
                    created_at, temp_sjcjId, temp_original_sjcjId
                )
                SELECT user_id, NULL, content, quote_content, 
                       created_at, temp_sjcjId, temp_original_sjcjId
                FROM source_db.post
            """)
            
            dst_cur.execute("DETACH DATABASE source_db")
            dst_conn.commit()
            
        except Exception as e:
            # 即使出错也不要破坏进度条显示
            pbar.write(f"❌ 错误 ({f_path.name}): {e}")
            dst_conn.rollback()

    # ----------------------------------------------------
    # 5. 修复关联关系 (因为是单条SQL，无法显示百分比，使用计时器)
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("⏳ 进入 Phase 2: 索引与关联修复 (请耐心等待，不要关闭)")
    print("="*50)

    print("⚡ [1/2] 正在构建临时索引 (加速 UPDATE 的关键)...")
    start_idx = time.time()
    
    dst_cur.execute("CREATE INDEX IF NOT EXISTS idx_temp_sjcjId ON post(temp_sjcjId);")
    dst_cur.execute("CREATE INDEX IF NOT EXISTS idx_temp_orig_id ON post(temp_original_sjcjId);")
    
    print(f"   ✅ 索引构建完成 | 耗时: {time.time() - start_idx:.2f}s")
    
    print("🔗 [2/2] 正在批量修复引用关系 (UPDATE)...")
    start_upd = time.time()
    
    dst_cur.execute("""
        UPDATE post
        SET original_post_id = (
            SELECT p2.post_id 
            FROM post AS p2
            WHERE p2.temp_sjcjId = post.temp_original_sjcjId
        )
        WHERE temp_original_sjcjId IS NOT NULL;
    """)
    dst_conn.commit()
    print(f"   ✅ 关系修复完成 | 耗时: {time.time() - start_upd:.2f}s")

    # ----------------------------------------------------
    # 6. 清理
    # ----------------------------------------------------
    print("🧹 清理临时索引...")
    dst_cur.execute("DROP INDEX idx_temp_sjcjId;")
    dst_cur.execute("DROP INDEX idx_temp_orig_id;")
    dst_conn.commit()
    dst_conn.close()
    
    print("\n🎉 全部任务完成!")

if __name__ == "__main__":
    merge_fast_immutable()