import sys
from pathlib import Path
import os
import sqlite3
import time
import typing
import multiprocessing
from datetime import datetime

# 尝试导入 orjson
try:
    import orjson as json_lib
except ImportError:
    import json as json_lib
    print("⚠️ 建议 pip install orjson 以获得最大提速", file=sys.stderr)

# --- 配置 ---
INPUT_DIRECTORY = 'data/raw/'
FINAL_DB_FILE = 'data/output/oasis_final.db' # 直接生成最终库
PROCESSED_LOG_FILE = Path(INPUT_DIRECTORY) / "processed_files_etl.log"
# --------------------

def parse_logic(data):
    """
    [核心清洗逻辑]
    从原始 JSON 中提取最终字段。这是原 Step 2 的 parse_log_line 逻辑。
    """
    try:
        comment_pojo = data.get('commentPojo')
        content_pojo = data.get('contentPojo')
        comment_fwd_pojo = data.get('commentForwardPojo')
        content_fwd_pojo = data.get('contentForwardPojo')
        content_root_pojo = data.get('contentRootPojo')

        source_sjcjId = None
        source_original_sjcjId = None
        content = ""
        quote_content = None

        # 逻辑复刻
        if comment_pojo:
            source_sjcjId = comment_pojo.get('sjcjId')
            content = comment_pojo.get('sjqxTitle', "")
            if comment_fwd_pojo:
                source_original_sjcjId = comment_fwd_pojo.get('sjcjId')
                quote_content = comment_fwd_pojo.get('sjcjContent') or comment_fwd_pojo.get('sjqxTitle')
            elif content_pojo:
                source_original_sjcjId = content_pojo.get('sjcjId')
                quote_content = content_pojo.get('sjqxContent') or content_pojo.get('sjcjContent')
        elif content_pojo:
            source_sjcjId = content_pojo.get('sjcjId')
            content = content_pojo.get('sjqxContent', "")
            if content_fwd_pojo:
                if content_root_pojo and content_root_pojo.get('sjcjId'):
                    source_original_sjcjId = content_root_pojo.get('sjcjId')
                else:
                    source_original_sjcjId = content_fwd_pojo.get('sjcjId')
                if content_root_pojo:
                    quote_content = content_root_pojo.get('sjqxContent') or content_root_pojo.get('sjcjContent')
                quote_content = quote_content or content_fwd_pojo.get('sjqxContent') or content_fwd_pojo.get('sjcjContent')
            else:
                source_original_sjcjId = None
                quote_content = None
        else:
            return None

        # 提取时间戳并格式化
        # 假设优先使用 contentPojo 的时间，如果没有则尝试 commentPojo
        ts = None
        if content_pojo: ts = content_pojo.get('sjcjPublished')
        if not ts and comment_pojo: ts = comment_pojo.get('sjcjPublished')
        
        created_at = None
        if ts:
            # 在 Worker 里做时间格式化，分担主进程压力
            created_at = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

        # 提取 User ID
        user_id = None
        author = data.get('authorContentPojo') or data.get('authorCommentPojo')
        if author:
            user_id = author.get('sjcjId')

        if not user_id or not source_sjcjId:
            return None

        return (
            str(user_id),
            None, # original_post_id (留空，最后 Link)
            content,
            quote_content,
            created_at,
            str(source_sjcjId),       # temp_sjcjId
            str(source_original_sjcjId) if source_original_sjcjId else None # temp_original_sjcjId
        )

    except Exception:
        return None

def _worker_process_file(filepath: Path, queue: multiprocessing.Queue, batch_size=2000):
    """
    [Worker] 读取 -> 解析 -> 提取 -> 发送
    """
    try:
        local_batch = []
        with open(filepath, 'rb') as f:
            for line in f:
                try:
                    data = json_lib.loads(line)
                    parsed_row = parse_logic(data)
                    
                    if parsed_row:
                        local_batch.append(parsed_row)
                    
                    if len(local_batch) >= batch_size:
                        queue.put(('DATA', local_batch))
                        local_batch = []
                except:
                    continue
        
        if local_batch:
            queue.put(('DATA', local_batch))
            
        queue.put(('FILE_DONE', filepath.name))
        
    except Exception as e:
        queue.put(('ERROR', f"{filepath.name}: {str(e)}"))

class UnifiedProcessor:
    def __init__(self, input_dir=INPUT_DIRECTORY, db_path=FINAL_DB_FILE):
        self.input_dir = Path(input_dir)
        self.db_path = db_path
        self.log_path = PROCESSED_LOG_FILE
        self._processed_files = self._load_log()

    def _load_log(self):
        processed = set()
        if self.log_path.exists():
            with open(self.log_path, 'r', encoding='utf-8') as f:
                processed = {line.strip() for line in f if line.strip()}
        return processed

    def _init_db(self):
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.execute("PRAGMA cache_size = -500000;") # 2GB 缓存
        cur.execute("PRAGMA temp_store = MEMORY;")
        
        # 直接创建最终表结构 (无索引，为了写入快)
        cur.execute("""
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
        return conn

    def _finalize_db(self, conn):
        """写入完成后：建索引 + 建立关联"""
        # print("\n🔨 [Phase 2] 正在构建索引...")
        t0 = time.time()
        cur = conn.cursor()
        
        # # 批量建索引
        cur.execute("CREATE INDEX IF NOT EXISTS idx_post_userid ON post (user_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_post_temp_id ON post (temp_sjcjId);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_post_temp_orig ON post (temp_original_sjcjId);")
        conn.commit()
        print(f"-> 索引构建耗时: {time.time()-t0:.2f}s")
        
        print("🔗 [Phase 3] 正在建立引用关联 (Linking)...")
        t0 = time.time()
        # 自关联更新
        cur.execute("""
            UPDATE post
            SET original_post_id = (
                SELECT p2.post_id FROM post AS p2
                WHERE p2.temp_sjcjId = post.temp_original_sjcjId
            )
            WHERE post.temp_original_sjcjId IS NOT NULL;
        """)
        conn.commit()
        print(f"-> 关联耗时: {time.time()-t0:.2f}s")

    def run(self):
        files = [f for f in self.input_dir.glob('*.txt') if f.name not in self._processed_files]
        if not files:
            print("没有新文件需要处理。")
            return

        print(f"🚀 开始处理 {len(files)} 个文件 (One-Pass ETL)...")
        
        manager = multiprocessing.Manager()
        queue = manager.Queue(maxsize=100)
        num_workers = max(1, os.cpu_count() - 1)
        
        pool = multiprocessing.Pool(num_workers)
        for f in files:
            pool.apply_async(_worker_process_file, args=(f, queue))
        pool.close()

        conn = self._init_db()
        cur = conn.cursor()
        
        processed_buffer = []
        insert_buffer = []
        total_inserted = 0
        files_done = 0
        start_time = time.time()
        
        try:
            while files_done < len(files):
                msg, payload = queue.get()
                
                if msg == 'DATA':
                    insert_buffer.extend(payload)
                    if len(insert_buffer) >= 50000:
                        cur.execute("BEGIN TRANSACTION;")
                        cur.executemany("""
                            INSERT INTO post (
                                user_id, original_post_id, content, quote_content, 
                                created_at, temp_sjcjId, temp_original_sjcjId
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, insert_buffer)
                        conn.commit()
                        total_inserted += len(insert_buffer)
                        insert_buffer = []
                        print(f"\r  -> 已处理: {total_inserted} 条 / 文件进度: {files_done}/{len(files)}", end="")
                        
                elif msg == 'FILE_DONE':
                    files_done += 1
                    processed_buffer.append(payload)
                    # 批量写日志
                    if len(processed_buffer) >= 10:
                        with open(self.log_path, 'a') as f:
                            for name in processed_buffer: f.write(name + '\n')
                        processed_buffer = []
                        
                elif msg == 'ERROR':
                    print(f"\n❌ {payload}")
                    files_done += 1

            # 写入剩余数据
            if insert_buffer:
                cur.execute("BEGIN TRANSACTION;")
                cur.executemany("""
                    INSERT INTO post (
                        user_id, original_post_id, content, quote_content, 
                        created_at, temp_sjcjId, temp_original_sjcjId
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, insert_buffer)
                conn.commit()

            # 记录剩余日志
            if processed_buffer:
                with open(self.log_path, 'a') as f:
                    for name in processed_buffer: f.write(name + '\n')

            print(f"\n✅ 数据迁移完成 (耗时 {time.time()-start_time:.2f}s)")
            
            # # 执行后续步骤
            # self._finalize_db(conn)
            
        finally:
            pool.terminate()
            conn.close()

if __name__ == "__main__":
    Path(INPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    Path(FINAL_DB_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    proc = UnifiedProcessor()
    proc.run()