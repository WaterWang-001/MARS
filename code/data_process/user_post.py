import sys
from pathlib import Path
import os
import sqlite3
import time
import multiprocessing
import shutil
from datetime import datetime
from collections import defaultdict

# 优先使用 orjson
try:
    import orjson as json_lib
except ImportError:
    import json as json_lib
    print("⚠️ 建议 pip install orjson 以获得最大提速", file=sys.stderr)

# ================= 配置区域 =================
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path('.').resolve()



PLATFORM_MAP = {
    "新浪微博": "weibo", "微博": "weibo", "微博视频号": "weibo",
    "今日头条": "toutiao", "今日头条微头条": "toutiao", "西瓜视频": "toutiao",
    "小红书": "xiaohongshu",
    "豆瓣": "douban",
    "知乎": "zhihu"
}

# ================= 极速工具函数 (保持不变) =================

def get_platform_key(raw_site):
    if not raw_site: return "other"
    if raw_site in PLATFORM_MAP: return PLATFORM_MAP[raw_site]
    site_str = str(raw_site)
    if "微博" in site_str: return "weibo"
    if "头条" in site_str or "西瓜" in site_str: return "toutiao"
    if "小红书" in site_str: return "xiaohongshu"
    if "豆瓣" in site_str: return "douban"
    if "知乎" in site_str: return "zhihu"
    return "other"

def _get_text(raw_text):
    if not raw_text: return ""
    return raw_text.strip()

def _extract_meta(data):
    comment_pojo = data.get('commentPojo')
    content_pojo = data.get('contentPojo')
    source_id = None
    ts = None
    
    if comment_pojo:
        source_id = comment_pojo.get('sjcjId')
        ts = comment_pojo.get('sjcjPublished')
    elif content_pojo:
        source_id = content_pojo.get('sjcjId')
        ts = content_pojo.get('sjcjPublished')
    
    created_at = None
    if ts:
        try: 
            created_at = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
        except: pass
    
    user_id = None
    author = data.get('authorContentPojo') or data.get('authorCommentPojo')
    if author:
        user_id = author.get('sjcjId') or author.get('sjcjAuthorAccount')
        
    return source_id, user_id, created_at

def _get_post_body_simple(pojo, is_xhs=False):
    if not pojo: return ""
    if is_xhs:
        desc = pojo.get('sjcjDesc')
        raw = desc if desc else (pojo.get('sjqxContent') or pojo.get('sjcjContent') or "")
    else:
        raw = (pojo.get('sjqxContent') or pojo.get('sjcjContent') or pojo.get('sjqxSummary') or "")
    media_text = pojo.get('media_text') or pojo.get('medita_text') or ""
    media_text = _get_text(media_text)
    raw_text = _get_text(raw)
    if media_text and media_text not in raw_text:
        return f"{raw_text}\n{media_text}" if raw_text else media_text
    return raw_text

# ================= 引用链构建器 (保持不变) =================

def _build_quote_chain(data, skip_content_pojo=False, is_xhs=False):
    quote_parts = []
    source_original_id = None
    root_pojo = data.get('commentRootPojo') or data.get('commentForwardPojo')
    content_pojo = data.get('contentPojo')

    if root_pojo:
        source_original_id = root_pojo.get('sjcjId')
        r_text = (root_pojo.get('sjqxTitle') or root_pojo.get('sjcjContent') or "")
        r_text = _get_text(r_text)
        if r_text:
            quote_parts.append(r_text)

    if content_pojo and not skip_content_pojo:
        if not source_original_id:
            source_original_id = content_pojo.get('sjcjId')
        c_text = _get_post_body_simple(content_pojo, is_xhs=is_xhs)
        c_title = _get_text(content_pojo.get('sjqxTitle'))
        if c_title and c_title not in c_text:
            c_combo = f"{c_title}\n{c_text}"
        else:
            c_combo = c_text
        if c_combo:
            quote_parts.append(c_combo)

    quote_content = " // ".join(quote_parts)
    return quote_content, source_original_id

# ================= 平台解析逻辑 (保持不变) =================
def parse_douban(data):
    source_id, user_id, created_at = _extract_meta(data)
    if not user_id or not source_id: return None
    comment_pojo = data.get('commentPojo')
    content_pojo = data.get('contentPojo')
    title, content, quote_content = "", "", ""
    source_original_id = None
    if comment_pojo:
        content = _get_text(comment_pojo.get('sjqxTitle'))
        if content_pojo: title = _get_text(content_pojo.get('sjqxTitle'))
        quote_content, source_original_id = _build_quote_chain(data)
    elif content_pojo:
        title = _get_text(content_pojo.get('sjqxTitle'))
        content = _get_post_body_simple(content_pojo)
        root = data.get('contentRootPojo')
        if root:
            source_original_id = root.get('sjcjId')
            q_title = _get_text(root.get('sjqxTitle'))
            q_body = _get_post_body_simple(root)
            quote_content = f"【{q_title}】\n{q_body}" if q_title else q_body
    return (user_id, None, title, content, quote_content, created_at, str(source_id), str(source_original_id) if source_original_id else None)

def parse_toutiao(data):
    source_id, user_id, created_at = _extract_meta(data)
    if not user_id or not source_id: return None
    comment_pojo = data.get('commentPojo')
    content_pojo = data.get('contentPojo')
    title, content, quote_content = "", "", ""
    source_original_id = None
    if comment_pojo:
        content = _get_text(comment_pojo.get('sjqxTitle'))
        if content_pojo: title = _get_text(content_pojo.get('sjqxTitle'))
        quote_content, source_original_id = _build_quote_chain(data)
    elif content_pojo:
        title = _get_text(content_pojo.get('sjqxTitle'))
        content = _get_post_body_simple(content_pojo)
        root = data.get('contentRootPojo')
        if root:
            source_original_id = root.get('sjcjId')
            quote_content = _get_post_body_simple(root)
    return (user_id, None, title, content, quote_content, created_at, str(source_id), str(source_original_id) if source_original_id else None)

def parse_xiaohongshu(data):
    source_id, user_id, created_at = _extract_meta(data)
    if not user_id or not source_id: return None
    comment_pojo = data.get('commentPojo')
    content_pojo = data.get('contentPojo')
    title, content, quote_content = "", "", ""
    source_original_id = None
    if comment_pojo:
        content = _get_text(comment_pojo.get('sjqxTitle'))
        if content_pojo: title = _get_text(content_pojo.get('sjqxTitle'))
        quote_content, source_original_id = _build_quote_chain(data, is_xhs=True)
    elif content_pojo:
        title = _get_text(content_pojo.get('sjqxTitle'))
        content = _get_post_body_simple(content_pojo, is_xhs=True)
    return (user_id, None, title, content, quote_content, created_at, str(source_id), str(source_original_id) if source_original_id else None)

def parse_zhihu(data):
    source_id, user_id, created_at = _extract_meta(data)
    if not user_id or not source_id: return None
    comment_pojo = data.get('commentPojo')
    content_pojo = data.get('contentPojo')
    title, content, quote_content = "", "", ""
    source_original_id = None
    if comment_pojo:
        content = _get_text(comment_pojo.get('sjqxTitle'))
        raw_url = comment_pojo.get('sjcjUrl', '') or comment_pojo.get('sjqxWebpageUrl', '')
        is_comment_on_answer = 'commentId=' in raw_url
        if content_pojo: title = _get_text(content_pojo.get('sjqxTitle'))
        skip_rule = not is_comment_on_answer
        quote_content, source_original_id = _build_quote_chain(data, skip_content_pojo=skip_rule)
    elif content_pojo:
        title = _get_text(content_pojo.get('sjqxTitle'))
        content = _get_post_body_simple(content_pojo)
        root = data.get('contentRootPojo')
        if root:
            source_original_id = root.get('sjcjId')
            q_title = _get_text(root.get('sjqxTitle'))
            q_body = _get_post_body_simple(root)
            quote_content = f"【{q_title}】\n{q_body}" if q_title else q_body
    return (user_id, None, title, content, quote_content, created_at, str(source_id), str(source_original_id) if source_original_id else None)

def parse_weibo(data):
    source_id, user_id, created_at = _extract_meta(data)
    if not user_id or not source_id: return None
    comment_pojo = data.get('commentPojo')
    content_pojo = data.get('contentPojo')
    title, content, quote_content = "", "", ""
    source_original_id = None
    if comment_pojo:
        content = _get_text(comment_pojo.get('sjqxTitle'))
        quote_content, source_original_id = _build_quote_chain(data)
    elif content_pojo:
        content = _get_post_body_simple(content_pojo)
        quote_target = data.get('commentForwardPojo') or data.get('contentRootPojo')
        if quote_target:
            source_original_id = quote_target.get('sjcjId')
            quote_content = _get_post_body_simple(quote_target)
    return (user_id, None, title, content, quote_content, created_at, str(source_id), str(source_original_id) if source_original_id else None)

# ================= 调度与执行 =================

def dispatch_parse(data):
    site_keys = ['sjqxCaptureWebsite', 'sjcjCaptureWebsite', 'sjqxCaptureWebsiteNew']
    def _get_site_from_pojo(pojo):
        if not pojo: return None
        for key in site_keys:
            val = pojo.get(key)
            if val: return val
        return None

    raw_site = _get_site_from_pojo(data.get('contentPojo'))
    if not raw_site:
        raw_site = _get_site_from_pojo(data.get('commentPojo'))
    
    platform = get_platform_key(raw_site)
    
    row_data = None
    if platform == 'weibo': row_data = parse_weibo(data)
    elif platform == 'toutiao': row_data = parse_toutiao(data)
    elif platform == 'xiaohongshu': row_data = parse_xiaohongshu(data)
    elif platform == 'douban': row_data = parse_douban(data)
    elif platform == 'zhihu': row_data = parse_zhihu(data)
    else: row_data = parse_weibo(data)
        
    if row_data: return platform, row_data
    return None

def _worker_process_file(filepath: Path, queue: multiprocessing.Queue, batch_size=50000):
    try:
        local_batch = [] 
        with open(filepath, 'rb') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json_lib.loads(line)
                    res = dispatch_parse(data) 
                    if res and res[1]:
                        local_batch.append(res)
                    if len(local_batch) >= batch_size:
                        queue.put(('DATA', local_batch))
                        local_batch = []
                except: continue
        
        if local_batch: queue.put(('DATA', local_batch))
        queue.put(('FILE_DONE', filepath.name))
    except Exception as e: 
        queue.put(('ERROR', f"{filepath.name}: {str(e)}"))

class MultiDBProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        # 获取当前处理的日期 (假设 input_dir 的最后一级是日期，例如 .../2025-06-15)
        self.target_date = self.input_dir.name
        
        # 1. 设置最终输出目录 (NFS 原目录)
        self.final_output_dir = Path(output_dir)
        # 确保输出目录也包含日期层级 (如果传入的不是日期结尾，则追加日期)
        if self.final_output_dir.name != self.target_date:
            self.final_output_dir = self.final_output_dir / self.target_date
        
        self.final_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"🎯 最终输出目录 (NFS): {self.final_output_dir}")

        # 2. 设置临时工作目录 (/root/sztemp/process/{date})
        # 避开 NFS 锁，速度快，支持断点续传
        self.temp_work_dir = Path(f"/root/sztemp/process/{self.target_date}")
        self.temp_work_dir.mkdir(parents=True, exist_ok=True)
        print(f"🚀 临时工作目录 (Temp): {self.temp_work_dir}")

        # 使用 data_process 注入的日志路径（若未注入，则回退到默认）
        self.log_path = Path(PROCESSED_LOG_FILE) if PROCESSED_LOG_FILE else (self.final_output_dir / "processed_posts.log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._processed_files = self._load_log()
        self.db_conns = {} 

    def _load_log(self):
        processed = set()
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r', encoding='utf-8') as f:
                    processed = {line.strip() for line in f if line.strip()}
            except: pass
        return processed

    def _get_db_conn(self, platform):
        if platform not in self.db_conns or self.db_conns[platform] is None:
            # 🔥 数据库文件指向临时目录
            db_path = self.temp_work_dir / f"posts_{platform}.db"
            
            # 检查是否已有文件（断点续传）
            is_new_db = not db_path.exists()
            if is_new_db:
                print(f"   ✨ Creating new DB for [{platform}] in temp dir.")
            else:
                # 只有在第一次连接时打印一次 Resume
                pass 
                
            # 使用本地临时文件连接
            conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=60)
            cur = conn.cursor()
            
            # 🔥 极速配置 (因为是本地盘/Temp，不怕锁，可以全速写入)
            cur.execute("PRAGMA journal_mode = OFF;") 
            cur.execute("PRAGMA synchronous = OFF;")
            cur.execute("PRAGMA cache_size = -2000000;") # 2GB
            cur.execute("PRAGMA temp_store = MEMORY;")
            
            # 如果是新文件，创建表结构
            cur.execute("""
            CREATE TABLE IF NOT EXISTS post (
                post_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, 
                original_post_id INTEGER,
                title TEXT,
                content TEXT,
                quote_content TEXT,
                created_at DATETIME,
                temp_sjcjId TEXT, 
                temp_original_sjcjId TEXT
            );
            """)
            
            self.db_conns[platform] = conn
            
        return self.db_conns[platform]

    def _flush_buffer(self, platform, rows):
        try:
            conn = self._get_db_conn(platform)
            cur = conn.cursor()
            cur.execute("BEGIN TRANSACTION;")
            cur.executemany("""
                INSERT INTO post (
                    user_id, original_post_id, title, content, quote_content, 
                    created_at, temp_sjcjId, temp_original_sjcjId
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
        except Exception as e:
            print(f"\n❌ Temp DB Write Error [{platform}]: {e}")
            # 如果临时盘写入出错（如空间满），重置连接
            try: self.db_conns[platform].close()
            except: pass
            self.db_conns[platform] = None

    def run(self):
        files = list(self.input_dir.rglob('*.txt'))
        files = [f for f in files if f.name not in self._processed_files]
        if not files:
            print("No new files.")
            return
        
        print(f"🚀 正在处理 {len(files)} 个文件 (Input: NFS -> Temp: /root/sztemp)...")
        
        manager = multiprocessing.Manager()
        queue = manager.Queue(maxsize=50)
        pool = multiprocessing.Pool(max(1, os.cpu_count() - 2))
        
        for f in files:
            pool.apply_async(_worker_process_file, args=(f, queue))
        pool.close()
        
        buffers = defaultdict(list)
        total_inserted = 0
        files_done = 0
        
        try:
            while files_done < len(files):
                msg, payload = queue.get()
                
                if msg == 'DATA':
                    for platform, row in payload:
                        buffers[platform].append(row)
                    
                    # 缓冲区写满则写入 (Temp 盘写入快，批次 10万)
                    for platform, rows in buffers.items():
                        if len(rows) >= 100000: 
                            self._flush_buffer(platform, rows)
                            total_inserted += len(rows)
                            print(f"\r  -> [Temp] Inserted {len(rows)} to [{platform}]. Total: {total_inserted}", end="")
                            buffers[platform] = []
                            
                elif msg == 'FILE_DONE':
                    files_done += 1
                    self._write_log([payload])
                    if files_done % 10 == 0:
                        print(f"\n📄 Processed {payload} ({files_done}/{len(files)})")
                        
                elif msg == 'ERROR':
                    print(f"\n❌ {payload}")
                    files_done += 1
            
            # 写入剩余数据
            for platform, rows in buffers.items():
                if rows:
                    self._flush_buffer(platform, rows)
            
            print(f"\n✅ 数据写入完成! 总条数: {total_inserted}")
            
            # 1. 建立索引 (在 Temp 盘进行，速度最快)
            self._finalize_indexes()
            
            # 2. 关闭所有连接，释放文件句柄 (必须关闭才能移动)
            for conn in self.db_conns.values():
                if conn: conn.close()
            self.db_conns = {} 

            # 🔥 3. 搬运文件：从 Temp -> NFS 原目录
            print(f"\n🚚 正在将数据库从临时目录 {self.temp_work_dir} 移回原目录 {self.final_output_dir}...")
            
            for db_file in self.temp_work_dir.glob("*.db"):
                target_path = self.final_output_dir / db_file.name
                
                # 如果目标存在，先删除（确保覆盖）
                if target_path.exists():
                    target_path.unlink()
                
                print(f"  -> Moving {db_file.name} ({db_file.stat().st_size / 1024 / 1024:.2f} MB)...")
                try:
                    shutil.move(str(db_file), str(target_path))
                except OSError as e:
                    print(f"  ⚠️ Move failed ({e}), trying copy + delete...")
                    try:
                        shutil.copy2(str(db_file), str(target_path))
                        db_file.unlink()
                    except Exception as e2:
                        print(f"  ❌ Copy failed: {e2}")

            # 🔥 4. 删除临时目录
            print(f"🧹 删除临时目录: {self.temp_work_dir}")
            try:
                shutil.rmtree(self.temp_work_dir)
            except Exception as e:
                print(f"⚠️ 临时目录删除失败 (可手动删除): {e}")

            print("🎉 全部完成！")
            
        finally:
            pool.terminate()

    def _write_log(self, file_names):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            for name in file_names: f.write(name + '\n')

    def _finalize_indexes(self):
        print("\n🔨 正在临时目录建立索引...")
        for platform, conn in self.db_conns.items():
            print(f"  -> Indexing {platform}...")
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_uid ON post(user_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_temp_id ON post(temp_sjcjId);")
                conn.commit()
            except Exception as e:
                print(f"  ❌ Indexing failed for {platform}: {e}")

if __name__ == "__main__":
    MultiDBProcessor().run()