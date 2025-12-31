import sqlite3
import os
import threading  # 导入 threading
from typing import List, Dict, Set

class DBClient:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # 1. 初始化一个线程锁
        self.lock = threading.Lock()

    def connect(self):
        # 保持原有逻辑，建议 timeout 设置稍大一点
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False, timeout=60.0)

    def get_user_posts(self, user_id: str, limit: int = 20) -> List[Dict]:
        posts_list = []
        
        # 2. 使用锁包裹整个数据库操作过程
        # 这会强制所有线程排队读取 DB，防止文件锁冲突
        with self.lock:
            try:
                conn = self.connect()
                cursor = conn.cursor()
                # 假设表名为 post 和 ground_truth_post，请根据实际情况调整
                query = """
                    SELECT content, quote_content, created_at 
                    FROM post WHERE user_id = ?
                    ORDER BY created_at DESC LIMIT ?
                """
                cursor.execute(query, (user_id, limit))
                for row in cursor.fetchall():
                    content = row[0].strip() if row[0] else ""
                    quote = row[1].strip() if row[1] else ""
                    if content or quote:
                        posts_list.append({
                            "content": content,
                            "quote_content": quote,
                            "created_at": row[2]
                        })
                conn.close()
            except Exception as e:
                # 打印具体的错误，方便调试
                print(f"DB Error for user {user_id}: {e}")
                # 可选：如果报错，稍微等一下，虽然加锁后应该不会再报错了
                
        return posts_list

    def get_processed_users(self, log_file: str) -> Set[str]:
        # 读取日志文件通常很快，冲突概率低，但为了保险也可以加锁，
        # 或者因为只在程序开始运行一次，不加也可以。
        if not os.path.exists(log_file):
            return set()
        with open(log_file, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())

    # log_processed_user 在你的主程序里似乎已经加了外部锁(log_lock)，这里可以保持原样
    def log_processed_user(self, log_file: str, user_id: str):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{user_id}\n")
    
    def get_user_posts_after(self, user_id, after_time):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT post_id, original_post_id, content, quote_content, 
                   created_at, temp_sjcjId, temp_original_sjcjId
            FROM posts 
            WHERE user_id = ? AND created_at > ?
            ORDER BY created_at ASC
        """, (str(user_id), str(after_time)))
        return [dict(row) for row in cursor.fetchall()]