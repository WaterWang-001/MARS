import sqlite3
import os
from datetime import datetime, timedelta
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

    def get_batch_posts(self, user_ids: List[str], max_count: int = 100) -> Dict[str, List[Dict]]:
        """
        批量读取用户帖子。
        1. 第一步：批量统计每个用户的发帖总数 (COUNT)。
        2. 第二步：过滤掉发帖数超过 max_count 的用户 (视为刷屏/营销号)。
        3. 第三步：对剩下的合格用户，批量读取内容。
        """
        if not user_ids:
            return {}

        results = {}
        # 初始化 Key，确保被过滤的用户返回空列表
        for uid in user_ids:
            results[str(uid)] = []

        with self.lock:
            try:
                conn = self.connect()
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # -------------------------------------------------------
                # Step 1: 批量统计数量 (极速，只走索引)
                # -------------------------------------------------------
                placeholders = ','.join(['?'] * len(user_ids))
                
                stats_sql = f"""
                    SELECT user_id, COUNT(*) as total_count
                    FROM post 
                    WHERE user_id IN ({placeholders})
                    GROUP BY user_id
                """
                
                cursor.execute(stats_sql, tuple(user_ids))
                stats_rows = cursor.fetchall()
                
                valid_uids = []
                
                for row in stats_rows:
                    uid = str(row['user_id'])
                    total = row['total_count']
                    
            
                    if total > max_count:
                        # 你可以在这里打印日志，或者保持静默
                        # print(f"⏩ [Filter] User {uid} skipped ({total} posts > {max_count})")
                        continue
                    
                    valid_uids.append(uid)
                # 如果全部都被过滤掉了，直接返回
                if not valid_uids:
                    conn.close()
                    return results

                # -------------------------------------------------------
                # Step 2: 批量读取详情 (只读合格的用户)
                # -------------------------------------------------------
                valid_placeholders = ','.join(['?'] * len(valid_uids))
                
                fetch_sql = f"""
                    SELECT user_id, post_id, original_post_id, content, quote_content, created_at
                    FROM post 
                    WHERE user_id IN ({valid_placeholders})
                    ORDER BY created_at DESC
                """
                
                cursor.execute(fetch_sql, tuple(valid_uids))
                rows = cursor.fetchall()
                
                for row in rows:
                    post_data = dict(row)
                    if post_data.get('content') is None:
                        post_data['content'] = ""
                    
                    if post_data.get('quote_content') is None:
                        post_data['quote_content'] = ""
                    uid = str(post_data['user_id'])
                    results[uid].append(post_data)
                
                conn.close()
                
            except Exception as e:
                print(f"❌ DB Filtered-Batch Error: {e}")
                # 出错不中断，返回空字典
                return results
        
        return results

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
    
    def get_user_posts_incremental(self, user_id, after_time):
        results = []
        # 加上锁，防止多线程操作 SQLite 报错
        with self.lock:
            try:
                # 修正点：将 self._get_conn() 改为 self.connect()
                conn = self.connect() 
                cursor = conn.cursor()
                
                # 注意：请确认表名是 'post' 还是 'posts'？你的 get_user_posts 用的是 'post'
                cursor.execute("""
                    SELECT post_id, original_post_id, content, quote_content, 
                           created_at
                    FROM post 
                    WHERE user_id = ? AND created_at > ?
                    ORDER BY created_at ASC
                """, (str(user_id), str(after_time)))
                
                # 将 tuple 转换为 dict，避免 dict(row) 可能失效的问题
                columns = [col[0] for col in cursor.description]
                for row in cursor.fetchall():
                    d = dict(zip(columns, row))
                    
                    raw_time = d.get('created_at')
                    if raw_time:
                        d['created_at'] = self._utc_to_gmt8(raw_time)
                    
                    results.append(d)
                    results.append(dict(zip(columns, row)))
                    
                conn.close()
            except Exception as e:
                print(f"DB Error (After) for user {user_id}: {e}")
                return []
                
        return results
    def _utc_to_gmt8(self, time_str):
        """将数据库取出的 UTC 时间转换为 GMT+8"""
        if not time_str: 
            return ""
        try:
            # 假设数据库存的是标准格式 "YYYY-MM-DD HH:MM:SS"
            dt_utc = datetime.strptime(str(time_str), "%Y-%m-%d %H:%M:%S")
            dt_gmt8 = dt_utc + timedelta(hours=8)
            return dt_gmt8.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            # 如果解析失败（格式不对），返回原值，避免崩溃
            return str(time_str)