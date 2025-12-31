import sqlite3
import argparse
from pathlib import Path
import time
from datetime import datetime
import sys

# --- 配置 ---
# 默认时间点，可由命令行覆盖
DEFAULT_SPLIT_TIME = "2025-06-02 16:30:00"

class DatabaseSplitter:
    """
    负责将统一的 post 表拆分为 post (校准集) 和 ground_truth_post (真值集)。
    使用纯 SQL 操作，速度极快。
    """
    def __init__(self, db_path: str, split_time_str: str):
        self.db_path = Path(db_path)
        self.split_time = split_time_str
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"❌ 数据库不存在: {self.db_path}")

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        # 性能优化配置
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA cache_size = -500000;") # 2GB 缓存
        return conn

    def run(self):
        print(f"🚀 开始分割数据库: {self.db_path.name}")
        print(f"⏱️  分割时间点 (Split Time): {self.split_time}")
        
        start_time = time.time()
        conn = self._get_conn()
        cur = conn.cursor()

        try:
            # 1. 检查是否存在 unified 'post' 表
            # 我们假设之前的 UnifiedProcessor 生成的是 'post' 表
            # 为了分割，我们先把它重命名为 'all_posts_backup'
            print("\n📦 [Step 1] 备份原始数据...")
            
            # 检查表是否存在
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='post';")
            if not cur.fetchone():
                print("❌ 错误: 数据库中找不到 'post' 表。请确保你运行了 UnifiedProcessor。")
                return

            # 如果 backup 已经存在（可能是上次中断了），先删除
            cur.execute("DROP TABLE IF EXISTS all_posts_backup;")
            
            # 重命名 post -> all_posts_backup
            # 这样我们就可以新建两个干净的表 post 和 ground_truth_post
            cur.execute("ALTER TABLE post RENAME TO all_posts_backup;")
            print("-> 原始 'post' 表已重命名为 'all_posts_backup'")

            # 2. 创建新表结构
            print("\n🏗️ [Step 2] 创建目标表结构 (Calibration & Ground Truth)...")
            
            # 定义表结构的 SQL (与 UnifiedProcessor 保持一致)
            create_table_sql = """
            (
                post_id INTEGER PRIMARY KEY, -- 注意：这里去掉 AUTOINCREMENT，因为我们要保留原ID
                user_id TEXT, 
                original_post_id INTEGER,
                content TEXT,
                quote_content TEXT,
                created_at DATETIME,
                temp_sjcjId TEXT, 
                temp_original_sjcjId TEXT
            )
            """
            
            cur.execute(f"CREATE TABLE post {create_table_sql};")
            cur.execute(f"CREATE TABLE ground_truth_post {create_table_sql};")
            
            # 3. 数据分发 (Bulk Insert)
            print("\n🔄 [Step 3] 执行数据分发 (SQL Bulk Insert)...")
            t_split = time.time()
            
            columns = "post_id, user_id, original_post_id, content, quote_content, created_at, temp_sjcjId, temp_original_sjcjId"
            
            # 分割逻辑：
            # Calibration: created_at <= split_time
            # Ground Truth: created_at > split_time
            
            print("  -> 正在提取 Calibration Set (post)...")
            cur.execute("BEGIN TRANSACTION;")
            cur.execute(f"""
                INSERT INTO post ({columns})
                SELECT {columns} FROM all_posts_backup
                WHERE created_at <= ?
            """, (self.split_time,))
            conn.commit()
            
            print("  -> 正在提取 Ground Truth Set (ground_truth_post)...")
            cur.execute("BEGIN TRANSACTION;")
            cur.execute(f"""
                INSERT INTO ground_truth_post ({columns})
                SELECT {columns} FROM all_posts_backup
                WHERE created_at > ?
            """, (self.split_time,))
            conn.commit()
            
            print(f"-> 分割耗时: {time.time() - t_split:.2f}s")

            # 4. 重建索引
            print("\n🔨 [Step 4] 重建索引...")
            for table in ['post', 'ground_truth_post']:
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_userid ON {table} (user_id);")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_tempid ON {table} (temp_sjcjId);")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_temporig ON {table} (temp_original_sjcjId);")
            conn.commit()

            # 5. 重新建立链接 (Relinking)
            # 因为数据分开了，我们需要确保 original_post_id 指向正确的表
            # 注意：我们保留了原始 post_id，所以 ID 是不变的，这极大简化了逻辑
            # 但是，ground_truth 可能会引用 post (calibration) 里的数据，或者 gt 里的数据
            
            print("\n🔗 [Step 5] 验证引用完整性 (Relinking)...")
            # 理论上 UnifiedProcessor 已经计算好了 original_post_id。
            # 如果 post_id 没变，且 original_post_id 存的是 ID 数字，那么关系依然是有效的。
            # 唯一的区别是跨表查询。
            
            # 但为了保险（防止之前的 link 指向了错误的范围），我们这里重新刷一遍 Link 逻辑
            # 特别是 Ground Truth，它既可能引用 GT，也可能引用 Calibration
            
            t_link = time.time()
            
            # 5.1 Link Calibration (只引用自己)
            cur.execute("""
                UPDATE post
                SET original_post_id = (
                    SELECT p2.post_id FROM post AS p2
                    WHERE p2.temp_sjcjId = post.temp_original_sjcjId
                )
                WHERE post.temp_original_sjcjId IS NOT NULL;
            """)
            
            # 5.2 Link Ground Truth (优先找 GT，找不到找 Calib)
            # COALESCE 会返回第一个非空值
            cur.execute("""
                UPDATE ground_truth_post
                SET original_post_id = COALESCE(
                    (SELECT p_cal.post_id FROM post AS p_cal WHERE p_cal.temp_sjcjId = ground_truth_post.temp_original_sjcjId),
                    (SELECT p_gt.post_id FROM ground_truth_post AS p_gt WHERE p_gt.temp_sjcjId = ground_truth_post.temp_original_sjcjId)
                )
                WHERE ground_truth_post.temp_original_sjcjId IS NOT NULL;
            """)
            conn.commit()
            print(f"-> Link 修复耗时: {time.time() - t_link:.2f}s")

            # 6. 清理
            print("\n🧹 [Step 6] 清理临时数据...")
            # 默认保留 backup 还是删除？为了节省空间，通常建议删除，或者询问用户
            # 这里我们为了安全，选择删除，因为原数据是 Raw 生成的，随时可以重跑
            cur.execute("DROP TABLE all_posts_backup;")
            cur.execute("VACUUM;") # 释放磁盘空间 (这一步可能稍慢，如果不需要可以注释掉)
            print("-> 临时表已清理，数据库已压缩。")

            # 统计
            count_cal = cur.execute("SELECT COUNT(*) FROM post").fetchone()[0]
            count_gt = cur.execute("SELECT COUNT(*) FROM ground_truth_post").fetchone()[0]

            print(f"\n✅ 分割完成！")
            print(f"   📊 Calibration Set (post): {count_cal} 条")
            print(f"   📊 Ground Truth Set (gt) : {count_gt} 条")
            print(f"   ⏱️ 总耗时: {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"\n❌ 分割过程中出错: {e}")
            # 尝试回滚 (如果 backup 存在，恢复它)
            try:
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='all_posts_backup';")
                if cur.fetchone():
                    print("⚠️ 正在尝试恢复原始 'post' 表...")
                    cur.execute("DROP TABLE IF EXISTS post;")
                    cur.execute("DROP TABLE IF EXISTS ground_truth_post;")
                    cur.execute("ALTER TABLE all_posts_backup RENAME TO post;")
                    print("✅ 恢复成功。数据库回到初始状态。")
            except:
                pass
            sys.exit(1)
        finally:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description="Split Unified DB into Calibration and Ground Truth tables")
    parser.add_argument("--db", required=True, help="Path to the unified SQLite database")
    parser.add_argument("--split_time", default=DEFAULT_SPLIT_TIME, help=f"Cutoff time (default: {DEFAULT_SPLIT_TIME})")
    
    args = parser.parse_args()
    
    splitter = DatabaseSplitter(args.db, args.split_time)
    splitter.run()

if __name__ == "__main__":
    main()