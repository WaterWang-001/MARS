import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
import time
import sys
from tqdm import tqdm

class OasisPostProcessor:
    def __init__(self,
                 source_db: str,
                 oasis_db: str,
                 calibration_end: datetime = datetime(2025,6,2,16,30,0),
                 ground_truth_end: datetime = datetime(2025,6,2,16,45,0),
                 batch_size: int = 50000,
                 create_calibration: bool = True,
                 create_ground_truth: bool = True,
                 skip_count: bool = True):  # 新增：默认跳过 COUNT
        self.SOURCE_DB_PATH = source_db
        self.OASIS_DB_PATH = oasis_db
        self.CALIBRATION_END_TIME = calibration_end
        self.GROUND_TRUTH_END_TIME = ground_truth_end
        self.BATCH_SIZE = batch_size
        self.CREATE_CALIBRATION_SET = create_calibration
        self.CREATE_GROUND_TRUTH_SET = create_ground_truth
        self.SKIP_COUNT = skip_count

    def _init_pragmas(self, conn):
        cur = conn.cursor()
        cur.execute("PRAGMA busy_timeout = 5000;")
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.execute("PRAGMA cache_size = -500000;")
        cur.execute("PRAGMA temp_store = MEMORY;")
        conn.commit()

    def create_target_table_schema(self, target_conn, table_name: str):
        """
        阶段1：只创建表结构和必要的唯一约束，不创建辅助索引。
        """
        cur = target_conn.cursor()
        # 仅保留 temp_sjcjId 的 UNIQUE 约束用于 INSERT OR IGNORE 去重
        # 移除 user_id 的索引，移除 temp_original_sjcjId 的索引，改为最后创建
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            post_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT, 
            original_post_id INTEGER,
            content TEXT DEFAULT '',
            quote_content TEXT,
            created_at DATETIME,
            num_likes INTEGER DEFAULT 0,
            num_dislikes INTEGER DEFAULT 0,
            num_shares INTEGER DEFAULT 0,
            num_reports INTEGER DEFAULT 0,
            temp_sjcjId TEXT UNIQUE, 
            temp_original_sjcjId TEXT,
            FOREIGN KEY(user_id) REFERENCES user(user_id),
            FOREIGN KEY(original_post_id) REFERENCES {table_name}(post_id)
        );
        """)
        target_conn.commit()

    def build_indexes(self, target_conn, table_name: str):
        """
        阶段2：数据迁移完成后，批量构建索引。
        这是提速的关键：Bulk Create Index 比 Incremental Create 快得多。
        """
        print(f"🔨 正在为 {table_name} 构建索引 (Index Building)...")
        start = time.time()
        cur = target_conn.cursor()
        
        # 1. UserID 索引 (用于查询)
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_userid ON {table_name} (user_id);")
        
        # 2. 引用ID 索引 (加速 UPDATE 里的 WHERE 子句)
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_temp_original ON {table_name} (temp_original_sjcjId);")
        
        # 3. 自身ID 索引 (加速 UPDATE 里的子查询 SELECT ... WHERE temp_sjcjId = ...)
        # 注意：UNIQUE 约束通常已隐含索引，但显式确认一下无害
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_temp_sjcjId ON {table_name} (temp_sjcjId);")
        
        target_conn.commit()
        print(f"-> 索引构建完成，耗时: {time.time() - start:.2f}s")

    @staticmethod
    def parse_log_line(data_json: str):
        # ... (保持原有的解析逻辑不变) ...
        try:
            data = json.loads(data_json)
            comment_pojo = data.get('commentPojo')
            content_pojo = data.get('contentPojo')
            comment_fwd_pojo = data.get('commentForwardPojo')
            content_fwd_pojo = data.get('contentForwardPojo')
            content_root_pojo = data.get('contentRootPojo')

            source_sjcjId = None
            source_original_sjcjId = None
            content = ""
            quote_content = None

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
                return (None, None, None, None)
            return (source_sjcjId, source_original_sjcjId, content, quote_content)
        except (json.JSONDecodeError, TypeError):
            return (None, None, None, None)

    def migrate_data(self, source_conn, target_conn, table_name: str, sql_filter: str,
                     filter_params: tuple, do_count: bool = True):
        source_cur = source_conn.cursor()
        target_cur = target_conn.cursor()
        source_cur.row_factory = None

        total_records = None
        if do_count:
            print(f"正在计算 {table_name} 的待处理数据总量...")
            count_sql = f"SELECT COUNT(*) FROM content {sql_filter}"
            source_cur.execute(count_sql, filter_params)
            total_records = source_cur.fetchone()[0]
            print(f"-> 共发现 {total_records} 条记录。")

        source_cur.execute(
            f"SELECT user_id, timestamp, data_json FROM content {sql_filter}",
            filter_params
        )

        insert_batch = []
        total_rows = 0
        error_rows = 0
        start_time = time.time()

        # 使用事务包裹整个大循环的若干个批次，进一步减少 I/O
        target_cur.execute("BEGIN TRANSACTION")
        
        with tqdm(total=total_records if total_records is not None else None,
                  desc=f"Migrating {table_name}", unit="row") as pbar:
            while True:
                rows = source_cur.fetchmany(self.BATCH_SIZE)
                if not rows:
                    break
                
                current_batch_parsed = []
                
                for row in rows:
                    total_rows += 1
                    # 索引访问比解包稍快
                    timestamp_ms = row[1]
                    
                    try:
                        # 优化：简单的字符串格式化比 datetime 转换快
                        # 但为了稳健性保留 datetime，除非这是极端瓶颈
                        created_at = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')
                        
                        parsed = self.parse_log_line(row[2])
                        # (source_sjcjId, source_original_sjcjId, content, quote_content) = parsed
                        
                        if parsed[0] is None:
                            error_rows += 1
                            continue
                        
                        current_batch_parsed.append((
                            str(row[0]), # user_id
                            None,        # original_post_id (wait for link)
                            parsed[2],   # content
                            parsed[3],   # quote_content
                            created_at,
                            parsed[0],   # temp_sjcjId
                            parsed[1]    # temp_original_sjcjId
                        ))

                    except Exception:
                        error_rows += 1
                        continue
                
                if current_batch_parsed:
                    target_cur.executemany(f"""
                        INSERT OR IGNORE INTO {table_name} (
                            user_id, original_post_id, content, quote_content, created_at,
                            temp_sjcjId, temp_original_sjcjId
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, current_batch_parsed)
                    
                    # 定期提交，防止 WAL 文件过大
                    if total_rows % (self.BATCH_SIZE * 5) == 0:
                        target_conn.commit()
                        target_cur.execute("BEGIN TRANSACTION")
                    
                    pbar.update(len(rows))

        target_conn.commit() # 最终提交
        
        end_time = time.time()
        inserted_rows = target_cur.execute(f"SELECT COUNT(post_id) FROM {table_name}").fetchone()[0]
        
        return {
            "total_rows": total_rows,
            "inserted_rows": inserted_rows,
            "error_rows": error_rows,
            "elapsed": end_time - start_time
        }

    def link_table_post(self, conn, table_name):
        print(f"🔗 正在构建 {table_name} 的引用关系 (Linking)...")
        cur = conn.cursor()
        start_link = time.time()

        # 注意：这里的索引已经在 build_indexes 中创建了，所以这里直接跑 UPDATE 即可
        # 数据库引擎会自动利用索引加速 UPDATE 的 WHERE 和 子查询

        if table_name == "post":
            # 自引用更新
            cur.execute("""
                UPDATE post
                SET original_post_id = (
                    SELECT p2.post_id FROM post AS p2
                    WHERE p2.temp_sjcjId = post.temp_original_sjcjId
                )
                WHERE post.temp_original_sjcjId IS NOT NULL;
            """)
        else:
            # 引用 post 或 ground_truth_post
            # 使用 COALESCE 查找两个表
            cur.execute(f"""
                UPDATE {table_name}
                SET original_post_id = COALESCE(
                    (SELECT p_cal.post_id FROM post AS p_cal WHERE p_cal.temp_sjcjId = {table_name}.temp_original_sjcjId),
                    (SELECT p_gt.post_id FROM ground_truth_post AS p_gt WHERE p_gt.temp_sjcjId = {table_name}.temp_original_sjcjId)
                )
                WHERE {table_name}.temp_original_sjcjId IS NOT NULL;
            """)
            
        conn.commit()
        print(f"-> 关系构建完成，耗时: {time.time() - start_link:.2f}s")

    def run(self):
        if not os.path.exists(self.SOURCE_DB_PATH):
            raise FileNotFoundError(f"源数据库不存在: {self.SOURCE_DB_PATH}")
        
        print(f"🚀 开始处理...")
        
        cal_end_ms = int(self.CALIBRATION_END_TIME.timestamp() * 1000)
        gt_end_ms = int(self.GROUND_TRUTH_END_TIME.timestamp() * 1000)
        
        try:
            # 统一连接，避免反复开关
            with sqlite3.connect(self.OASIS_DB_PATH) as target_conn:
                self._init_pragmas(target_conn)
                
                with sqlite3.connect(self.SOURCE_DB_PATH) as source_conn:
                    
                    if self.CREATE_CALIBRATION_SET:
                        print("\n=== 阶段 1: 生成 Calibration 集 (post) ===")
                        # 1. 建表（无索引）
                        self.create_target_table_schema(target_conn, "post")
                        # 2. 迁移数据
                        stats = self.migrate_data(
                            source_conn, target_conn, "post",
                            "WHERE timestamp <= ?", (cal_end_ms,),
                            do_count=not self.SKIP_COUNT
                        )
                        print(f"Post 迁移完成: {stats['elapsed']:.2f}s")
                        # 3. 建索引（批量）
                        self.build_indexes(target_conn, "post")
                        # 4. 关联更新
                        self.link_table_post(target_conn, "post")

                    if self.CREATE_GROUND_TRUTH_SET:
                        print("\n=== 阶段 2: 生成 Ground Truth 集 (ground_truth_post) ===")
                        # 1. 建表
                        self.create_target_table_schema(target_conn, "ground_truth_post")
                        # 2. 迁移
                        stats = self.migrate_data(
                            source_conn, target_conn, "ground_truth_post",
                            "WHERE timestamp > ? AND timestamp <= ?", (cal_end_ms, gt_end_ms),
                            do_count=not self.SKIP_COUNT
                        )
                        print(f"GT 迁移完成: {stats['elapsed']:.2f}s")
                        # 3. 建索引
                        self.build_indexes(target_conn, "ground_truth_post")
                        # 4. 关联
                        self.link_table_post(target_conn, "ground_truth_post")
                        
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise

# ... (main 函数保持不变) ...
def main():
    print("-" * 50)
    print(" OASIS Post Database Processor Test Interface")
    print("-" * 50)
    
    # 1. 获取输入数据库路径
    source_db = 'MARS/data/output/2025-06-15/user_post_database.db'

    oasis_db = 'MARS/data/output/2025-06-15/oasis_post_database.db'
    
    print(f"\n配置确认:")
    print(f"源: {source_db}")
    print(f"目标: {oasis_db}")
    
    confirm = input("是否开始处理? (y/n): ").lower()
    if confirm != 'y':
        print("已取消。")
        return

    processor = OasisPostProcessor(
        source_db=source_db,
        oasis_db=oasis_db,
        calibration_end=datetime(2025, 6, 15, 16, 00, 0),
        ground_truth_end=datetime(2025, 6, 16, 00, 00, 0),
        batch_size=20000 
    )
    
    start = time.time()
    processor.run()
    print(f"\n✅ 全部任务完成，总耗时: {time.time() - start:.2f}秒")
    print(f"输出文件位于: {oasis_db}")

if __name__ == "__main__":
    main()