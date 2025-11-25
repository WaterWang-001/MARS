import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
import time
import sys
from tqdm import tqdm  # 引入进度条库

class OasisPostProcessor:
    """
    将原有 oasis_post.py 的逻辑封装为类接口。
    """
    def __init__(self,
                 source_db: str,
                 oasis_db: str,
                 calibration_end: datetime = datetime(2025,6,2,16,30,0),
                 ground_truth_end: datetime = datetime(2025,6,2,16,45,0),
                 batch_size: int = 50000,
                 create_calibration: bool = True,
                 create_ground_truth: bool = True):
        self.SOURCE_DB_PATH = source_db
        self.OASIS_DB_PATH = oasis_db
        self.CALIBRATION_END_TIME = calibration_end
        self.GROUND_TRUTH_END_TIME = ground_truth_end
        self.BATCH_SIZE = batch_size
        self.CREATE_CALIBRATION_SET = create_calibration
        self.CREATE_GROUND_TRUTH_SET = create_ground_truth

    def create_target_table(self, target_conn, table_name: str):
        cur = target_conn.cursor()
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
        # 显式为 user_id 创建索引，极大加速后续按用户查询的速度
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_userid ON {table_name} (user_id);")
        target_conn.commit()

    @staticmethod
    def parse_log_line(data_json: str):
        try:
            data = json.loads(data_json)
            # 尝试不同的字段组合来提取内容
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

    def migrate_data(self, source_conn, target_conn, table_name: str, sql_filter: str, filter_params: tuple):
        source_cur = source_conn.cursor()
        target_cur = target_conn.cursor()
        
        # 优化写入性能
        target_cur.execute("PRAGMA journal_mode = WAL;")
        target_cur.execute("PRAGMA synchronous = NORMAL;")
        
        # 1. 获取总行数用于进度条
        print(f"正在计算 {table_name} 的待处理数据总量...")
        count_sql = f"SELECT COUNT(*) FROM content {sql_filter}"
        source_cur.execute(count_sql, filter_params)
        total_records = source_cur.fetchone()[0]
        print(f"-> 共发现 {total_records} 条记录。")

        # 2. 开始查询数据
        source_cur.execute(
            f"SELECT user_id, timestamp, data_json FROM content {sql_filter}",
            filter_params
        )

        insert_batch = []
        total_rows = 0
        error_rows = 0
        start_time = time.time()

        # 3. 使用 tqdm 包装循环
        with tqdm(total=total_records, desc=f"Migrating {table_name}", unit="row") as pbar:
            while True:
                rows = source_cur.fetchmany(self.BATCH_SIZE)
                if not rows:
                    break
                
                for row in rows:
                    total_rows += 1
                    user_id = row[0]
                    timestamp_ms = row[1]
                    data_json = row[2]
                    try:
                        created_at = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        error_rows += 1
                        continue
                    
                    parsed = self.parse_log_line(data_json)
                    (source_sjcjId, source_original_sjcjId, content, quote_content) = parsed
                    
                    if source_sjcjId is None:
                        error_rows += 1
                        continue
                    
                    insert_batch.append((
                        str(user_id),
                        None,
                        content,
                        quote_content,
                        created_at,
                        source_sjcjId,
                        source_original_sjcjId
                    ))
                
                # 批量插入
                if insert_batch:
                    target_cur.executemany(f"""
                        INSERT OR IGNORE INTO {table_name} (
                            user_id, original_post_id, content, quote_content, created_at,
                            temp_sjcjId, temp_original_sjcjId
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, insert_batch)
                    target_conn.commit()
                    
                    # 更新进度条
                    pbar.update(len(rows)) # 注意这里更新的是读取的行数，而不是插入的行数（因为有过滤）
                    insert_batch = []
        
        end_time = time.time()
        inserted_rows = target_cur.execute(f"SELECT COUNT(post_id) FROM {table_name}").fetchone()[0]
        
        return {
            "total_rows": total_rows,
            "inserted_rows": inserted_rows,
            "error_rows": error_rows,
            "elapsed": end_time - start_time
        }

    def link_table_post(self, conn, table_name):
        print(f"正在构建 {table_name} 的引用关系 (Linking)... 这可能需要一些时间...")
        cur = conn.cursor()
        start_link = time.time()

        if table_name == "post":
            cur.execute("CREATE INDEX IF NOT EXISTS idx_post_temp_original ON post (temp_original_sjcjId);")
            conn.commit()
            cur.execute("""
                UPDATE post
                SET original_post_id = (
                    SELECT p2.post_id FROM post AS p2
                    WHERE p2.temp_sjcjId = post.temp_original_sjcjId
                )
                WHERE post.temp_original_sjcjId IS NOT NULL;
            """)
            conn.commit()
        else:
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_temp_original ON {table_name} (temp_original_sjcjId);")
            conn.commit()
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
        print(f"源数据库: {self.SOURCE_DB_PATH}")
        print(f"目标数据库: {self.OASIS_DB_PATH}")
        
        cal_end_ms = int(self.CALIBRATION_END_TIME.timestamp() * 1000)
        gt_end_ms = int(self.GROUND_TRUTH_END_TIME.timestamp() * 1000)
        
        try:
            if self.CREATE_CALIBRATION_SET:
                print("\n=== 阶段 1: 生成 Calibration 集 (post) ===")
                # 如果存在目标库，先删除，保证重新生成
                if os.path.exists(self.OASIS_DB_PATH) and self.CREATE_CALIBRATION_SET and self.CREATE_GROUND_TRUTH_SET:
                    # 只有当想完全重跑时才删除，这里逻辑可以根据需求调整
                    # 简单起见，如果是全新的运行，我们假设用户希望覆盖
                    pass 

                with sqlite3.connect(self.SOURCE_DB_PATH) as source_conn:
                    with sqlite3.connect(self.OASIS_DB_PATH) as target_conn:
                        self.create_target_table(target_conn, "post")
                        
                        stats = self.migrate_data(source_conn, target_conn, "post", "WHERE timestamp <= ?", (cal_end_ms,))
                        print(f"Calibration 完成: 读取 {stats['total_rows']}, 插入 {stats['inserted_rows']}, 错误/跳过 {stats['error_rows']}, 耗时 {stats['elapsed']:.2f}s")
                        
                        self.link_table_post(sqlite3.connect(self.OASIS_DB_PATH), "post")

            if self.CREATE_GROUND_TRUTH_SET:
                print("\n=== 阶段 2: 生成 Ground Truth 集 (ground_truth_post) ===")
                with sqlite3.connect(self.SOURCE_DB_PATH) as source_conn:
                    with sqlite3.connect(self.OASIS_DB_PATH) as target_conn:
                        self.create_target_table(target_conn, "ground_truth_post")
                        
                        stats = self.migrate_data(source_conn, target_conn, "ground_truth_post", "WHERE timestamp > ? AND timestamp <= ?", (cal_end_ms, gt_end_ms))
                        print(f"Ground Truth 完成: 读取 {stats['total_rows']}, 插入 {stats['inserted_rows']}, 错误/跳过 {stats['error_rows']}, 耗时 {stats['elapsed']:.2f}s")
                        
                        self.link_table_post(sqlite3.connect(self.OASIS_DB_PATH), "ground_truth_post")
                        
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    print("-" * 50)
    print(" OASIS Post Database Processor Test Interface")
    print("-" * 50)
    
    # 1. 获取输入数据库路径
    default_source = '/remote-home/JuelinW/oasis_project/MARS/data/output/2025-06-14/user_post_database.db'
    user_input = input(f"请输入源数据库路径 [回车使用默认值: {default_source}]: ").strip()
    
    source_db = user_input if user_input else default_source
    
    if not os.path.exists(source_db):
        print(f"错误: 找不到文件 {source_db}")
        return

    # 2. 定义输出路径
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f'/remote-home/JuelinW/oasis_project/MARS/data/output/2025-06-14/oasis_database_test_{timestamp_str}.db'
    oasis_db = default_output # 这里为了测试方便直接生成一个带时间戳的新文件，防止覆盖重要数据
    
    print(f"\n配置确认:")
    print(f"源: {source_db}")
    print(f"目标: {oasis_db}")
    
    confirm = input("是否开始处理? (y/n): ").lower()
    if confirm != 'y':
        print("已取消。")
        return

    # 3. 运行处理器
    # 注意：这里的时间您可以根据您的实际数据调整
    processor = OasisPostProcessor(
        source_db=source_db,
        oasis_db=oasis_db,
        calibration_end=datetime(2025, 6, 2, 16, 30, 0), # 请根据您的数据实际时间范围调整
        ground_truth_end=datetime(2025, 6, 2, 16, 45, 0),
        batch_size=10000 # 测试时可以调小一点看看效果
    )
    
    start = time.time()
    processor.run()
    print(f"\n✅ 全部任务完成，总耗时: {time.time() - start:.2f}秒")
    print(f"输出文件位于: {oasis_db}")

if __name__ == "__main__":
    main()