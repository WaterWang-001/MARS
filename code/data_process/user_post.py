import json
import sys
from pathlib import Path
import os
import gc
import sqlite3
import time
import typing
from typing import List

# --- 配置 (保持不变) ---
INPUT_DIRECTORY = 'data/raw/'
PERMANENT_DB_FILE = 'data/user_post/user_post_database.db'
# --------------------

# 新增：记录已处理文件的日志文件
PROCESSED_LOG_FILE = Path(INPUT_DIRECTORY) / ".processed_files.log"


class UserPostProcessor:
    """
    将原始 .txt 文件流式写入 SQLite 的封装类，支持断点续传。
    """
    def __init__(self, input_directory=INPUT_DIRECTORY, db_path=PERMANENT_DB_FILE, log_path=PROCESSED_LOG_FILE):
        self.input_directory = Path(input_directory)
        self.db_path = db_path
        
        # --- [修复点] ---
        # 强制将 log_path 转换为 Path 对象，防止传入 string 时报错
        self.log_path = Path(log_path) if log_path else None
        # ----------------
        
        self._processed_files = self._load_processed_log()

    def _load_processed_log(self) -> typing.Set[str]:
        """从日志文件中加载已处理的文件名集合。"""
        processed_set = set()
        
        # 检查路径是否存在且是文件
        if self.log_path and self.log_path.exists() and self.log_path.is_file():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    processed_set = {line.strip() for line in f if line.strip()}
                # 这里的 .name 现在安全了，因为 self.log_path 肯定是 Path 对象
                print(f"[Log] 已从 {self.log_path.name} 加载 {len(processed_set)} 个已处理文件记录。")
            except Exception as e:
                print(f"[Log] 警告: 无法读取日志文件 {self.log_path}: {e}", file=sys.stderr)
        
        return processed_set

    def _record_processed_log(self, filename: str):
        """将成功处理的文件名写入日志文件。"""
        if not self.log_path:
            return

        try:
            # 确保父目录存在，防止首次写入报错
            if not self.log_path.parent.exists():
                self.log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{filename}\n")
            self._processed_files.add(filename)
        except Exception as e:
            print(f"[Log] 警告: 无法写入日志文件 {self.log_path}: {e}", file=sys.stderr)
    def collect_file_list(self) -> List[Path]:
        if not self.input_directory.is_dir():
            raise FileNotFoundError(f"输入目录不存在: {self.input_directory}")
        
        all_files = list(self.input_directory.glob('*.txt'))
        
        # 核心修改：筛选未处理的文件
        file_list = [f for f in all_files if f.name not in self._processed_files]
        
        if not file_list:
            if len(all_files) > 0 and len(all_files) == len(self._processed_files):
                 print(f"在目录 '{self.input_directory}' 中发现所有 {len(all_files)} 个 .txt 文件均已处理，无需重复运行。")
            else:
                raise FileNotFoundError(f"在目录 '{self.input_directory}' 中找不到新的 .txt 文件需要处理。")
            return []
            
        print(f"[File] 发现 {len(all_files)} 个 .txt 文件，其中 {len(file_list)} 个待处理。")
        return file_list

    # 静态方法保持不变
    @staticmethod
    def get_user_id(pojo):
        if not pojo: return None
        return pojo.get('sjcjId')

    @staticmethod
    def get_post_timestamp(pojo):
        if not pojo: return None
        return pojo.get('sjcjPublished')

    @staticmethod
    def get_comment_timestamp(pojo):
        if not pojo: return None
        return pojo.get('sjcjPublished')

    # --- 重写 Pass 1 ---

    def _initialize_db(self, db_path: str) -> sqlite3.Connection:
        """连接数据库并进行初始化设置，支持增量写入。"""
        print(f"  -> 正在连接数据库: {db_path} ...")
        
        # 连接数据库 (如果文件不存在会自动创建)
        # isolation_level=None 启用自动提交，配合 WAL 提高写入速度
        conn = sqlite3.connect(db_path, isolation_level=None)
        cur = conn.cursor()
        
        # 优化1: 设置高性能 pragma
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        
        # 优化2: 创建表 (IF NOT EXISTS 支持增量写入)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS content (
            user_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            data_json TEXT NOT NULL
        );
        """)
        
        # 优化3: 确保索引存在 (这必须在 Pass 1 完成时执行，但提前创建 IF NOT EXISTS 也是安全的)
        # 我们把它留在 process_and_store_to_db 的末尾执行，以确保数据完整性
        
        return conn


    def process_and_store_to_db(self, file_list: List[Path], db_path: str):
        """
        Pass 1 (重写): 遍历所有文件，将数据流式存入 SQLite 数据库，支持断点续传。
        """
        if not file_list:
            return

        print(f"--- 🚀 Pass 1 (DB): 正在将 {len(file_list)} 个新文件流式传输到数据库... ---")
        
        total_line_count = 0
        total_error_count = 0
        post_count = 0
        comment_count = 0
        
        BATCH_SIZE = 50000 
        insert_batch = []
        
        conn = self._initialize_db(db_path)
        cur = conn.cursor()
        
        start_time = time.time()
        
        try:
            for filepath in file_list:
                print(f"  -> 正在处理: {filepath.name}")
                
                # 开始事务（可选，但对于批处理是好的实践）
                # cur.execute("BEGIN TRANSACTION;") 
                
                current_file_line_count = 0
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            total_line_count += 1
                            current_file_line_count += 1
                            
                            try:
                                data = json.loads(line.strip())
                                
                                # ... (数据提取逻辑保持不变) ...
                                author_content_pojo = data.get('authorContentPojo')
                                author_comment_pojo = data.get('authorCommentPojo')
                                content_pojo = data.get('contentPojo', {})
                                comment_pojo = data.get('commentPojo', {})
                                
                                user_id = None
                                timestamp = None
                                
                                if author_content_pojo and not author_comment_pojo:
                                    user_id = self.get_user_id(author_content_pojo)
                                    timestamp = self.get_post_timestamp(content_pojo)
                                    if user_id and timestamp is not None:
                                        post_count += 1
                                        
                                elif author_comment_pojo:
                                    user_id = self.get_user_id(author_comment_pojo)
                                    timestamp = self.get_comment_timestamp(comment_pojo)
                                    if user_id and timestamp is not None:
                                        comment_count += 1
                                
                                if user_id and timestamp is not None:
                                    insert_batch.append(
                                        (user_id, timestamp, json.dumps(data, ensure_ascii=False))
                                    )
                                
                                # 批量插入
                                if len(insert_batch) >= BATCH_SIZE:
                                    cur.executemany(
                                        "INSERT INTO content (user_id, timestamp, data_json) VALUES (?, ?, ?)",
                                        insert_batch
                                    )
                                    insert_batch = []
                                    
                            except (json.JSONDecodeError, Exception) as e:
                                if current_file_line_count % 10000 == 0:
                                    print(f"⚠️ 文件 {filepath.name} Line {line_num}: 处理时发生错误: {e}", file=sys.stderr)
                                total_error_count += 1
                                continue
                                
                    # 插入本文件剩余的批次数据
                    if insert_batch:
                        cur.executemany(
                            "INSERT INTO content (user_id, timestamp, data_json) VALUES (?, ?, ?)",
                            insert_batch
                        )
                        insert_batch = []

                    # 核心修改：记录日志！
                    self._record_processed_log(filepath.name)
                    print(f"  -> {filepath.name} 处理成功并记录到日志。")

                except Exception as e:
                    print(f"❌ 错误: 无法读取文件 {filepath.name}. 错误: {e}", file=sys.stderr)
                    # 在这里不记录日志，下次运行会重试该文件
                    total_error_count += 1
            
            # --- 最终优化：创建索引 (如果数据库是新的或者没有索引) ---
            end_time = time.time()
            print("\n  -> 正在为数据库创建/检查索引 (user_id, timestamp)...")
            index_start_time = time.time()
            # 使用 IF NOT EXISTS 确保不会重复创建
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_ts ON content (user_id, timestamp);") 
            index_end_time = time.time()
            print(f"  -> 索引创建/检查完成! 耗时: {index_end_time - index_start_time:.2f} 秒")

            print(f"\nPass 1 增量完成: 共处理 {total_line_count} 行, {total_error_count} 行解析/处理失败。")
            print(f"  -> 耗时: {end_time - start_time:.2f} 秒")
            print(f"  -> 共 {post_count + comment_count} 条新内容存入数据库 {db_path}")

        except Exception as e:
            print(f"❌ Pass 1 发生致命错误: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            conn.close() 

    def run(self):
        file_list = self.collect_file_list()
        self.process_and_store_to_db(file_list, self.db_path)
        gc.collect()
        return self.db_path

# main 和 __name__ 部分保持不变
def main():
    # 确保日志文件路径指向正确，即使目录不存在也能创建
    Path(INPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    Path(PERMANENT_DB_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    proc = UserPostProcessor(input_directory=INPUT_DIRECTORY, db_path=PERMANENT_DB_FILE)
    try:
        proc.run()
        print("\n🎉 全部处理完成。")
        print(f"✅ 最终数据库已成功保存到: {PERMANENT_DB_FILE}")
    except Exception as e:
        # 如果 collect_file_list 抛出 FileNotFoundError，可能是没有新文件，不一定是致命错误
        if "找不到新的 .txt 文件" in str(e):
             print(f"\n[Note] {e}")
        else:
            print(f"\n❌ 发生未知错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()