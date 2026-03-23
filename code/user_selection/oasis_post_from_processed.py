"""
从 data_process 产出的平台 DB（posts_{platform}.db）中筛选用户子集，
写入统一的 oasis DB，并按时间切分 calibration / ground_truth 两张表。

用法：
    collector = ProcessedPostCollector(
        input_dirs=["MARS_result/data/output/2025-06-15"],
        user_list=["uid1", "uid2"],
        oasis_db="data/oasis/oasis_database.db",
        calibration_end="2025-06-15 16:00:00",
        ground_truth_end="2025-06-16 00:00:00",
    )
    collector.run()
"""

import sqlite3
import time
import sys
from pathlib import Path
from typing import List, Optional


class ProcessedPostCollector:
    """
    从 data_process 产出的 posts_{platform}.db 中按 user_id 筛选，
    合并写入统一 oasis DB，并按时间拆分 calibration / ground_truth。
    """

    OASIS_POST_SCHEMA = """
    CREATE TABLE IF NOT EXISTS {table} (
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
        FOREIGN KEY(original_post_id) REFERENCES {table}(post_id)
    );
    """

    def __init__(
        self,
        input_dirs: List[str],
        oasis_db: str,
        user_list: Optional[List[str]] = None,
        calibration_end: Optional[str] = None,
        ground_truth_end: Optional[str] = None,
        chunk_size: int = 500,
    ):
        self.input_dirs = [Path(d) for d in input_dirs]
        self.oasis_db = Path(oasis_db)
        self.user_set = set(str(u) for u in user_list) if user_list else None
        self.calibration_end = calibration_end
        self.ground_truth_end = ground_truth_end
        self.chunk_size = chunk_size

    # ---- 内部工具 ----

    def _find_platform_dbs(self) -> List[Path]:
        """在各 input_dir 中找到 posts_*.db 文件。"""
        dbs = []
        for d in self.input_dirs:
            if not d.exists():
                print(f"[WARN] 目录不存在，跳过: {d}", file=sys.stderr)
                continue
            # 支持直接指向日期目录或其上级
            found = list(d.glob("posts_*.db"))
            if not found:
                # 也尝试在子目录中查找（例如 output/2025-06-15/posts_weibo.db）
                found = list(d.rglob("posts_*.db"))
            dbs.extend(found)
        if not dbs:
            print(f"[WARN] 未在 {self.input_dirs} 中找到 posts_*.db 文件", file=sys.stderr)
        return sorted(set(dbs))

    @staticmethod
    def _init_pragmas(conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.execute("PRAGMA cache_size = -500000;")
        cur.execute("PRAGMA temp_store = MEMORY;")
        conn.commit()

    def _merge_title_content(self, title: Optional[str], content: Optional[str]) -> str:
        """将 title 合并到 content 前面（data_process 产出有 title 列，oasis 没有）。"""
        title = (title or "").strip()
        content = (content or "").strip()
        if title and title not in content:
            return f"{title}\n{content}" if content else title
        return content

    def _read_source_posts(self, src_conn: sqlite3.Connection) -> List[tuple]:
        """从源 DB 读取 post 表数据，按 user_set 筛选。返回统一格式 tuple 列表。"""
        cur = src_conn.cursor()

        # 检查表是否存在
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='post';")
        if not cur.fetchone():
            return []

        # 检查是否有 title 列
        cur.execute("PRAGMA table_info(post);")
        columns = {row[1] for row in cur.fetchall()}
        has_title = "title" in columns

        if self.user_set:
            user_list = list(self.user_set)
            results = []
            for i in range(0, len(user_list), self.chunk_size):
                chunk = user_list[i : i + self.chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                if has_title:
                    sql = f"""SELECT user_id, title, content, quote_content, created_at,
                                     temp_sjcjId, temp_original_sjcjId
                              FROM post WHERE user_id IN ({placeholders})"""
                else:
                    sql = f"""SELECT user_id, NULL, content, quote_content, created_at,
                                     temp_sjcjId, temp_original_sjcjId
                              FROM post WHERE user_id IN ({placeholders})"""
                cur.execute(sql, tuple(chunk))
                results.extend(cur.fetchall())
            return results
        else:
            # 无筛选，全部读取
            if has_title:
                cur.execute("""SELECT user_id, title, content, quote_content, created_at,
                                      temp_sjcjId, temp_original_sjcjId FROM post""")
            else:
                cur.execute("""SELECT user_id, NULL, content, quote_content, created_at,
                                      temp_sjcjId, temp_original_sjcjId FROM post""")
            return cur.fetchall()

    def _insert_into_target(self, target_conn: sqlite3.Connection, table_name: str, rows: List[tuple]):
        """将行写入目标表。rows 格式: (user_id, title, content, quote_content, created_at, temp_sjcjId, temp_original_sjcjId)"""
        if not rows:
            return 0

        cur = target_conn.cursor()
        batch = []
        for row in rows:
            user_id, title, content, quote_content, created_at, sjcj_id, orig_sjcj_id = row
            merged_content = self._merge_title_content(title, content)
            batch.append((user_id, merged_content, quote_content, created_at, sjcj_id, orig_sjcj_id))

        cur.execute("BEGIN TRANSACTION;")
        cur.executemany(f"""
            INSERT OR IGNORE INTO {table_name}
                (user_id, content, quote_content, created_at, temp_sjcjId, temp_original_sjcjId)
            VALUES (?, ?, ?, ?, ?, ?)
        """, batch)
        target_conn.commit()
        return len(batch)

    def _build_links(self, conn: sqlite3.Connection, table_name: str):
        """通过 temp_sjcjId / temp_original_sjcjId 建立 original_post_id 引用。"""
        print(f"  -> 建立 {table_name} 的引用关系...")
        cur = conn.cursor()
        start = time.time()

        if table_name == "post":
            cur.execute("""
                UPDATE post
                SET original_post_id = (
                    SELECT p2.post_id FROM post AS p2
                    WHERE p2.temp_sjcjId = post.temp_original_sjcjId
                )
                WHERE post.temp_original_sjcjId IS NOT NULL;
            """)
        else:
            # ground_truth_post 可能引用 calibration 的 post 或自身
            cur.execute(f"""
                UPDATE {table_name}
                SET original_post_id = COALESCE(
                    (SELECT p.post_id FROM post AS p
                     WHERE p.temp_sjcjId = {table_name}.temp_original_sjcjId),
                    (SELECT g.post_id FROM {table_name} AS g
                     WHERE g.temp_sjcjId = {table_name}.temp_original_sjcjId)
                )
                WHERE {table_name}.temp_original_sjcjId IS NOT NULL;
            """)
        conn.commit()
        print(f"     完成 ({time.time() - start:.2f}s)")

    def _build_indexes(self, conn: sqlite3.Connection, table_name: str):
        """为目标表建立索引。"""
        cur = conn.cursor()
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_userid ON {table_name}(user_id);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_temp_id ON {table_name}(temp_sjcjId);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_temp_orig ON {table_name}(temp_original_sjcjId);")
        conn.commit()

    # ---- 主流程 ----

    def run(self):
        start_all = time.time()

        # 1. 发现平台 DB
        platform_dbs = self._find_platform_dbs()
        if not platform_dbs:
            print("没有找到任何 posts_*.db 文件，退出。")
            return
        print(f"发现 {len(platform_dbs)} 个平台 DB 文件:")
        for db in platform_dbs:
            print(f"  - {db}")

        # 2. 准备目标 DB
        self.oasis_db.parent.mkdir(parents=True, exist_ok=True)
        # 如果已存在则删除重建
        if self.oasis_db.exists():
            self.oasis_db.unlink()

        target_conn = sqlite3.connect(str(self.oasis_db))
        self._init_pragmas(target_conn)

        # 3. 收集所有符合条件的行，按时间分组
        all_cal_rows = []
        all_gt_rows = []
        all_rows_no_split = []

        for db_path in platform_dbs:
            print(f"\n读取: {db_path.name} ...")
            try:
                src_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            except Exception:
                src_conn = sqlite3.connect(str(db_path))

            try:
                rows = self._read_source_posts(src_conn)
                print(f"  -> 读取到 {len(rows)} 条记录")

                if self.calibration_end or self.ground_truth_end:
                    for row in rows:
                        created_at = row[4]  # created_at 字段
                        if not created_at:
                            all_cal_rows.append(row)
                            continue
                        if self.calibration_end and created_at <= self.calibration_end:
                            all_cal_rows.append(row)
                        elif self.ground_truth_end and created_at <= self.ground_truth_end:
                            all_gt_rows.append(row)
                        # 超过 ground_truth_end 的丢弃
                else:
                    all_rows_no_split.extend(rows)
            finally:
                src_conn.close()

        # 4. 写入目标 DB
        if self.calibration_end or self.ground_truth_end:
            # 有时间切分
            print(f"\n写入 calibration (post): {len(all_cal_rows)} 条")
            target_conn.execute(self.OASIS_POST_SCHEMA.format(table="post"))
            target_conn.commit()
            self._insert_into_target(target_conn, "post", all_cal_rows)
            self._build_indexes(target_conn, "post")
            self._build_links(target_conn, "post")

            print(f"写入 ground_truth (ground_truth_post): {len(all_gt_rows)} 条")
            target_conn.execute(self.OASIS_POST_SCHEMA.format(table="ground_truth_post"))
            target_conn.commit()
            self._insert_into_target(target_conn, "ground_truth_post", all_gt_rows)
            self._build_indexes(target_conn, "ground_truth_post")
            self._build_links(target_conn, "ground_truth_post")
        else:
            # 无时间切分，全部写入 post 表
            print(f"\n写入 post 表（无时间切分）: {len(all_rows_no_split)} 条")
            target_conn.execute(self.OASIS_POST_SCHEMA.format(table="post"))
            target_conn.commit()
            self._insert_into_target(target_conn, "post", all_rows_no_split)
            self._build_indexes(target_conn, "post")
            self._build_links(target_conn, "post")

        # 5. 统计
        cur = target_conn.cursor()
        tables = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for t in tables:
            count = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t}: {count} 条")

        target_conn.close()
        print(f"\n完成! 输出: {self.oasis_db} ({time.time() - start_all:.2f}s)")
