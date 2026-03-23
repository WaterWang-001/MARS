import json
import os
import sqlite3
import threading
import csv
import time


class BenchStateManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self.max_write_retries = int(os.getenv("BENCH_DB_WRITE_RETRIES", "15"))
        self.retry_sleep_sec = float(os.getenv("BENCH_DB_RETRY_SLEEP_SEC", "0.05"))
        self._init_db()

    def _get_conn(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, timeout=60.0)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA synchronous=OFF;")
            self._local.conn.execute("PRAGMA busy_timeout=60000;")
            self._local.conn.execute("PRAGMA temp_store=MEMORY;")
            self._local.conn.execute("PRAGMA wal_autocheckpoint=1000;")
        return self._local.conn

    def _execute_write_with_retry(self, operation):
        last_exc = None
        for attempt in range(self.max_write_retries):
            try:
                return operation()
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "database is locked" not in msg:
                    raise
                last_exc = exc
                time.sleep(self.retry_sleep_sec * (attempt + 1))
        raise last_exc

    def flush(self):
        conn = self._get_conn()
        conn.commit()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_full_state(
                user_id TEXT PRIMARY KEY,
                last_cursor_time TEXT,
                interest_tree_snapshot TEXT,
                profile_snapshot TEXT,
                dataset_role TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_pending_buffer (
                post_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                original_post_id INTEGER,
                title TEXT,
                content TEXT,
                quote_content TEXT,
                created_at DATETIME,
                valid_tags_json TEXT
            )
            """
        )
        cursor.execute("PRAGMA table_info(user_pending_buffer)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if "title" not in existing_columns:
            cursor.execute("ALTER TABLE user_pending_buffer ADD COLUMN title TEXT")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_buffer_user_id ON user_pending_buffer (user_id)")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS global_tag_registry (
                tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag_text TEXT NOT NULL UNIQUE,
                mapped_tag TEXT,
                frequency_daily INTEGER DEFAULT 0,
                frequency_total INTEGER DEFAULT 0,
                embedding BLOB,
                status INTEGER DEFAULT 0
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tag_text ON global_tag_registry (tag_text)")
        conn.commit()
        conn.close()

    def get_user_state(self, user_id):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT last_cursor_time, dataset_role FROM user_full_state WHERE user_id = ?", (str(user_id),))
        row = cursor.fetchone()
        if row:
            return {
                "last_cursor_time": row["last_cursor_time"],
                "dataset_role": row["dataset_role"],
                "buffered_posts": self.get_buffered_posts(user_id),
            }
        return {"last_cursor_time": "2000-01-01 00:00:00", "dataset_role": None, "buffered_posts": []}

    def update_user_progress(self, user_id, new_cursor_time, dataset_role="normal", interest_tree=None):
        tree_str = json.dumps(interest_tree, ensure_ascii=False) if interest_tree else ""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            def _op():
                cursor.execute(
                    """
                    SELECT last_cursor_time, dataset_role, interest_tree_snapshot
                    FROM user_full_state
                    WHERE user_id = ?
                    """,
                    (str(user_id),),
                )
                existing = cursor.fetchone()
                if existing:
                    existing_cursor = existing["last_cursor_time"]
                    existing_role = existing["dataset_role"]
                    existing_tree = existing["interest_tree_snapshot"] or ""
                    if (
                        str(existing_cursor) == str(new_cursor_time)
                        and str(existing_role) == str(dataset_role)
                        and existing_tree == tree_str
                    ):
                        return
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO user_full_state
                    (user_id, last_cursor_time, interest_tree_snapshot, profile_snapshot, dataset_role)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(user_id), str(new_cursor_time), tree_str, "", dataset_role),
                )
                conn.commit()

            self._execute_write_with_retry(_op)

    def update_user_buffer(self, user_id, posts_list, new_cursor_time, dataset_role=None):
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            def _op():
                cursor.execute("DELETE FROM user_pending_buffer WHERE user_id = ?", (str(user_id),))
                if posts_list:
                    data_to_insert = [
                        (
                            str(user_id),
                            p.get("original_post_id"),
                            p.get("title", ""),
                            p.get("content", ""),
                            p.get("quote_content", ""),
                            p.get("created_at"),
                            json.dumps(p.get("valid_tags", []), ensure_ascii=False),
                        )
                        for p in posts_list
                    ]
                    cursor.executemany(
                        """
                        INSERT INTO user_pending_buffer (user_id, original_post_id, title, content, quote_content, created_at, valid_tags_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        data_to_insert,
                    )
                cursor.execute(
                    """
                    INSERT INTO user_full_state (user_id, last_cursor_time, dataset_role) VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        last_cursor_time = excluded.last_cursor_time,
                        dataset_role = COALESCE(excluded.dataset_role, user_full_state.dataset_role)
                    """,
                    (str(user_id), new_cursor_time, dataset_role),
                )
                conn.commit()

            self._execute_write_with_retry(_op)

    def clear_buffer_and_update_progress(self, user_id, new_cursor_time, dataset_role="normal", interest_tree=None):
        tree_str = json.dumps(interest_tree, ensure_ascii=False) if interest_tree else ""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            def _op():
                cursor.execute("DELETE FROM user_pending_buffer WHERE user_id = ?", (str(user_id),))
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO user_full_state
                    (user_id, last_cursor_time, interest_tree_snapshot, profile_snapshot, dataset_role)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(user_id), str(new_cursor_time), tree_str, "", dataset_role),
                )
                conn.commit()

            self._execute_write_with_retry(_op)

    def get_buffered_posts(self, user_id):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT title, content, quote_content, created_at, valid_tags_json, original_post_id FROM user_pending_buffer WHERE user_id = ? ORDER BY created_at ASC",
            (str(user_id),),
        )
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            tags_str = d.pop("valid_tags_json", "[]")
            d["valid_tags"] = json.loads(tags_str) if tags_str else []
            results.append(d)
        return results

    def clear_user_buffer(self, user_id):
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            def _op():
                cursor.execute("DELETE FROM user_pending_buffer WHERE user_id = ?", (str(user_id),))
                conn.commit()

            self._execute_write_with_retry(_op)

    def batch_upsert_tags(self, tags_list):
        if not tags_list:
            return
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            from collections import Counter

            tag_counts = Counter(tags_list)

            def _op():
                cursor.executemany(
                    """
                    INSERT INTO global_tag_registry (tag_text, frequency_daily, frequency_total)
                    VALUES (?, ?, ?)
                    ON CONFLICT(tag_text) DO UPDATE SET
                        frequency_daily = frequency_daily + excluded.frequency_daily,
                        frequency_total = frequency_total + excluded.frequency_total
                    """,
                    [(tag, count, count) for tag, count in tag_counts.items()],
                )
                conn.commit()

            self._execute_write_with_retry(_op)

    def reset_daily_anchor_stats(self, export_csv_path=None):
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            try:
                if export_csv_path:
                    os.makedirs(os.path.dirname(export_csv_path), exist_ok=True)
                    cursor.execute(
                        """
                        SELECT tag_id, tag_text, frequency_daily, frequency_total
                        FROM global_tag_registry
                        WHERE frequency_daily > 0
                        ORDER BY frequency_daily DESC
                        """
                    )
                    rows = cursor.fetchall()
                    if rows:
                        with open(export_csv_path, "w", newline="", encoding="utf-8-sig") as f:
                            writer = csv.writer(f)
                            writer.writerow(["anchor_id", "anchor_text", "daily_freq", "total_freq"])
                            writer.writerows(rows)

                def _op():
                    cursor.execute("UPDATE global_tag_registry SET frequency_daily = 0 WHERE frequency_daily > 0")
                    conn.commit()

                self._execute_write_with_retry(_op)
            except Exception:
                conn.rollback()
                raise

    def reset_daily_tag_stats(self, export_csv_path=None):
        self.reset_daily_anchor_stats(export_csv_path=export_csv_path)
