import sqlite3
import json
import os
import threading

class StateManager:
    def __init__(self, db_path="MARS/data/output/processing_state.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self):
        """获取线程安全的连接"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """初始化数据库结构，包含自动迁移逻辑"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        
        # 性能优化
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        
        cursor = conn.cursor()
        
        # ==========================================
        # 1. 自动迁移逻辑 (Migration Logic)
        # ==========================================
        
        # 检查是否还存在旧名字的表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_interest_state'")
        old_table_exists = cursor.fetchone()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_full_state'")
        new_table_exists = cursor.fetchone()
        
        if old_table_exists and not new_table_exists:
            print("⚠️ [Migration] Renaming table 'user_interest_state' -> 'user_full_state'...")
            try:
                cursor.execute("ALTER TABLE user_interest_state RENAME TO user_full_state")
            except Exception as e:
                print(f"❌ Rename failed: {e}")

        # ==========================================
        # 2. 创建/确认主状态表 (user_full_state)
        # ==========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_full_state (
                user_id TEXT PRIMARY KEY,
                last_cursor_time TEXT ,
                interest_tree_snapshot TEXT,
                profile_snapshot TEXT
            )
        """)
        
        # 检查列是否存在 (防止旧DB缺少 profile_snapshot)
        cursor.execute("PRAGMA table_info(user_full_state)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'profile_snapshot' not in columns:
            print("⚠️ [Migration] Adding missing column 'profile_snapshot'...")
            try:
                cursor.execute("ALTER TABLE user_full_state ADD COLUMN profile_snapshot TEXT")
            except Exception as e:
                print(f"❌ Add column failed: {e}")

        # ==========================================
        # 3. 创建 Buffer 表
        # ==========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_pending_buffer (
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
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_buffer_user_id ON user_pending_buffer (user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_buffer_created_at ON user_pending_buffer (created_at)")
        
        conn.commit()
        conn.close()

    # --- 状态管理 (General State) ---
    
    def get_user_state(self, user_id):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # [修改] 从 user_full_state 读取
        try:
            cursor.execute(
                "SELECT last_cursor_time, interest_tree_snapshot, profile_snapshot FROM user_full_state WHERE user_id = ?", 
                (str(user_id),)
            )
            row = cursor.fetchone()
        except sqlite3.OperationalError:
            row = None
        
        if row:
            return {
                "last_cursor_time": row["last_cursor_time"],
                "interest_tree": json.loads(row["interest_tree_snapshot"]) if row["interest_tree_snapshot"] else [],
                "profile": json.loads(row["profile_snapshot"]) if row["profile_snapshot"] else {}
            }
        
        # 默认空状态
        return {
            "last_cursor_time": [], 
            "interest_tree": [], 
            "profile": {} 
        }

    def update_user_state(self, user_id, new_cursor_time, merged_tree, new_profile):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        tree_json = json.dumps(merged_tree, ensure_ascii=False)
        profile_json = json.dumps(new_profile, ensure_ascii=False)
        
        # [修改] 写入 user_full_state
        cursor.execute("""
            INSERT INTO user_full_state (user_id, last_cursor_time, interest_tree_snapshot, profile_snapshot)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                last_cursor_time = excluded.last_cursor_time,
                interest_tree_snapshot = excluded.interest_tree_snapshot,
                profile_snapshot = excluded.profile_snapshot
        """, (str(user_id), new_cursor_time, tree_json, profile_json))
        conn.commit()

    # --- Buffer 管理 (保持不变) ---
    
    def add_to_buffer(self, user_id, posts_rows):
        if not posts_rows: return
        conn = self._get_conn()
        cursor = conn.cursor()
        data_to_insert = []
        for p in posts_rows:
            data_to_insert.append((
                str(user_id),
                p.get('original_post_id'),
                p.get('content', ''),
                p.get('quote_content', ''),
                p.get('created_at'),
                p.get('temp_sjcjId'),
                p.get('temp_original_sjcjId')
            ))
        cursor.executemany("""
            INSERT INTO user_pending_buffer (
                user_id, original_post_id, content, quote_content, 
                created_at, temp_sjcjId, temp_original_sjcjId
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data_to_insert)
        conn.commit()

    def get_buffered_posts(self, user_id):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT content, quote_content, created_at 
            FROM user_pending_buffer 
            WHERE user_id = ? 
            ORDER BY created_at ASC
        """, (str(user_id),))
        return [dict(row) for row in cursor.fetchall()]

    def clear_user_buffer(self, user_id):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_pending_buffer WHERE user_id = ?", (str(user_id),))
        conn.commit()