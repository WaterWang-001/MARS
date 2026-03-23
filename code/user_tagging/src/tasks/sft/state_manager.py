import sqlite3
import json
import os
import threading

class StateManager:
    def __init__(self, db_path):
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
        
        # 性能优化: 开启 WAL 模式允许并发读写
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        
        cursor = conn.cursor()
        
        # ==========================================
        # Part A: 用户状态与Buffer (在线Pipeline使用)
        # ==========================================
        
        # 1. 迁移旧表名逻辑 (Keep existing logic)
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

        # 2. 主状态表 user_full_state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_full_state (
                user_id TEXT PRIMARY KEY,
                last_cursor_time TEXT,
                interest_tree_snapshot TEXT,
                profile_snapshot TEXT,
                persona_summary TEXT,
                dataset_role TEXT
            )
        """)

        # 字段检查与自动迁移
        cursor.execute("PRAGMA table_info(user_full_state)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'profile_snapshot' not in columns:
            cursor.execute("ALTER TABLE user_full_state ADD COLUMN profile_snapshot TEXT")
        if 'dataset_role' not in columns:
            cursor.execute("ALTER TABLE user_full_state ADD COLUMN dataset_role TEXT")
        if 'persona_summary' not in columns:
            cursor.execute("ALTER TABLE user_full_state ADD COLUMN persona_summary TEXT")

        # 3. Buffer 表 user_pending_buffer
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_pending_buffer (
                post_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, 
                original_post_id INTEGER,
                content TEXT,
                quote_content TEXT,
                created_at DATETIME,
                temp_sjcjId TEXT, 
                temp_original_sjcjId TEXT,
                valid_tags_json TEXT
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_buffer_user_id ON user_pending_buffer (user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_buffer_created_at ON user_pending_buffer (created_at)")
        
        # Buffer 表字段迁移
        cursor.execute("PRAGMA table_info(user_pending_buffer)")
        buffer_columns = [info[1] for info in cursor.fetchall()]
        if 'valid_tags_json' not in buffer_columns:
            cursor.execute("ALTER TABLE user_pending_buffer ADD COLUMN valid_tags_json TEXT")

        # ==========================================
        # Part B: 全局标签与图谱 (离线清洗/聚类使用)
        # ==========================================

        # 4. 全局标签注册表 (Global_Tag_Registry)
        # 存储所有出现过的 L2/L3 标签，用于去重、向量化和状态管理
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS global_tag_registry (
                tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag_text TEXT NOT NULL UNIQUE,       -- 原始标签文本
                tag_level TEXT,                      -- 'L2' 或 'L3'
                embedding BLOB,                      -- 语义向量 (Bytes)
                status INTEGER DEFAULT 0,            -- 0: 新增待处理, 1: 已固化(锚点), 2: 待归并, 3: 废弃/噪声
                canonical_id INTEGER,                -- 指向标准词 ID (Self-Reference)
                cluster_id INTEGER,                  -- 聚类簇 ID
                
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # 索引：加速查找和状态过滤
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tag_text ON global_tag_registry (tag_text)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tag_status ON global_tag_registry (status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tag_canonical ON global_tag_registry (canonical_id)")

        # 5. 全局分类拓扑图 (Global_Taxonomy_Graph)
        # 存储 L1->L2->L3 的链条关系，用于构建全局兴趣图谱
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS global_taxonomy_graph (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                
                -- 链条内容 --
                l1_label TEXT NOT NULL,
                l2_tag_id INTEGER NOT NULL,          -- 关联 registry 表
                l3_tag_id INTEGER,                   -- 关联 registry 表 (可为空)
                
                -- 链条指纹 (用于去重) --
                edge_hash TEXT UNIQUE,               -- MD5(l1 + l2_id + l3_id)
                
                -- 统计信息 --
                frequency INTEGER DEFAULT 1,         -- 该路径出现的次数 (热度)
                last_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                -- 有效性 --
                is_active BOOLEAN DEFAULT 1,         -- 1: 有效, 0: 已被清洗逻辑废弃
                
                FOREIGN KEY(l2_tag_id) REFERENCES global_tag_registry(tag_id),
                FOREIGN KEY(l3_tag_id) REFERENCES global_tag_registry(tag_id)
            );
        """)
        # 索引：加速路径查询和热度统计
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_hash ON global_taxonomy_graph (edge_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_l2 ON global_taxonomy_graph (l2_tag_id)")
        
        conn.commit()
        conn.close()

    # --- 状态管理 (General State) ---
    
    def get_user_state(self, user_id):
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT last_cursor_time, interest_tree_snapshot, profile_snapshot, persona_summary, dataset_role FROM user_full_state WHERE user_id = ?",
                (str(user_id),)
            )
            row = cursor.fetchone()
        except sqlite3.OperationalError:
            row = None

        if row:
            return {
                "last_cursor_time": row["last_cursor_time"],
                "interest_tree": json.loads(row["interest_tree_snapshot"]) if row["interest_tree_snapshot"] else [],
                "profile": json.loads(row["profile_snapshot"]) if row["profile_snapshot"] else {},
                "persona_summary": row["persona_summary"] or "",
                "dataset_role": row["dataset_role"]
            }
        return {
            "last_cursor_time": "2000-01-01 00:00:00",
            "interest_tree": [],
            "profile": {},
            "persona_summary": "",
            "dataset_role": None
        }

    def update_user_state(self, user_id, new_cursor_time, merged_tree, new_profile, dataset_role=None, persona_summary=None):
        conn = self._get_conn()
        cursor = conn.cursor()
        tree_json = json.dumps(merged_tree, ensure_ascii=False)
        profile_json = json.dumps(new_profile, ensure_ascii=False)

        cursor.execute("""
            INSERT INTO user_full_state (user_id, last_cursor_time, interest_tree_snapshot, profile_snapshot, persona_summary, dataset_role)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                last_cursor_time = excluded.last_cursor_time,
                interest_tree_snapshot = excluded.interest_tree_snapshot,
                profile_snapshot = excluded.profile_snapshot,
                persona_summary = excluded.persona_summary,
                dataset_role = excluded.dataset_role
        """, (str(user_id), new_cursor_time, tree_json, profile_json, persona_summary or "", dataset_role))
        conn.commit()

    # --- Buffer 管理 ---
    
    def add_to_buffer(self, user_id, posts_rows):
        if not posts_rows: return
        conn = self._get_conn()
        cursor = conn.cursor()
        data_to_insert = []
        for p in posts_rows:
            tags_json = json.dumps(p.get('valid_tags', []), ensure_ascii=False)
            data_to_insert.append((
                str(user_id),
                p.get('original_post_id'),
                p.get('content', ''),
                p.get('quote_content', ''),
                p.get('created_at'),
                p.get('temp_sjcjId'),
                p.get('temp_original_sjcjId'),
                tags_json 
            ))
            
        cursor.executemany("""
            INSERT INTO user_pending_buffer (
                user_id, original_post_id, content, quote_content, 
                created_at, temp_sjcjId, temp_original_sjcjId, valid_tags_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, data_to_insert)
        conn.commit()

    def get_buffered_posts(self, user_id):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT content, quote_content, created_at, valid_tags_json 
            FROM user_pending_buffer 
            WHERE user_id = ? 
            ORDER BY created_at ASC
        """, (str(user_id),))
        
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            tags_str = d.pop('valid_tags_json', '[]')
            d['valid_tags'] = json.loads(tags_str) if tags_str else []
            results.append(d)
        return results

    def clear_user_buffer(self, user_id):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_pending_buffer WHERE user_id = ?", (str(user_id),))
        conn.commit()