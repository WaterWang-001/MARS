# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import os
import os.path as osp
import sqlite3
from typing import Any, Dict, List

SCHEMA_DIR = "social_platform/schema"
DB_DIR = "db"
DB_NAME = "social_media.db"

# 基础固定表 SQL 文件 (不需要动态生成的)
USER_SCHEMA_SQL = "user.sql"
POST_SCHEMA_SQL = "post.sql"
FOLLOW_SCHEMA_SQL = "follow.sql"
MUTE_SCHEMA_SQL = "mute.sql"
LIKE_SCHEMA_SQL = "like.sql"
DISLIKE_SCHEMA_SQL = "dislike.sql"
REPORT_SCHEAM_SQL = "report.sql"
TRACE_SCHEMA_SQL = "trace.sql"
REC_SCHEMA_SQL = "rec.sql"
COMMENT_SCHEMA_SQL = "comment.sql"
COMMENT_LIKE_SCHEMA_SQL = "comment_like.sql"
COMMENT_DISLIKE_SCHEMA_SQL = "comment_dislike.sql"
PRODUCT_SCHEMA_SQL = "product.sql"
GROUP_SCHEMA_SQL = "chat_group.sql"
GROUP_MEMBER_SCHEMA_SQL = "group_member.sql"
GROUP_MESSAGE_SCHEMA_SQL = "group_message.sql"

def get_db_path() -> str:
    curr_file_path = osp.abspath(__file__)
    parent_dir = osp.dirname(osp.dirname(curr_file_path))
    db_dir = osp.join(parent_dir, DB_DIR)
    os.makedirs(db_dir, exist_ok=True)
    db_path = osp.join(db_dir, DB_NAME)
    return db_path

def get_schema_dir_path() -> str:
    curr_file_path = osp.abspath(__file__)
    parent_dir = osp.dirname(osp.dirname(curr_file_path))
    schema_dir = osp.join(parent_dir, SCHEMA_DIR)
    return schema_dir

def _execute_schema_if_not_exists(
    cursor: sqlite3.Cursor, schema_dir: str, sql_file_name: str, table_name: str
):
    """创建静态固定表 (从 SQL 文件读取)"""
    try:
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        )
        if not cursor.fetchone():
            print(f"  -> Creating static table '{table_name}'...")
            sql_path = osp.join(schema_dir, sql_file_name)
            if not osp.exists(sql_path):
                print(f"     [WARNING] SQL file {sql_path} missing!")
                return
            with open(sql_path, "r", encoding='utf-8') as sql_file:
                sql_script = sql_file.read()
            cursor.executescript(sql_script)
        else:
            # print(f"  -> Table '{table_name}' exists.")
            pass
    except sqlite3.Error as e:
        print(f"  -> [ERROR] creating '{table_name}': {e}")


# --- 【核心修改】动态创建态度表函数 ---
def _create_dynamic_attitude_table(cursor: sqlite3.Cursor, metric_name: str):
    """
    根据指标名称动态创建表。
    表名: log_attitude_{metric_name}
    结构: 使用用户提供的通用 Schema
    """
    table_name = f"{metric_name}"
    
    try:
        # 1. 检查表是否存在
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        )
        if cursor.fetchone():
            # print(f"  -> Dynamic table '{table_name}' already exists.")
            return

        print(f"  -> Creating dynamic metric table: '{table_name}'")

        # 2. 使用您提供的通用 Schema
        # 注意：这里使用了 f-string 插入 table_name
        sql_script = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            time_step INTEGER,
            user_id TEXT,       
            agent_id INTEGER,   
            agent_type TEXT,
            metric_type TEXT,
            attitude_score REAL
        );
        """
        cursor.executescript(sql_script)
        
    except sqlite3.Error as e:
        print(f"  -> [ERROR] creating dynamic table '{table_name}': {e}")


# --- 【核心修改】主初始化函数 ---
def create_db(
    db_path: str | None = None, 
    attitude_metrics: List[str] | None = None
):
    r"""
    初始化数据库。
    Args:
        db_path: 数据库路径。
        attitude_metrics: 需要追踪的态度指标列表 (例如 ['lifestyle_culture', 'trust'])。
                          系统会为每个指标创建一个 log_attitude_{metric} 表。
    """
    schema_dir = get_schema_dir_path()
    if db_path is None:
        db_path = get_db_path()

    # 默认指标列表（如果调用者没传，可以使用这些默认值，或者留空）
    if attitude_metrics is None:
        attitude_metrics = []

    print(f"Initializing Database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 1. 创建静态业务表 (User, Post, Relation 等)
        static_tables = [
            (USER_SCHEMA_SQL, "user"),
            (POST_SCHEMA_SQL, "post"),
            (FOLLOW_SCHEMA_SQL, "follow"),
            (MUTE_SCHEMA_SQL, "mute"),
            (LIKE_SCHEMA_SQL, "like"),
            (DISLIKE_SCHEMA_SQL, "dislike"),
            (REPORT_SCHEAM_SQL, "report"),
            (TRACE_SCHEMA_SQL, "trace"),
            (REC_SCHEMA_SQL, "rec"),
            (COMMENT_SCHEMA_SQL, "comment"),
            (COMMENT_LIKE_SCHEMA_SQL, "comment_like"),
            (COMMENT_DISLIKE_SCHEMA_SQL, "comment_dislike"),
            (PRODUCT_SCHEMA_SQL, "product"),
            (GROUP_SCHEMA_SQL, "chat_group"),
            (GROUP_MEMBER_SCHEMA_SQL, "group_member"),
            (GROUP_MESSAGE_SCHEMA_SQL, "group_message"),
        ]

        for sql_file, table_name in static_tables:
            _execute_schema_if_not_exists(cursor, schema_dir, sql_file, table_name)

        # 2. 动态创建态度指标表
        # 总是创建一个 'average' 表作为基础汇总（可选，看你需求）
        _create_dynamic_attitude_table(cursor, "average")

        # 创建传入的自定义指标表
        if attitude_metrics:
            print(f"  -> configuring dynamic metrics: {attitude_metrics}")
            for metric in attitude_metrics:
                _create_dynamic_attitude_table(cursor, metric)

        conn.commit()
        print("  -> Database initialization complete.\n")

    except sqlite3.Error as e:
        print(f"An error occurred while creating tables: {e}")

    return conn, cursor


def print_db_tables_summary():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print(f"=== Database Schema Summary ({len(tables)} tables) ===")
    for table in tables:
        table_name = table[0]
        if table_name.startswith("sqlite_"): continue
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"• {table_name:<35} | Columns: {len(columns)}")
    
    conn.close()
def fetch_table_from_db(cursor: sqlite3.Cursor,
                        table_name: str) -> List[Dict[str, Any]]:
    cursor.execute(f"SELECT * FROM {table_name}")
    columns = [description[0] for description in cursor.description]
    data_dicts = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return data_dicts


def fetch_rec_table_as_matrix(cursor: sqlite3.Cursor) -> List[List[int]]:
    # First, query all user_ids from the user table, assuming they start from
    # 1 and are consecutive
    cursor.execute("SELECT user_id FROM user ORDER BY user_id")
    user_ids = [row[0] for row in cursor.fetchall()]

    # Then, query all records from the rec table
    cursor.execute(
        "SELECT user_id, post_id FROM rec ORDER BY user_id, post_id")
    rec_rows = cursor.fetchall()
    # Initialize a dictionary, assigning an empty list to each user_id
    user_posts = {user_id: [] for user_id in user_ids}
    # Fill the dictionary with the records queried from the rec table
    for user_id, post_id in rec_rows:
        if user_id in user_posts:
            user_posts[user_id].append(post_id)
    # Convert the dictionary into matrix form
    matrix = [user_posts[user_id] for user_id in user_ids]
    return matrix


if __name__ == "__main__":
    # === 测试用例 ===
    
    # 假设这是你在模拟配置文件中定义的指标
    simulation_metrics = [
        "lifestyle_culture", 
        "sport_ent", 
        "sci_health", 
        "politics_econ",
        "trustworthiness" # 新增测试指标
    ]
    
    # 初始化数据库，传入指标
    create_db(attitude_metrics=simulation_metrics)
    
    # 打印结果验证
    print_db_tables_summary()