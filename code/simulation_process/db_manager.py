import os
import sqlite3
import logging
import time
from typing import List


DB_LOCK_RETRY_TIMES = 12
DB_LOCK_RETRY_SLEEP_SECONDS = 1.0
DB_BUSY_TIMEOUT_MS = 15000


def _is_lock_error(exc: Exception) -> bool:
    return "database is locked" in str(exc).lower()


def _connect_with_retry(db_path: str, logger: logging.Logger) -> sqlite3.Connection:
    last_error: sqlite3.Error | None = None
    for attempt in range(1, DB_LOCK_RETRY_TIMES + 1):
        try:
            conn = sqlite3.connect(db_path, timeout=DB_BUSY_TIMEOUT_MS / 1000)
            conn.execute(f"PRAGMA busy_timeout = {DB_BUSY_TIMEOUT_MS}")
            return conn
        except sqlite3.Error as exc:
            last_error = exc
            if _is_lock_error(exc) and attempt < DB_LOCK_RETRY_TIMES:
                logger.warning(
                    f"数据库连接被锁(第 {attempt}/{DB_LOCK_RETRY_TIMES} 次)，等待 {DB_LOCK_RETRY_SLEEP_SECONDS:.1f}s 后重试..."
                )
                time.sleep(DB_LOCK_RETRY_SLEEP_SECONDS)
                continue
            raise

    assert last_error is not None
    raise last_error


def _execute_with_retry(
    conn: sqlite3.Connection,
    cursor: sqlite3.Cursor,
    sql: str,
    params: tuple = (),
    logger: logging.Logger | None = None,
) -> sqlite3.Cursor:
    for attempt in range(1, DB_LOCK_RETRY_TIMES + 1):
        try:
            return cursor.execute(sql, params)
        except sqlite3.Error as exc:
            if _is_lock_error(exc) and attempt < DB_LOCK_RETRY_TIMES:
                if logger is not None:
                    logger.warning(
                        f"SQL 执行被锁(第 {attempt}/{DB_LOCK_RETRY_TIMES} 次)，等待 {DB_LOCK_RETRY_SLEEP_SECONDS:.1f}s 后重试... sql={sql.strip()[:80]}"
                    )
                time.sleep(DB_LOCK_RETRY_SLEEP_SECONDS)
                continue
            raise

    return cursor


def _commit_with_retry(conn: sqlite3.Connection, logger: logging.Logger | None = None) -> None:
    for attempt in range(1, DB_LOCK_RETRY_TIMES + 1):
        try:
            conn.commit()
            return
        except sqlite3.Error as exc:
            if _is_lock_error(exc) and attempt < DB_LOCK_RETRY_TIMES:
                if logger is not None:
                    logger.warning(
                        f"数据库提交被锁(第 {attempt}/{DB_LOCK_RETRY_TIMES} 次)，等待 {DB_LOCK_RETRY_SLEEP_SECONDS:.1f}s 后重试..."
                    )
                time.sleep(DB_LOCK_RETRY_SLEEP_SECONDS)
                continue
            raise

def reset_simulation_tables(
    db_path: str,
    tables_to_keep: List[str],
    logger: logging.Logger,
    calibration_cutoff: str | None = None
):
    """
    重置OASIS数据库，删除所有模拟结果表，但保留指定的核心数据表。

    如果数据库文件不存在，它只会记录一条信息，因为OASIS的
    'make' 流程稍后会自动创建它。

    参数:
        db_path (str): 数据库文件路径。
        tables_to_keep (List[str]): 不应被删除的表名列表。
        logger (logging.Logger): 用于记录操作的日志记录器实例。
        calibration_cutoff (Optional[str]): 若提供，则额外删除 post 表中
            created_at 晚于此时间(含)的模拟帖子，避免多次运行累积。
    """
    if os.path.exists(db_path):
        logger.warning(f"数据库 {db_path} 已存在。将重置表，但保留: {', '.join(tables_to_keep)}")

        conn = None # 初始化 conn
        try:
            # 1. 连接到数据库
            conn = _connect_with_retry(db_path, logger)
            cursor = conn.cursor()

            # 2. 获取所有表的列表
            _execute_with_retry(cursor=cursor, conn=conn, sql="SELECT name FROM sqlite_master WHERE type='table';", logger=logger)
            all_tables = [row[0] for row in cursor.fetchall()]

            tables_to_drop = []

            # 3. 找出所有需要删除的表
            for table_name in all_tables:
                if table_name not in tables_to_keep:
                    tables_to_drop.append(table_name)

            # 4. 逐个删除这些表
            if tables_to_drop:
                logger.warning(f"将删除以下模拟结果表: {', '.join(tables_to_drop)}")
                for table_name in tables_to_drop:
                    _execute_with_retry(cursor=cursor, conn=conn, sql=f"DROP TABLE IF EXISTS {table_name}", logger=logger)
                _commit_with_retry(conn, logger)
                logger.info("数据库重置完成。")
            else:
                logger.info("没有找到需要删除的模拟结果表。")

            # 5. 清理历史模拟帖子（created_at >= 0）
            # 说明：OASIS 在模拟帖子上常使用数值时间步(如 0,1,2...)，需要在每次模拟开始前清空。
            numeric_cleanup_deleted = 0
            try:
                _execute_with_retry(
                    cursor=cursor,
                    conn=conn,
                    logger=logger,
                    sql=
                    """
                    DELETE FROM post
                    WHERE (
                        typeof(created_at) IN ('integer', 'real')
                        AND CAST(created_at AS REAL) >= 0
                    )
                    OR (
                        typeof(created_at) = 'text'
                        AND created_at NOT LIKE '%-%'
                        AND created_at NOT LIKE '%:%'
                        AND created_at NOT LIKE '% %'
                        AND CAST(created_at AS REAL) >= 0
                    )
                    """,
                )
                numeric_cleanup_deleted = cursor.rowcount
                _commit_with_retry(conn, logger)
                logger.info(f"已清理 {numeric_cleanup_deleted} 条 created_at >= 0 的历史模拟帖子。")
            except sqlite3.Error as e:
                logger.error(f"清理 created_at >= 0 的模拟帖子失败: {e}")

            # 6. 删除历史帖子（created_at >= calibration_cutoff）
            if calibration_cutoff:
                cutoff_sql = calibration_cutoff.replace('T', ' ')
                try:
                    _execute_with_retry(
                        cursor=cursor,
                        conn=conn,
                        sql="DELETE FROM post WHERE created_at >= ?",
                        params=(cutoff_sql,),
                        logger=logger,
                    )
                    deleted = cursor.rowcount
                    _commit_with_retry(conn, logger)
                    logger.info(
                        f"已按校准时间清理 {deleted} 条 created_at >= {cutoff_sql} 的帖子。"
                    )
                except sqlite3.Error as e:
                    logger.error(f"删除过期模拟帖子失败: {e}")

        except sqlite3.Error as e:
            logger.error(f"重置数据库时出错: {e}")
        finally:
            if conn:
                conn.close()

    else:
        logger.info(f"数据库 {db_path} 不存在，将(在env.make中)创建新库。")
