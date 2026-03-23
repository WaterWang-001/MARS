# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# ... (版权信息) ...
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import ast
import asyncio
import json
import time 
import logging
from typing import List, Optional, Union,Dict,Any
from collections import defaultdict 
from datetime import datetime 

import pandas as pd
import numpy as np
import tqdm
import sqlite3
from camel.memories import MemoryRecord
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelManager, ModelFactory
from camel.types import OpenAIBackendRole


# 【!! 关键 !!】 导入你的 4+1 Agent
from oasis.social_agent.agent_custom import (
    BaseAgent, SocialAgent, AuthorityAgent, KOLAgent, 
    ActiveCreatorAgent, NormalUserAgent, # (Tier 1 - LLM)
    HeuristicAgent, LurkerAgent # (Tier 2 - ABM)
)
from oasis.social_platform import Channel, Platform
from oasis.social_platform.config import Neo4jConfig, UserInfo
from oasis.social_platform.typing import ActionType, RecsysType
from oasis.social_agent.agent_graph import AgentGraph # (保留T1 Agent的类型提示)
from oasis.social_agent.agent_action import SocialAction

# --- [!! 保持不变: Tier 定义 !!] ---
TIER_1_LLM_GROUPS = {
    "权威媒体/大V",
    "活跃KOL",
    "活跃创作者",
    "普通用户" 
}
TIER_1_CLASS_MAP = {
    "权威媒体/大V": AuthorityAgent,
    "活跃KOL": KOLAgent,
    "活跃创作者": ActiveCreatorAgent,
    "普通用户": NormalUserAgent, 
    "default": SocialAgent
}

TIER_2_HEURISTIC_GROUPS = {
    "潜水用户"
}
TIER_2_CLASS_MAP = {
    "潜水用户": LurkerAgent,
    "default": HeuristicAgent
}
# --- [!! 定义结束 !!] ---


# --- [!! 保持不变: 辅助函数 !!] ---
def _parse_follow_list(follow_str: str) -> List[int]:
    if not follow_str or follow_str == "[]" or pd.isna(follow_str):
        return []
    try:
        stripped_str = follow_str.strip("[]")
        if not stripped_str:
            return []
        ids_str_list = stripped_str.split(',')
        return [
            int(id_str.strip()) for id_str in ids_str_list if id_str.strip()
        ]
    except Exception as e:
        logging.warning(f"⚠️ 警告: _parse_follow_list 失败，输入: '{follow_str}', 错误: {e}")
        return []


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger("agents_generator")
    
    df['user_id'] = df['user_id'].astype(str)
    df['user_char'] = df['user_char'].fillna('')
    df['description'] = df['description'].fillna('')
    df['following_agentid_list'] = df['following_agentid_list'].fillna('[]')
        
    return df


def _resolve_attitude_table_name(metric: str, existing_tables: set[str]) -> Optional[str]:
    candidates: List[str] = []
    base = metric
    if base.startswith("log_"):
        candidates.append(base)
        base = base[4:]
    else:
        candidates.append(f"log_{base}")
    candidates.append(base)
    if base.startswith("attitude_"):
        stripped = base[len("attitude_"):]
        if stripped:
            candidates.append(stripped)
        if base == "attitude_average":
            candidates.append("average")
    for name in dict.fromkeys([c for c in candidates if c]):
        if name in existing_tables:
            return name
    return None

# --- [!! 修正: 查询 'post' 表 !!] ---
def _load_initial_posts_from_db(db_path: str) -> dict[str, List[tuple[Optional[str], Optional[str]]]]:
    """
    (辅助函数) 从 'post' 表加载帖子，用于 T1 Agent 的 Memory。
    """
    logger = logging.getLogger("agents_generator")
    # [!! 修正 !!]
    logger.info(f"(Graph Build) 正在从 {db_path} 的 'post' 表预加载所有初始帖子...")
    
    initial_posts_map = defaultdict(list)
    
    try:
        if ":memory:" not in db_path:
            db_uri = f'file:{db_path}?mode=ro'
            conn = sqlite3.connect(db_uri, uri=True)
        else:
            logger.warning("(Graph Build) 正在从 :memory: 数据库加载帖子。这可能不会按预期工作，除非DB已填充。")
            conn = sqlite3.connect(db_path)
            
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # [!! 修正: 查询 'post' !!]
        cur.execute(
            "SELECT user_id, content, quote_content FROM post ORDER BY created_at"
        )
        
        count = 0
        for row in cur:
            user_id_str = str(row['user_id']) 
            initial_posts_map[user_id_str].append(
                (row['content'], row['quote_content'])
            )
            count += 1
        
        cur.close()
        conn.close()
        
        # [!! 修正 !!]
        logger.info(f"(Graph Build) 成功从 'post' 加载 {count} 条初始帖子, "
                    f"分布在 {len(initial_posts_map)} 个用户中。")
        return initial_posts_map
        
    except sqlite3.Error as e:
        # [!! 修正 !!]
        logger.error(f"❌ (Graph Build) 无法从 {db_path} 读取 'post' 表: {e}")
        logger.error("   将继续执行, 但所有 T1 agent 的 memory 都会是空的。")
        return initial_posts_map
    except Exception as e:
         logger.error(f"❌ (Graph Build) _load_initial_posts_from_db 发生意外错误: {e}")
         return initial_posts_map
# --- [!! 修正结束 !!] ---

    
def _preload_agent_memory(
    agent: BaseAgent, 
    initial_posts: List[tuple[Optional[str], Optional[str]]] 
):
    # ... (此函数不变) ...
    logger = logging.getLogger("agents_generator")
    
    if not initial_posts:
        return

    try:
        post_count = 0
        
        for post_tuple in initial_posts:
            user_comment_raw, original_post_raw = post_tuple
            
            user_comment = ""
            if isinstance(user_comment_raw, bytes):
                user_comment = user_comment_raw.decode('utf-8', 'replace').strip()
            elif isinstance(user_comment_raw, str):
                user_comment = user_comment_raw.strip()
                
            original_post = ""
            if isinstance(original_post_raw, bytes):
                original_post = original_post_raw.decode('utf-8', 'replace').strip()
            elif isinstance(original_post_raw, str):
                original_post = original_post_raw.strip()

        
            text_to_load_in_memory = ""
            if user_comment:
                text_to_load_in_memory = f"[用户评论]\n{user_comment}"
                if original_post:
                    text_to_load_in_memory += f"\n\n[转发的原帖]\n{original_post}"
            elif original_post:
                text_to_load_in_memory = f"[转发的原帖]\n{original_post}"
            else:
                continue 

            action_content = json.dumps({
                "reason": "This is an initial post from my history.",
                "functions": [
                    {
                        "name": "post",
                        "arguments": {
                            "content": text_to_load_in_memory
                        }
                    }
                ]
            })
            
            agent_msg = BaseMessage.make_user_message(
                role_name=OpenAIBackendRole.ASSISTANT.value, 
                content=action_content
            )
            
            agent.memory.write_record(
                MemoryRecord(message=agent_msg, 
                             role_at_backend=OpenAIBackendRole.ASSISTANT)
            )
            post_count += 1
        
        if post_count > 0:
            logger.debug(f"(Graph Build) 成功为 Agent {agent.agent_id} "
                         f"预加载了 {post_count} 条帖子到 Memory。")
    
    except Exception as e:
        logger.error(f"❌ (Graph Build) 预加载 Memory 失败 for agent "
                     f"{agent.agent_id}: {e}")

def _extract_attitude_score(data_source: Union[pd.Series, Dict], metric: str) -> float:
    """
    从数据源 (Pandas Series 或 Dict) 中提取 'initial_{metric}' 或 '{metric}'。
    如果不存在或无法转换为 float，返回 0.0。
    """
    keys_to_try = [f"initial_{metric}", metric]
    
    for k in keys_to_try:
        if k in data_source:
            val = data_source[k]
            # 处理 Pandas 的 NaN / None / 空字符串
            if pd.isna(val) or val == "":
                return 0.0
            try:
                return float(val)
            except (ValueError, TypeError):
                continue # 尝试下一个 key
    
    return 0.0

def create_and_register_single_agent(
    agent_id: int,
    user_profile: Dict[str, Any], # 包含 name, username, bio, group, attitudes
    platform: Platform,
    db_path: str,
    model: Optional[Any],
    available_actions: list[ActionType],
    current_time_step: int, # 明确指定当前时间步 (初始化是0, 动态是T)
    attitude_metrics: List[str],
    is_dynamic_injection: bool = False # 标记是否为动态注入(用于日志区分)
) -> BaseAgent:
    """
    创建一个 Agent 实例，并在数据库中注册它（User表 & 初始态度Log表）。
    """
    logger = logging.getLogger("agents_generator")
    
    # 1. 构造 UserInfo
    # 确保 attitude 数据在 other_info 中，以便 Agent.__init__ 读取
    other_info = {
        "user_profile": user_profile.get("user_char", ""),
        "original_user_id": str(user_profile.get("user_id", agent_id)),
        "group": user_profile.get("group", "default"),
        "initial_attitude_avg": user_profile.get("initial_attitude_avg", 0.0)
    }
    
    # 将态度指标注入 profile
    if attitude_metrics:
        for metric in attitude_metrics:
            # 使用辅助函数安全提取，默认为 0.0
            val = _extract_attitude_score(user_profile, metric)
            other_info[metric] = val

    full_profile = {
        "nodes": [], "edges": [], "other_info": other_info
    }

    user_info = UserInfo(
        name=user_profile.get("name", f"User_{agent_id}"),
        user_name=user_profile.get("username", f"user_{agent_id}"),
        description=user_profile.get("bio", ""),
        profile=full_profile,
        recsys_type='twitter'
    )

    # 2. 实例化 Agent (根据 Group 选择类型)
    group_name = other_info["group"]
    
    # 这里简单判断：InterventionBot 或 TIER_1 里的都是 SocialAgent (LLM)
    # 如果需要支持动态插入 ABM Agent，可以在这里加逻辑
    if group_name in TIER_2_CLASS_MAP:
        AgentClass = TIER_2_CLASS_MAP[group_name]
        agent = AgentClass(
            agent_id=agent_id,
            env=SocialAction(agent_id=agent_id, channel=platform.channel),
            db_path=db_path,
            user_info=user_info
        )
    else:
        # 默认为 SocialAgent (LLM)
        AgentClass = SocialAgent # 或者从 TIER_1_CLASS_MAP 拿
        agent = AgentClass(
            agent_id=agent_id,
            user_info=user_info,
            model=model,
            available_actions=available_actions,
            channel=platform.channel,
            db_path=db_path
        )
    
    # 设置当前时间步，以便后续 save_attitude 使用
    agent.current_time_step = current_time_step

    # 3. 注册到 Platform (写入 user 表)
    # 注意：如果是批量初始化，为了性能通常是收集 list 后 executemany
    # 但对于动态插入(数量少)或者为了代码复用，单次 execute 也可以接受
    if is_dynamic_injection:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO user (user_id, agent_id, user_name, name, bio, created_at, num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
                (str(agent_id), agent_id, user_info.user_name, user_info.name, user_info.description, datetime.now())
            )
            conn.commit()
            conn.close()
            
            # 别忘了注册 Group 到 Platform (我们在 Platform 加的那个方法)
            if hasattr(platform, 'register_agent_group'):
                platform.register_agent_group(agent_id, group_name)
                
        except Exception as e:
            logger.error(f"❌ 单个 Agent {agent_id} 注册 DB 失败: {e}")

    # 4. 写入初始态度 Log (Time Step T)
    # 这一步非常重要，让新 Agent 在 Log 表里有“出生记录”
    # Agent 类里已经封装了 save_attitude_to_db，直接利用它
    agent.save_attitude_to_db()

    return agent

# --- [!! 新的统一函数: `generate_and_register_agents` !!] ---
async def generate_and_register_agents(
    profile_path: str,
    db_path: str,  
    platform: Platform,
    model: Optional[Union[BaseModelBackend, List[BaseModelBackend],
                          ModelManager]] = None,
    available_actions: list[ActionType] = None,
    CALIBRATION_END: str = None, 
    TIME_STEP_MINUTES: int = 3,
    attitude_metrics: List[str] = None
) -> List[BaseAgent]:
    
    logger = logging.getLogger("agents_generator")
    if attitude_metrics is None:
        attitude_metrics = []
        logger.warning("⚠️ 未传入 attitude_metrics，Agent 将不包含态度初始值！")

    # 1. 预加载历史帖子
    initial_posts_map = _load_initial_posts_from_db(db_path)
    
    agent_list: List[BaseAgent] = []
    
    # 2. 加载数据 (直接指定 user_id 为 int)
    logger.info(f"(Agent Gen) 正在从 {profile_path} 加载并清洗所有用户数据...")
    try:
        # 移除了 dtype={'user_id': str}，让 pandas 自动推断或强制为 int
        all_user_info = pd.read_csv(profile_path, index_col=0) 
        all_user_info['user_id'] = all_user_info['user_id'].astype(int)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        logger.error(f"❌ (Agent Gen) 找不到或用户文件 {profile_path} 为空。")
        return agent_list
    except KeyError as e:
        logger.error(f"❌ (Agent Gen) CSV 文件缺少必需的列: {e}")
        return agent_list

    tier1_info = all_user_info[all_user_info['group'].isin(TIER_1_LLM_GROUPS)]
    tier2_info = all_user_info[all_user_info['group'].isin(TIER_2_HEURISTIC_GROUPS)]
    
    # 3. 预计算关注图 (全部使用整数操作)
    logger.info("... (Agent Gen) 正在预计算关注图...")
    followings_map = defaultdict(set) 
    followers_map = defaultdict(set)
    all_agent_ids_set = set(all_user_info.index) # 索引即为 agent_id

    for agent_id, row in all_user_info.iterrows():
        # _parse_follow_list 内部应确保返回整数列表
        followee_list = _parse_follow_list(row["following_agentid_list"])
        for followee_id in followee_list:
            followee_id = int(followee_id) # 强制确保是数字
            if followee_id in all_agent_ids_set:
                followings_map[agent_id].add(followee_id)
                followers_map[followee_id].add(agent_id)
    
    # 4. 辅助函数：构建 Profile
    def _build_dynamic_profile(row, metric_list):
        avg_score = _extract_attitude_score(row, "attitude_avg")
        other_info = {
            "user_profile": row["user_char"],
            "original_user_id": int(row["user_id"]),
            "following_agentid_list": row["following_agentid_list"], 
            "group": row["group"],
            "initial_attitude_avg": avg_score
        }
        if metric_list:
            for metric in metric_list:
                other_info[metric] = _extract_attitude_score(row, metric)
        return {"nodes": [], "edges": [], "other_info": other_info}

    # 5. 准备批量写入容器
    sign_up_list = []
    follow_list = []
    agent_id_to_type_map = {} 
    
    # --- 步骤 A: Heuristic Agents (Tier 2) ---
    for agent_id, row in tqdm.tqdm(tier2_info.iterrows(), total=len(tier2_info), desc="Building Heuristic Agents"):
        user_id = int(row["user_id"])
        
        AgentClass = TIER_2_CLASS_MAP.get(row["group"], TIER_2_CLASS_MAP["default"])
        profile = _build_dynamic_profile(row, attitude_metrics)
        
        user_info = UserInfo(
            name=row["username"], user_name=row["name"], 
            description=row["description"], profile=profile, recsys_type='twitter',
        )
        
        agent = AgentClass(
            agent_id=agent_id,
            env=SocialAction(agent_id=agent_id, channel=platform.channel),
            db_path=db_path,
            user_info=user_info
        )
        agent_list.append(agent)
        
        # 准备数据
        sign_up_list.append((
            user_id, agent_id, row["username"], row["name"], row["description"], 
            datetime.now(), len(followings_map[agent_id]), len(followers_map[agent_id])
        ))
        for fid in followings_map[agent_id]:
            follow_list.append((agent_id, fid, datetime.now()))
        
        agent_id_to_type_map[str(agent_id)] = ('ABM', 'internal_state')

    # --- 步骤 B: LLM Agents (Tier 1) ---
    for agent_id, row in tqdm.tqdm(tier1_info.iterrows(), total=len(tier1_info), desc="Building LLM Agents"):
        user_id = int(row["user_id"])
        
        profile_data = row.to_dict()
        profile_data["user_id"] = user_id # 确保是数值
        
        agent = create_and_register_single_agent(
            agent_id=agent_id,
            user_profile=profile_data,
            platform=platform,
            db_path=db_path,
            model=model,
            available_actions=available_actions,
            current_time_step=0,
            attitude_metrics=attitude_metrics,
            is_dynamic_injection=False 
        )
        agent_list.append(agent)

        # 内存预加载
        posts = initial_posts_map.get(user_id, [])
        _preload_agent_memory(agent, posts)
        
        sign_up_list.append((
            user_id, agent_id, row["username"], row["name"], row["description"], 
            datetime.now(), len(followings_map[agent_id]), len(followers_map[agent_id])
        ))
        for fid in followings_map[agent_id]:
            follow_list.append((agent_id, fid, datetime.now()))

        agent_id_to_type_map[str(agent_id)] = ('LLM', 'external_expression')

    # --- 6. 批量写入数据库 ---
    logger.info(f"... (Agent Gen) 正在批量写入 {len(sign_up_list)} 个用户 ...")
    user_sql = "INSERT OR IGNORE INTO user (user_id, agent_id, user_name, name, bio, created_at, num_followings, num_followers) VALUES (?,?,?,?,?,?,?,?)"
    platform.pl_utils._execute_many_db_command(user_sql, sign_up_list, commit=True)

    follow_sql = "INSERT OR IGNORE INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)"
    platform.pl_utils._execute_many_db_command(follow_sql, follow_list, commit=True)

    # --- 7. T<0 日志处理 ---
    if CALIBRATION_END and attitude_metrics:
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(post)")
                cols = {r[1] for r in cursor.fetchall()}
                
                if not set(attitude_metrics).issubset(cols) or 'attitude_annotated' not in cols:
                    logger.warning("⚠️ 数据库缺少指标列，跳过历史日志计算。")
                else:
                    att_cols_sql = ", ".join([f"T1.{c}" for c in attitude_metrics])
                    query = f"""
                        SELECT T1.created_at, T1.user_id, T2.agent_id, {att_cols_sql}
                        FROM post AS T1 
                        INNER JOIN user AS T2 ON T1.user_id = T2.user_id
                        WHERE T1.created_at < ? AND T1.attitude_annotated = 1
                    """
                    cal_dt = datetime.fromisoformat(CALIBRATION_END)
                    df_hist = pd.read_sql_query(query, conn, params=(cal_dt.strftime("%Y-%m-%d %H:%M:%S"),))
                    
                    if not df_hist.empty:
                        # 确保 ID 是整数以便映射
                        df_hist['agent_id'] = df_hist['agent_id'].astype(int)
                        
                        # 计算 Time Step
                        df_hist['dt'] = pd.to_datetime(df_hist['created_at'])
                        df_hist['time_step'] = -(((cal_dt - df_hist['dt']).dt.total_seconds() // (TIME_STEP_MINUTES * 60)) + 1).astype(int)
                        
                        df_grp = df_hist.groupby(['time_step', 'user_id', 'agent_id'])[attitude_metrics].mean().reset_index()
                        
                        batch_log = []
                        for r in df_grp.itertuples():
                            key = str(r.agent_id) if r.agent_id is not None else None
                            a_type, m_type = agent_id_to_type_map.get(key, (None, None))
                            if not a_type: continue
                            
                            vals = [getattr(r, m) for m in attitude_metrics if getattr(r, m) is not None]
                            avg_v = np.mean(vals) if vals else 0.0
                            
                            scores = {m: getattr(r, m) for m in attitude_metrics}
                            scores['attitude_average'] = avg_v
                            
                            for dim, val in scores.items():
                                if val is not None:
                                    batch_log.append((r.time_step, r.user_id, r.agent_id, a_type, m_type, val, dim))
                        
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        existing_tables = {row[0] for row in cursor.fetchall()}

                        # 按表名分发插入
                        for ts, uid, aid, atype, mtype, score, dim in batch_log:
                            table_name = _resolve_attitude_table_name(dim, existing_tables)
                            if not table_name:
                                continue
                            cursor.execute(
                                f"INSERT INTO {table_name} (time_step, user_id, agent_id, agent_type, metric_type, attitude_score) VALUES (?,?,?,?,?,?)",
                                (ts, uid, aid, atype, mtype, score)
                            )
                        conn.commit()
                        logger.info(f"成功插入 {len(batch_log)} 条 T<0 日志。")
        except Exception as e:
            logger.error(f"❌ T<0 日志处理失败: {e}")
    
    if attitude_metrics:
        logger.info(f"... (Agent Gen) 正在将所有 Agent 的初始态度 (Step 0) 写入对应指标表...")
        
        # 准备数据容器： { "log_attitude_trust": [(time_step, user_id, ...), ...], "log_attitude_risk": [...] }
        # 这样可以针对每个表做 executemany
        insert_batches = defaultdict(list)
        
        # 包含 Average 表
        all_metrics_to_insert = attitude_metrics + ["attitude_average"]
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # 获取已存在的表名，防止报错
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            valid_tables = []
            for m in all_metrics_to_insert:
                tbl = f"log_{m}" # 例如 log_attitude_trust
                if tbl in existing_tables:
                    valid_tables.append(m)
                else:
                    logger.warning(f"⚠️ 表 '{tbl}' 不存在，跳过该指标的初始值写入。")

            # 遍历所有 Agent，提取 profile 中的数据
            count_records = 0
            for agent in agent_list:
                other_info = agent.user_info.profile["other_info"]
                
                uid = other_info.get("original_user_id")
                aid = agent.agent_id
                aid_int = agent.agent_id_int if hasattr(agent, "agent_id_int") else int(aid)
                # 获取类型信息
                atype, mtype = agent_id_to_type_map.get(str(aid), ("Unknown", "Unknown"))
                
                for metric in valid_tables:
                    val = 0.0
                    table_name = f"log_{metric}"
                    
                    if metric == "attitude_average":
                        val = other_info.get("initial_attitude_avg", 0.0)
                    else:
                        # profile 里的 key 就是 metric 本身 (在 _build_dynamic_profile 里设置的)
                        val = other_info.get(metric, 0.0)
                    
                    # 构造插入 Tuple: (time_step=0, user_id, agent_id, agent_type, metric_type, score)
                    insert_batches[table_name].append(
                        (0, uid, aid_int, atype, mtype, float(val))
                    )
                    count_records += 1
            
            # 执行批量插入
            for tbl_name, data_rows in insert_batches.items():
                if not data_rows: continue
                sql = f"""
                INSERT INTO {tbl_name} 
                (time_step, user_id, agent_id, agent_type, metric_type, attitude_score) 
                VALUES (?, ?, ?, ?, ?, ?)
                """
                cursor.executemany(sql, data_rows)
            
            conn.commit()
            logger.info(f"✅ (Agent Gen) 成功写入 {count_records} 条初始态度记录 (Step 0)。")
            
        except Exception as e:
            logger.error(f"❌ (Agent Gen) 写入初始态度时发生错误: {e}")
        finally:
            conn.close()
    
    return agent_list
# --- [!! 统一函数结束 !!] ---



def connect_platform_channel(
    channel: Channel,
    agent_list: List[BaseAgent] | None = None,
) -> List[BaseAgent]:
    """
    (已修改) 
    将平台 channel 注入到 *已创建* 的 Agent 实例列表中。
    """
    if agent_list is None:
        agent_list = []
        
    for agent in agent_list:
        if hasattr(agent, 'channel'):
             agent.channel = channel
        if hasattr(agent, 'env') and hasattr(agent.env, 'action') and isinstance(agent.env, SocialAction):
             agent.env.channel = channel
             agent.env.action.channel = channel

    return agent_list


async def generate_custom_agents(
    platform: Platform, 
    agent_list: List[BaseAgent] | None = None,
) -> List[BaseAgent]: 
    """
    (!! 已重构 !!)
    
    在新的 `generate_and_register_agents` 流程中, 此函数 (在 env.reset() 中被调用) 
    的唯一职责是 *重新连接* Platform 和 Channel 实例。
    """
    logger = logging.getLogger("agents_generator")
    
    if agent_list is None:
        agent_list = []
    
    logger.info(f"... (generate_custom_agents) 正在将 Platform 和 Channel 实例重新连接到 {len(agent_list)} 个 Agents ...")
    
    channel = platform.channel
    
    for agent in agent_list:
        # T1 Agents (SocialAgent) - channel 已在创建时注入
        if hasattr(agent, 'channel'):
             agent.channel = channel
        
        # T2 Agents (HeuristicAgent) - 必须注入 platform
        if hasattr(agent, 'env') and isinstance(agent.env, SocialAction):
             agent.env.channel = channel
             # [!! 2. 关键修复: SocialAction 确实有 platform 属性 (我们在上一步已添加) !!]
             agent.env.platform = platform 
             agent.env.action.channel = channel
             # [!! 3. 关键修复: SocialAction.action 就是 SocialAction 自己 !!]
             agent.env.action.platform = platform 

    logger.info("... (generate_custom_agents) 注入完成。")
    
    return agent_list
# --- [!! 修正结束 !!] ---


async def generate_reddit_agent_graph(
    # ... (此函数保持不变) ...
    profile_path: str,
    model: Optional[Union[BaseModelBackend, List[BaseModelBackend],
                          ModelManager]] = None,
    available_actions: list[ActionType] = None,
) -> AgentGraph:
    agent_graph = AgentGraph()
    with open(profile_path, "r") as file:
        agent_info = json.load(file)
    async def process_agent(i):
        profile = { "nodes": [], "edges": [], "other_info": {}, }
        profile["other_info"]["user_profile"] = agent_info[i]["persona"]
        profile["other_info"]["mbti"] = agent_info[i]["mbti"]
        profile["other_info"]["gender"] = agent_info[i]["gender"]
        profile["other_info"]["age"] = agent_info[i]["age"]
        profile["other_info"]["country"] = agent_info[i]["country"]
        user_info = UserInfo(
            name=agent_info[i]["username"],
            description=agent_info[i]["bio"],
            profile=profile,
            recsys_type="reddit",
        )
        agent = SocialAgent(
            agent_id=i,
            user_info=user_info,
            agent_graph=agent_graph,
            model=model,
            available_actions=available_actions,
        )
        agent_graph.add_agent(agent)
    tasks = [process_agent(i) for i in range(len(agent_info))]
    await asyncio.gather(*tasks)
    return agent_graph