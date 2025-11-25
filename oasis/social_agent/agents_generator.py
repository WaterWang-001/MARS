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
# --- [!! 辅助函数结束 !!] ---


# --- [!! 删除: `generate_agents`, `generate_agents_100w`, `generate_twitter_agent_graph` !!] ---
# (已删除)
# --- [!! 删除结束 !!] ---
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
    CALIBRATION_END: str = None, # 接收字符串
    TIME_STEP_MINUTES: int = 3,
    attitude_metrics: List[str] = None
) -> List[BaseAgent]:
    
    logger = logging.getLogger("agents_generator")
    ATTITUDE_COLUMNS = [
        'attitude_lifestyle_culture', 'attitude_sport_ent',
        'attitude_sci_health', 'attitude_politics_econ'
    ]
    if attitude_metrics is None:
        attitude_metrics = []
        logger.warning("⚠️ 未传入 attitude_metrics，Agent 将不包含态度初始值！")
    # 1. 预加载历史帖子
    initial_posts_map = _load_initial_posts_from_db(db_path)
    
    agent_list: List[BaseAgent] = []
    
    # 2. 加载并拆分所有用户
    logger.info(f"(Agent Gen) 正在从 {profile_path} 加载并清洗所有用户数据...")
    try:
        all_user_info = pd.read_csv(profile_path, index_col=0, dtype={'user_id': str})
        all_user_info = _clean_dataframe(all_user_info)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        logger.error(f"❌ (Agent Gen) 找不到或用户文件 {profile_path} 为空。模拟无法启动。")
        return agent_list
    except KeyError as e:
        logger.error(f"❌ (Agent Gen) CSV 文件缺少必需的列: {e}")
        return agent_list

    tier1_info = all_user_info[all_user_info['group'].isin(TIER_1_LLM_GROUPS)]
    tier2_info = all_user_info[all_user_info['group'].isin(TIER_2_HEURISTIC_GROUPS)]
    
    logger.info(f"(Agent Gen) 数据加载完毕: {len(tier1_info)} 个 [Tier 1 LLM], {len(tier2_info)} 个 [Tier 2 ABM]")
    
    # 3. 预计算关注图
    logger.info("... (Agent Gen) 正在预计算关注图...")
    followings_map = defaultdict(set) 
    followers_map = defaultdict(set)
    all_agent_ids_in_csv = set(all_user_info.index.astype(str)) 

    for agent_id_str, row in all_user_info.iterrows():
        agent_id_str = str(agent_id_str)
        followee_list = _parse_follow_list(row["following_agentid_list"])
        for followee_id in followee_list:
            followee_id_str = str(followee_id)
            if followee_id_str in all_agent_ids_in_csv:
                followings_map[agent_id_str].add(followee_id_str)
                followers_map[followee_id_str].add(agent_id_str)
    logger.info(f"... (Agent Gen) 关注图构建完成。{len(followings_map)} 个用户有关注列表。")

    # 4. 准备批量数据库写入
    sign_up_list = []
    follow_list = []
    agent_id_to_type_map = {} 
    
    def _build_dynamic_profile(row, metric_list):
        """根据传入的 metric_list，从 CSV row 中提取 initial_{metric}"""
        
        # 安全提取 average
        avg_score = _extract_attitude_score(row, "attitude_avg")
        
        other_info = {
            "user_profile": row["user_char"],
            "original_user_id": row["user_id"],
            "following_agentid_list_str": row["following_agentid_list"], 
            "group": row["group"],
            "initial_attitude_avg": avg_score
        }
        
        # [!! 修正 !!] 安全提取各项指标
        if metric_list:
            for metric in metric_list:
                val = _extract_attitude_score(row, metric)
                other_info[metric] = val
            
        return {
            "nodes": [], "edges": [], "other_info": other_info
        }
    # --- 5. 遍历并创建数据 (分两步) ---
    
    # 步骤 A: 遍历 Heuristic Agents (Tier 2)
    logger.info(f"(Agent Gen) 正在为 {len(tier2_info)} 个 Heuristic Agents (Tier 2) 创建对象...")
    for agent_id, row in tqdm.tqdm(tier2_info.iterrows(), total=len(tier2_info), desc="Building Heuristic Agents (Fast)"):
        
        agent_sim_id = int(agent_id)
        agent_sim_id_str = str(agent_id)
        
        user_name = row["username"]
        name = row["name"]
        bio = row["description"]
        user_id_str = row["user_id"]
        group_name = row["group"]
        
        AgentClass = TIER_2_CLASS_MAP.get(group_name, TIER_2_CLASS_MAP["default"])

        profile = _build_dynamic_profile(row, attitude_metrics)
        user_info = UserInfo(
            name=user_name, user_name=name, description=bio,
            profile=profile, recsys_type='twitter',
        )
        
        agent_env = SocialAction(
            agent_id=agent_sim_id, 
            channel=platform.channel
        )
       
        agent = AgentClass(
                agent_id=agent_sim_id,
                env=agent_env,
                db_path=db_path,
                user_info=user_info
            )
        # agent.group = group_name
        agent_list.append(agent)
        
        # (准备 DB 写入)
        num_followings = len(followings_map.get(agent_sim_id_str, set()))
        num_followers = len(followers_map.get(agent_sim_id_str, set()))
        sign_up_list.append((
            user_id_str, agent_sim_id, user_name, name, bio, 
            datetime.now(), num_followings, num_followers
        ))
        for follow_id_str in followings_map.get(agent_sim_id_str, set()):
            follow_list.append((agent_sim_id, int(follow_id_str), datetime.now()))
        
        agent_id_to_type_map[agent_sim_id] = ('ABM', 'internal_state')

            
    # 步骤 B: 遍历 LLM Agents (Tier 1)
    logger.info(f"(Agent Gen) 正在为 {len(tier1_info)} 个 LLM Agents (Tier 1) 创建对象...")
    for agent_id, row in tqdm.tqdm(tier1_info.iterrows(), total=len(tier1_info), desc="Building LLM Agents"):
        agent_sim_id = int(agent_id)
        
        # 将 row (Series) 转为 dict，并规范化字段名以匹配 create_single_agent 的要求
        profile_data = row.to_dict()
        
        
        # 实例化 (不进行 DB 写入，因为我们要批量写)
        agent = create_and_register_single_agent(
            agent_id=agent_sim_id,
            user_profile=profile_data,
            platform=platform,
            db_path=db_path,
            model=model,
            available_actions=available_actions,
            current_time_step=0, # 初始化是 0
            attitude_metrics=attitude_metrics,
            is_dynamic_injection=False # 标记为 False，跳过单次 DB 写入
        )
        
        agent_list.append(agent)

        posts_for_this_agent = initial_posts_map.get(user_id_str, [])
        _preload_agent_memory(agent, posts_for_this_agent)
        
        # (准备 DB 写入)
        num_followings = len(followings_map.get(agent_sim_id_str, set()))
        num_followers = len(followers_map.get(agent_sim_id_str, set()))
        sign_up_list.append((
            user_id_str, agent_sim_id, user_name, name, bio, 
            datetime.now(), num_followings, num_followers
        ))
        for follow_id_str in followings_map.get(agent_sim_id_str, set()):
            follow_list.append((agent_sim_id, int(follow_id_str), datetime.now()))

        agent_id_to_type_map[agent_sim_id] = ('LLM', 'external_expression')

    logger.info(f"(Agent Gen) 成功创建 {len(agent_list)} 个 agent (T1+T2) 在内存中。")
    
    # --- 6. 批量写入数据库 (不变) ---
    logger.info("... (Agent Gen) 正在批量注册用户到数据库 ...")
    user_insert_query = (
        f"INSERT OR IGNORE INTO user (user_id, agent_id, user_name, name, bio, "
        f"created_at, num_followings, num_followers) VALUES "
        f"(?, ?, ?, ?, ?, ?, ?, ?)"
    )
    platform.pl_utils._execute_many_db_command(user_insert_query,
                                               sign_up_list,
                                               commit=True)
    logger.info(f"... (Agent Gen) 成功注册 {len(sign_up_list)} 个用户。")

    logger.info("... (Agent Gen) 正在批量注册关注关系到数据库 ...")
    follow_insert_query = (
        "INSERT OR IGNORE INTO follow (follower_id, followee_id, created_at) "
        "VALUES (?, ?, ?)")
    platform.pl_utils._execute_many_db_command(follow_insert_query,
                                              follow_list,
                                              commit=True)
    logger.info(f"... (Agent Gen) 成功插入 {len(follow_list)} 条关注关系。")

    # --- 7. [!! 修正: 注入 T<0 日志 !!] ---
    if CALIBRATION_END and attitude_metrics:
        logger.info(f"... (Agent Gen) 准备计算 T<0 历史日志 (Metrics: {attitude_metrics})...")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 1. 【新增】Schema 检查：确认数据库中是否有这些列
            # 获取 post 表的所有列名
            cursor.execute("PRAGMA table_info(post)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            # 检查是否缺少必要的指标列
            missing_metrics = [m for m in attitude_metrics if m not in existing_columns]
            
            # 检查是否缺少 attitude_annotated 标记列
            is_annotated_column_missing = 'attitude_annotated' not in existing_columns

            if missing_metrics or is_annotated_column_missing:
                logger.warning(
                    f"⚠️ (Agent Gen) 跳过 T<0 日志生成: 数据库 'post' 表缺少必要的列。"
                    f"\n   - 缺失指标: {missing_metrics}"
                    f"\n   - 缺失标记列: {'attitude_annotated' if is_annotated_column_missing else 'None'}"
                    f"\n   (这通常意味着您跳过了 'OasisAttitudeProcessor' 的初始化步骤，这是正常的，但历史数据将不会有态度记录。)"
                )
                # 关闭连接并跳过后续逻辑，但不中断程序
                cursor.close()
                conn.close()
            else:
                # 2. 列存在，安全执行原来的逻辑
                logger.info("... (Agent Gen) 数据库 Schema 校验通过，开始提取历史数据...")
                
                # 动态生成 SQL 列名
                att_cols_sql = ", ".join([f"T1.{col}" for col in attitude_metrics])
                
                query = f"""
                SELECT 
                    T1.created_at, T1.user_id, T2.agent_id, {att_cols_sql}
                FROM post AS T1 
                INNER JOIN user AS T2 ON T1.user_id = T2.user_id
                WHERE T1.created_at < ? AND T1.attitude_annotated = 1
                """
                
                calibration_dt = datetime.fromisoformat(CALIBRATION_END)
                df_history = pd.read_sql_query(
                    query, conn, params=(calibration_dt.strftime("%Y-%m-%d %H:%M:%S"),)
                )
                
                if not df_history.empty:
                    # 计算 Time Step
                    df_history['created_at_dt'] = pd.to_datetime(df_history['created_at'])
                    delta_seconds = (calibration_dt - df_history['created_at_dt']).dt.total_seconds()
                    time_step_col = -((delta_seconds // (TIME_STEP_MINUTES * 60)) + 1).astype(int)
                    df_history['time_step'] = time_step_col
                    
                    # Group By
                    df_grouped = df_history.groupby(['time_step', 'user_id', 'agent_id'])[attitude_metrics].mean().reset_index()
                    
                    # 映射类型
                    df_grouped['agent_type_metric'] = df_grouped['agent_id'].map(agent_id_to_type_map)
                    df_grouped = df_grouped.dropna(subset=['agent_type_metric'])
                    df_grouped['agent_type'] = df_grouped['agent_type_metric'].apply(lambda x: x[0])
                    df_grouped['metric_type'] = df_grouped['agent_type_metric'].apply(lambda x: x[1])

                    # 插入日志
                    all_dims_to_log = attitude_metrics + ['attitude_average']
                    batch_insert_data = []
                    
                    for row in df_grouped.itertuples(index=False):
                        # 动态获取所有指标的值
                        scores_dict = {col: getattr(row, col) for col in attitude_metrics}
                        valid_scores = [s for s in scores_dict.values() if s is not None]
                        scores_dict['attitude_average'] = np.mean(valid_scores) if valid_scores else 0.0
                        
                        for dim_name in all_dims_to_log:
                            table_name = f"log_{dim_name}"
                            score = scores_dict.get(dim_name)
                            if score is not None:
                                batch_insert_data.append((
                                    table_name, row.time_step, row.user_id, row.agent_id,
                                    row.agent_type, row.metric_type, score
                                ))
                    
                    # 执行 SQL 插入
                    # cursor 已经在上面创建了，这里直接用
                    for item in batch_insert_data:
                        # item: (table, ts, uid, aid, atype, mtype, score)
                        sql = f"INSERT INTO {item[0]} (time_step, user_id, agent_id, agent_type, metric_type, attitude_score) VALUES (?, ?, ?, ?, ?, ?)"
                        cursor.execute(sql, item[1:])
                    conn.commit()
                    logger.info(f"... (Agent Gen) T<0 日志插入完成: {len(batch_insert_data)} 条")
                else:
                    logger.info("... (Agent Gen) 没有找到符合条件的 T<0 历史数据。")
                
                cursor.close()
                conn.close()
                
        except Exception as e:
            logger.error(f"❌ (Agent Gen) 处理 T<0 日志时发生错误: {e}")
            # 确保连接关闭（如果在 try 块中初始化了 conn）
            try:
                conn.close()
            except:
                pass
    
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
                # 获取类型信息
                atype, mtype = agent_id_to_type_map.get(aid, ("Unknown", "Unknown"))
                
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
                        (0, uid, aid, atype, mtype, float(val))
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