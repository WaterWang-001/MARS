# =========== agent_custom.py ===========

import random
import asyncio
import sqlite3
import json
import logging
import numpy as np
from typing import TYPE_CHECKING, Optional, Union, List, Callable, Dict
from datetime import datetime

# CAMEL 导入
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import OpenAIBackendRole
from camel.toolkits import FunctionTool

# OASIS 导入
from oasis.social_agent.agent import SocialAgent as OriginalOasisAgent
from oasis.social_agent.agent_action import SocialAction
from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType
# 引入 OASIS 定义的动作列表用于日志过滤
from oasis.social_agent.agent import ALL_SOCIAL_ACTIONS 

# 引入刚才抽离的态度工具处理器
from oasis.social_agent.agent_attitude import AttitudeToolHandler

agent_log = logging.getLogger("social.agent")


def _resolve_attitude_table_name(metric: str, existing_tables: set[str]) -> Optional[str]:
    """Find the actual table name for a given attitude metric."""
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
    seen = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        if name in existing_tables:
            return name
    return None

# --- BaseAgent (状态管理与数据库写入 - 保持不变) ---
class BaseAgent:
    def __init__(self, agent_id: int, user_info: UserInfo, channel: Channel | None = None, db_path: str | None = None, **kwargs):
        self.agent_id = str(agent_id)
        self._agent_id_int = int(agent_id)
        self.db_path = db_path
        self.user_info = user_info
        self.channel = channel or Channel()
        action = SocialAction(self._agent_id_int, self.channel)
        if db_path:
            action.db_path = db_path
        self.env = SocialEnvironment(action)
        self.group = user_info.profile["other_info"].get("group", "default")
        self.current_time_step = 0
        self.attitude_scores: Dict[str, float] = {}
        self._init_attitudes_from_profile()

    @property
    def agent_id_int(self) -> int:
        return self._agent_id_int

    def _init_attitudes_from_profile(self):
        try:
            profile_info = self.user_info.profile["other_info"]
            for key, value in profile_info.items():
                if key.startswith("attitude_") and key != "initial_attitude_avg":
                    try:
                        self.attitude_scores[key] = float(value)
                    except (ValueError, TypeError):
                        self.attitude_scores[key] = 0.0
        except Exception:
            pass

    def save_attitude_to_db(self):
        if not self.db_path or not self.attitude_scores: return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            original_user_id = self.user_info.profile["other_info"].get("original_user_id", str(self.agent_id))
            agent_type = "ABM" if isinstance(self, HeuristicAgent) else "LLM"
            metric_type = "internal_state" if agent_type == "ABM" else "external_expression"
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}

            for metric, score in self.attitude_scores.items():
                table_name = _resolve_attitude_table_name(metric, existing_tables)
                if not table_name:
                    continue
                sql = f"INSERT INTO {table_name} (time_step, user_id, agent_id, agent_type, metric_type, attitude_score) VALUES (?, ?, ?, ?, ?, ?)"
                cursor.execute(sql, (self.current_time_step, original_user_id, self.agent_id_int, agent_type, metric_type, score))
            
            if self.attitude_scores:
                avg_score = np.mean(list(self.attitude_scores.values()))
                avg_table = _resolve_attitude_table_name("attitude_average", existing_tables)
                if avg_table:
                    cursor.execute(
                        f"INSERT INTO {avg_table} (time_step, user_id, agent_id, agent_type, metric_type, attitude_score) VALUES (?, ?, ?, ?, ?, ?)",
                        (self.current_time_step, original_user_id, self.agent_id_int, agent_type, metric_type, avg_score),
                    )
            conn.commit()
        except Exception as e:
            print(f"Error saving attitude for agent {self.agent_id}: {e}")
        finally:
            conn.close()

    async def step(self):
        await self.env.action.do_nothing()


# --- SocialAgent (LLM) 核心修改 ---
class SocialAgent(OriginalOasisAgent, BaseAgent):
    """
    Tier 1 (重) LLM Agent。
    """
    def __init__(self,
                 agent_id: int,
                 user_info: UserInfo,
                 user_info_template: str | None = None,
                 db_path: str | None = None,
                 available_actions: list[ActionType] = None, # 显式接收
                 **kwargs):
        
        # 1. 处理 Persona
        if user_info_template:
            base_persona = user_info.profile["other_info"].get("user_profile", "")
            user_info.profile["other_info"]["user_profile"] = \
                user_info_template.format(base_persona=base_persona)
        
        # 2. 初始化 BaseAgent (为了拿到 attitude_scores)
        BaseAgent.__init__(self, agent_id=agent_id, user_info=user_info, db_path=db_path)
        
        # 3. 创建态度工具
        self.attitude_handler = AttitudeToolHandler(self)
        self.attitude_update_tool = self.attitude_handler.create_tool()
        
        # 4. 【修正】初始化 OriginalOasisAgent
        # - 传入 available_actions：父类会据此筛选 SocialAction 并存入 self.action_tools
        # - 传入 tools=[态度工具]：父类会将此列表与 action_tools 合并，注册给 LLM
        OriginalOasisAgent.__init__(
            self, 
            agent_id=str(agent_id), 
            user_info=user_info, 
            available_actions=available_actions, 
            tools=[self.attitude_update_tool], 
            **kwargs
        )
        if hasattr(self, 'env') and hasattr(self.env, 'action'):
            self.env.action.agent_id = self.agent_id_int
            if db_path:
                self.env.action.db_path = db_path
                self.env.db_path = db_path
        if hasattr(self, "memory") and hasattr(self.memory, "agent_id"):
            self.memory.agent_id = self.agent_id
        
        self.group = user_info.profile["other_info"].get("group", "default")

    async def perform_action_by_llm(self):
        """
        执行 LLM 思考与行动。
        """
        env_prompt = await self.env.to_text_prompt()
        
        # 1. 构建 Prompt：强制要求双重动作
        attitude_str = "\n".join([f"- {k}: {v:.2f}" for k, v in self.attitude_scores.items()])
        
        # 【关键修正】获取真正可用的社交动作名称列表
        # self.action_tools 是父类根据 available_actions 过滤后生成的 FunctionTool 列表
        if hasattr(self, 'action_tools') and self.action_tools:
            valid_social_actions = [t.func.__name__ for t in self.action_tools]
        else:
            # Fallback: 如果没设置，父类可能默认加载了所有 actions
            valid_social_actions = [t.func.__name__ for t in self.env.action.get_openai_function_list()]

        prompt_content = (
            f"You are a social media user interacting with a platform.\n"
            f"Here is your current internal attitude state (-1.0 to 1.0):\n{attitude_str}\n\n"
            f"Here is the environment:\n{env_prompt}\n\n"
            f"You MUST perform TWO parallel actions in this single turn using tool calls:\n"
            f"1. Call `{AttitudeToolHandler.TOOL_NAME}` to update your internal attitudes (or keep them same).\n"
            f"2. Call ONE social action tool from the following AVAILABLE list: {valid_social_actions}.\n"
        )

        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=prompt_content
        )
        
        try:
            agent_log.info(f"Agent {self.agent_id} observing environment...")
            
            # 2. 执行 LLM 调用 (CAMEL 会自动执行 Tool Calls)
            response = await self.astep(user_msg)
            
            # 3. 记录日志 (此时 Tool 已经执行完毕)
            if response.info and 'tool_calls' in response.info:
                for tool_call in response.info['tool_calls']:
                    # 这里的 tool_name 应该是 update_internal_attitude 或 create_post 等
                    agent_log.info(f"Agent {self.agent_id} executed: {tool_call.tool_name}")

            # 4. 写入当前态度到 DB (无论变没变)
            self.save_attitude_to_db()
            
            return response
            
        except Exception as e:
            agent_log.error(f"Agent {self.agent_id} step error: {e}")
            return e


# --- HeuristicAgent (Tier 2) 保持不变 ---
class HeuristicAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        for k in ["model", "available_actions", "user_info_template", "tools", "single_iteration", "interview_record"]:
            kwargs.pop(k, None)
        super().__init__(*args, **kwargs)

# --- 子类定义 (保持不变) ---
class AuthorityAgent(SocialAgent):
    def __init__(self, *args, **kwargs):
        kwargs["user_info_template"] = "You are an authoritative media outlet... Persona: {base_persona}"
        super().__init__(*args, **kwargs)

class KOLAgent(SocialAgent):
    def __init__(self, *args, **kwargs):
        kwargs["user_info_template"] = "You are an active KOL... Persona: {base_persona}"
        super().__init__(*args, **kwargs)
        
class ActiveCreatorAgent(SocialAgent):
    def __init__(self, *args, **kwargs):
        kwargs["user_info_template"] = "You are a highly active content creator... Persona: {base_persona}"
        super().__init__(*args, **kwargs)

class NormalUserAgent(SocialAgent):
    def __init__(self, *args, **kwargs):
        kwargs["user_info_template"] = "You are a typical user... Persona: {base_persona}"
        super().__init__(*args, **kwargs)

# --- LurkerAgent (ABM) 保持不变 ---
# (此处省略，与之前确认的 LurkerAgent 逻辑完全一致)
class LurkerAgent(HeuristicAgent):
    def __init__(self, agent_id: str, env: SocialAction, db_path: str, **kwargs):
        super().__init__(agent_id=agent_id, env=env, db_path=db_path, **kwargs)
        self.confidence_threshold = 0.5 
        self.convergence_mu = 0.2        
        self.base_action_prob = 0.05     

    def get_authors_attitudes(self, user_ids: List[str]) -> List[Dict[str, float]]:
        if not user_ids: return []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        attitude_vectors = []
        try:
            unique_users = list(set(user_ids))
            if not unique_users: return []
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}
            user_vectors = {uid: {} for uid in unique_users}
            for metric in self.attitude_scores.keys():
                table_name = _resolve_attitude_table_name(metric, existing_tables)
                if not table_name:
                    continue
                placeholders = ','.join('?' for _ in unique_users)
                sql = (
                    f"SELECT user_id, attitude_score FROM {table_name} "
                    f"WHERE user_id IN ({placeholders}) GROUP BY user_id HAVING time_step = MAX(time_step)"
                )
                try:
                    cursor.execute(sql, unique_users)
                    for row in cursor.fetchall():
                        user_vectors[row[0]][metric] = float(row[1])
                except sqlite3.OperationalError:
                    continue
            for uid in user_ids:
                if uid in user_vectors and user_vectors[uid]:
                    attitude_vectors.append(user_vectors[uid])
            return attitude_vectors
        except Exception as e:
            print(f"LurkerAgent query error: {e}")
            return []
        finally:
            conn.close()

    async def step(self):
        try:
            refresh_response = await self.env.action.refresh()
            posts_list = refresh_response.get("posts", [])
            if not refresh_response.get("success") or not posts_list:
                self.save_attitude_to_db()
                await self.env.action.do_nothing()
                return
        except Exception:
            self.save_attitude_to_db()
            await self.env.action.do_nothing()
            return

        author_ids = [str(post.get('user_id')) for post in posts_list if post.get('user_id')]
        author_attitude_vectors = self.get_authors_attitudes(author_ids)

        if author_attitude_vectors:
            new_scores = self.attitude_scores.copy()
            for dim in self.attitude_scores.keys():
                a_i = self.attitude_scores[dim]
                candidate_attitudes = []
                for vec in author_attitude_vectors:
                    a_j = vec.get(dim)
                    if a_j is not None and abs(a_i - a_j) < self.confidence_threshold:
                        candidate_attitudes.append(a_j)
                if candidate_attitudes:
                    target = random.choice(candidate_attitudes)
                    delta = self.convergence_mu * (target - a_i)
                    new_scores[dim] = max(-1.0, min(1.0, a_i + delta))
            self.attitude_scores = new_scores

        self.save_attitude_to_db()
        
        overall_extremity = np.mean([abs(s) for s in self.attitude_scores.values()]) if self.attitude_scores else 0
        action_probability = self.base_action_prob + (overall_extremity * 0.10)
        
        if random.random() < (1.0 - action_probability):
            await self.env.action.do_nothing()
        else:
            if random.random() < 0.5 and posts_list:
                post_to_like = random.choice(posts_list)
                await self.env.action.like_post(post_to_like["post_id"])
            else:
                await self.env.action.do_nothing()