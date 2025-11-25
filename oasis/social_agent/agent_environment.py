# =========== oasis/social_agent/agent_environment.py ===========
from __future__ import annotations
import sqlite3
import json
from abc import ABC, abstractmethod
from string import Template
import logging 

from oasis.social_agent.agent_action import SocialAction
try:
    from oasis.social_platform.database import get_db_path
except ImportError:
    def get_db_path(): return "data/oasis/oasis_database.db"

env_log = logging.getLogger("oasis.environment")


class Environment(ABC):
    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    
    # --- 基础模板 ---
    followers_env_template = Template("You are followed by these users:\n$followers_info")
    follows_env_template = Template("You are following these users:\n$follows_info")
    
    # 广播 (Broadcast)
    broadcast_env_template = Template(
        "You see the following global broadcast messages:\n$broadcasts"
    )

    # 帖子 (Posts)
    posts_env_template = Template(
        "After refreshing, you see some posts:\n$posts")

  
    intervention_env_template = Template(
        "\n[MANDATORY SPECIAL INSTRUCTION]\n"
        "You have received private instructions that MUST OVERRIDE your default persona:\n"
        "$instructions\n"
        "REQUIRED Attitude Adjustment: $attitude_target\n"
        "You are required to:\n"
        "1. IMMEDIATELY update your internal attitude to align with the target above.\n"
        "2. Perform actions (e.g., create_post) that strictly support these instructions.\n"
    )
    
    # --- 主环境模板 ---
    env_template = Template(
        "## Social Connections\n"
        "$followers_env\n\n"  
        "$follows_env\n\n"
        
        "## Platform Content\n"    
        "$broadcast_env\n"
        "$posts_env\n\n"
        
        "$intervention_env\n"
        
        "## Action Guidance\n"
        "Based on your profile, your internal attitude, "
        "and the content, pick ONE action."
    )
 

    def __init__(self, action: SocialAction):
        self.action = action
        self.db_path = getattr(action, 'db_path', None) or get_db_path()

    def get_posts_env(self, refresh_data: dict) -> str:
        if refresh_data.get("success") and refresh_data.get("posts"):
            posts = refresh_data["posts"]
            simplified_posts = []
            for p in posts:
                p_str = (f"- [PostID: {p.get('post_id')}] User {p.get('user_name')}: "
                         f"{p.get('content')}")
                if p.get('quote_content'):
                    p_str += f" (Quoting: {p.get('quote_content')})"
                simplified_posts.append(p_str)
            
            posts_env = "\n".join(simplified_posts)
            return self.posts_env_template.substitute(posts=posts_env)
        else:
            return "After refreshing, there are no existing posts."
    

    def get_broadcast_env(self, refresh_data: dict) -> str:
        broadcasts = refresh_data.get("broadcast_messages")
        if broadcasts:
            # 简单的列表格式化
            bc_str = "\n".join([f"- [SYSTEM]: {msg}" for msg in broadcasts])
            return self.broadcast_env_template.substitute(broadcasts=bc_str)
        else:
            return "(No broadcast messages)" 

    # --- [新增] 解析定向干预 ---
    def get_intervention_env(self, refresh_data: dict) -> str:
        """
        从 refresh 结果中提取 intervention_instructions 并生成 Prompt
        """
        instructions = refresh_data.get("intervention_instructions", [])
        if not instructions:
            return "" # 如果没有指令，返回空字符串
        
        # 聚合多条指令
        text_list = []
        attitude_changes = {}
        
        for item in instructions:
            # item 结构: {'type': 'bribery', 'content': '...', 'attitude_target': {...}}
            ctype = item.get('type', 'intervention').upper()
            content = item.get('content', '')
            
            text_list.append(f"- [{ctype}]: {content}")
            
            # 合并态度目标
            if item.get('attitude_target'):
                attitude_changes.update(item['attitude_target'])
        
        # 如果没有明确的态度改变，提示保持现状或根据文本推断
        att_str = json.dumps(attitude_changes, ensure_ascii=False) if attitude_changes else "(Infer from instructions)"
        
        return self.intervention_env_template.substitute(
            instructions="\n".join(text_list),
            attitude_target=att_str
        )
  
    
    async def get_followers_env(self) -> str:
        agent_id = self.action.agent_id
        try:
            if not self.db_path or ":memory:" in self.db_path:
                return self.followers_env_template.substitute(followers_info=": (DB not available)")

            conn = sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True)
            cursor = conn.cursor()
            
            query = """
            SELECT T2.name, T2.bio, T2.user_name
            FROM follow AS T1
            LEFT JOIN user AS T2 ON T1.follower_id = T2.agent_id
            WHERE T1.followee_id = ?
            LIMIT 10 
            """
            cursor.execute(query, (agent_id,))
            results = cursor.fetchall()
            conn.close()

            if not results:
                return self.followers_env_template.substitute(followers_info=": no one yet.")
            
            formatted_list = []
            for row in results:
                name = row[0] or "Unknown"
                bio = (row[1] or "")[:50] + "..." if row[1] and len(row[1]) > 50 else (row[1] or "")
                user_name = row[2] or "unknown"
                formatted_list.append(f"- {name} (@{user_name}): {bio}")
            
            return self.followers_env_template.substitute(
                followers_info=f" (Recent 10):\n" + "\n".join(formatted_list)
            )
        except Exception as e:
            env_log.error(f"Error in get_followers_env: {e}")
            return self.followers_env_template.substitute(followers_info=": error.")
 
    async def get_follows_env(self) -> str:
        agent_id = self.action.agent_id
        try:
            if not self.db_path or ":memory:" in self.db_path:
                return self.follows_env_template.substitute(follows_info=": (DB not available)")

            conn = sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True)
            cursor = conn.cursor()
            
            query = """
            SELECT T2.name, T2.bio, T2.user_name
            FROM follow AS T1
            LEFT JOIN user AS T2 ON T1.followee_id = T2.agent_id
            WHERE T1.follower_id = ?
            LIMIT 10
            """
            cursor.execute(query, (agent_id,))
            results = cursor.fetchall()
            conn.close()

            if not results:
                return self.follows_env_template.substitute(follows_info=": no one yet.")
            
            formatted_list = []
            for row in results:
                name = row[0] or "Unknown"
                bio = (row[1] or "")[:50] + "..." if row[1] and len(row[1]) > 50 else (row[1] or "")
                user_name = row[2] or "unknown"
                formatted_list.append(f"- {name} (@{user_name}): {bio}")
            
            return self.follows_env_template.substitute(
                follows_info=f" (Recent 10):\n" + "\n".join(formatted_list)
            )
        except Exception as e:
            env_log.error(f"Error in get_follows_env: {e}")
            return self.follows_env_template.substitute(follows_info=": error.")
    
    async def get_group_env(self) -> str:
        return ""

    # --- [修改] 集成所有部分 ---
    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_followers: bool = True, 
        include_follows: bool = True,
    ) -> str:
        
        # 1. 获取 Refresh 数据
        refresh_data = {}
        if include_posts:
            try:
                # Platform.refresh 现在会返回 posts, broadcast_messages, intervention_instructions
                refresh_data = await self.action.refresh()
            except Exception as e:
                env_log.error(f"Error during action.refresh(): {e}")
                refresh_data = {"success": False, "posts": [], "broadcast_messages": [], "intervention_instructions": []}

        # 2. 异步获取关系
        followers_env = (await self.get_followers_env() if include_followers else "")
        follows_env = (await self.get_follows_env() if include_follows else "")
        
        # 3. 格式化内容
        posts_env = (self.get_posts_env(refresh_data) if include_posts else "")
        broadcast_env = (self.get_broadcast_env(refresh_data) if include_posts else "")
        
        # [新增] 格式化干预指令
        intervention_env = (self.get_intervention_env(refresh_data) if include_posts else "")

        # 4. 生成最终 Prompt
        return self.env_template.safe_substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
            broadcast_env=broadcast_env,
            intervention_env=intervention_env # <--- 注入
        )