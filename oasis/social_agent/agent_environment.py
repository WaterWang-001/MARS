
from __future__ import annotations
import sqlite3
import json
from abc import ABC, abstractmethod
from string import Template
import logging # <-- 新增

from oasis.social_agent.agent_action import SocialAction
from oasis.social_platform.database import get_db_path

# (设置一个 logger)
env_log = logging.getLogger("oasis.environment")


class Environment(ABC):

    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    
    # --- [!! 1. 修改 followers 模板 !!] ---
    followers_env_template = Template("You are followed by these users$followers_info")
    # --- [!! 修改结束 !!] ---

    follows_env_template = Template("You are following these users$follows_info")
    
    broadcast_env_template = Template(
        "You see the following global broadcast messages: $broadcasts"
    )

    posts_env_template = Template(
        "After refreshing, you see some posts $posts")

    groups_env_template = Template(
        "And there are many group chat channels $all_groups\n"
        "And You are already in some groups $joined_groups\n"
        "You receive some messages from them $messages\n"
        "You can join the groups you are interested, "
        "leave the groups you already in, send messages to the group "
        "you already in.\n"
        "You must make sure you can only send messages to the group you "
        "are already in")
    
    env_template = Template(
        "$groups_env\n"
        "$followers_env\n"  
        "$follows_env\n"    
        "$broadcast_env\n"
        "$posts_env\n"
        "pick one you want to perform action that best "
        "reflects your current inclination based on your profile and "
        "posts content. Do not limit your action in just `like` to like posts")
 

    def __init__(self, action: SocialAction):
        self.action = action

    def get_posts_env(self, refresh_data: dict) -> str:
        # ... (此函数不变) ...
        if refresh_data.get("success") and refresh_data.get("posts"):
            posts_env = json.dumps(refresh_data["posts"], indent=4)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
        return posts_env
    

    def get_broadcast_env(self, refresh_data: dict) -> str:
        # ... (此函数不变) ...
        broadcasts = refresh_data.get("broadcast_messages")
        
        if broadcasts:
            broadcast_env = json.dumps(broadcasts, indent=4)
            broadcast_env = self.broadcast_env_template.substitute(
                broadcasts=broadcast_env
            )
        else:
            broadcast_env = "" 
        return broadcast_env
  
    
    # --- [!! 3. 重写 get_followers_env !!] ---
    async def get_followers_env(self) -> str:
        r""" (已修改) 获取关注此 agent 的用户的 name 和 bio。"""
        agent_id = self.action.agent_id
        try:
            db_path = get_db_path()
            # (使用只读模式)
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
            cursor = conn.cursor()
            
            # (使用 LEFT JOIN 查询关注者的信息)
            query = """
            SELECT T2.name, T2.bio, T2.user_name
            FROM follow AS T1
            LEFT JOIN user AS T2 ON T1.follower_id = T2.agent_id
            WHERE T1.followee_id = ?
            """
            cursor.execute(query, (agent_id,))
            results = cursor.fetchall()
            conn.close()

            if not results:
                return self.followers_env_template.substitute(
                    followers_info=": no one yet."
                )
            
            formatted_list = []
            for row in results:
                # (处理 "查不到就记空" 的情况)
                name = row[0] or "Unknown Name"
                bio = row[1] or "No bio"
                user_name = row[2] or "unknown_user"
                
                if len(bio) > 75:
                    bio = bio[:72] + "..."
                    
                formatted_list.append(
                    f"- {name} (@{user_name}): {bio}"
                )
            
            joined_string = "\n".join(formatted_list)
            return self.followers_env_template.substitute(
                followers_info=f" ({len(results)} total):\n{joined_string}"
            )
            
        except Exception as e:
            env_log.error(f"Error in get_followers_env (Agent {agent_id}): {e}", exc_info=True)
            return self.followers_env_template.substitute(
                followers_info=": an error occurred while fetching."
            )
    # --- [!! 修改结束 !!] ---
 
    async def get_follows_env(self) -> str:
        r""" (已修改) 获取此 agent 正在关注的用户的 name 和 bio。"""
        agent_id = self.action.agent_id
        try:
            db_path = get_db_path()
            # (使用只读模式)
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
            cursor = conn.cursor()
            
            # (使用 LEFT JOIN, 即使 followee_id 不在 user 表中也能安全运行)
            query = """
            SELECT T2.name, T2.bio, T2.user_name
            FROM follow AS T1
            LEFT JOIN user AS T2 ON T1.followee_id = T2.agent_id
            WHERE T1.follower_id = ?
            """
            cursor.execute(query, (agent_id,))
            results = cursor.fetchall()
            conn.close()

            if not results:
                return self.follows_env_template.substitute(
                    follows_info=": no one yet."
                )
            
            formatted_list = []
            for row in results:
                # (处理 "查不到就记空" 的情况)
                name = row[0] or "Unknown Name"
                bio = row[1] or "No bio"
                user_name = row[2] or "unknown_user"
                
                # (截断 bio 以保持 prompt 简洁)
                if len(bio) > 75:
                    bio = bio[:72] + "..."
                    
                formatted_list.append(
                    f"- {name} (@{user_name}): {bio}"
                )
            
            joined_string = "\n".join(formatted_list)
            return self.follows_env_template.substitute(
                follows_info=f" ({len(results)} total):\n{joined_string}"
            )
            
        except Exception as e:
            env_log.error(f"Error in get_follows_env (Agent {agent_id}): {e}", exc_info=True)
            return self.follows_env_template.substitute(
                follows_info=": an error occurred while fetching."
            )
    
    async def get_group_env(self) -> str:
        # ... (此函数不变) ...
        try:
            groups = await self.action.listen_from_group()
            if groups.get("success"):
                all_groups = json.dumps(groups.get("all_groups", {}))
                joined_groups = json.dumps(groups.get("joined_groups", []))
                messages = json.dumps(groups.get("messages", {}))
                groups_env = self.groups_env_template.substitute(
                    all_groups=all_groups,
                    joined_groups=joined_groups,
                    messages=messages,
                )
            else:
                groups_env = "No groups."
        except Exception as e:
            env_log.warning(f"get_group_env 失败: {e}")
            groups_env = "No groups."
        return groups_env

    # --- [!! 4. 确保 to_text_prompt 默认值 !!] ---
    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_followers: bool = True, # <--- 确保为 True
        include_follows: bool = True,    # <--- 确保为 True
    ) -> str:
        
        # 1. (效率优化) 只调用一次 refresh()
        refresh_data = {}
        if include_posts:
            try:
                refresh_data = await self.action.refresh()
            except Exception as e:
                env_log.error(f"Error during action.refresh(): {e}", exc_info=True)
                refresh_data = {"success": False, "error": str(e), "posts": [], "broadcast_messages": []}

        # 2. 获取其他异步部分
        # (修改: 如果不 include, 则返回空字符串)
        followers_env = (await self.get_followers_env()
                         if include_followers else "")
        follows_env = (await self.get_follows_env()
                       if include_follows else "")
        groups_env = await self.get_group_env()
        
        # 3. (修改) 调用非异步辅助函数
        posts_env = (self.get_posts_env(refresh_data) 
                     if include_posts else "")
        broadcast_env = (self.get_broadcast_env(refresh_data)
                         if include_posts else "")

        # 4. (修改) 注入所有 substitution
        return self.env_template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
            broadcast_env=broadcast_env, 
            # groups_env=groups_env,
        )
    # --- [!! 修改结束 !!] ---