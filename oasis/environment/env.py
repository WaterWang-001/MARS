# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import asyncio
import logging
import os
from datetime import datetime
from typing import List, Union, Optional 

from oasis.environment.env_action import (LLMAction, ManualAction,
                                          HeuristicAction)
from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_graph import AgentGraph
try:
    from oasis.social_agent.agent_custom import BaseAgent
except ImportError:
    # (如果 agent_custom.py 不存在, 则回退)
    from oasis.social_agent.agent import BaseAgent 
    
from oasis.social_agent.agents_generator import generate_custom_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import (ActionType, DefaultPlatformType,
                                          RecsysType)

# ... (日志配置不变) ...
log_dir = "./log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
env_log = logging.getLogger("oasis.env")
env_log.setLevel("INFO")
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_handler = logging.FileHandler(f"{log_dir}/oasis-{current_time}.log",
                                   encoding="utf-8")
file_handler.setLevel("INFO")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
env_log.addHandler(file_handler)


class OasisEnv:

    def __init__(
        self,
        agent_graph: Optional[Union[AgentGraph, List[BaseAgent]]],
        platform: Union[DefaultPlatformType, Platform],
        database_path: str = None,
        semaphore: int = 128,
        intervention_file_path: str = None,
    ) -> None:
        r"""Init the oasis environment.

        Args:
            agent_graph: The Agent list or AgentGraph (legacy).
                         (在新的 100w 流程中, 这在 'make' 时为 None)
            ...
        """
        # Initialize the agent graph
        self.agent_graph = agent_graph
        # ... (其余 __init__ 逻辑不变) ...
        self.llm_semaphore = asyncio.Semaphore(semaphore)
        if isinstance(platform, DefaultPlatformType):
            if database_path is None:
                raise ValueError(
                    "database_path is required for DefaultPlatformType")
            self.platform = platform
            if platform == DefaultPlatformType.TWITTER:
                self.channel = Channel()
                self.platform = Platform(
                    db_path=database_path,
                    channel=self.channel,
                    # recsys_type="twhin-bert",
                    recsys_type="random",
                    refresh_rec_post_count=2,
                    max_rec_post_len=2,
                    following_post_count=3,
                    intervention_file_path=intervention_file_path
                )
                self.platform_type = DefaultPlatformType.TWITTER
            elif platform == DefaultPlatformType.REDDIT:
                self.channel = Channel()
                self.platform = Platform(
                    db_path=database_path,
                    channel=self.channel,
                    recsys_type="reddit",
                    allow_self_rating=True,
                    show_score=True,
                    max_rec_post_len=100,
                    refresh_rec_post_count=5,
                )
                self.platform_type = DefaultPlatformType.REDDIT
            else:
                raise ValueError(f"Invalid platform: {platform}. Only "
                                 "DefaultPlatformType.TWITTER or "
                                 "DefaultPlatformType.REDDIT are supported.")
        elif isinstance(platform, Platform):
            if database_path != platform.db_path:
                env_log.warning("database_path is not the same as the "
                                "platform.db_path, using the platform.db_path")
            self.platform = platform
            self.channel = platform.channel
            if platform.recsys_type == RecsysType.REDDIT:
                self.platform_type = DefaultPlatformType.REDDIT
            else:
                self.platform_type = DefaultPlatformType.TWITTER
            if intervention_file_path:
                if hasattr(self.platform, '_load_interventions_from_csv'):
                    env_log.info(f"Loading interventions from {intervention_file_path} into pre-existing platform...")
                    # (这是一个我们即将在 Platform 类中创建的方法)
                    self.platform._load_interventions_from_csv(intervention_file_path)
                else:
                    env_log.warning("OasisEnv received 'intervention_file_path' but the pre-supplied Platform object has no '_load_interventions_from_csv' method.")
        else:
            raise ValueError(
                f"Invalid platform: {platform}. You should pass a "
                "DefaultPlatformType or a Platform instance.")
        

    async def reset(self) -> None:
        r"""Start the platform and sign up the agents."""
        self.platform_task = asyncio.create_task(self.platform.running())
        

        self.agent_graph = await generate_custom_agents(
            platform=self.platform, 
            agent_list=self.agent_graph
        )
        

    async def _perform_llm_action(self, agent):
        # ... (此方法不变) ...
        async with self.llm_semaphore:
            return await agent.perform_action_by_llm()

    async def _perform_interview_action(self, agent, interview_prompt: str):
        # ... (此方法不变) ...
        async with self.llm_semaphore:
            return await agent.perform_interview(interview_prompt)

    async def step(
        self, 
        actions: dict[SocialAgent, Union[ManualAction, LLMAction, HeuristicAction,
                                               List[Union[ManualAction,
                                                          LLMAction,
                                                          HeuristicAction]]]]
        # --- [!! 修改结束 !!] ---
    ) -> None:
        r"""Update the recommendation system and perform the actions."""
        
        # ... (step 内部的所有逻辑不变) ...
        
        await self.platform.update_rec_table()
        env_log.info("update rec table.")

        tasks = []
        for agent, action in actions.items():
            if isinstance(action, list):
                for single_action in action:
                    if isinstance(single_action, ManualAction):
                        if single_action.action_type == ActionType.INTERVIEW:
                            interview_prompt = single_action.action_args.get(
                                "prompt", "")
                            tasks.append(
                                self._perform_interview_action(
                                    agent, interview_prompt))
                        else:
                            tasks.append(
                                agent.perform_action_by_data(
                                    single_action.action_type,
                                    **single_action.action_args))
                    elif isinstance(single_action, LLMAction):
                        tasks.append(self._perform_llm_action(agent))
                    
                    elif isinstance(single_action, HeuristicAction):
                        if hasattr(agent, 'step') and callable(agent.step):
                            tasks.append(agent.step())
                        else:
                            env_log.warning(f"Agent {agent.agent_id} "
                                            "received HeuristicAction but "
                                            "has no .step() method.")
            else:
                if isinstance(action, ManualAction):
                    if action.action_type == ActionType.INTERVIEW:
                        interview_prompt = action.action_args.get("prompt", "")
                        tasks.append(
                            self._perform_interview_action(
                                agent, interview_prompt))
                    else:
                        tasks.append(
                            agent.perform_action_by_data(
                                action.action_type, **action.action_args))
                elif isinstance(action, LLMAction):
                    tasks.append(self._perform_llm_action(agent))
                
                elif isinstance(action, HeuristicAction):
                    if hasattr(agent, 'step') and callable(agent.step):
                        tasks.append(agent.step())
                    else:
                        env_log.warning(f"Agent {agent.agent_id} "
                                        "received HeuristicAction but "
                                        "has no .step() method.")

        await asyncio.gather(*tasks)
        env_log.info("performed all actions.")
        
        if self.platform_type == DefaultPlatformType.TWITTER:
            self.platform.sandbox_clock.time_step += 1

    async def close(self) -> None:
        # ... (此方法不变) ...
        await self.channel.write_to_receive_queue(
            (None, None, ActionType.EXIT))
        await self.platform_task
        env_log.info("Simulation finished! Please check the results in the "
                     f"database: {self.platform.db_path}. Note that the trace "
                     "table stored all the actions of the agents.")