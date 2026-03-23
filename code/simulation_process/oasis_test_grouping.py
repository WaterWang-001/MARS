import asyncio
import os
import sys
import logging
import random
import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import List, Set, Dict, Any, Iterable, Tuple, Optional
import sqlite3

# 优先使用本地 MARS/oasis 包而非 pip 安装的外部包
_MARS_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _MARS_ROOT not in sys.path:
    sys.path.insert(0, _MARS_ROOT)

from camel.models import OpenAICompatibleModel

# 引入新的 Annotator
from attitude_annotator import OpenAIAttitudeAnnotator, VLLMAttitudeAnnotator

import oasis
from oasis import (ActionType, LLMAction, ManualAction, HeuristicAction)
from oasis import generate_and_register_agents
from oasis.social_agent import BaseAgent
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType
from oasis.social_platform import Platform

from db_manager import reset_simulation_tables
from intervention_processor import InterventionProcessor 


# --- 全局配置 ---
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE_PATH = BASE_DIR / ".env"


def _load_local_env(path: Path) -> dict[str, str]:
    env_data: dict[str, str] = {}
    if not path.exists():
        return env_data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        env_data[key] = value
        os.environ.setdefault(key, value)
    return env_data


_LOCAL_ENV_CACHE = _load_local_env(ENV_FILE_PATH)

# Tier 1: "重" LLM Agents (初始化慢, 运行慢)
TIER_1_LLM_GROUPS = {
    "权威媒体/大V",
    "活跃KOL",
    "活跃创作者",
    "普通用户" 
}

# Tier 2: "轻" ABM Agents (初始化快, 运行快)
TIER_2_HEURISTIC_GROUPS = {
    "潜水用户"
}

# 时间设置
CALIBRATION_END = "2025-06-02T16:30:00"
TIME_STEP_MINUTES = 5

# 模拟总步数（支持通过环境变量覆盖）
TOTAL_STEPS = int(os.getenv("MARS_TOTAL_STEPS", "2"))

# 路径配置（可通过环境变量覆盖）
DEFAULT_PROFILE_PATH = str(BASE_DIR / "oasis_agent_init.csv")
DEFAULT_DB_PATH = str(BASE_DIR / "oasis_database.db")
DEFAULT_INTERVENTION_FILE = str(BASE_DIR / "intervention_messages.csv")

PROFILE_PATH = os.getenv("MARS_PROFILE_PATH", DEFAULT_PROFILE_PATH)
DB_PATH = os.getenv("MARS_DB_PATH", DEFAULT_DB_PATH)
INTERVENTION_FILE_PATH = os.getenv("MARS_INTERVENTION_PATH", DEFAULT_INTERVENTION_FILE)

# 态度配置支持 Json 覆盖
DEFAULT_ATTITUDE_CONFIG = {
    'attitude_TNT': "Evaluate the user's sentiment towards TNT."
}
_attitude_config_override = os.getenv("MARS_ATTITUDE_CONFIG_JSON")
if _attitude_config_override:
    try:
        ATTITUDE_CONFIG = json.loads(_attitude_config_override)
        if not isinstance(ATTITUDE_CONFIG, dict):
            ATTITUDE_CONFIG = DEFAULT_ATTITUDE_CONFIG
    except json.JSONDecodeError:
        ATTITUDE_CONFIG = DEFAULT_ATTITUDE_CONFIG
else:
    ATTITUDE_CONFIG = DEFAULT_ATTITUDE_CONFIG

ATTITUDE_METRICS_LIST = list(ATTITUDE_CONFIG.keys())

DEFAULT_MODEL_NAME = "gpt-4o-mini"
MODEL_NAME = os.getenv("MARS_MODEL_NAME", DEFAULT_MODEL_NAME)
MODEL_BASE_URL = os.getenv("MARS_MODEL_BASE_URL", "").strip()
MODEL_API_KEY = os.getenv("MARS_MODEL_API_KEY", "").strip()



async def main():
    # --- 1. 日志配置 ---
    log_dir = BASE_DIR / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = log_dir / f"oasis_test_{current_time}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_file_path), encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志将保存到: {str(log_file_path)}")

    if not MODEL_BASE_URL or not MODEL_API_KEY:
        logger.error("MARS_MODEL_BASE_URL 或 MARS_MODEL_API_KEY 未配置，请通过 .env 或控制台填写。")
        raise RuntimeError("Missing OpenAI-compatible endpoint configuration")

    # --- 2. 模型初始化 ---
    logger.info("正在初始化 Agent 模型...")
    logger.info(f"OpenAICompatibleModel: {MODEL_NAME} @ {MODEL_BASE_URL}")
    model = OpenAICompatibleModel(
        model_type=MODEL_NAME,
        model_config_dict={
            "temperature": 0.5,
            "max_tokens": 8192
        },
        api_key=MODEL_API_KEY,
        url=MODEL_BASE_URL 
    )
    # model = VLLMModel(
    #     model_type="model/qwen/Qwen2.5-7B-Instruct",
    #     model_config_dict={
    #         "temperature": 0.5
    #     }
    # )

    logger.info("Agent 模型初始化完毕。")
    
    # --- 3. Attitude 配置 ---
    logger.info(f"态度指标列表: {ATTITUDE_METRICS_LIST}")

    
    # if ENABLE_ATTITUDE_ANNOTATION:
    #     logger.info("正在初始化 AttitudeAnnotator (LLM 标注器)...")
    #     annotator = VLLMAttitudeAnnotator(
    #         model_name="model/qwen/Qwen2.5-7B-Instruct",
    #         attitude_config=ATTITUDE_CONFIG,
    #         concurrency_limit=1
    #     )
    #     logger.info("AttitudeAnnotator 初始化完毕。")
    # else:
    #     logger.info("AttitudeAnnotator 已禁用。")

    
    # --- 4. 路径配置 ---
    available_actions = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
        ActionType.QUOTE_POST
    ]

    profile_path = PROFILE_PATH
    db_path = DB_PATH
    # 干预文件路径
    intervention_file_path = INTERVENTION_FILE_PATH
  
    # --- 5. 数据库重置 ---
    logger.info("步骤 1: 正在重置数据库...")
    tables_to_keep = [
        'post', 
        'ground_truth_post', 
        'sqlite_sequence'
    ]
    reset_simulation_tables(
        db_path=db_path,
        tables_to_keep=tables_to_keep,
        logger=logger,
        calibration_cutoff=CALIBRATION_END
    )

    # --- 6. 环境创建 ---
    logger.info("步骤 2: 正在创建 Oasis 环境 (platform)...")
    
    # 【修改点 1】: 这里的 intervention_file_path 设为 None 
    # 因为我们不再让 Platform 自己加载 CSV，而是后面用 Processor 处理
    env = oasis.make(
            agent_graph=None, 
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=db_path,
            attitude_metrics=ATTITUDE_METRICS_LIST 
    )
    logger.info("环境和 Platform 已创建。")

    # --- 7. Agent 生成 ---
    logger.info(f"步骤 3: 正在从 {profile_path} 生成、注册并回填所有 Agents...")
    agent_list: List[BaseAgent] = await generate_and_register_agents(
        profile_path=profile_path,
        db_path=db_path,
        platform=env.platform, 
        model=model,
        available_actions=available_actions,
        CALIBRATION_END=CALIBRATION_END,
        TIME_STEP_MINUTES=TIME_STEP_MINUTES,
        attitude_metrics=ATTITUDE_METRICS_LIST
    )
    logger.info(f"Agent 生成和注册完毕, 共 {len(agent_list)} 个 agents。")
    env.agent_graph = agent_list
    print(f"已注册的 Agent ID 范围: {[agent.agent_id for agent in agent_list]}")
    current_max_agent_id = max((getattr(agent, "agent_id_int", int(agent.agent_id)) for agent in agent_list), default=0)

  
    if os.path.exists(intervention_file_path):
        logger.info(f"步骤 4: 检测到干预文件 {intervention_file_path}，正在进行预处理...")
        
        # 初始化处理器
        int_processor = InterventionProcessor(db_path=db_path)
        
        # 执行处理：读取 CSV -> 按组/ID筛选 Agent -> 写入 DB
        int_processor.process_and_distribute(
            csv_path=intervention_file_path,
            agent_list=agent_list
        )
        logger.info("干预指令已写入数据库 (agent_intervention / intervention_message)。")
    else:
        logger.warning(f"步骤 4: 未找到干预文件 {intervention_file_path}，跳过干预处理。")


    # --- 9. 环境重置 ---
    logger.info("步骤 5: 正在执行环境重置 (env.reset)...")
    await env.reset()
    logger.info("环境重置完毕。")
    
    # --- 10. Attitude Logger 初始化 ---
    logger.info("正在初始化 SimulationAttitudeLogger...")
    # attitude_logger = SimulationAttitudeLogger(
    #     db_path=db_path,
    #     attitude_columns=ATTITUDE_METRICS_LIST,
    #     tier_1_groups=TIER_1_LLM_GROUPS,
    #     tier_2_groups=TIER_2_HEURISTIC_GROUPS
    # )
    
    # --- 11. 模拟循环 ---
    TIER_1_ACTIVATION_RATES = {
        "权威媒体/大V": 0.8,
        "活跃KOL": 0.7,
        "活跃创作者": 0.6,
        "普通用户": 0.3, 
    }
    TIER_2_ACTIVATION_RATES = {
        "潜水用户": 0.1, 
    }
    
    for step in range(TOTAL_STEPS):
        current_step = step + 1 
        logger.info(f"--- 🚀 Simulation Step {current_step} / {TOTAL_STEPS} ---")
        
        # 11.1 动态激活器
        llm_agents_to_run = [] 
        heuristic_agents_to_run = [] 
        new_agents, current_max_agent_id = int_processor.execute_dynamic_registrations(
            env=env,
            current_step=current_step,
            current_max_agent_id=current_max_agent_id,
            model=model,
            available_actions=available_actions,
            attitude_metrics=ATTITUDE_METRICS_LIST,
            agent_list=agent_list 
        )
        if new_agents:
            # 新注册的 Agent 必须在当前步运行以执行任务
            llm_agents_to_run.extend(new_agents)
        if isinstance(env.agent_graph, list):
            total_active_pool = [(agent.agent_id, agent) for agent in env.agent_graph]
        else:
            total_active_pool = env.agent_graph.get_agents()
        
        for agent_id, agent in total_active_pool:
            group = agent.group
            if agent in llm_agents_to_run:
                continue
            if group in TIER_1_LLM_GROUPS:
                if random.random() < TIER_1_ACTIVATION_RATES.get(group, 0.0):
                    llm_agents_to_run.append(agent)
            elif group in TIER_2_HEURISTIC_GROUPS:
                if random.random() < TIER_2_ACTIVATION_RATES.get(group, 0.0):
                    heuristic_agents_to_run.append(agent)
                    
        logger.info(f"动态激活器: {len(llm_agents_to_run)} 个 LLM, {len(heuristic_agents_to_run)} 个 Heuristic 被激活。")

        # 【关键】更新 Agent 的内部时间步，确保 Log 写入正确时间
        active_agents = llm_agents_to_run + heuristic_agents_to_run
        for agent in active_agents:
            agent.current_time_step = current_step

        # 11.2 构建 Action 字典
        all_actions = {}
        all_actions.update({agent: LLMAction() for agent in llm_agents_to_run})
        all_actions.update({agent: HeuristicAction() for agent in heuristic_agents_to_run})

        # 11.3 执行 Step
        if all_actions:
            logger.info(f"即将为 {len(all_actions)} 个 agents统一执行 actions...")
            await env.step(all_actions)
        else:
            logger.info("本轮无 Agent 激活，跳过 step。")
        
        # 11.4 Attitude 标注
        # if annotator:
        #     try:
        #         logger.info(f"--- 🛠️ Maintenance Phase (after step {current_step}) - Attitude annotation ---")
        #         logger.info("... 正在标注 'post' 表中的新帖子 ...")
                
        #         await annotator.annotate_table(
        #             db_path=db_path, 
        #             table_name="post", 
        #             only_sim_posts=True, 
        #             batch_size=50 
        #         )
        #         logger.info("--- ✅ 'post' 表标注完成 ---")
        #     except Exception as e:
        #         logger.error(f"Attitude 标注失败: {e}", exc_info=True)
        


    await env.close()
    logger.info("--- Simulation Finished ---")
        

if __name__ == "__main__":
    asyncio.run(main())