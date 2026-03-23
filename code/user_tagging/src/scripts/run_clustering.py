import sys
import os
import yaml
import logging
import sqlite3
import time

# 将 user_tagging/ 加入 path
current_dir = os.path.dirname(os.path.abspath(__file__))        # src/scripts/
package_root = os.path.dirname(os.path.dirname(current_dir))    # user_tagging/
sys.path.insert(0, package_root)

from src.core.db_client import DBClient
from src.core.api_client import APIClient
from src.tasks.sft.state_manager import StateManager
from src.tasks.clustering.clustering_manager import ClusteringManager

# ================= 配置 =================
LOG_DIR = os.path.join(package_root, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'offline_clustering.log')),
        logging.StreamHandler()
    ]
)
# =======================================

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    logging.info(">>> Starting Offline Clustering Job...")
    
    # 1. 加载配置
    config_path = os.path.join(package_root, 'configs', 'settings.yaml')
    if not os.path.exists(config_path):
        logging.error(f"Config file not found at {config_path}")
        return
    
    config = load_config(config_path)
    
    # 2. 路径配置 (指向同一个 State DB)
    # 注意：这里只需要 State DB，不需要 Post DB
    state_db_path = 'remote-home/JuelinW/oasis_project/MARS/data/output/tagging_result/processing_state.db'
    
    if not os.path.exists(state_db_path):
        logging.error(f"State DB not found at {state_db_path}. No data to cluster.")
        return

    # 3. 初始化组件
    llm_cfg = config['llm_service']
    emb_cfg = config.get('embedding_service', llm_cfg) # 假设有单独的 embedding 配置，否则复用 LLM
    
    # A. 状态管理器
    state_manager = StateManager(state_db_path)
    
    # B. LLM Client (用于给聚类生成 Label)
    api_client = APIClient(
        api_key=llm_cfg['api_key'], 
        base_url=llm_cfg['base_url'], 
        model_name=llm_cfg['model_name'],
        timeout=120
    )
    
    # C. Embedding Client (用于向量化 Tag)
    # 如果你的 APIClient 已经支持 get_embeddings，直接复用
    # 如果是单独的 AzureOpenAI / HuggingFace，请在这里初始化
    logging.info("Initializing Embedding Client...")
    embedding_client = api_client 
    
    # D. Clustering Manager
    # 注意：这里不需要 PromptManager，因为 ClusteringManager 内部生成简单 Prompt
    # 也不需要 DBClient，因为它只操作 State DB
    manager = ClusteringManager(
        state_manager=state_manager,
        api_client=api_client,
        embedding_client=embedding_client,
        prompt_manager=None 
    )
    
    # 4. 执行重型任务
    try:
        start_time = time.time()
     
        manager.run_clustering_job(batch_size=2000) 
        
        duration = time.time() - start_time
        logging.info(f"<<< Offline Clustering Finished in {duration:.2f} seconds.")
        
    except Exception as e:
        logging.exception(f"Critical Error during offline clustering: {e}")

if __name__ == "__main__":
    main()