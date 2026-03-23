import pandas as pd
import json
import os
import yaml
from typing import Any

def load_csv(file_path):
    print(f"📂 Loading {file_path} in chunks...")
    user_list = []
    chunk_size = 100000 
    
    try:
        with pd.read_csv(file_path, chunksize=chunk_size, dtype=str, on_bad_lines='skip', lineterminator='\n') as reader:
            
            for i, chunk in enumerate(reader):
                user_list.extend(chunk.to_dict('records'))
                # 每处理 5 个 chunk (50万行) 打印一次，避免 log 刷屏
                if i % 5 == 0:
                    print(f"  - Loaded {(i + 1) * chunk_size} rows...", end="", flush=True)

        print("Done!")
        
        return pd.DataFrame(user_list)

    except Exception as e:
        print(f"\n❌ Error loading CSV: {e}")
      
        return pd.DataFrame()

def save_json(data: Any, file_path: str):
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save JSON: {e}")

def append_jsonl(data: Any, file_path: str):
    """
    [新增] 追加写入 JSON Lines 文件
    适合处理流式数据，每一行是一个独立的 JSON 对象
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文正常显示
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Failed to append JSONL: {e}")

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_processed_ids(log_path):
    if not os.path.exists(log_path):
        return set()
    print(f"📜 Found resume log: {log_path}")
    with open(log_path, 'r', encoding='utf-8') as f:
        processed = set(line.strip() for line in f if line.strip())
    print(f"   Loaded {len(processed)} previously processed IDs.")
    return processed