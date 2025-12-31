import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging
import sys
import multiprocessing
import time
import shutil
from typing import List, Dict, Tuple

# --- 尝试导入极速 JSON 库 ---
try:
    import orjson
    JSON_LIB = orjson
except ImportError:
    import json
    JSON_LIB = json

# --- 配置 ---
PROJECT_ROOT = Path(__file__).parent.parent
MARS_ROOT = PROJECT_ROOT / "MARS"
INPUT_DIRECTORY = PROJECT_ROOT / "data" / "raw"
LOG_FOLDER = PROJECT_ROOT / "logs"
TEMP_OUTPUT_DIR = PROJECT_ROOT / "data" / "temp_edges" # 临时文件夹

OUTPUT_MATRIX_CSV = MARS_ROOT / "scripts" / "attention_matrix_edges_COMBINED.csv"
OUTPUT_SCORES_CSV = MARS_ROOT / "scripts" / "total_attention_scores_COMBINED.csv"

# 权重配置 (作为全局变量方便多进程访问)
WEIGHTS = {
    'strong': 3,
    'moderate': 2,
    'weak': 1
}

# --- 静态辅助函数 (Worker 专用) ---

def _normalize_user_id_static(raw) -> str:
    """静态版本的 ID 标准化，移除 self 引用加速"""
    if raw is None: return ""
    # 最常见的情况放在最前面
    if isinstance(raw, str): return raw.strip().lstrip("@")
    if isinstance(raw, (int, float)): return str(int(raw))
    if isinstance(raw, dict):
        # 优化：按概率高低排序
        for k in ("uid", "id", "user_id", "uid_str", "userid"):
            val = raw.get(k)
            if val: return str(val)
    return str(raw)

def _extract_mentions_static(record: dict) -> List[Tuple[str, int]]:
    """
    静态提取函数
    返回: List[(target_id, weight_value)]
    """
    candidates = None
    # 优化：减少循环，直接查找最可能的键
    # 按照数据中出现的频率排序 keys
    for key in ("sjcjMentions", "mentions", "atUserList", "at_users", "atUser"):
        val = record.get(key)
        if val:
            candidates = val
            break
            
    if not candidates:
        return []

    # 字符串处理
    if isinstance(candidates, str):
        if candidates.startswith('['): # 看起来像 JSON
            try:
                candidates = JSON_LIB.loads(candidates)
            except:
                return []
        else:
            candidates = [c.strip() for c in candidates.split(",") if c.strip()]

    if not candidates:
        return []

    out = []
    for item in candidates:
        strength_val = 2 # moderate default
        target = ""

        if isinstance(item, dict):
            # 提取 ID
            target = _normalize_user_id_static(
                item.get("uid") or item.get("id") or item.get("user_id") or item.get("name")
            )
            
            # 判断权重
            # 优化：将 lookup 逻辑简化
            typ = str(item.get("type") or item.get("interaction") or "").lower()
            if typ:
                if "mention" in typ or "at" in typ:
                    strength_val = 3 # strong
                elif any(x in typ for x in ("repost", "forward", "foc")):
                    strength_val = 2 # moderate
                else:
                    strength_val = 1 # weak
        else:
            # 简单字符串列表
            target = _normalize_user_id_static(item)
            strength_val = 3 # 通常纯列表是 @提及

        if target:
            out.append((target, strength_val))
    return out

def process_file_worker(filepath: Path) -> str:
    """
    [Worker 进程]
    处理单个文件 -> 在内存中聚合 -> 保存为临时 CSV
    返回: 临时 CSV 的路径 (或错误信息)
    """
    local_counts = defaultdict(float)
    
    try:
        # 使用二进制模式读取，配合 orjson
        with open(filepath, 'rb') as f:
            for line in f:
                try:
                    rec = JSON_LIB.loads(line)
                    
                    # 1. 提取 Source
                    # 优化：链式 get 比较慢，按概率分层
                    actor = rec.get("sjcjId") # 假设这是最常见的
                    if not actor:
                        actor = rec.get("user_id") or rec.get("uid") or rec.get("actor_id") or rec.get("mid")
                    
                    actor = _normalize_user_id_static(actor)
                    if not actor: continue

                    # 2. 提取 Targets
                    mentions = _extract_mentions_static(rec)
                    
                    # 3. 局部聚合 (Map 阶段的核心优化)
                    # 我们在这里就把重复的边合并了，大大减少后续的数据量
                    for tgt, weight in mentions:
                        if tgt and tgt != actor:
                            local_counts[(actor, tgt)] += weight

                except Exception:
                    continue # 忽略单行错误

        # 如果没有数据，直接返回 None
        if not local_counts:
            return None

        # 将局部聚合结果写入临时 CSV
        # 格式: source, target, weight
        temp_file = TEMP_OUTPUT_DIR / f"{filepath.stem}_{int(time.time()*1000)}.csv"
        
        # 转换为列表写入，比 DataFrame 快
        with open(temp_file, 'w', encoding='utf-8') as out_f:
            out_f.write("source,target,weight\n") # Header
            for (src, tgt), w in local_counts.items():
                out_f.write(f"{src},{tgt},{w}\n")
                
        return str(temp_file)

    except Exception as e:
        return f"ERROR: {filepath.name} - {str(e)}"

class FastRelationshipPipeline:
    def __init__(self):
        self.input_directory = INPUT_DIRECTORY
        TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # 清理旧的临时文件
        for f in TEMP_OUTPUT_DIR.glob("*.csv"):
            f.unlink()

    def run(self):
        txt_files = list(self.input_directory.glob('*.txt'))
        if not txt_files:
            print("❌ 没有找到文件")
            return

        print(f"🚀 [Phase 1] 并行处理 {len(txt_files)} 个文件 (Map阶段)...")
        
        temp_csv_files = []
        start_time = time.time()
        
        # 使用所有 CPU 核心
        cpu_count = multiprocessing.cpu_count()
        # 如果文件很多，chunksize 设置大一点可以减少任务切换开销
        chunk_size = max(1, len(txt_files) // (cpu_count * 4))
        
        with multiprocessing.Pool(processes=cpu_count) as pool:
            # imap_unordered 可以在结果生成时立即获取，适合进度条
            results = pool.imap_unordered(process_file_worker, txt_files, chunksize=chunk_size)
            
            for res in results:
                if res:
                    if res.startswith("ERROR"):
                        print(f"  ⚠️ {res}")
                    else:
                        temp_csv_files.append(res)
                        if len(temp_csv_files) % 10 == 0:
                            print(f"\r  -> 已生成 {len(temp_csv_files)} 个中间聚合文件...", end="")

        print(f"\n✅ Phase 1 完成。耗时: {time.time() - start_time:.2f}s")
        print(f"🚀 [Phase 2] 合并中间结果 (Reduce阶段)...")

        self.aggregate_and_save(temp_csv_files)

    def aggregate_and_save(self, csv_files: List[str]):
        """
        读取所有临时 CSV，进行最终的 GroupBy Sum，然后保存
        """
        if not csv_files:
            print("没有产生任何数据。")
            return

        # 优化：分块合并 (防止一次性读取几百个CSV导致内存爆炸)
        # 如果 csv_files 很多，我们分批读取 concat，最后再 groupby
        
        chunk_size = 50 # 每次合并 50 个文件
        combined_dfs = []
        
        total_files = len(csv_files)
        print(f"  -> 正在合并 {total_files} 个文件片段...")

        for i in range(0, total_files, chunk_size):
            chunk_files = csv_files[i : i + chunk_size]
            try:
                # 读取这一批文件
                df_chunk = pd.concat((pd.read_csv(f) for f in chunk_files), ignore_index=True)
                
                # 在这一批次内先做一次预聚合 (Pre-aggregation)
                # 这样可以保持内存中的 DataFrame 比较小
                df_grouped = df_chunk.groupby(['source', 'target'], as_index=False)['weight'].sum()
                combined_dfs.append(df_grouped)
                
                print(f"\r    -> 进度: {min(i + chunk_size, total_files)}/{total_files}", end="")
            except Exception as e:
                print(f"合并出错: {e}")

        print("\n  -> 正在执行最终全局聚合...")
        
        # 将所有预聚合的小块合并
        final_df = pd.concat(combined_dfs, ignore_index=True)
        final_df = final_df.groupby(['source', 'target'], as_index=False)['weight'].sum()
        
        print(f"  -> 最终结果: {len(final_df)} 条边。正在保存...")
        
        # 保存结果
        OUTPUT_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
        
        # 排序
        final_df.sort_values(by=['source', 'weight'], ascending=[True, False], inplace=True)
        final_df.to_csv(OUTPUT_MATRIX_CSV, index=False, encoding='utf-8-sig')
        
        # 计算分数表
        scores = final_df.groupby('source')['weight'].sum().sort_values(ascending=False)
        scores.to_csv(OUTPUT_SCORES_CSV, header=['total_out_score'], encoding='utf-8-sig')
        
        print(f"✅ 处理完毕！\n  边表: {OUTPUT_MATRIX_CSV}\n  分值: {OUTPUT_SCORES_CSV}")
        
        # 清理临时文件
        try:
            shutil.rmtree(TEMP_OUTPUT_DIR)
        except:
            pass

if __name__ == "__main__":
    # Windows 下 multiprocessing 必需
    logging.basicConfig(level=logging.INFO)
    pipeline = FastRelationshipPipeline()
    pipeline.run()