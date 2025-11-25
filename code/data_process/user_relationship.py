import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging
import sys
from typing import Iterable, Tuple, Dict, Set, Any, Optional
from tqdm import tqdm  # 引入进度条库

# --- 配置路径 ---
PROJECT_ROOT = Path(__file__).parent.parent
MARS_ROOT = PROJECT_ROOT / "MARS"
INPUT_DIRECTORY = PROJECT_ROOT / "data" / "raw"
LOG_FOLDER = PROJECT_ROOT / "logs"

# 输出文件路径
OUTPUT_MATRIX_CSV = MARS_ROOT / "scripts" / "attention_matrix_edges_COMBINED.csv"
OUTPUT_SCORES_CSV = MARS_ROOT / "scripts" / "total_attention_scores_COMBINED.csv"

# --- 配置 Logging ---
LOG_FOLDER.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FOLDER / "relationship_generation.log", encoding='utf-8')
        # 移除 StreamHandler 以保护进度条
    ]
)

# --- 权重配置 ---
WEIGHTS = {
    'strong': 3,    # Mention / At
    'moderate': 2,  # Repost / Comment (Direct)
    'weak': 1       # Secondary Repost/Comment
}

class UserRelationshipPipeline:
    """
    用户关系抽取流水线 (带进度监控版)
    - 遍历原始数据
    - 提取交互 (Mention, Repost, etc.)
    - 计算加权有向边
    - 导出 CSV
    """

    def __init__(self, input_directory: Path = INPUT_DIRECTORY, weights: Dict[str, int] = WEIGHTS):
        self.input_directory = input_directory
        self.weights = weights
        
        # 核心数据结构：内存中聚合
        # Key: (source_id, target_id), Value: total_weight
        self.interaction_counts = defaultdict(float)
        self.users_found = set()
        self.file_stats = {}

    def _normalize_user_id(self, raw: Any) -> str:
        """标准化用户ID"""
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw.strip().lstrip("@")
        if isinstance(raw, (int,)):
            return str(raw)
        if isinstance(raw, dict):
            for k in ("uid", "id", "user_id", "userid", "uid_str"):
                if k in raw:
                    return str(raw[k])
        return str(raw)

    def _extract_mentions(self, record: dict) -> Iterable[Tuple[str, str]]:
        """从单条记录中提取 (target_user_id, strength_type)"""
        candidates = []
        # 尝试从不同字段获取 mention 信息
        for key in ("sjcjMentions", "mentions", "at_users", "atUserList", "atUser"):
            if key in record and record[key]:
                candidates = record[key]
                break
        
        if isinstance(candidates, str):
            try:
                candidates = json.loads(candidates)
            except Exception:
                # 简单的逗号分隔容错
                candidates = [c.strip() for c in candidates.split(",") if c.strip()]
        
        if not candidates:
            return []

        out = []
        for item in candidates:
            strength = "moderate" # 默认权重
            target = ""

            if isinstance(item, dict):
                target = self._normalize_user_id(
                    item.get("uid") or item.get("id") or item.get("user_id") or 
                    item.get("uid_str") or item.get("name")
                )
                
                # 简单的类型判断逻辑
                typ = None
                for k in ("type", "interaction", "relation"):
                    if k in item:
                        typ = str(item[k]).lower()
                        break
                
                if typ:
                    if "mention" in typ or "at" in typ:
                        strength = "strong"
                    elif any(x in typ for x in ["p-repost", "foc", "forward", "repost"]):
                        strength = "moderate"
                    else:
                        strength = "weak"
            else:
                # 如果只是简单的字符串列表，默认为 strong (通常是 @列表)
                target = self._normalize_user_id(item)
                strength = "strong"
            
            if target:
                out.append((target, strength))
        return out

    def process_single_file(self, filepath: Path, outer_pbar):
        """处理单个文件"""
        filename = filepath.name
        logging.info(f"Processing file: {filename}")
        outer_pbar.set_description(f"正在处理: {filename}")

        line_count = 0
        valid_interactions = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # 内层进度条：显示行处理速度
                with tqdm(f, unit="line", leave=False, desc="Lines") as line_pbar:
                    for line in line_pbar:
                        line_count += 1
                        line = line.strip()
                        if not line: continue

                        try:
                            rec = json.loads(line)
                            
                            # 1. 提取发起者 (Source)
                            actor = rec.get("actor_id") or rec.get("user_id") or rec.get("uid") or rec.get("from_user") or rec.get("mid")
                            actor = self._normalize_user_id(actor)
                            
                            if not actor: continue
                            
                            self.users_found.add(actor)

                            # 2. 提取目标 (Targets)
                            mentions = self._extract_mentions(rec)
                            
                            for tgt, strength in mentions:
                                if not tgt or tgt == actor: continue # 忽略自环
                                
                                self.users_found.add(tgt)
                                
                                # 3. 累加权重
                                weight = self.weights.get(strength, 1)
                                self.interaction_counts[(actor, tgt)] += float(weight)
                                valid_interactions += 1
                            
                            # 实时更新显示
                            if line_count % 500 == 0:
                                line_pbar.set_postfix({
                                    "Edges": len(self.interaction_counts),
                                    "Nodes": len(self.users_found)
                                })

                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            # 偶尔的数据错误不应中断流程
                            logging.debug(f"Line parse error in {filename}: {e}")
                            continue
            
            logging.info(f"Done {filename}: {line_count} lines, {valid_interactions} interactions found.")
            
        except Exception as e:
            logging.error(f"Failed to open {filename}: {e}")

    def process_all_files(self):
        """主处理循环"""
        txt_files = sorted(list(self.input_directory.glob('*.txt')))
        
        if not txt_files:
            logging.warning(f"No .txt files found in {self.input_directory}")
            print(f"⚠️ 在 {self.input_directory} 中未找到 .txt 文件")
            return
        
        print(f"🚀 开始构建关系矩阵，共 {len(txt_files)} 个文件...")
        logging.info(f"Found {len(txt_files)} files")

        # 外层进度条：文件级别
        with tqdm(total=len(txt_files), unit="file", desc="总体进度") as pbar:
            for filepath in txt_files:
                self.process_single_file(filepath, pbar)
                pbar.update(1)
        
        print(f"\n✨ 处理完成！")
        print(f"📊 统计信息: 节点数 {len(self.users_found)}, 边数 {len(self.interaction_counts)}")
        logging.info(f"Total Nodes: {len(self.users_found)}, Total Edges: {len(self.interaction_counts)}")

    def save_results(self):
        """保存结果到 CSV"""
        if not self.interaction_counts:
            print("⚠️ 没有提取到任何交互数据，跳过保存。")
            return

        print("💾 正在生成 DataFrame 并保存...")
        
        # 1. 转换为 DataFrame
        rows = [{"source": src, "target": tgt, "weight": w} 
                for (src, tgt), w in self.interaction_counts.items()]
        
        edges_df = pd.DataFrame(rows)
        
        # 2. 按 Source 和 权重 排序，方便查看
        edges_df = edges_df.sort_values(["source", "weight"], ascending=[True, False])
        
        # 3. 计算总分 (Total Influence Score / Out-Degree Weight)
        total_scores = edges_df.groupby("source")["weight"].sum().sort_values(ascending=False)
        
        # 4. 确保目录存在
        OUTPUT_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
        
        # 5. 写入文件
        edges_df.to_csv(OUTPUT_MATRIX_CSV, index=False, encoding='utf-8-sig')
        total_scores.to_csv(OUTPUT_SCORES_CSV, header=["total_out_score"], encoding='utf-8-sig')
        
        print(f"✅ 边表已保存: {OUTPUT_MATRIX_CSV}")
        print(f"✅ 总分表已保存: {OUTPUT_SCORES_CSV}")
        logging.info(f"Saved edges to {OUTPUT_MATRIX_CSV} and scores to {OUTPUT_SCORES_CSV}")

def main():
    print(f"📂 输入目录: {INPUT_DIRECTORY}")
    
    pipeline = UserRelationshipPipeline()
    
    # 1. 处理所有文件
    pipeline.process_all_files()
    
    # 2. 保存结果
    pipeline.save_results()

if __name__ == "__main__":
    main()