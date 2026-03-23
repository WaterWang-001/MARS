import os
import json
import threading
import numpy as np
from collections import defaultdict, Counter

class DatasetManager:
    def __init__(self, output_dir, sft_quality_config=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # --- 文件路径 ---
        self.sft_file = os.path.join(output_dir, "sft_training.jsonl")
        self.stats_file = os.path.join(output_dir, "cold_start_stats.json")

        self._lock = threading.Lock()

        # ================= 配置参数（可由 YAML 覆盖） =================
        sqc = sft_quality_config or {}

        self.MIN_TOTAL_CONTENT_LEN = 50

        # SFT 置信度比例控制 (High:Mid:NA = 7:2:1)
        conf_ratio = sqc.get("confidence_ratio", {})
        self.SFT_CONF_TARGET = {
            "High": conf_ratio.get("high", 0.7),
            "Mid": conf_ratio.get("mid", 0.2),
            "NA": conf_ratio.get("na", 0.1),
        }
        self.SFT_CONF_CAP = {"High": 1.0, "Mid": 0.25, "NA": 0.15}

        self.SFT_FOCUS_THRESHOLD = sqc.get("focus_threshold", 0.4)
        self.SFT_GROUNDING_THRESHOLD = sqc.get("grounding_threshold", 0.2)

        self.L1_BALANCE_START_THRESHOLD = 200
        self.L1_OVERFLOW_RATIO = sqc.get("l1_overflow_ratio", 1.5)

        self.CONFIDENCE_SCORE_MAP = {
            "High": 5, "Medium-High": 4, "Medium": 3, "Medium-Low": 2, "Low": 1
        }

        self.stats = self._load_json(self.stats_file, default={
            "sft": {"confidence": {"High": 0, "Mid": 0, "NA": 0}, "l1": {}, "total": 0},
        })

    def _load_json(self, path, default):
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return default

    def _save_stats(self):
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f)

    # ======================================================
    #  核心逻辑：SFT vs Normal 分类
    # ======================================================

    def classify_and_save_user(self, user_id, user_data, posts_text, interest_tree, current_role):
        """
        根据 LLM 生成结果，决定用户归属为 SFT 或 Normal 并落盘。
        Return: (save_status, final_role)
        """
        if current_role == "sft":
            return "Role Inherited", current_role

        with self._lock:
            # 0. 基础内容检查
            if not self._check_content_richness(posts_text):
                return "Content too short", "normal"

            primary_l1 = self._extract_primary_l1(interest_tree)

            # 1. 结构完整性 (L1->L2->L3)
            if not self._check_tree_structure(interest_tree):
                return "SFT Rejected (Structure Incomplete)", "normal"

            # 2. 语义质量检查 (Focus, InfoGain, Grounding)
            is_high_quality, quality_reason = self._evaluate_sft_quality(interest_tree, posts_text)
            if not is_high_quality:
                return f"SFT Rejected ({quality_reason})", "normal"

            # 3. 置信度配额 (7:2:1)
            confidence_bin = self._extract_confidence(interest_tree)
            if not self._check_sft_confidence_quota(confidence_bin):
                return f"SFT Rejected (Conf Quota: {confidence_bin})", "normal"

            # 4. L1 领域均衡
            if not self._check_l1_quota(primary_l1):
                return f"SFT Rejected (L1 Quota: {primary_l1})", "normal"

            # 满足所有条件 -> 入库
            self._write_jsonl(self.sft_file, {
                "user_id": user_id,
                "input": posts_text,
                "output": interest_tree,
                "confidence": confidence_bin,
                "l1": primary_l1
            })
            self._update_stats(confidence=confidence_bin, l1=primary_l1)

            return f"Saved to SFT ({confidence_bin})", "sft"

        return "Normal User", "normal"

    # ======================================================
    #  辅助判定逻辑
    # ======================================================
    def _evaluate_sft_quality(self, interest_tree, posts_text):
        """
        评估生成的兴趣树是否具有作为 SFT 训练数据的价值。
        检查维度：
        1. 聚焦度 (Focus): 是否有明显的主导兴趣，拒绝大杂烩。
        2. 信息增益 (Info Gain): L3 是否比 L2 更具体。
        3. 证据支撑 (Grounding): 标签是否在原文中有一定体现 (防严重幻觉)。
        """
        data = interest_tree.get('interest_tree', []) if isinstance(interest_tree, dict) else interest_tree
        if not isinstance(data, list): return False, "Invalid Format"

        all_l3_tags = []
        l1_counts = Counter()
        total_leaf_nodes = 0

        for l1 in data:
            l1_name = l1.get('interest_L1', 'Unknown')
            if 'children' in l1:
                for l2 in l1['children']:
                    l2_name = l2.get('interest_L2', '')

                    if 'children' in l2:
                        for l3 in l2['children']:
                            l3_name = l3.get('interest_L3', '')
                            if l3_name == l2_name and len(l3_name) < 4:
                                return False, "Low Info Gain (L2==L3)"

                            all_l3_tags.append(l3_name)
                            l1_counts[l1_name] += 1
                            total_leaf_nodes += 1

        if total_leaf_nodes == 0:
            return False, "Empty Leaves"

        # Focus Score
        if l1_counts:
            most_common_l1, count = l1_counts.most_common(1)[0]
            dominance_ratio = count / total_leaf_nodes
            if dominance_ratio < self.SFT_FOCUS_THRESHOLD:
                return False, f"Low Focus (Max L1 Ratio: {dominance_ratio:.2f})"

        # Grounding Score
        hit_count = 0
        for tag in all_l3_tags:
            clean_tag = tag.split('(')[0].strip()
            if clean_tag in posts_text:
                hit_count += 1

        grounding_score = hit_count / len(all_l3_tags) if all_l3_tags else 0
        if grounding_score < self.SFT_GROUNDING_THRESHOLD:
            return False, f"Low Grounding (Score: {grounding_score:.2f})"

        return True, "Pass"

    def _check_tree_structure(self, interest_tree):
        """检查 SFT 结构完整性 (L1->L2->L3)"""
        if not interest_tree: return False

        data = interest_tree.get('interest_tree', []) if isinstance(interest_tree, dict) else interest_tree
        if not isinstance(data, list): return False

        for l1 in data:
            if l1.get('children'):
                for l2 in l1['children']:
                    if l2.get('children'):
                        return True
        return False

    def _check_sft_confidence_quota(self, confidence_bin):
        """SFT 置信度在线平衡 (7:2:1)"""
        stats = self.stats["sft"]
        total = stats["total"]

        if total < self.L1_BALANCE_START_THRESHOLD:
            return True

        curr_ratio = stats["confidence"].get(confidence_bin, 0) / total
        limit = self.SFT_CONF_CAP.get(confidence_bin, 1.0)

        return curr_ratio < limit

    def _check_l1_quota(self, l1_label):
        """L1 领域均匀分布采样"""
        if not l1_label or l1_label == "Unknown":
            return False

        stats = self.stats["sft"]
        total = stats["total"]
        l1_counts = stats["l1"]

        if total < self.L1_BALANCE_START_THRESHOLD:
            return True

        current_count = l1_counts.get(l1_label, 0)
        unique_l1_count = len(l1_counts) if l1_counts else 1
        avg_count = total / unique_l1_count

        if current_count > (avg_count * self.L1_OVERFLOW_RATIO):
            return False

        return True

    def _extract_primary_l1(self, interest_tree):
        """从树中提取主要的 L1 标签"""
        data = interest_tree.get('interest_tree', []) if isinstance(interest_tree, dict) else interest_tree
        if not data or not isinstance(data, list):
            return "Unknown"

        best_l1 = "Unknown"
        max_children = -1

        for node in data:
            l1_name = node.get('interest_L1')
            children_count = len(node.get('children', []))
            if children_count > max_children:
                max_children = children_count
                best_l1 = l1_name

        return best_l1

    def _update_stats(self, confidence=None, l1=None):
        """更新统计数据并落盘"""
        self.stats["sft"]["total"] += 1

        if confidence:
            self.stats["sft"]["confidence"][confidence] = self.stats["sft"]["confidence"].get(confidence, 0) + 1

        if l1:
            self.stats["sft"]["l1"][l1] = self.stats["sft"]["l1"].get(l1, 0) + 1

        self._save_stats()

    # ======================================================
    #  基础工具
    # ======================================================

    def _write_jsonl(self, path, record):
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _check_content_richness(self, posts_text):
        if not posts_text: return False
        clean = posts_text.replace("Content:", "").replace("Quote:", "").replace("\n", "")
        return len(clean) >= self.MIN_TOTAL_CONTENT_LEN

    def _extract_confidence(self, interest_tree):
        scores = []
        def traverse(nodes):
            if isinstance(nodes, list):
                for node in nodes:
                    if isinstance(node, dict):
                        conf = node.get('confidence', node.get('Confidence', 'Low'))
                        score = self.CONFIDENCE_SCORE_MAP.get(conf, 1)
                        scores.append(score)
                        if 'children' in node: traverse(node['children'])
            elif isinstance(nodes, dict): traverse([nodes])

        data = interest_tree.get('interest_tree', []) if isinstance(interest_tree, dict) else interest_tree
        traverse(data)

        if not scores: return "NA"
        avg = sum(scores) / len(scores)

        if avg >= 3.5: return "High"
        elif avg >= 1.5: return "Mid"
        else: return "NA"
