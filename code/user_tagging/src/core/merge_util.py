import json

# ==========================================
# 1. 统一的基础配置与工具 (Shared Utils)
# ==========================================

# 置信度映射表
CONFIDENCE_MAP = {"Low": 1, "Medium-Low": 2, "Medium": 3, "Medium-High": 4, "High": 5}
SCORE_MAP = {v: k for k, v in CONFIDENCE_MAP.items()}

# 全局统一的无效值集合 (归一化为小写)
NA_VALUES = {'na', 'n/a', 'unknown', 'none', 'null', 'nan', 'nil', '未知', '无', '其他', '不详', '待定'}

def _get_score(conf_str):
    """获取置信度分数 (1-5)"""
    return CONFIDENCE_MAP.get(str(conf_str), 3)

def _is_na(value):
    """
    [核心判断] 判断值是否为无意义占位符
    适用于 Interest 的 L2/L3 名称，以及 Profile 的 value 字段
    """
    if not value:
        return True
    return str(value).strip().lower() in NA_VALUES

def _is_valid(value):
    """_is_na 的反义词，方便逻辑阅读"""
    return not _is_na(value)

# ==========================================
# 2. Interest Tree 归并逻辑 (树状/列表结构)
# ==========================================

def merge_interest_trees(old_tree, new_tree):
    """
    合并兴趣树：L1/L2/L3 层级合并 + NA 驱逐
    """
    if not old_tree:
        return _clean_tree_impurities(new_tree)
    if not new_tree:
        return old_tree

    merged_map = {node["L1"]: node for node in old_tree}

    for new_l1 in new_tree:
        l1_name = new_l1["L1"]
        # L1 不太可能是 NA，略过检查

        if l1_name not in merged_map:
            # Case 1: 全新 L1，清洗后追加
            cleaned_node = _clean_tree_node(new_l1)
            if cleaned_node:
                merged_map[l1_name] = cleaned_node
        else:
            # Case 2: 旧 L1 存在
            curr_l1 = merged_map[l1_name]
            
            # Update L1 Confidence
            if _get_score(new_l1.get("confidence")) > _get_score(curr_l1.get("confidence")):
                curr_l1["confidence"] = new_l1.get("confidence")
                curr_l1["reasoning"] = new_l1.get("reasoning", curr_l1.get("reasoning"))

            # --- [L2 驱逐逻辑] ---
            # 如果新树带来了有效 L2，删除旧树里的 NA L2
            if any(_is_valid(n.get("L2")) for n in new_l1.get("sub_domains", [])):
                curr_l1["sub_domains"] = [n for n in curr_l1.get("sub_domains", []) if _is_valid(n.get("L2"))]

            # --- [L2 合并] ---
            l2_map = {n["L2"]: n for n in curr_l1.get("sub_domains", [])}
            
            for new_l2 in new_l1.get("sub_domains", []):
                l2_name = new_l2["L2"]
                
                # Rule 1: 入站拦截 NA
                if _is_na(l2_name): continue

                if l2_name not in l2_map:
                    # 全新 L2
                    cleaned_l2 = _clean_l2_node(new_l2)
                    curr_l1.setdefault("sub_domains", []).append(cleaned_l2)
                else:
                    curr_l2 = l2_map[l2_name]
                    # Update L2 Confidence
                    if _get_score(new_l2.get("confidence")) > _get_score(curr_l2.get("confidence")):
                        curr_l2["confidence"] = new_l2.get("confidence")
                        if new_l2.get("meta_type"): curr_l2["meta_type"] = new_l2.get("meta_type")

                    # --- [L3 驱逐逻辑] ---
                    # 如果新树带来了有效 L3，删除旧树里的 NA L3
                    if any(_is_valid(e.get("L3")) for e in new_l2.get("entities", [])):
                        curr_l2["entities"] = [e for e in curr_l2.get("entities", []) if _is_valid(e.get("L3"))]

                    # --- [L3 合并] ---
                    l3_map = {e["L3"]: e for e in curr_l2.get("entities", [])}
                    for new_l3 in new_l2.get("entities", []):
                        l3_name = new_l3["L3"]
                        
                        # Rule 1: 入站拦截 NA
                        if _is_na(l3_name): continue

                        if l3_name not in l3_map:
                            curr_l2.setdefault("entities", []).append(new_l3)
                        else:
                            # Rule 3: 强者生存
                            existing_l3 = l3_map[l3_name]
                            if _get_score(new_l3.get("confidence")) > _get_score(existing_l3.get("confidence")):
                                existing_l3["confidence"] = new_l3.get("confidence")

    return list(merged_map.values())

# --- Interest Tree 清洗辅助 ---
def _clean_tree_impurities(tree):
    cleaned = []
    for l1 in tree:
        node = _clean_tree_node(l1)
        if node: cleaned.append(node)
    return cleaned

def _clean_tree_node(l1_node):
    valid_subs = []
    for l2 in l1_node.get("sub_domains", []):
        if _is_valid(l2.get("L2")):
            valid_subs.append(_clean_l2_node(l2))
    l1_node["sub_domains"] = valid_subs
    return l1_node

def _clean_l2_node(l2_node):
    valid_ents = [e for e in l2_node.get("entities", []) if _is_valid(e.get("L3"))]
    l2_node["entities"] = valid_ents
    return l2_node

# ==========================================
# 3. Profile 归并逻辑 (字典/扁平结构)
# ==========================================

def merge_profiles(old_profile, new_profile):
    """
    合并用户画像 (通用版：支持 Demographic 和 Firmographic)
    """
    if not old_profile: return new_profile
    if not new_profile: return old_profile

   
    DEMO_KEYS = ["gender", "age", "education_level", "residence_type", "relationship_status", "job_type", "personal_income", "family_income"]
    FIRM_KEYS = ["industry", "sub_industry", "scale", "business_type", "service_scope", "location"]
    
    # 合并所有可能的 Key
    all_target_keys = set(DEMO_KEYS) | set(FIRM_KEYS)
    
    merged = old_profile.copy()

    for key in all_target_keys:
        old_node = old_profile.get(key, {})
        new_node = new_profile.get(key, {})
        
        old_val = old_node.get("value")
        new_val = new_node.get("value")

        # Rule 1: 入站拦截 NA
        if _is_na(new_val): continue

        # Rule 2: 存量驱逐 (Old is NA)
        if _is_na(old_val):
            merged[key] = new_node
            continue

        # Rule 3: 强者生存 (Confidence Bidding)
        old_score = _get_score(old_node.get("confidence"))
        new_score = _get_score(new_node.get("confidence"))
        
        if new_score >= old_score:
            merged[key] = new_node

    return merged