import logging
import pandas as pd
import spacy
import re
import time
import os
from core.merge_util import merge_interest_trees, merge_profiles
import concurrent.futures
import json
class TaggingService:
    def __init__(self, db_client, prompt_manager, api_client, stopwords_path, state_manager,config):
        self.db_client = db_client
        self.prompt_manager = prompt_manager
        self.api_client = api_client
        self.state_manager = state_manager
        self.stopwords_path = stopwords_path
        self.config=config
        self.quality_config = self.config.get('quality', {
            "repetition_threshold": 0.3, 
            "min_avg_time_gap": 60
        })
        self.trigger_threshold = self.config.get('trigger_threshold', 3)
        self.max_posts_per_call = self.config.get('max_posts_per_call', 20)

        try:
            print("[TaggingService] Loading Spacy model (zh_core_web_sm)...")
            self.nlp = spacy.load("zh_core_web_sm")
            self.ignored_ent_labels = {'DATE', 'TIME', 'PERCENT', 'CARDINAL', 'QUANTITY', 'MONEY', 'ORDINAL', 'LAW'}
            self.custom_stop_words = set()
            self._load_stopwords(self.stopwords_path)
        except Exception as e:
            print(f"Spacy Init Error: {e}")

    def _load_stopwords(self, path):
        """加载停用词文件到内存"""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            self.custom_stop_words.add(word)
                print(f"✅ Loaded {len(self.custom_stop_words)} custom stopwords from {path}")
                for w in self.custom_stop_words:
                    self.nlp.vocab[w].is_stop = True
            except Exception as e:
                print(f"⚠️ Failed to load stopwords from {path}: {e}")
        else:
            print(f"⚠️ Stopwords file not found at {path}. Using default Spacy list only.")

    def _extract_syntax_candidates(self, text_list):
        # 1. 拼接与基础处理 (保持之前的优化：先截断后清洗)
        clean_texts = [str(t).strip() for t in text_list if pd.notna(t) and str(t).strip()]
        full_text = " ".join(clean_texts)

        full_text = re.sub(r'<[^>]+>', '', full_text)
        full_text = re.sub(r'http[s]?://\S+', '', full_text)
        full_text = re.sub(r'\[.*?\]', '', full_text)
        full_text = re.sub(r'\b[a-zA-Z0-9]{8,}\b', '', full_text)
            
        if not full_text.strip():
            return ""

        # 2. Spacy 推理
        doc = self.nlp(full_text)
        
        candidates_pool = {
            "NER": {},
            "PROPN": {},
            "NOUN": {}
        }
        
        seen_texts = set()
        VALID_DEP_TAGS = {'nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr'}

        # --- 策略 A: 命名实体 (NER) ---
        for ent in doc.ents:
            text = ent.text.strip()
            label = ent.label_
            
            if (label not in self.ignored_ent_labels and 
                len(text) > 1 and 
                text not in self.custom_stop_words):
                
                candidates_pool["NER"][text] = label
                seen_texts.add(text)

        # --- 策略 B: 句法筛选 (PROPN vs NOUN) ---
        for token in doc:
            text = token.text.strip()
            
            if (len(text) > 1 and 
                text not in seen_texts and 
                not token.is_stop and 
                not token.is_punct and 
                not text.isdigit() and
                not re.match(r'^[^\w]', text) and 
                text not in self.custom_stop_words):

                # 1. 专有名词 PROPN: 只要出现就保留 (通常是品牌、特定称呼)
                if token.pos_ == 'PROPN':
                    candidates_pool["PROPN"][text] = "PROPN"
                    seen_texts.add(text)
                
                # 2. 普通名词 NOUN: 必须通过句法检查 (无黑名单逻辑)
                elif token.pos_ == 'NOUN':
                    # 【核心修改】只保留充当核心成分的名词
                    # 过滤掉 tmod(时间), clf(量词), advmod(修饰) 等边缘成分
                    if token.dep_ in VALID_DEP_TAGS:
                        candidates_pool["NOUN"][text] = "NOUN"
                        seen_texts.add(text)

        # 4. 组装输出 (优先级: NER > PROPN > NOUN)
        final_list = []
        
        # NER 全部保留
        for text, label in candidates_pool["NER"].items():
            final_list.append(f"{text} ({label})")
            
        # PROPN 全部保留
        for text, label in candidates_pool["PROPN"].items():
            final_list.append(f"{text} ({label})")
            
        # NOUN 限制数量 (例如最多 20 个，且由 Spacy 认为重要的词构成)
        # 按长度排序，优先保留长词 (通常信息量更大)
        noun_list = [f"{text} (NOUN)" for text, label in candidates_pool["NOUN"].items()]
        noun_list.sort(key=lambda x: len(x), reverse=True)
        final_list.extend(noun_list[:20]) 

        return ", ".join(final_list[:100])

    def _normalize_gender(self, val):
        if pd.isna(val): return "NA"
        s = str(val).lower()
        if s in ['f', 'female', '女']: return "Female"
        if s in ['m', 'male', '男']: return "Male"
        return "NA"

    def _clean_text_for_stat(self, text: str) -> str:
        if not text: return ""
        text = str(text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        return text.strip()

    def _format_posts_for_llm(self, posts: list) -> str:
        """
        [新增] 将 Buffer 中的帖子列表格式化为结构清晰的文本块
        优势：
        1. 增加序号 [1], [2]，方便 LLM 在 Reasoning 中引用（如"根据第1条..."）。
        2. 显式区分 Content 和 Quote，防止 LLM 把引用内容误认为是用户观点。
        """
        if not posts:
            return ""

        formatted_lines = []
        for i, p in enumerate(posts, 1):
            # 安全获取并去除首尾空格，处理 None/NaN 情况
            content = str(p.get('content', '') or '').strip()
            quote = str(p.get('quote_content', '') or '').strip()
            
            # 如果是 'nan' 字符串（pandas 读取常见问题），也视为空
            if content.lower() == 'nan': content = ""
            if quote.lower() == 'nan': quote = ""

            # 跳过完全为空的脏数据
            if not content and not quote:
                continue

            # 构造格式: "[i] Content: {内容} | Quote: {引用}"
            # 使用列表 join 避免字符串拼接的额外空格问题
            parts = [f"[{i}] Content: {content}"]
            
            if quote:
                parts.append(f"| Quote: {quote}")


            formatted_lines.append(" ".join(parts))

        return "\n".join(formatted_lines)
    
    def _validate_posts_quality(self, posts_rows: list) -> tuple:
        """校验帖子质量：反垃圾、反机器人"""
        if not posts_rows: return True, "VALID"
        valid_contents = []
        timestamps = []
        rep_threshold = self.quality_config.get('repetition_threshold', 0.3)
        gap_threshold = self.quality_config.get('min_avg_time_gap', 60)
        for row in posts_rows:
            clean_c = self._clean_text_for_stat(row.get('content', ""))
            if clean_c: valid_contents.append(clean_c)
            if row.get('created_at'):
                try: timestamps.append(pd.to_datetime(row['created_at']))
                except: pass

        if len(valid_contents) >= 3:
            unique = len(set(valid_contents))
            if (1.0 - unique/len(valid_contents)) > rep_threshold:
                return False, "SPAM: High Repetition"

        if len(timestamps) > 4:
            timestamps.sort()
            diffs = [(timestamps[i+1]-timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            valid_diffs = [d for d in diffs if d < 3600]
            if len(valid_diffs) > 3 and (sum(valid_diffs)/len(valid_diffs)) < gap_threshold:
                return False, "BOT: Abnormal Frequency"
        
        return True, "VALID"
    
    def _prepare_context(self, user_data: dict):
        """
        [核心辅助] 清洗用户元数据，判断机构类型
        注意：这里不获取帖子，只处理 Profile 字段
        """
        user_id = str(user_data.get('user_id', '')).strip()
        username = user_data.get('username', 'Unknown')
        
        bio = user_data.get('bio', '')
        if pd.isna(bio) or str(bio).lower() == 'nan':
            bio = "NA"
        else:
            bio = str(bio).strip()

        # 认证信息清洗
        verified = user_data.get('verified', False)
        verified_type = str(user_data.get('verified_type', ''))
        verified_reason = str(user_data.get('verified_reason', ''))
        verified_info = f"Verified: {verified} (Type: {verified_type}, Info: {verified_reason})"

        # [关键] 机构判断逻辑 (根据你提供的规则)
        # 1, 2, 3, 7 通常对应政府、企业、媒体等蓝V
        is_org = str(verified_type) in ['1', '2', '3', '7']
        mapped_type_name = "机构" if is_org else "个人"

        gender = self._normalize_gender(user_data.get('gender'))
        
        # 注册时间清洗
        reg_raw = str(user_data.get('registration_time', ''))
        reg_time = reg_raw.split('T')[0] if 'T' in reg_raw else reg_raw
        if not reg_time or reg_time.lower() == 'nan': reg_time = "NA"

        # 地点清洗
        prov = str(user_data.get('province', ''))
        city = str(user_data.get('city', ''))
        if prov.lower() == 'nan': prov = ''
        if city.lower() == 'nan': city = ''
        location = f"{prov} {city}".strip() or "NA"

        return {
            "user_id": user_id,
            "username": username,
            "bio": bio,
            "verified_info": verified_info,
            "verified_type": verified_type,
            "mapped_type_name": mapped_type_name,
            "is_org": is_org, # Boolean Flag
            "gender": gender,
            "reg_time": reg_time,
            "location": location
        }

    
    def process_user_incremental(self, user_data, batch_date):
        """
        核心流程：增量 + 机构分支 + 并行处理
        """
        # 1. 预处理上下文 (包含机构判断)
        ctx = self._prepare_context(user_data)
        user_id = ctx['user_id']
        
        # 2. 获取状态
        state = self.state_manager.get_user_state(user_id)
        last_cursor_time = state["last_cursor_time"]
        old_tree = state["interest_tree"]
        current_profile = state["profile"]

        # 3. DB 增量获取
        new_source_posts = self.db_client.get_user_posts_after(user_id, last_cursor_time)
        
        # 4. 质量校验
        is_valid, reason = self._validate_posts_quality(new_source_posts)
        if not is_valid:
            self.state_manager.update_user_state(user_id, batch_date, old_tree, current_profile)
            return {"status": "skipped_quality", "reason": reason, "user_id": user_id}

        # 5. 入 Buffer
        if new_source_posts:
            self.state_manager.add_to_buffer(user_id, new_source_posts)
            self.state_manager.update_user_state(user_id, batch_date, old_tree, current_profile)

        # 6. 取 Buffer & 阈值检查
        all_buffered = self.state_manager.get_buffered_posts(user_id)
        if len(all_buffered) < self.trigger_threshold:
            return {"status": "buffered", "count": len(all_buffered), "user_id": user_id}
        if len(all_buffered) > self.max_posts_per_call:
            # all_buffered 是按时间正序排列的，[-N:] 取最后(最近)的 N 条
            all_buffered = all_buffered[-self.max_posts_per_call:]
        # --- 触发 LLM 任务 ---

        posts = self._format_posts_for_llm(all_buffered)
        raw_contents = [str(p.get('content', '') or '') for p in all_buffered]
        # Task A: Interest Tagging (通用)
        def run_interest_task():
            candidates = self._extract_syntax_candidates([ctx['bio']] + raw_contents)
            prompt = self.prompt_manager.get_interest_prompt(
                username=ctx['username'], 
                bio=ctx['bio'], 
                posts=posts, 
                candidate_entities=candidates
            )
            return self.api_client.call_api(prompt)

        # Task B: Profile Update (根据 is_org 分支)
        def run_profile_task():
            if ctx['is_org']:
                return self.api_client.call_api(
                    self.prompt_manager.get_firmographic_prompt(
                        username=ctx['username'],
                        bio=ctx['bio'],
                        posts=posts,
                        verified_info=ctx['verified_info'],
                        mapped_type_name=ctx['mapped_type_name'],
                        verified_type=ctx['verified_type']
                    )
                )
             
            else:
                return self.api_client.call_api(
                    self.prompt_manager.get_demographic_prompt(
                        username=ctx['username'],
                        bio=ctx['bio'],
                        posts=posts,
                        gender_reported=ctx['gender'],
                        reg_time=ctx['reg_time'],
                        location_reported=ctx['location'],
                        verified_info=ctx['verified_info']
                    )
                )

        # --- 并行执行 ---
        new_partial_tree = []
        new_partial_profile = {}
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_interest = executor.submit(run_interest_task)
                future_profile = executor.submit(run_profile_task)
                
                if future_interest.result():
                    res = future_interest.result()
                    new_partial_tree = res.get("interest_tree", [])
                
                if future_profile.result():
                    res = future_profile.result()
                    # 兼容不同 Prompt 的返回 Key
                    if "firmographic_profile" in res:
                        new_partial_profile = res["firmographic_profile"]
                    elif "demographic_profile" in res:
                        new_partial_profile = res["demographic_profile"]
                    else:
                        new_partial_profile = res

        except Exception as e:
            logging.error(f"[User {user_id}] Parallel Task Error: {e}")
            return None

        # --- 归并逻辑 ---
        # Interest Merge
        merged_tree = merge_interest_trees(old_tree, new_partial_tree)
        
        # Profile Merge (通用函数处理两套字段)
        merged_profile = merge_profiles(current_profile, new_partial_profile)
        
        # 将元数据回写到 Profile 方便下游使用
        merged_profile["user_type"] = ctx['mapped_type_name'] # "个人" 或 "机构"
        merged_profile["is_org"] = ctx['is_org']
        
        # 7. 清理 & 保存
        self.state_manager.clear_user_buffer(user_id)
        self.state_manager.update_user_state(user_id, batch_date, merged_tree, merged_profile)
        
        return {
            "status": "updated",
            "user_id": user_id,
            "user_type": ctx['mapped_type_name'],
            "interest_tree": merged_tree,
            "profile": merged_profile
        }