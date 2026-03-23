import os
import re
import logging
import spacy
import pandas as pd

class NLPProcessor:
    def __init__(self, stopwords_path):
        self.stopwords_path = stopwords_path
        self.nlp = None
        self.custom_stop_words = set()
        self.ignored_ent_labels = {'DATE', 'TIME', 'PERCENT', 'CARDINAL', 'QUANTITY', 'MONEY', 'ORDINAL', 'LAW'}
        
        self._init_spacy()

    def _init_spacy(self):
        """初始化 Spacy 模型和停用词"""
        try:
            logging.info("[NLPProcessor] Loading Spacy model (zh_core_web_sm)...")
            self.nlp = spacy.load("zh_core_web_sm")
            self._load_stopwords()
        except Exception as e:
            logging.error(f"[NLPProcessor] Spacy Init Error: {e}")
            # 可以在这里抛出异常或者设置一个标记，视系统容错策略而定

    def _load_stopwords(self):
        """加载停用词文件到内存"""
        if os.path.exists(self.stopwords_path):
            try:
                with open(self.stopwords_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            self.custom_stop_words.add(word)
                
                logging.info(f"✅ Loaded {len(self.custom_stop_words)} custom stopwords from {self.stopwords_path}")
                
                # 将自定义停用词注册到 Spacy
                if self.nlp:
                    for w in self.custom_stop_words:
                        self.nlp.vocab[w].is_stop = True
            except Exception as e:
                logging.warning(f"⚠️ Failed to load stopwords from {self.stopwords_path}: {e}")
        else:
            logging.warning(f"⚠️ Stopwords file not found at {self.stopwords_path}. Using default Spacy list only.")

    def extract_candidates(self, text_list):
        """
        [核心功能] 从文本列表中提取句法候选词 (NER + PROPN + NOUN)
        """
        if not self.nlp:
            return ""

        # 1. 拼接与基础处理
        clean_texts = [str(t).strip() for t in text_list if pd.notna(t) and str(t).strip()]
        full_text = " ".join(clean_texts)
        
        # 基础清洗正则
        full_text = re.sub(r'<[^>]+>', '', full_text)
        full_text = re.sub(r'http[s]?://\S+', '', full_text)
        full_text = re.sub(r'\[.*?\]', '', full_text)
        full_text = re.sub(r'\b[a-zA-Z0-9]{8,}\b', '', full_text)
            
        if not full_text.strip(): return ""

        # 2. Spacy 推理
        # 增加长度限制防止内存溢出，Spacy对超长文本处理较慢
        if len(full_text) > 100000: 
            full_text = full_text[:100000]
            
        doc = self.nlp(full_text)
        
        candidates_pool = {"NER": {}, "PROPN": {}, "NOUN": {}}
        seen_texts = set()
        VALID_DEP_TAGS = {'nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr'}

        # --- 策略 A: 命名实体 (NER) ---
        for ent in doc.ents:
            text = ent.text.strip()
            if (ent.label_ not in self.ignored_ent_labels and len(text) > 1 and text not in self.custom_stop_words):
                candidates_pool["NER"][text] = ent.label_
                seen_texts.add(text)

        # --- 策略 B: 句法筛选 (PROPN vs NOUN) ---
        for token in doc:
            text = token.text.strip()
            if (len(text) > 1 and text not in seen_texts and not token.is_stop and 
                not token.is_punct and not text.isdigit() and not re.match(r'^[^\w]', text) and 
                text not in self.custom_stop_words):
                
                if token.pos_ == 'PROPN':
                    candidates_pool["PROPN"][text] = "PROPN"
                    seen_texts.add(text)
                elif token.pos_ == 'NOUN' and token.dep_ in VALID_DEP_TAGS:
                    candidates_pool["NOUN"][text] = "NOUN"
                    seen_texts.add(text)

        # 3. 组装输出
        final_list = []
        for text, label in candidates_pool["NER"].items(): 
            final_list.append(f"{text} ({label})")
        for text, label in candidates_pool["PROPN"].items(): 
            final_list.append(f"{text} ({label})")
        
        # 名词按长度降序保留前20
        noun_list = [f"{text} (NOUN)" for text, _ in candidates_pool["NOUN"].items()]
        noun_list.sort(key=lambda x: len(x), reverse=True)
        final_list.extend(noun_list[:20]) 

        return ", ".join(final_list[:100])