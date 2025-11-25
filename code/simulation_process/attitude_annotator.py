import asyncio
import sqlite3
import re
import json
import logging
import time
import math
from tqdm.asyncio import tqdm_asyncio
from typing import Dict, List, Optional, Tuple,Union
from openai import AsyncOpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
# ... logger 定义保持不变 ...
logger = logging.getLogger("AttitudeAnnotation")


# =====================================================================
# 1. 基类 (已添加超时和健壮性)
# =====================================================================

class BaseAttitudeAnnotator:
    """
    (基类) 态度标注器的共享逻辑。
    支持自定义指标说明 + 实时流式打印 + 错误重试。
    """
    
    def __init__(
        self, 
        api_key: str, 
        base_url: Optional[str], 
        # 【修改 1】这里改为接收字典(推荐)或列表(兼容)
        # 格式: {"attitude_trust": "Description of trust...", "attitude_risk": "..."}
        attitude_config: Union[Dict[str, str], List[str]], 
        concurrency_limit: int = 100, 
        log_interval_posts: int = 100,
        api_timeout_seconds: int = 120
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.api_timeout_seconds = api_timeout_seconds
        
        self.client = AsyncOpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
            timeout=self.api_timeout_seconds
        )
        
        # 【修改 2】解析配置，分离出 列名列表 和 描述映射
        if isinstance(attitude_config, list):
            # 如果用户只传了列表，默认描述为空
            self.attitude_columns = attitude_config
            self.attitude_descriptions = {col: "Analyze the sentiment intensity for this aspect." for col in attitude_config}
        elif isinstance(attitude_config, dict):
            # 如果用户传了字典，提取列名和描述
            self.attitude_columns = list(attitude_config.keys())
            self.attitude_descriptions = attitude_config
        else:
            raise ValueError("attitude_config must be a list of strings or a dict {name: description}")

        self.api_semaphore = asyncio.Semaphore(concurrency_limit)
        self.log_interval_posts = max(1, log_interval_posts) 
        
        self.control_chars_regex = re.compile(r'[\x00-\x1F\x7F-\x9F\u200B-\u200D\uFEFF]')
        self.logger = logger
        self.logger.info(
            f"{self.__class__.__name__} initialized.\n"
            f"   -> Columns: {self.attitude_columns}\n"
            f"   -> Concurrency: {concurrency_limit}"
        )

    def _clean_text(self, text) -> str:
        if text is None: return ""
        cleaned = str(text)
        cleaned = self.control_chars_regex.sub(' ', cleaned)
        return cleaned.strip()

    # 【修改 3】动态生成包含具体定义的 Prompt
    def _get_system_prompt(self) -> str:
        # 构建指标说明文本
        # 格式如: 
        # - attitude_trust: Measures how much the user trusts the information.
        # - attitude_risk: Measures the perceived danger.
        metric_instructions = []
        for col, desc in self.attitude_descriptions.items():
            metric_instructions.append(f"- **{col}**: {desc}")
        
        metrics_block = "\n".join(metric_instructions)
        json_keys_str = ", ".join([f'"{col}"' for col in self.attitude_columns])
        
        return f"""
        You are a content analysis expert. Analyze the user's post.
        
        The post will be provided in one or both of the following forms:
        -   `[User Comment]`: The comment written by the user.
        -   `[Forwarded Original Post]`: The post that the user forwarded or quoted.

        == Your Core Task ==
        Your task is to analyze the sentiment/attitude of the **`[User Comment]`**.
        1.  If `[User Comment]` exists, analyze it. `[Forwarded Original Post]` is only context.
        2.  If `[User Comment]` does not exist, analyze `[Forwarded Original Post]`.

        == Scoring Definitions ==
        Please analyze the text based on the following specific metrics and definitions:
        
        {metrics_block}

        == Output Format ==
        1.  Use a continuous score from -1.0 (Negative/Low) to 1.0 (Positive/High).
        2.  Return a SINGLE JSON object. Do NOT output markdown code blocks.
        3.  The JSON keys MUST be exactly: [{json_keys_str}].
        4.  Values must be floats (e.g., 0.5, -0.2, 0.0).
        """

    async def _get_attitude_scores_from_llm(self, content: str) -> Dict[str, float]:
        raise NotImplementedError("Subclass must implement.")

    def _setup_database_columns(self, conn: sqlite3.Connection, table_name: str):
        self.logger.info(f"Setting up columns for table '{table_name}'...")
        cur = conn.cursor()
        all_columns_to_add = self.attitude_columns + ['attitude_annotated']
        
        for col in all_columns_to_add:
            col_type = "REAL DEFAULT 0.0" if col != "attitude_annotated" else "INTEGER DEFAULT 0"
            try:
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass 
                
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_annotated ON {table_name} (attitude_annotated);")
        conn.commit()
        cur.close()

    async def _process_post(self, post_id: str, content: str, quote_content: str) -> Optional[Tuple[str, Optional[Dict[str, float]], Optional[str]]]:
        try:
            user_comment = self._clean_text(content)
            original_post = self._clean_text(quote_content)
            
            text_to_annotate = ""
            if user_comment:
                text_to_annotate = f"[User Comment]\n{user_comment}"
                if original_post:
                    text_to_annotate += f"\n\n[Forwarded Original Post]\n{original_post}"
            elif original_post:
                text_to_annotate = f"[Forwarded Original Post]\n{original_post}"
            else:
                return (post_id, None, 'skipped') 

            async with self.api_semaphore:
                start_t = time.time()
                scores = await self._get_attitude_scores_from_llm(text_to_annotate)
                duration = time.time() - start_t
                
                preview = text_to_annotate.replace('\n', ' ')[:30]
                print(f"✅ [ID:{post_id}] {duration:.2f}s | {preview}... -> {json.dumps(scores, ensure_ascii=False)}")
            
            return (post_id, scores, 'success')

        except Exception as e:
            print(f"❌ [ID:{post_id}] Failed: {e}")
            return None

    async def annotate_table(self, db_path: str, table_name: str, only_sim_posts: bool = True, batch_size: int = 50):
        self.logger.info(f"--- 🚀 Starting Annotation for '{table_name}' ---")
        
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            self._setup_database_columns(conn, table_name)
            
            cur = conn.cursor()
            base_query = f"SELECT post_id, content, quote_content FROM {table_name} WHERE attitude_annotated = 0"
            query_sql = f"{base_query} AND created_at NOT LIKE '%-%'" if only_sim_posts else base_query
            
            cur.execute(query_sql)
            all_posts = cur.fetchall()
            cur.close()

            total_count = len(all_posts)
            if total_count == 0:
                print("没有需要处理的数据。")
                return
            
            print(f"共发现 {total_count} 条待处理数据，指标: {self.attitude_columns}")

            pbar = tqdm_asyncio(total=total_count, desc="Processing", unit="post")
            
            global_processed = 0
            global_skipped = 0
            
            for i in range(0, total_count, batch_size):
                batch_data = all_posts[i : i + batch_size]
                tasks = []
                
                for post_id, content, quote_content in batch_data:
                    tasks.append(self._process_post(post_id, content, quote_content))
                
                batch_results = []
                for future in asyncio.as_completed(tasks):
                    result = await future
                    pbar.update(1) 
                    batch_results.append(result)
                
                update_data = []
                for res in batch_results:
                    if res is None: continue 
                    
                    post_id = res[0]
                    status = res[2]
                    
                    if status == 'skipped':
                        global_skipped += 1
                        zeros = tuple(0.0 for _ in self.attitude_columns)
                        update_data.append(zeros + (post_id,))
                    elif status == 'success':
                        global_processed += 1
                        scores = res[1]
                        scores_tuple = tuple(scores.get(col, 0.0) for col in self.attitude_columns)
                        update_data.append(scores_tuple + (post_id,))

                if update_data:
                    write_cur = conn.cursor()
                    try:
                        set_sql_parts = [f"{col} = ?" for col in self.attitude_columns]
                        update_sql = f"UPDATE {table_name} SET {', '.join(set_sql_parts)}, attitude_annotated = 1 WHERE post_id = ?"
                        write_cur.executemany(update_sql, update_data)
                        conn.commit()
                    except Exception as e:
                        self.logger.error(f"DB Commit Error: {e}")
                    finally:
                        write_cur.close()

            pbar.close()
            print(f"任务完成。处理: {global_processed}, 跳过: {global_skipped}")

        except Exception as e:
            self.logger.error(f"Critical Error: {e}")
        finally:
            if conn: conn.close()

class VLLMAttitudeAnnotator(BaseAttitudeAnnotator):
    """
    (子类) 使用本地 Transformers 库直接加载模型进行标注。
    接口参数与 VLLMAttitudeAnnotator 保持一致，以便无缝替换。
    """
    
    def __init__(
        self, 
        model_name: str, 
        attitude_config: Union[Dict[str, str], List[str]],
        base_url: str = "http://localhost:8000/v1",   # [兼容参数] 内部忽略
        api_key: str = "vllm",                        # [兼容参数] 内部忽略
        concurrency_limit: int = 1,                 # [兼容参数] 内部会被强制覆盖为 1
        log_interval_posts: int = 10,
        api_timeout_seconds: int = 120                # [兼容参数] 内部忽略
    ):
        """
        初始化本地 Transformer 标注器。
        注意：虽然接收 concurrency_limit 参数，但为了防止显存溢出，
        内部强制将并发数设为 1 (串行处理)。
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
        # 1. 强制覆盖并发限制
        # 本地模型推理是显存密集型操作，不能并行处理，否则会 OOM
        real_concurrency = 1
        if concurrency_limit > 1:
            print(f"⚠️ [LocalTransformer] Detected concurrency_limit={concurrency_limit}.")
            print(f"   -> Forcing concurrency to {real_concurrency} to prevent CUDA OOM on local inference.")

        # 2. 调用基类初始化
        # 我们传递 dummy 的 url 和 key，因为 Base 类可能会初始化 OpenAI client，但我们不会去用它
        super().__init__(
            api_key="dummy", 
            base_url="http://dummy", 
            attitude_config=attitude_config, 
            concurrency_limit=real_concurrency,  # 关键：传 1 给基类 Semaphore
            log_interval_posts=log_interval_posts,
            api_timeout_seconds=api_timeout_seconds
        )
        
        self.model_path = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.logger.info(f"--- 🐢 Initializing Local Transformer ---")
        self.logger.info(f"   -> Path: {self.model_path}")
        self.logger.info(f"   -> Device: {self.device}")
        self.logger.info("   -> Loading model... (this may take time)")

        # 3. 加载 Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise e

        # 4. 加载 Model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",          # 自动分配显存
                torch_dtype=torch.float16,  # 半精度节省显存
                trust_remote_code=True
            )
            self.model.eval() # 设为评估模式
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise e
            
        self.logger.info("✅ Local model loaded successfully!")

    def _sync_inference(self, prompt: str) -> str:
        """
        同步推理函数 (运行在独立线程中)
        """
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        # 应用模版
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 编码
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 推理
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                temperature=0.1,  # 低温
                do_sample=True,   # 开启采样以支持 temperature
                top_p=0.9
            )
        
        # 解码 (只取新生成的 token)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    async def _get_attitude_scores_from_llm(self, content: str) -> Dict[str, float]:
        """
        覆盖基类方法。
        使用 run_in_executor 将同步的 transformer 推理包装为异步，避免阻塞事件循环。
        """
        loop = asyncio.get_running_loop()
        
        try:
            # 在线程池中运行同步推理，防止卡死主线程
            json_text = await loop.run_in_executor(None, self._sync_inference, content)
            
            # 以下复用之前的 JSON 清洗逻辑
            if "{" in json_text and "}" in json_text:
                start = json_text.index("{")
                end = json_text.rindex("}") + 1
                json_text = json_text[start:end]
            
            parsed = json.loads(json_text)
            
            validated_scores = {}
            for col in self.attitude_columns:
                val = parsed.get(col, 0.0)
                validated_scores[col] = float(val) if isinstance(val, (int, float)) else 0.0
            return validated_scores
            
        except Exception as e:
            # 抛出异常让上层记录为 Failed
            self.logger.error(f"Local Inference Error: {e}")
            raise e

# =====================================================================
# 2. vLLM (开源) 子类
# =====================================================================

class _VLLMAttitudeAnnotator(BaseAttitudeAnnotator):
    """(子类) 使用 vLLM (本地/开源) 兼容 API 进行标注。"""
    
    def __init__(
        self, 
        model_name: str, 
        attitude_config: Union[Dict[str, str], List[str]],
        base_url: str = "http://localhost:8000/v1",  
        api_key: str = "vllm",  
        concurrency_limit: int = 100,
        log_interval_posts: int = 100,
        api_timeout_seconds: int = 120  # [新]
    ):
        """
        初始化 vLLM 标注器。
        """
        self.model_name = model_name
        # [修改] 调用基类的 __init__，传入所有参数
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            attitude_config=attitude_config,
            concurrency_limit=concurrency_limit,
            log_interval_posts=log_interval_posts,
            api_timeout_seconds=api_timeout_seconds # [新]
        )
        self.logger.info(f"VLLM Annotator using model: {self.model_name}")

    # ... _get_attitude_scores_from_llm 方法保持不变 ...
    async def _get_attitude_scores_from_llm(self, content: str) -> Dict[str, float]:
        # (此方法无需修改)
        cleaned_content = self._clean_text(content)
        # _get_system_prompt 已经是动态的了，这里直接调用
        system_prompt = self._get_system_prompt()
        json_text = "" 
        try:
            # 1. 执行 API 调用 (如果超时，在此处抛出异常)
            async with self.api_semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": cleaned_content}
                    ],
                    temperature=0.0,
                    max_tokens=512
                )
            json_text = response.choices[0].message.content
            
            # 2. JSON 提取/清理
            try:
                start_index = json_text.index("{")
                end_index = json_text.rindex("}")
                json_text = json_text[start_index : end_index + 1]
            except ValueError:
                self.logger.warning(f"  -> Could not find '{{' or '}}' in response. Trying to parse anyway.")

            # 3. JSON 解析和验证 (只捕获 JSON 相关的错误)
            parsed_json = json.loads(json_text)
            validated_scores = {}
            for col in self.attitude_columns:
                value = parsed_json.get(col)
                if not isinstance(value, (int, float)):
                    self.logger.warning(f"  -> Invalid data type for key '{col}'. Got: {value}. Defaulting to 0.0")
                    validated_scores[col] = 0.0
                else:
                    validated_scores[col] = float(value)
            return validated_scores
            
        except json.JSONDecodeError as e:
            # 捕获 JSON 解析失败，返回默认分数 (视为有效结果，但不写入数据库)
            self.logger.error(f"  -> LLM JSON parsing failed: {e}. Raw response: '{json_text}'")
            # 修复：在这里**重新抛出异常**，让 _process_post 捕获并返回 None
            raise # ⬅️ 关键修改！让 _process_post 记录为失败
        except Exception as e:
            # 捕获其他非 API/非 JSON 错误 (例如 index 找不到等)
            self.logger.error(f"  -> Unexpected error in LLM processing: {e}. Raw response: '{json_text}'")
            # 修复：重新抛出异常，让 _process_post 捕获并返回 None
            raise # ⬅️ 关键修改！让 _process_post 记录为失败

# =====================================================================
# 3. OpenAI (闭源) 子类
# =====================================================================

class OpenAIAttitudeAnnotator(BaseAttitudeAnnotator):
    """(子类) 使用 OpenAI (闭源) 兼容 API (如 gpt-4o-mini) 进行标注。"""

    def __init__(
        self, 
        model_name: str, 
        api_key: str,
        attitude_columns: List[str],
        base_url: Optional[str] = None,  
        concurrency_limit: int = 100,
        log_interval_posts: int = 10,
        api_timeout_seconds: int = 120  # [新]
    ):
        """
        初始化 OpenAI 标注器。
        """
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("有效的 OpenAI API_KEY (sk-...) 未提供。")
            
        self.model_name = model_name
        # [修改] 调用基类的 __init__，传入所有参数
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            attitude_columns=attitude_columns,
            concurrency_limit=concurrency_limit,
            log_interval_posts=log_interval_posts,
            api_timeout_seconds=api_timeout_seconds # [新]
        )
        self.logger.info(f"OpenAI Annotator using model: {self.model_name}")

    # ... _get_attitude_scores_from_llm 方法保持不变 ...
    async def _get_attitude_scores_from_llm(self, content: str) -> Dict[str, float]:
        # (此方法无需修改)
        default_scores = {col: 0.0 for col in self.attitude_columns}
        if not content or not isinstance(content, str):
            return default_scores
        
        cleaned_content = self._clean_text(content)
        if not cleaned_content:
            return default_scores
        
        system_prompt = self._get_system_prompt()

        try:
            # (此调用现在受 self.api_timeout_seconds 限制)
            async with self.api_semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": cleaned_content}
                    ],
                    response_format={"type": "json_object"}, 
                    temperature=0.0
                )
            json_text = response.choices[0].message.content
            
            parsed_json = json.loads(json_text)
            
            validated_scores = {}
            for col in self.attitude_columns:
                value = parsed_json.get(col)
                if not isinstance(value, (int, float)):
                    self.logger.warning(f"  -> Invalid data type for key '{col}'. Got: {value}. Defaulting to 0.0")
                    validated_scores[col] = 0.0
                else:
                    validated_scores[col] = float(value)
            return validated_scores
        except Exception as e:
            self.logger.error(f"  -> LLM call/parse failed: {e}")
            return default_scores