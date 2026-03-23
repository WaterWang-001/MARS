import json
import re
import os
import time
import logging
import torch
# 尝试导入必要的库
try:
    import openai
    from openai import OpenAI, APITimeoutError, APIConnectionError
except ImportError:
    openai = None
    OpenAI = None
    APITimeoutError = Exception
    APIConnectionError = Exception

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class APIClient:
    def __init__(self, 
                 api_key: str, 
                 base_url: str, 
                 model_name: str,
                 embedding_model_path: str, 
                 mode: str = "remote", 
                 timeout: float = 60.0):
        """
        :param api_key: API Key (仅 Remote 模式需要)
        :param base_url: API 地址 或 本地 LLM 路径
        :param model_name: LLM 模型名称 (vLLM serve name)
        :param embedding_model_path: 本地 BGE Embedding 模型路径
        :param mode: "remote" (vLLM/OpenAI) 或 "local" (Transformers)
        :param timeout: 请求超时时间
        """
        self.model_name = model_name
        self.mode = mode
        self.timeout = float(timeout)
        self.embedding_model_path = embedding_model_path

        # ======================================================
        # 1. 统一加载本地 Embedding 模型 (BGE)
        # ======================================================
        if SentenceTransformer is None:
            raise ImportError("统一使用本地 BGE 模型需要安装: pip install sentence-transformers")

        if not os.path.exists(self.embedding_model_path):
            print(f"⚠️ Warning: 本地路径不存在: {self.embedding_model_path}，尝试从 HuggingFace Hub 下载...")
        
        print(f"🔌 Loading Local Embedding Model: {self.embedding_model_path} ...")
        try:
            # 始终加载到 CPU，不仅速度足够快，还能为本地 LLM 腾出显存
            # normalize_embeddings=True 确保输出向量已归一化，利于后续余弦相似度计算
            self.local_embedder = SentenceTransformer(self.embedding_model_path,device="cpu")
            print(f"✅ Embedding Model Loaded (CPU).")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")

        # ======================================================
        # 2. 初始化 LLM 客户端 (根据 Mode)
        # ======================================================
        
        # --- Remote Mode (vLLM / OpenAI) ---
        if self.mode == "remote":
            if openai is None:
                raise ImportError("Remote mode requires: pip install openai")
            
            self.client = OpenAI(
                api_key=api_key, 
                base_url=base_url,
                max_retries=3
            )
            print(f"✅ LLM Client (Remote): {base_url} | Model: {model_name}")
            
        # --- Local Mode (HuggingFace Transformers) ---
        elif self.mode == "local":
            print(f"⏳ Loading Local LLM: {base_url} ...")
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM
            except ImportError:
                raise ImportError("Local mode requires: pip install torch transformers accelerate")

            self.tokenizer = AutoTokenizer.from_pretrained(base_url, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_url, 
                device_map="auto", 
                torch_dtype=torch.float16, 
                trust_remote_code=True
            )
            print("✅ Local LLM Loaded.")
            
        else:
            raise ValueError("Mode must be 'remote' or 'local'")

    # ======================================================
    #  Chat Completions (LLM)
    # ======================================================

    def call_api(self, prompt: str, temperature: float = 0.1) -> dict:
        """统一 LLM 调用入口
        :param prompt: 提示词内容
        :param temperature: 采样温度（默认 0.1）。
        """
        try:
            if self.mode == "remote":
                return self._call_remote_api(prompt, temperature=temperature)
            else:
                return self._call_local_model(prompt, temperature=temperature)
        except Exception as e:
            # 可以在这里加重试逻辑或日志
            raise e 

    def _call_remote_api(self, prompt: str, temperature: float = 0.1) -> dict:
        use_json_mode = False
        # 简单判断是否启用 JSON Mode
        if "deepseek" in self.model_name.lower() or "json" in self.model_name.lower(): 
            use_json_mode = True 
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Output strictly valid JSON."},
            {"role": "user", "content": prompt}
        ]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
        }
        
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            # 将超时设置为调用级别（OpenAI Python SDK 支持 with_options）
            client = self.client.with_options(timeout=self.timeout)
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return self._parse_json(content)
        except Exception as e:
            # Fallback: 如果服务端报错不支持 response_format，则去掉参数重试
            if "response_format" in str(e) and use_json_mode:
                del kwargs["response_format"]
                client = self.client.with_options(timeout=self.timeout)
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return self._parse_json(content)
            raise e

    def _call_local_model(self, prompt: str, temperature: float = 0.1) -> dict:
        messages = [
            {"role": "system", "content": "Output strictly valid JSON."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024, 
                temperature=float(temperature),
                do_sample=True 
            )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self._parse_json(response_text)

    # ======================================================
    #  Embeddings (Always Local BGE)
    # ======================================================

    def get_embeddings(self, texts: list) -> list:
        """
        获取文本向量。
        统一使用本地加载的 BGE 模型。
        """
        if not texts:
            return []
            
        try:
            # normalize_embeddings=True 对聚类（Cosine Similarity）至关重要
            # convert_to_numpy=True 后转 list，便于 JSON 序列化
            embeddings = self.local_embedder.encode(
                texts, 
                batch_size=32, 
                normalize_embeddings=True, 
                convert_to_numpy=True
            )
            return embeddings.tolist()
            
        except Exception as e:
            print(f"❌ Local Embedding Error: {e}")
            raise e

    # ======================================================
    #  Utils
    # ======================================================

    def _parse_json(self, content: str) -> dict:
        if not content: return {}
        content = content.strip()
        # 移除 Markdown 代码块
        content = re.sub(r'^```json\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                content = content[start : end + 1]
        except Exception: pass

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ JSON Parse Error. Preview: {content[:50]}...")
            return {}