import json
import re
import os
import time

try:
    import openai
    from openai import OpenAI, APITimeoutError, APIConnectionError
except ImportError:
    openai = None
    OpenAI = None
    APITimeoutError = Exception
    APIConnectionError = Exception

class APIClient:
    def __init__(self, api_key: str, base_url: str, model_name: str, mode: str = "remote", timeout: float = 60.0):
        """
        :param api_key: API Key
        :param base_url: API 地址
        :param model_name: 模型名称 (vLLM 部署时的 serve name)
        :param mode: "remote" (vLLM/OpenAI) 或 "local" (Transformers)
        :param timeout: 请求超时时间 (秒)
        """
        self.model_name = model_name
        self.mode = mode
        self.timeout = float(timeout) # [修改] 接收并存储超时设置
        
        if self.mode == "remote":
            if openai is None:
                raise ImportError("请安装 openai 包: pip install openai")
            
            # 初始化 OpenAI 客户端
            self.client = OpenAI(
                api_key=api_key, 
                base_url=base_url,
                max_retries=1  # 快速失败，不要在库内部死锁
            )
            print(f"✅ API Client Init: {base_url} | Model: {model_name} | Timeout: {self.timeout}s")
            
        elif self.mode == "local":
            print(f"⏳ Loading Local Model: {base_url} ...")
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
            print("✅ Local Model Loaded.")
        else:
            raise ValueError("Mode must be 'remote' or 'local'")

    def call_api(self, prompt: str) -> dict:
        """
        统一调用入口，异常由上层 Service 捕获
        """
        try:
            if self.mode == "remote":
                return self._call_remote_api(prompt)
            else:
                return self._call_local_model(prompt)
        except Exception as e:
            # 打印日志并向上抛出，让 TaggingService 决定是重试还是记录 Error
            # print(f"❌ Inference Error: {e}") 
            raise e 

    def _call_remote_api(self, prompt: str) -> dict:
        """调用兼容 OpenAI 格式的 API"""
        
        # 针对特定模型启用 JSON Mode (可选，Qwen 2.5 通常不需要强制 JSON Mode 也能遵循指令)
        use_json_mode = False
        if "deepseek" in self.model_name.lower() or "json" in self.model_name.lower(): 
            use_json_mode = True 
            
        messages = [
            # [修改] 简化 System Prompt，因为业务 Prompt 里已经定义了详细的角色
            {"role": "system", "content": "You are a helpful assistant. Output strictly valid JSON."},
            {"role": "user", "content": prompt}
        ]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "timeout": self.timeout # [修改] 使用配置的超时时间
        }
        
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return self._parse_json(content)
            
        except APITimeoutError:
            print(f"⚠️ [Timeout] Request timed out after {self.timeout}s.")
            raise # 抛出，让外层捕获
            
        except APIConnectionError:
            print(f"⚠️ [Connection] Failed to connect to vLLM/API.")
            raise

        except Exception as e:
            # Fallback: 如果模型不支持 JSON Mode 参数，回退到普通模式重试
            if "response_format" in str(e) and use_json_mode:
                # print(f"⚠️ JSON Mode not supported, retrying text mode...")
                del kwargs["response_format"]
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return self._parse_json(content)
            raise e

    def _call_local_model(self, prompt: str) -> dict:
        """本地 Transformers 推理"""
        messages = [
            {"role": "system", "content": "Output strictly valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024, 
                temperature=0.1,
                do_sample=True 
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self._parse_json(response_text)

    def _parse_json(self, content: str) -> dict:
        """
        鲁棒的 JSON 解析器
        """
        if not content:
            return {}
            
        content = content.strip()
        
        # 去除 Markdown 代码块标记
        content = re.sub(r'^```json\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        
        # 尝试提取第一个 { ... } 块，防止 LLM 在 JSON 前后废话
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                content = content[start : end + 1]
        except Exception:
            pass

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 如果解析失败，返回空字典而不是由这里报错
            # 这样 Service 层只会因为拿到空数据而做相应处理，不会 Crash
            print(f"⚠️ JSON Parse Error. Preview: {content[:50]}...")
            return {}