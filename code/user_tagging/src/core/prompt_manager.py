import os
import logging

class PromptManager:
    def __init__(self, prompt_dir: str):
        self.prompt_dir = prompt_dir
        self.templates = {}
        self.logger = logging.getLogger("PromptManager")
        self._load_templates()

    def _load_templates(self):
        """预加载所有 .md/.txt 提示词模板"""
        if not os.path.exists(self.prompt_dir):
            self.logger.warning(f"Prompt directory not found: {self.prompt_dir}")
            return

        for filename in os.listdir(self.prompt_dir):
            # 支持 .md 和 .txt
            if filename.endswith(".md") or filename.endswith(".txt"):
                name = os.path.splitext(filename)[0]
                try:
                    with open(os.path.join(self.prompt_dir, filename), 'r', encoding='utf-8') as f:
                        self.templates[name] = f.read()
                except Exception as e:
                    self.logger.error(f"Failed to load template {filename}: {e}")
        
        self.logger.info(f"Loaded {len(self.templates)} templates: {list(self.templates.keys())}")

    def get_interest_prompt(self, username, bio, posts, candidate_entities, prev_persona=""):
        """
        生成兴趣标签提取 Prompt (流式模式：传入历史 persona 作为上下文先验)
        :param posts: 已经拼接好的帖子字符串
        :param prev_persona: 上一批次的 persona_summary (Internal Memory State S_{n-1})
        """
        template = self.templates.get('interest_prompt')
        if not template:
            raise ValueError("Template 'interest_prompt' not found!")

        if prev_persona:
            persona_block = f"""## 当前用户画像（基于历史观测积累的认知中枢 / Internal Memory State）
{prev_persona}"""
        else:
            persona_block = """## 当前用户画像
暂无历史观测，这是该用户的第一个活动批次。请从零开始构建画像。"""

        return template.format(
            username=username,
            bio=bio,
            posts=posts,
            candidate_entities=candidate_entities,
            persona_block=persona_block
        )

    @staticmethod
    def _build_profile_block(prev_profile: dict) -> str:
        """构建历史人口学/机构画像上下文块，注入 prompt 作为先验"""
        if not prev_profile:
            return "## 历史画像\n暂无历史观测，这是该用户的第一个活动批次。请从零开始推断。"
        # 过滤掉辅助字段，只保留属性槽
        skip_keys = {"user_type", "is_org"}
        lines = []
        for key, node in prev_profile.items():
            if key in skip_keys:
                continue
            if isinstance(node, dict):
                val = node.get("tag") or node.get("value", "NA")
                conf = node.get("confidence", "NA")
                lines.append(f"- {key}: {val} (置信度: {conf})")
            else:
                lines.append(f"- {key}: {node}")
        if not lines:
            return "## 历史画像\n暂无历史观测，这是该用户的第一个活动批次。请从零开始推断。"
        summary = "\n".join(lines)
        return f"""## 历史画像（基于此前批次的推断结果）
{summary}

**更新原则**：若新证据的置信度高于历史值，则覆写；若新证据不足或置信度更低，则保留历史推断。"""

    def get_demographic_prompt(self, username, bio, posts, gender_reported, reg_time, location_reported, verified_info, prev_profile=None):
        template = self.templates.get('demographic_prompt')
        if not template:
            self.logger.error("Template 'demographic_prompt' not found!")
            return ""

        return template.format(
            username=username,
            bio=bio,
            posts_content=posts,
            gender_reported=gender_reported,
            reg_time=reg_time,
            location_reported=location_reported,
            verified_info=verified_info,
            prev_profile_block=self._build_profile_block(prev_profile)
        )

    def get_firmographic_prompt(self, username, bio, verified_type, mapped_type_name, verified_info, posts, prev_profile=None):
        template = self.templates.get('firmographic_prompt')
        return template.format(
            username=username,
            bio=bio,
            posts_content=posts,
            verified_type=verified_type,
            mapped_type_name=mapped_type_name,
            verified_info=verified_info,
            prev_profile_block=self._build_profile_block(prev_profile)
        )