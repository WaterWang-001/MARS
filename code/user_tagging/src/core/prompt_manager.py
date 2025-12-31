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

    def get_interest_prompt(self, username, bio, posts, candidate_entities):
        """
        生成兴趣标签提取 Prompt
        :param posts: 已经拼接好的帖子字符串
        """
        template = self.templates.get('interest_prompt')
        if not template:
            raise ValueError("Template 'interest_prompt' not found!")
            
        return template.format(
            username=username,
            bio=bio,
            posts=posts,  
            candidate_entities=candidate_entities
        )

    def get_demographic_prompt(self, username, bio, gender, reg_time, location, verified_info, posts):
        template = self.templates.get('demographic_prompt')
        return template.format(
            username=username,
            bio=bio,
            posts_content=posts,
            gender_reported=gender,
            reg_time=reg_time,
            location_reported=location,
            verified_info=verified_info
        )

    def get_firmographic_prompt(self, username, bio, verified_type, mapped_type_name, verified_info, posts):
        template = self.templates.get('firmographic_prompt')
        return template.format(
            username=username,
            bio=bio,
            posts_content=posts,
            verified_type=verified_type,
            mapped_type_name=mapped_type_name,
            verified_info=verified_info
        )