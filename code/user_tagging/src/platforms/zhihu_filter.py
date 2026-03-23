import re
from .base_filter import BasePlatformFilter

class ZhihuFilter(BasePlatformFilter):
    def __init__(self, config: dict, blacklist_file: str | None = None):
        super().__init__(config=config, blacklist_file=blacklist_file)
        
        # ==========================================
        # 1. 知乎专属营销/引流词库 (知识变现、做号党与导流特征)
        # ==========================================
        self.marketing_keywords = {
            # 公众号引流
            "公号", "公众号", "同名公众号", "首发于", "转载请联系", 
            # 营销变现 & 私域引流
            "关注我", "引流", "私信", "联系方式", "付费咨询", "专栏", "盐选专栏",
            "点击链接", "裙", "扣1", "VX", "+V", "加微", "最低价",
            "评论区留言", "私信领取", "留下一个小❤️", "点个赞吧", "详细介绍", "同款链接",
            # 小说/故事搬运特征
            "未完待续", "点击卡片阅读", "无偿分享", "后续在",
            # 灰产/中介/SEO做号特征 (针对考研、旅游、足彩等)
            "招生简章", "报考条件", "比分预测", "胜负分析", "定制游"
        }
        
        # ==========================================
        # 2. 深度注入到底层 shared_filter
        # ==========================================
        self.shared_filter.marketing_keywords = set(self.marketing_keywords)
        
        # 知乎特色的第一人称代词
        self.shared_filter.first_person_pronouns = {
            "我", "答主", "题主", "鄙人", "小透明", "本人"
        }
        
        # 知乎特色的黑话与情绪词汇 (识别真实人类答主的极佳特征)
        self.shared_filter.implicit_personality_words = {
            "谢邀", "有一说一", "不请自来", "匿了", "实名反对", "不敢苟同", 
            "利益相关", "刚下飞机", "圈子太小", "懂的都懂", 
            "大受震撼", "蚌埠住了", "麻了", "破防", "无语"
        }
        
        self.shared_filter.fake_first_person_patterns = [
            "关注我的公众号", "我的专栏", "我的主页", "私信我", "我的盐选"
        ]

    def get_hash_text(self, post: dict) -> str:
        return f"{post.get('title', '')} {post.get('content', '')}"

    def check_fatal_risk(self, post: dict) -> tuple[bool, str]:
        if not isinstance(post, dict):
            return True, "Invalid Post Type"
            
        # 1. 营销词拦截 (直接用原文本)
        text = self.get_hash_text(post)
        for kw in self.marketing_keywords:
            if kw in text:
                return True, f"Marketing Keyword ({kw})"
                
        # 2. 纯文本长度拦截 (防抖机灵和低质短句)
        # 必须使用脱水后的纯文本来计算长度
        clean_t = self.clean_text(post)
        if len(clean_t.strip()) < 50:
            return True, "Fatal: Text too short for Zhihu (Low Entropy)"
                
        return False, "Safe"

    # 匹配知乎"想法"类帖子的标题，格式为 "用户名 的想法"
    _XIANGFA_PATTERN = re.compile(r"^.+\s+的想法$")

    def extract_anchors(self, post: dict) -> list[str]:
        # 【核心逻辑】：知乎以完整的问题标题作为聚类锚点
        title = str(post.get("title", "")).strip()
        if not title:
            return []

        # 过滤"想法"类帖子：标题为 "用户名 的想法"，不是问答，不能作为 anchor
        if self._XIANGFA_PATTERN.match(title):
            return []

        # 注意：不调用 shared_filter.filter_hashtags，因为知乎的标题往往超过30个字符
        return self._clean_tags([title])

    def clean_text(self, post: dict) -> str:
        title = str(post.get("title", "") or "")
        raw_html = str(post.get("content", "") or "")
        
        # 暴力清除所有 HTML 标签 (彻底解决 data-rawwidth 等图片 base64 乱码残留)
        cleaned_body = re.sub(r'<[^>]+>', ' ', raw_html)
        
        # 调用父类方法进一步清理 URL 和连续空白符
        cleaned_body = self._strip_html(self._clean_urls(cleaned_body))
        return self._normalize_spaces(f"{title} {cleaned_body}")