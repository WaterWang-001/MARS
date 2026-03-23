import re
from .base_filter import BasePlatformFilter

class XiaohongshuFilter(BasePlatformFilter):
    def __init__(self, config: dict, blacklist_file: str | None = None):
        super().__init__(config=config, blacklist_file=blacklist_file)
        
        # 小红书锚点特征：单井号开头，遇到空格或换行停止
        self.hashtag_pattern = re.compile(r"#([^\s#]+)")
        
        # ==========================================
        # 1. 小红书专属营销/黑产词库 (史诗级扩充)
        # ==========================================
        self.marketing_keywords = {
            # 强引流与交易
            "主页有惊喜", "私信我", "滴滴我", "领券", "代购", "加v", "加V", "+v", "+V",
            "橱窗", "拼单", "某海鲜市场", "私我", "后台", "看简介", "薯店", "米米", 
            "米w", "rmb", "福利群", "小号", "备用号", "防走丢", "全网同名",
            # 营销与做号暗语
            "避雷", "智商税", "戳左下角", "链接在", "薅羊毛", "白菜价", 
            "福利款", "微瑕", "种草", "拔草", "平替", "贵替", "闭眼入", "无限回购", 
            "宝藏好物", "神仙好物", "作业", "抄作业", "自用", "亲测",
            # 兼职、招商与黑产
            "搞钱", "副业", "变现", "无偿分享", "资料包", "网盘", "提取码",
            "操盘手", "合伙人", "赛道", "知识付费", "招商", "实体店", "招代理",
            # 饭圈代拍与黄牛 (基于样本暴露的问题)
            "蹲全程", "拼房", "拼车", "代拍", "出图", "接拍",
            # 文玩/二手/微商特定高频词 (基于样本暴露的问题)
            "沉水", "手串", "老料", "回购", "包邮", "现货", "发货", "直邮", "专柜"
        }
        
        # ==========================================
        # 2. 商家简介拦截词库 (拦截官方号/店铺号)
        # ==========================================
        self.merchant_profile_keywords = {
            "发货", "实体店", "招商", "合伙人", "品牌官方", "旗舰店", 
            "V心", "➕v", "主理人", "主页", "橱窗", "全网同名", "店主", "运营"
        }

        # ==========================================
        # 3. 深度注入到底层 shared_filter
        # ==========================================
        self.shared_filter.marketing_keywords = set(self.marketing_keywords)
        self.shared_filter.junk_tag_suffixes = ["[话题]"]
        
        self.shared_filter.first_person_pronouns = {
            "我", "俺", "主播", "主包", "博主", "老子", "劳资", "本人"
        }
        
        self.shared_filter.implicit_personality_words = {
            "集美们", "绝绝子", "谁懂啊", "家人们", "姐妹们", "宝子们", "plmm",
            "无语", "离谱", "崩溃", "真香", "救命", "踩雷", "天花板", "yyds",
            "一整个", "狠狠", "拿捏", "锁死", "破防", "沦陷", "天菜"
        }
        
        self.shared_filter.fake_first_person_patterns = [
            "我的主页", "我的橱窗", "我的小店", "关注我", "私信我", "滴滴我", "找到我"
        ]

    def is_user_qualified(self, user_profile: dict) -> tuple[bool, str]:
        if not user_profile:
            return True, "Pass"

        desc = str(user_profile.get("description") or user_profile.get("bio") or "")
        name = str(user_profile.get("name", "") or "").lower()

        # 1. 商家引流词拦截
        combined_profile = f"{name} {desc}".lower()
        for kw in self.merchant_profile_keywords:
            if kw in combined_profile:
                return False, f"Fatal: Merchant Profile ({kw})"

        # 2. Bio 最短长度（小红书用户普遍有 bio，过短说明非活跃真人）
        profile_cfg = self.quality_rules.get("user_profile", {})
        min_bio_length = int(profile_cfg.get("min_bio_length", 0))
        if min_bio_length > 0:
            clean_desc = desc.strip()
            if clean_desc in ("NaN", "nan", "None", ""):
                clean_desc = ""
            if len(clean_desc) < min_bio_length:
                return False, f"Bio too short ({len(clean_desc)}<{min_bio_length})"

        return True, "Pass"

    def get_hash_text(self, post: dict) -> str:
        # 【修改】：引入 quote_content，彻底解决评论短缺上下文的问题
        return f"{post.get('title', '')} {post.get('content', '')} {post.get('quote_content', '')}"

    def check_fatal_risk(self, post: dict) -> tuple[bool, str]:
        if not isinstance(post, dict):
            return True, "Invalid Post Type"

        text = self.get_hash_text(post)
        low = text.lower()

        # 营销词拦截
        for kw in self.marketing_keywords:
            if kw in text:
                return True, f"Marketing Keyword ({kw})"
                
        # 排版格式风控：小红书常见的 SEO 堆砌拦截
        if low.count("|") > 3 or low.count("｜") > 3:
            return True, "Pipe Stuffing Format"

        return False, "Safe"

    def extract_anchors(self, post: dict) -> list[str]:
        anchor_text = f"{post.get('title', '')} {post.get('content', '')}"
        tags = self.hashtag_pattern.findall(anchor_text)
        cleaned_raw_tags = []
        for tag in tags:
            clean_tag = str(tag or "").strip().strip(".,!?;:。，！？；：'\"”’")
            if clean_tag:
                cleaned_raw_tags.append(clean_tag)
                
        return self.shared_filter.filter_hashtags(cleaned_raw_tags)

    def clean_text(self, post: dict) -> str:
        text = self.get_hash_text(post)
        text = self._clean_urls(text)
        return self._normalize_spaces(text)