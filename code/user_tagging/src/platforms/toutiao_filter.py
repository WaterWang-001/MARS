import re
import html
from .base_filter import BasePlatformFilter

class ToutiaoFilter(BasePlatformFilter):
    def __init__(self, config: dict, blacklist_file: str | None = None):
        super().__init__(config=config, blacklist_file=blacklist_file)
        
        # 修正正则，防止抓到 HTML 里的颜色代码(如 #FF0000;)
        self.hashtag_pattern = re.compile(r"#([\u4e00-\u9fa5a-zA-Z0-9_]+)")

        # ==========================================
        # 1. 头条专属企业/SEO矩阵号拦截词库 (拦截在起跑线)
        # ==========================================
        self.corporate_keywords = {
            "厂家", "生产厂家", "模型", "专营", "旗舰店", "批发", "机械", "加工", 
            "设备", "公司", "企业", "招商", "加盟", "车行", "二手车", "直供", 
            "零售", "回收", "供应", "定制"
        }
        
        # ==========================================
        # 2. 头条专属营销/做号党词库 (地毯式全覆盖)
        # ==========================================
        self.marketing_keywords = {
            # 【下沉电商 / 直播带货 / 薅羊毛】
            "包赔", "不粉包赔", "退换无忧", "运费险", "假一赔十", "上锅蒸", "开袋即食",
            "冰冰凉凉", "超大号", "干的快", "不起球", "不褪色", "纯棉", "源头工厂", 
            "厂家直销", "拍一发", "破破价", "清仓", "只要9块9", "只要9.9", "买一送",
            "点击下方", "小黄车", "商品卡", "橱窗", "点击左下角", "同款链接", 
            "直播间", "感谢老铁", "关注主播", "性价比高", "便宜好用", "赶紧囤",
            "包邮", "下单", "小店", "带货", "秒杀价", "点击下方卡片", "点击视频",

            # 【AI做号党 / 震惊体 / 八卦模板话术】
            "小编带你", "不转不是", "赶紧看", "惊呆了", "细思极恐", "深度长文",
            "建议收藏", "不看后悔", "万万没想到", "惊掉下巴", "刚刚传来", "突发", 
            "彻底瞒不住了", "背后真相", "到底是怎么回事", "让我们一起来看看",
            "家人们！", "咱接着往下看", "结语：", "前言：", "网友们瞬间炸开了锅", 
            "大家对这件事怎么看", "欢迎在评论区", "别忘了点赞关注", "感谢各位看官",
            "你怎么看？", "看完记得", "点个赞吧", "创作不易", "别划走", 
            "一定要看到最后", "太不可思议了", "看完泪目", "深度好文", "建议点赞收藏",

            # 【引流 / 灰产 / 微商骗局 / 区域SEO矩阵】
            "私信我", "进群", "点击链接", "代购", "免费领取", "点击头像", "主页",
            "网赚", "副业", "带你赚钱", "在家兼职", "了解一下", "加威", "加微",
            "微信号", "长按复制", "领券", "日入过百", "宝妈兼职", "无门槛", 
            "免费带徒", "一部手机就能", "月入过万", "扫码添加",
            "多少钱", "价格怎么算", "哪家好"  # 拦截诸如“邢台沙盘模型多少钱”
        }
        
        # ==========================================
        # 3. 深度注入到底层 shared_filter
        # ==========================================
        self.shared_filter.marketing_keywords = set(self.marketing_keywords)
        self.shared_filter.junk_tag_suffixes = ["头条", "挑战赛", "打卡季", "计划", "活动"]
        
        # 【扩充】：头条专属废话/流量标签拦截
        self.shared_filter.junk_hashtag_keywords.update({
            "随拍", "上热门", "头条创作挑战赛", "流量", "求赞", "互粉", "互关",
            "我要上微头条", "我要上热门", "记录真实生活", "记录生活", "上推荐",
            "日常", "日常vlog", "新人求过百", "互粉互赞", "微头条激励计划",
            "自媒体", "涨粉", "头条流量", "新人小白"
        })

        self.shared_filter.first_person_pronouns = {
            "我", "俺", "笔者", "小编", "老朽", "鄙人", "老铁", "咱"
        }
        
        self.shared_filter.implicit_personality_words = {
            "老铁", "说实话", "大实话", "坑人", "太难了", "扎心", "实话实说", 
            "不懂就问", "长见识", "气愤", "看哭", "感动", "良心话", "有一说一"
        }
        
        self.shared_filter.fake_first_person_patterns = [
            "我的主页", "关注我", "私信我", "联系我", "主页找我", "我的橱窗"
        ]

    def is_user_qualified(self, user_profile: dict) -> tuple[bool, str]:
        """
        【核心新增】：拦截头条上泛滥的工厂、机械、二手车等 B端 SEO 矩阵号
        """
        if not user_profile:
            return True, "Pass"
            
        desc = str(user_profile.get("description", "") or "").lower()
        name = str(user_profile.get("name", "") or "").lower()
        combined_profile = f"{name} {desc}"
        
        for kw in self.corporate_keywords:
            if kw in combined_profile:
                return False, f"Fatal: Corporate/SEO Account ({kw})"
                
        return True, "Pass"

    def get_hash_text(self, post: dict) -> str:
        return f"{post.get('title', '')} {post.get('content', '')} {post.get('quote_content', '')}"

    def check_fatal_risk(self, post: dict) -> tuple[bool, str]:
        if not isinstance(post, dict):
            return True, "Invalid Post Type"

        text = self.get_hash_text(post)
        low = text.lower()

        for kw in self.marketing_keywords:
            if kw in text:
                return True, f"Marketing Keyword ({kw})"

        if low.count("|") > 3 or low.count("｜") > 3 or low.count("~") > 4:
            return True, "Pipe Stuffing Format"

        # 只用用户自己写的内容做长度检查，不计入 title 和 quote_content
        user_text = self._get_user_original_text(post)
        if len(user_text.strip()) < 20:
            return True, "Fatal: Text too short for Toutiao (Low Entropy)"

        return False, "Safe"

    def _get_user_original_text(self, post: dict) -> str:
        """提取用户自己写的原创内容，排除 title 和 quote_content"""
        content = str(post.get("content", "") or "")
        quote_content = str(post.get("quote_content", "") or "")
        title = str(post.get("title", "") or "")
        if quote_content:
            # 评论帖：只算 content
            text = content
        else:
            # 原创帖：算 title + content
            text = f"{title} {content}"
        text = html.unescape(text)
        text = re.sub(r'<[^>]*>', ' ', text)
        text = self._clean_urls(text)
        return self._normalize_spaces(text)

    def extract_anchors(self, post: dict) -> list[str]:
        text = self.get_hash_text(post)
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', ' ', text)     # strip HTML first
        tags = self.hashtag_pattern.findall(text)
        cleaned_raw_tags = []
        _hex_re = re.compile(r'^[0-9A-Fa-f]{3,8}$')
        _strip_chars = '.,!?;:。，！？；：' + "'\"" + '“”‘’'
        for tag in tags:
            clean_tag = str(tag or "").strip().strip(_strip_chars)
            if clean_tag and not _hex_re.fullmatch(clean_tag):
                cleaned_raw_tags.append(clean_tag)
        return self.shared_filter.filter_hashtags(cleaned_raw_tags)

    def clean_text(self, post: dict) -> str:
        text = self.get_hash_text(post)
        
        # 1. 反转义 HTML 实体 (如将 &#34; 变回 ", &nbsp; 变回空格)
        text = html.unescape(text)
        
        # 2. 暴力清除头条残留的复杂 HTML (消灭 <p data-track="2"> 和长串的图片代码)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        text = self._clean_urls(text)
        return self._normalize_spaces(text)