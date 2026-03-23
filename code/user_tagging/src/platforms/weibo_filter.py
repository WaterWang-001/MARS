import re

from .base_filter import BasePlatformFilter


class WeiboFilter(BasePlatformFilter):
    def __init__(self, config: dict, blacklist_file: str | None = None):
        super().__init__(config=config, blacklist_file=blacklist_file)
        self.hashtag_pattern = re.compile(r"#([^#\n]+)#")
        self.fandom_keywords = {
            "抱走", "不约", "专注自家", "非官宣不约", "独美", "净化广场", "空瓶",
            "捞一下", "做数据", "打卡", "签到", "积分", "教程", "切号",
            "控评", "反黑", "卡黑", "洗白", "虐粉", "固粉", "撕逼", "拉踩",
            "艳压", "碰瓷", "带节奏", "按头", "鉴粉籍", "泥塑", "脱粉回踩",
            "打投", "催票", "集资", "应援", "艹数据", "草数据",  
            "打call", "转评赞", "互捞", "互赞", "互评", "互关", "互粉", "回关",
            "周边", "应援物", "后援会", "反黑站", "数据站", "投票组","周边中转站",
            "蒸煮", "正主", "爱豆", "墙头", "本命", "对家", "队友", "前队友",
            "私生", "站姐", "粉头", "大粉", "脂粉", "职粉", "散粉", "路人粉",
            "唯粉", "团粉", "CP粉", "毒唯", "黑粉", "黑子", "水军", "键盘侠",
            "喷子", "杠精", "柠檬精", "戏精", "白嫖", "bp",
            "糊逼", "糊穿地心", "糊出宇宙", "不可说", "盛世美颜", "ssmy",
            "神仙颜值", "绝绝子", "yyds", "awsl", "kswl", "szd", "sjd", "be", "he",
            "房子塌了", "塌房", "抠脚", "划水"
        }
        
        self.fandom_abbr = {
            "gzs", "lb", "hyq", "dbq", "xnl", "yxh", 
            "nbcs", "rnb", "top", "ace", "c位", "pb", "rs" 
        }
        
        # --- 2. 营销号关键词 (全量) ---
        self.marketing_keywords = {
            "领券", "优惠券", "大额券", "隐藏券", "内部券", "神券", "券后",
            "下单", "橱窗", "代购", "拼单", "凑单", "满赠", "加购", "锁单",
            "包邮", "满减", "秒杀", "限时", "手慢无", "库存", "尾款", "预售",
            "点击链接", "戳链接", "链接在", "同款链接", "下单链接", "传送门",
            "薅羊毛", "白菜价", "骨折价", "捡漏", "神车", "漏洞", "拍一发", "拍一走",
            "作业", "抄作业", "补货", "掉落", "福利款", "清仓", "孤品", "微瑕",
            "承接", "业务", "咨询", "热线", "联系电话", "专业办理", "一站式服务", "有限公司",
            "红包", "壁纸",
            "戳右边", "主页有惊喜", "主页领取", "看主页", "看置顶", "看简介",
            "薇信", "V信", "VX", "+V", "威信", "卫星", "公重号", "弓中号", "gzh", "公主号",
            "粉丝群", "进群", "福利群", "交流群", "上车", "禁言群", "裙",
            "某宝", "某东", "某多", "薯店", "米米", "米w", "rmb", "软妹币",
            "私我", "私信", "后台", "滴滴我", "dd我", "踢踢我", "丝我",
            "防走丢", "小号", "新号", "备用号",
            "种草", "拔草", "安利", "按头安利", "强烈推荐", "墙裂推荐", "吐血推荐",
            "闭眼入", "闭眼冲", "人手一个", "入手不亏", "无限回购", "囤货", "自用",
            "神仙好物", "宝藏好物", "宝藏店铺", "良心推荐", "真实反馈", "真实测评",
            "亲测", "避雷", "踩雷", "智商税", "平替", "贵替", "天花板", "yyds",
            "绝绝子", "谁懂啊", "咱就是说", "一整个", "狠狠", "拿捏", "锁死",
            "#投稿#", "私信投稿", "匿名投稿", "粉丝投稿", "树洞", "墙",
            "营销号", "搬运自", "授权转载", "侵删", "cr.", "cr:", "转侵删", "cr网络",
            "杀疯了", "封神", "美强惨", "天菜", "垂直入坑", "沦陷", "暴击", "破防",
            "谁懂", "家人们", "集美们", "姐妹们", "宝子们", "plmm", "xswl",
            "吃瓜", "瓜主", "实锤", "塌房", "反转", "洗白",
            "资料包", "资源包", "教程", "合集", "打包", "无偿分享", "自取",
            "搞钱", "搞米", "副业", "创业", "翻身", "逆袭", "上岸", "暴富",
            "变现", "实操", "落地", "闭环", "底层逻辑", "认知",
            "试看", "完整版", "网盘", "提取码", "扣1", "扣666", "评论区见",
            "带你", "一对一", "手把手", "小白", "0基础",
        }

        # --- 3. 垃圾/无意义标签 (全量) ---
        self.junk_hashtag_keywords = {
            "打投", "数据", "签到", "打卡", "积分", "教程", "反黑", "净化", "控评",
            "安利", "应援", "投票", "榜单", "链接", "集资", "催票", "预约", "销量",
            "转评", "赞评", "互捞", "互粉", "回关", "扩列", "任务", "福利", "白菜",
            "后援会", "反黑站", "数据站", "投票组", "官方粉丝群",
            "微博会员", "阳光信用", "粉丝红包", "新人", "报道", "扩关", "svip", "优惠", "公益",
            "小红书", "快手", "bilibili", "抖音", "淘宝", "天猫", "京东", "拼多多"
        }
        self._WEIBO_TAG_RE = re.compile(r"#([^#]+)#")
        self.shared_filter.fandom_keywords = set(self.fandom_keywords)
        self.shared_filter.marketing_keywords = set(self.marketing_keywords)
        self.shared_filter.junk_hashtag_keywords = set(self.junk_hashtag_keywords)
        self.shared_filter.fandom_abbr = set(self.fandom_abbr)
        self.shared_filter.junk_tag_suffixes = ["[话题]", "[超话]"]
        self.shared_filter.first_person_pronouns = {
            "我", "俺", "主播", "主包", "博主", "老子", "劳资", "本人"
        }
        self.shared_filter.implicit_personality_words = {
            "觉得", "以为", "打算", "想去", "猜", "怀疑", "不懂就问", "求助", "蹲一个",
            "好饿", "好困", "好累", "想吐", "吃撑", "笑吐", "气死", "烦死", "吓死",
            "无语", "离谱", "崩溃", "emo", "破防", "尴尬", "社死", "真香", "救命",
            "卧槽", "牛逼", "wc", "nb", "tmd", "傻逼", "有病", "服了"
        }
        self.shared_filter.fake_first_person_patterns = [
            "我的评分", "我的位置", "我的主页", "我的直播", "我的橱窗", "我的小店",
            "关注我", "私信我", "滴滴我", "联系我", "找到我"
        ]

    def get_hash_text(self, post: dict) -> str:
        if not isinstance(post, dict):
            return str(post or "")
        return f"{post.get('content', '')} {post.get('quote_content', '')}"

    def check_fatal_risk(self, post: dict) -> tuple[bool, str]:
        if not isinstance(post, dict):
            return True, "Invalid Post Type"

        text = str(post.get("content", "") or "")
        low = text.lower()

        for kw in self.fandom_keywords:
            if kw in text:
                return True, f"Fandom Keyword ({kw})"
        for kw in self.marketing_keywords:
            if kw in text:
                return True, f"Marketing Keyword ({kw})"
        for abbr in self.fandom_abbr:
            if abbr in low and len(abbr) > 2:
                return True, f"Fandom Abbr ({abbr})"
        if low.count("|") > 3 or low.count("｜") > 3:
            return True, "Pipe Stuffing Format"

        return False, "Safe"

    def extract_anchors(self, post: dict) -> list[str]:
        tags = self.hashtag_pattern.findall(self.get_hash_text(post))
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
