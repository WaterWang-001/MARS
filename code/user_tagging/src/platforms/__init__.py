"""Platform adapter layer."""

from .base_adapter import BasePlatformAdapter
from .base_filter import BasePlatformFilter
from .weibo_adapter import WeiboAdapter
from .weibo_filter import WeiboFilter
from .zhihu_adapter import ZhihuAdapter
from .zhihu_filter import ZhihuFilter
from .xiaohongshu_adapter import XiaohongshuAdapter
from .xiaohongshu_filter import XiaohongshuFilter
from .toutiao_adapter import ToutiaoAdapter
from .toutiao_filter import ToutiaoFilter
from .douban_adapter import DoubanAdapter
from .douban_filter import DoubanFilter

__all__ = [
    "BasePlatformAdapter",
    "BasePlatformFilter",
    "WeiboAdapter",
    "WeiboFilter",
    "ZhihuAdapter",
    "ZhihuFilter",
    "XiaohongshuAdapter",
    "XiaohongshuFilter",
    "ToutiaoAdapter",
    "ToutiaoFilter",
    "DoubanAdapter",
    "DoubanFilter",
]
