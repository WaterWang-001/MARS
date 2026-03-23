import json
import pandas as pd
from pathlib import Path
import random
import multiprocessing
import time
import os
import csv
from collections import defaultdict

# 尝试加速 JSON 解析
try:
    import orjson
    JSON_LIB = orjson
except ImportError:
    import json
    JSON_LIB = json

# ================= 配置区域 =================
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path('.').resolve()

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "user_profiles"
INPUT_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_FILES_LOG = OUTPUT_FOLDER / "processed_files_record.txt"

# 1. 字段映射表 (原始字段 -> CSV列名)
FIELD_MAPPING = {
    # 身份标识
    'sjcjId': 'user_id',
    'sjcjAuthorAccount': 'user_id', # Fallback
    'sjcjNickName': 'username',
    'sjcjDescription': 'bio',
    'sjcjGender': 'gender',
    # 'sjcjProfileImageUrl': 'avatar_url',  <-- [保持] 移除 avatar_url
    
    # 核心数据
    'sjcjFollowersCount': 'followers_count',
    'sjcjFriendsCount': 'following_count',
    'sjcjStatusesCount': 'posts_count',
    'sjcjProductionNum': 'posts_count',
    'sjcjFavouritesCount': 'favorites_count',
    'sjcjAttitudesCount': 'likes_received_count',
    
    # 位置信息
    'sjqxProvince': 'province',
    'sjqxCity': 'city',
    'sjcjLocation': 'location',
    'sjcjIpLocation': 'ip_location',
    
    # 时间与状态
    'sjcjRegistrationTime': 'registration_time',
    'sjqxLastPublished': 'last_active_time',
    'sjqxLastUpdatetime': 'last_update_time',
    'sjcjVerified': 'is_verified',
    'sjcjVerifiedType': 'verified_type',
    'sjqxSourceClient': 'device_source'
}

# [保持] 定义字段顺序列表
ORDERED_COLUMNS = [
    'user_id', 
    'username', 
    'gender', 
    'bio', 
    'is_verified', 
    'verified_type',
    'followers_count', 
    'following_count', 
    'posts_count', 
    'favorites_count', 
    'likes_received_count', 
    'province', 
    'city', 
    'location', 
    'ip_location', 
    'registration_time', 
    'last_active_time', 
    'last_update_time',
    'device_source',
    'platform',
    'is_core_user'
]

# 2. 平台字段白名单 (Schema)
PLATFORM_SCHEMA = {
    'weibo': {
        'user_id', 'username', 'bio', 'gender', 'followers_count', 'following_count', 
        'posts_count', 'favorites_count', 'province', 'city', 'location', 'ip_location', 
        'registration_time', 'is_verified', 'verified_type', 'device_source',
        'last_active_time', 'last_update_time'
    },
    'toutiao': {
        'user_id', 'username', 'bio', 'followers_count', 'following_count', 
        'posts_count', 'likes_received_count', 'province', 'city', 'ip_location', 
        'is_verified', 'verified_type', 'last_active_time', 'last_update_time'
    },
    'xiaohongshu': {
        'user_id', 'username', 'bio', 'gender', 'followers_count', 'following_count', 
        'posts_count', 'favorites_count', 'likes_received_count', 'province', 'city', 'ip_location', 
        'is_verified', 'verified_type', 'last_active_time', 'last_update_time'
    },
    'douban': {
        'user_id', 'username', 'followers_count', 'following_count', 
        'posts_count', 'favorites_count', 'likes_received_count', 'province', 'city', 'ip_location', 
        'last_active_time', 'last_update_time'
    },
    'zhihu': {
        'user_id', 'username', 'bio', 'followers_count', 'following_count', 
        'posts_count', 'favorites_count', 'likes_received_count', 'province', 'city', 'location', 'ip_location', 
        'is_verified', 'verified_type', 'last_active_time', 'last_update_time'
    }
}

# [更新] 增强版平台映射表
PLATFORM_MAP = {
    "新浪微博": "weibo",
    "微博": "weibo",
    "微博视频号": "weibo",  # 👈 关键修复
    "今日头条": "toutiao",
    "今日头条微头条": "toutiao",
    "西瓜视频": "toutiao",
    "小红书": "xiaohongshu",
    "豆瓣": "douban",
    "知乎": "zhihu"
}

# ================= 核心逻辑 =================

def get_platform_key(raw_site):
    """
    [同步更新] 增强型平台匹配逻辑 (与 Post 处理逻辑保持一致)
    """
    if not raw_site:
        return "other"
    
    # 1. 精确匹配
    if raw_site in PLATFORM_MAP:
        return PLATFORM_MAP[raw_site]
    
    # 2. 模糊匹配
    site_str = str(raw_site)
    if "微博" in site_str: return "weibo"
    if "头条" in site_str or "西瓜" in site_str: return "toutiao"
    if "小红书" in site_str: return "xiaohongshu"
    if "豆瓣" in site_str: return "douban"
    if "知乎" in site_str: return "zhihu"
    
    return "other"

def safe_bool(val):
    if val is None: return False
    if isinstance(val, bool): return val
    if isinstance(val, str):
        return val.lower() == 'true'
    return bool(val)

def process_file_worker(filepath):
    extracted_data = [] 
    seen_in_file = set()
    
    # [同步更新] 定义查找字段优先级
    site_keys = ['sjqxCaptureWebsite', 'sjcjCaptureWebsite', 'sjqxCaptureWebsiteNew']

    def _get_site_from_pojo(pojo):
        if not pojo: return None
        for key in site_keys:
            val = pojo.get(key)
            if val: return val
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = JSON_LIB.loads(line)
                    
                    # 1. 确定平台 [逻辑更新]
                    # 优先从 contentPojo 找，找不到再去 commentPojo
                    raw_site = _get_site_from_pojo(data.get('contentPojo'))
                    if not raw_site:
                        raw_site = _get_site_from_pojo(data.get('commentPojo'))
                    
                    platform = get_platform_key(raw_site)
                    schema = PLATFORM_SCHEMA.get(platform, PLATFORM_SCHEMA['weibo'])
                    
                    # 2. 定义提取目标 (Target List)
                    targets = []
                    
                    # (A) 基础提取
                    targets.append(data.get('authorContentPojo'))
                    targets.append(data.get('authorCommentPojo'))
                    
                    # (B) 扩展提取：原贴作者
                    if platform in ['weibo', 'douban', 'toutiao', 'zhihu']:
                        targets.append(data.get('authorContentRootPojo'))
                        
                    # (C) 深度提取：转发者
                    if platform == 'weibo':
                        targets.append(data.get('authorCommentForwardPojo'))
                    
                    for user_data in targets:
                        if not user_data: continue
                        
                        uid = user_data.get('sjcjId') or user_data.get('sjcjAuthorAccount')
                        if not uid: continue
                        uid = str(uid)
                        
                        dedup_key = (platform, uid)
                        if dedup_key in seen_in_file: continue
                        
                        # 3. 提取字段
                        user_dict = {'platform': platform}
                        for raw_field, clean_col in FIELD_MAPPING.items():
                            if clean_col in schema:
                                val = user_data.get(raw_field)
                                
                                # 数据清洗
                                if clean_col == 'is_verified': 
                                    val = safe_bool(val)
                                elif clean_col == 'gender':
                                    if val == 'n': val = 'unknown' 
                                    elif val == 'm': val = 'male'
                                    elif val == 'f': val = 'female'
                                
                                user_dict[clean_col] = val
                        
                        if 'user_id' not in user_dict: user_dict['user_id'] = uid
                        
                        # 4. 计算 Core User
                        followers = int(user_dict.get('followers_count', 0) or 0)
                        verified = user_dict.get('is_verified', False)
                        is_core = verified or (followers > 10000) or (followers > 1000 and random.random() < 0.3)
                        user_dict['is_core_user'] = is_core

                        extracted_data.append((platform, user_dict))
                        seen_in_file.add(dedup_key)
                        
                except Exception:
                    continue
        return (filepath.name, extracted_data)
    except Exception as e:
        return (filepath.name, f"ERROR: {str(e)}")

class ParallelProcessor:
    def __init__(self):
        self.existing_ids = set()
        self._load_processed_log()
        self._load_existing_ids()
        
    def _load_processed_log(self):
        self.processed_files = set()
        if PROCESSED_FILES_LOG.exists():
            with open(PROCESSED_FILES_LOG, 'r', encoding='utf-8') as f:
                self.processed_files = {line.strip() for line in f if line.strip()}

    def _load_existing_ids(self):
        if not OUTPUT_FOLDER.exists(): return
        print("⏳ Loading existing IDs...")
        for csv_file in OUTPUT_FOLDER.glob("*.csv"):
            try:
                platform = csv_file.stem.replace("user_profile_", "")
                df = pd.read_csv(csv_file, usecols=['user_id'], dtype={'user_id': str})
                for uid in df['user_id']:
                    self.existing_ids.add(f"{platform}_{uid}")
            except: pass
        print(f"✅ Loaded {len(self.existing_ids)} users.")

    def run(self):
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        files = [f for f in INPUT_FOLDER.glob('**/*.txt') if f.name not in self.processed_files]
        
        if not files:
            print("No new files.")
            return

        print(f"🚀 Processing {len(files)} files...")
        pool = multiprocessing.Pool(processes=max(1, os.cpu_count() - 2))
        
        batch_buffer = defaultdict(list)
        pending_files = []
        total_new = 0
        
        try:
            for filename, result in pool.imap_unordered(process_file_worker, files):
                if isinstance(result, str):
                    print(f"❌ {filename}: {result}")
                    continue
                
                for platform, user_dict in result:
                    key = f"{platform}_{user_dict['user_id']}"
                    if key not in self.existing_ids:
                        self.existing_ids.add(key)
                        batch_buffer[platform].append(user_dict)
                
                pending_files.append(filename)
                
                current_size = sum(len(v) for v in batch_buffer.values())
                if current_size >= 5000 or len(pending_files) >= 50:
                    self._flush(batch_buffer, pending_files)
                    total_new += current_size
                    print(f"⚡ Saved {current_size} users. Total: {total_new}")
                    batch_buffer = defaultdict(list)
                    pending_files = []
            
            if any(batch_buffer.values()) or pending_files:
                self._flush(batch_buffer, pending_files)
                
        finally:
            pool.close()
            pool.join()
            
        print(f"✅ Done! Added {total_new} users.")

    def _flush(self, buffer, files):
        for platform, rows in buffer.items():
            if not rows: continue
            file_path = OUTPUT_FOLDER / f"user_profile_{platform}.csv"
            
            # [保持] 按指定顺序排序字段
            valid_platform_cols = PLATFORM_SCHEMA.get(platform, PLATFORM_SCHEMA['weibo']).copy()
            valid_platform_cols.update(['platform', 'is_core_user'])
            
            final_columns = [col for col in ORDERED_COLUMNS if col in valid_platform_cols]
            
            df = pd.DataFrame(rows)
            # 容错：防止部分列缺失
            cols_to_use = [c for c in final_columns if c in df.columns]
            df = df[cols_to_use]

            write_header = not file_path.exists()
            df.to_csv(
                file_path,
                mode='a',
                header=write_header,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
                escapechar='\\',
            )
            
        with open(PROCESSED_FILES_LOG, 'a', encoding='utf-8') as f:
            for name in files: f.write(name + '\n')

if __name__ == "__main__":
    ParallelProcessor().run()