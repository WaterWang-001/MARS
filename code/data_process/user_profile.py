import json
import numpy as np
import pandas as pd
from pathlib import Path
import random
from datetime import datetime
import multiprocessing
import time
import os

# 尝试加速 JSON 解析
try:
    import orjson
    JSON_LIB = orjson
except ImportError:
    import json
    JSON_LIB = json

# 配置
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "user_profiles"
OUTPUT_PATH = OUTPUT_FOLDER / "user_profiles.csv"
INPUT_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_FILES_LOG = OUTPUT_FOLDER / "processed_files_record.txt"

# --- 静态辅助函数 (保持不变) ---
def is_core_user_static(followers, verified):
    if verified or followers > 10000:
        return True
    elif followers > 1000:
        return random.random() < 0.3
    else:
        return False

def process_file_worker(filepath):
    """Worker 进程：解析逻辑不变"""
    extracted_users = {}
    try:
        with open(filepath, 'rb') as f:
            for line in f:
                try:
                    data = JSON_LIB.loads(line)
                    targets = [
                        (data.get('authorContentPojo'), 'content_author'),
                        (data.get('authorCommentPojo'), 'comment_author'),
                        (data.get('authorCommentForwardPojo'), 'forward_author')
                    ]
                    for user_data, u_type in targets:
                        if not user_data: continue
                        uid = user_data.get('sjcjId')
                        if not uid: continue
                        uid = str(uid)
                        if uid in extracted_users: continue
                        
                        followers = user_data.get('sjcjFollowersCount', 0)
                        posts = user_data.get('sjcjStatusesCount', 0)
                        verified = user_data.get('sjcjVerified', False)
                        
                        user_dict = {
                            'user_id': uid,
                            'username': user_data.get('sjcjNickName', f'user_{uid}'),
                            'display_name': user_data.get('sjcjNickName', ''),
                            'gender': user_data.get('sjcjGender', 'unknown'),
                            'verified': verified,
                            'verified_type': user_data.get('sjcjVerifiedType', -1),
                            'bio': user_data.get('sjcjDescription', ''),
                            'followers_count': followers,
                            'following_count': user_data.get('sjcjFriendsCount', 0),
                            'posts_count': posts,
                            'favorites_count': user_data.get('sjcjFavouritesCount', 0),
                            'province': user_data.get('sjqxProvince', ''),
                            'city': user_data.get('sjqxCity', ''),
                            'location': user_data.get('sjcjLocation', ''),
                            'ip_location': user_data.get('sjcjIpLocation', ''),
                            'registration_time': user_data.get('sjcjRegistrationTime'),
                            'last_published': user_data.get('sjqxLastPublished'),
                            'source': user_data.get('sjqxSource', ''),
                            'source_mobile': user_data.get('sjqxSourceMobileV2', ''),
                            'profile_image_url': user_data.get('sjcjProfileImageUrl', ''),
                            'user_type': u_type,
                            'core_user': is_core_user_static(followers, verified)
                        }
                        extracted_users[uid] = user_dict
                except Exception:
                    continue
        return (filepath.name, list(extracted_users.values()))
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
        if OUTPUT_PATH.exists():
            print("⏳ Loading existing IDs...")
            # 优化：只读取必要的列
            df = pd.read_csv(OUTPUT_PATH, usecols=['user_id'], dtype={'user_id': str})
            self.existing_ids = set(df['user_id'])
            print(f"✅ Loaded {len(self.existing_ids)} IDs.")

    def run(self):
        # 1. 收集未处理文件
        files = [f for f in INPUT_FOLDER.glob('*.txt') if f.name not in self.processed_files]
        if not files:
            print("No new files.")
            return

        print(f"🚀 Processing {len(files)} files with {os.cpu_count()} cores...")
        
        pool = multiprocessing.Pool()
        
        total_new_users = 0
        batch_buffer = []
        
        # [新增] 待处理文件列表：用于记录当前 batch_buffer 对应哪些文件
        pending_files = [] 
        
        BATCH_SIZE = 1000
        write_header = not OUTPUT_PATH.exists()
        start_time = time.time()
        
        try:
            for filename, result in pool.imap_unordered(process_file_worker, files):
                if isinstance(result, str) and result.startswith("ERROR"):
                    print(f"❌ {filename}: {result}")
                    # 出错的文件我们不记录到 log，这样下次还会重试
                    continue
                
                # 收集当前文件里的新用户
                new_users_in_file = []
                for user in result:
                    uid = user['user_id']
                    if uid not in self.existing_ids:
                        self.existing_ids.add(uid)
                        new_users_in_file.append(user)
                
                batch_buffer.extend(new_users_in_file)
                
                # [核心修改] 不要立即写日志，而是加入待处理列表
                pending_files.append(filename)
                
                # 攒够一波数据，或者待确认的文件太多了，就执行写入
                if len(batch_buffer) >= BATCH_SIZE or len(pending_files) >= 50:
                    self._flush_data_and_logs(batch_buffer, pending_files, write_header)
                    
                    if batch_buffer: # 如果确实写入了数据，下次就不写 header 了
                        write_header = False
                    
                    total_new_users += len(batch_buffer)
                    print(f"⚡ Saved batch of {len(batch_buffer)}. Total new: {total_new_users}")
                    
                    # 清空缓冲区
                    batch_buffer = []
                    pending_files = []
            
            # 循环结束后，处理剩余的数据
            if batch_buffer or pending_files:
                self._flush_data_and_logs(batch_buffer, pending_files, write_header)
                total_new_users += len(batch_buffer)
                
        finally:
            pool.close()
            pool.join()
            
        print(f"Done! Added {total_new_users} users in {time.time()-start_time:.2f}s")

    def _flush_data_and_logs(self, data_buffer, file_list, write_header):
        """原子性操作：先存数据，再记日志"""
        
        # 1. 存数据 (如果有数据)
        if data_buffer:
            df = pd.DataFrame(data_buffer)
            # 显式转换列类型，避免警告
            if 'user_id' in df.columns:
                df['user_id'] = df['user_id'].astype(str)
                
            df.to_csv(OUTPUT_PATH, mode='a', header=write_header, index=False, encoding='utf-8', escapechar='\\')
        
        # 2. 只有数据写入成功（没报错），才更新日志
        if file_list:
            with open(PROCESSED_FILES_LOG, 'a', encoding='utf-8') as f:
                for fname in file_list:
                    f.write(f"{fname}\n")

if __name__ == "__main__":
    ParallelProcessor().run()