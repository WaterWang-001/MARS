import json
import pandas as pd
import numpy as np
from pathlib import Path
import random
from datetime import datetime
import logging
import sys
import os
from tqdm import tqdm

# 配置路径（相对于项目根目录）
PROJECT_ROOT = Path(__file__).parent.parent
MARS_ROOT = PROJECT_ROOT / "MARS"
OUTPUT_FOLDER = MARS_ROOT / "data" / "user_profiles"
OUTPUT_PATH = OUTPUT_FOLDER / "user_profiles.csv"
INPUT_FOLDER = PROJECT_ROOT / "data" / "raw"
LOG_FOLDER = PROJECT_ROOT / "logs"

# 记录已处理文件的记录表路径
PROCESSED_FILES_LOG = OUTPUT_FOLDER / "processed_files_record.txt"

# 配置 Logging
LOG_FOLDER.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FOLDER / "profile_generation.log", encoding='utf-8')
    ]
)

class WeiboToOASISConverter:
    """将微博数据转换为OASIS平台格式的转换器 (支持断点续传)"""
    
    def __init__(self):
        # 1. 内存中只存储 UserID 集合，用于去重，大大节省内存
        self.existing_user_ids = set()
        
        # 2. 加载已处理过的文件列表
        self.processed_filenames = self._load_processed_filenames()
        
        # 3. 如果输出文件存在，预加载已有的 UserID，防止断点续传时产生重复数据
        self._load_existing_user_ids()
        
        self.total_new_users_count = 0 # 本次运行新增用户数

    def _load_processed_filenames(self):
        """加载已处理文件记录"""
        if not PROCESSED_FILES_LOG.exists():
            return set()
        with open(PROCESSED_FILES_LOG, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())

    def _load_existing_user_ids(self):
        """预加载 CSV 中已存在的 UserID"""
        if not OUTPUT_PATH.exists():
            return
        
        print("🔄 正在加载已有 CSV 数据以进行去重...")
        try:
            # 只读取 user_id 列，速度快且省内存
            df = pd.read_csv(OUTPUT_PATH, usecols=['user_id'], dtype={'user_id': str})
            self.existing_user_ids = set(df['user_id'].unique())
            print(f"✅ 已加载 {len(self.existing_user_ids)} 个现有用户 ID")
            logging.info(f"Loaded {len(self.existing_user_ids)} existing user IDs.")
        except Exception as e:
            logging.error(f"Error loading existing CSV: {e}")
            print(f"⚠️ 加载旧 CSV 失败，可能会导致部分重复: {e}")

    def mark_file_as_processed(self, filename):
        """将文件标记为已处理"""
        with open(PROCESSED_FILES_LOG, 'a', encoding='utf-8') as f:
            f.write(f"{filename}\n")
        self.processed_filenames.add(filename)

    def extract_users_from_json(self, json_data):
        """从JSON数据中提取用户信息"""
        users = []
        if 'authorContentPojo' in json_data:
            users.append(self.parse_user(json_data['authorContentPojo'], 'content_author'))
        if 'authorCommentPojo' in json_data:
            users.append(self.parse_user(json_data['authorCommentPojo'], 'comment_author'))
        if 'authorCommentForwardPojo' in json_data:
            users.append(self.parse_user(json_data['authorCommentForwardPojo'], 'forward_author'))
        return users
    
    def parse_user(self, user_data, user_type):
        """解析单个用户数据"""
        user_id = user_data.get('sjcjId', '')
        if not user_id:
            return None
        
        # 这里的去重逻辑移到 process_single_file 中统一处理
        
        user = {
            'user_id': str(user_id), # 确保 ID 是字符串
            'username': user_data.get('sjcjNickName', f'user_{user_id}'),
            'display_name': user_data.get('sjcjNickName', ''),
            'gender': self.map_gender(user_data.get('sjcjGender', 'unknown')),
            'verified': user_data.get('sjcjVerified', False),
            'verified_type': user_data.get('sjcjVerifiedType', -1),
            'bio': user_data.get('sjcjDescription', ''),
            'followers_count': user_data.get('sjcjFollowersCount', 0),
            'following_count': user_data.get('sjcjFriendsCount', 0),
            'posts_count': user_data.get('sjcjStatusesCount', 0),
            'favorites_count': user_data.get('sjcjFavouritesCount', 0),
            'province': user_data.get('sjqxProvince', ''),
            'city': user_data.get('sjqxCity', ''),
            'location': user_data.get('sjcjLocation', ''),
            'ip_location': user_data.get('sjcjIpLocation', ''),
            'registration_time': self.format_timestamp(user_data.get('sjcjRegistrationTime')),
            'last_published': self.format_timestamp(user_data.get('sjqxLastPublished')),
            'source': user_data.get('sjqxSource', ''),
            'source_mobile': user_data.get('sjqxSourceMobileV2', ''),
            'profile_image_url': user_data.get('sjcjProfileImageUrl', ''),
            'user_type': user_type,
            'influence_score': self.calculate_influence_score(user_data),
            'core_user': self.is_core_user(user_data)
        }
        return user
    
    def map_gender(self, gender):
        gender_map = {'m': 'male', 'f': 'female'}
        return gender_map.get(gender, 'unknown')
    
    def format_timestamp(self, timestamp):
        if not timestamp:
            return None
        try:
            return datetime.fromtimestamp(timestamp/1000).isoformat()
        except:
            return None
    
    def calculate_influence_score(self, user_data):
        followers = user_data.get('sjcjFollowersCount', 0)
        posts = user_data.get('sjcjStatusesCount', 0)
        verified = user_data.get('sjcjVerified', False)
        score = np.log1p(followers) * 0.5 + np.log1p(posts) * 0.3
        if verified:
            score *= 1.5
        score = min(100, score * 5)
        return round(score, 2)
    
    def is_core_user(self, user_data):
        followers = user_data.get('sjcjFollowersCount', 0)
        verified = user_data.get('sjcjVerified', False)
        if verified or followers > 10000:
            return True
        elif followers > 1000:
            return random.random() < 0.3
        else:
            return False
    
    def append_batch_to_csv(self, new_users_dict):
        """将当前批次的新用户追加到 CSV"""
        if not new_users_dict:
            return

        df = pd.DataFrame(new_users_dict.values())
        
        # 如果文件不存在，需要写 header；如果存在，不需要写 header
        header = not OUTPUT_PATH.exists()
        
        try:
            # === 修改了下面这一行，增加了 escapechar='\\' ===
            df.to_csv(OUTPUT_PATH, mode='a', header=header, index=False, encoding='utf-8', escapechar='\\')
            self.total_new_users_count += len(df)
        except Exception as e:
            logging.error(f"Failed to append to CSV: {e}")
            print(f"❌ 写入 CSV 失败: {e}")
    def process_single_file(self, filepath, file_pbar):
        filename = filepath.name
        
        # 检查是否已处理
        if filename in self.processed_filenames:
            logging.info(f"Skipping {filename} (already processed)")
            file_pbar.set_description(f"跳过已处理: {filename}")
            return

        logging.info(f"Processing file: {filename}")
        file_pbar.set_description(f"正在处理: {filename}")
        
        # 当前文件的临时数据存储
        current_file_users = {}
        line_count = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                with tqdm(f, unit=" line", leave=False, desc="Lines") as line_pbar:
                    for line_num, line in enumerate(line_pbar, 1):
                        line_count += 1
                        try:
                            json_data = json.loads(line.strip())
                            users = self.extract_users_from_json(json_data)
                            
                            for user in users:
                                if user:
                                    uid = user['user_id']
                                    # 关键去重逻辑：
                                    # 1. 不在全局已存 ID 中
                                    # 2. 不在当前文件暂存中 (防止同一文件内重复)
                                    if uid not in self.existing_user_ids and uid not in current_file_users:
                                        current_file_users[uid] = user
                            
                            if line_count % 500 == 0:
                                line_pbar.set_postfix({
                                    "New Users": len(current_file_users)
                                })

                        except json.JSONDecodeError:
                            pass 
                        except Exception as e:
                            pass
            
            # === 文件处理完毕，执行保存逻辑 ===
            
            # 1. 写入 CSV
            if current_file_users:
                self.append_batch_to_csv(current_file_users)
                
                # 2. 更新内存中的 ID 集合
                self.existing_user_ids.update(current_file_users.keys())
            
            # 3. 记录文件已处理
            self.mark_file_as_processed(filename)
            
            logging.info(f"Done {filename}: Extracted {len(current_file_users)} new users")
            
        except Exception as e:
            logging.error(f"Failed to open {filename}: {e}")

    def process_all_files(self):
        txt_files = list(INPUT_FOLDER.glob('*.txt'))
        
        if not txt_files:
            logging.warning(f"No .txt files found in {INPUT_FOLDER}")
            return
        
        # 过滤掉已处理的文件，只统计剩余的用于显示进度
        remaining_files = [f for f in txt_files if f.name not in self.processed_filenames]
        skipped_count = len(txt_files) - len(remaining_files)
        
        print(f"📂 总文件数: {len(txt_files)}")
        print(f"⏭️ 跳过已处理: {skipped_count}")
        print(f"🚀 本次需处理: {len(remaining_files)}")
        
        if not remaining_files:
            print("✨ 所有文件都已处理完毕！")
            return

        # 只对剩下的文件进行循环
        with tqdm(total=len(remaining_files), unit="file", desc="总体进度") as pbar:
            for filepath in remaining_files:
                self.process_single_file(filepath, pbar)
                pbar.update(1)
        
        print(f"\n✨ 本次运行新增用户: {self.total_new_users_count}")
        print(f"📊 库中累计总用户: {len(self.existing_user_ids)}")

    def generate_report(self):
        """生成简易报告（基于最终CSV统计，比较耗时，可选）"""
        # 由于现在是增量写入，内存里没有所有数据，所以最后不再全量读取生成详细报告
        # 仅生成简单的计数报告
        report = f"""
    {'='*60}
    OASIS Data Conversion Report (Incremental Mode)
    {'='*60}
    Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Input: {INPUT_FOLDER}
    Output: {OUTPUT_PATH}
    
    Total Unique Users in DB: {len(self.existing_user_ids)}
    New Users Added This Run: {self.total_new_users_count}
    {'='*60}
    """
        return report
    
    def save_report(self):
        report = self.generate_report()
        report_path = OUTPUT_FOLDER / 'conversion_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print("\n📝 简单报告:")
        print(report)
    def save_to_csv(self):
        """
        [兼容性修复] 外部脚本会调用此方法。
        由于数据已经在 process_single_file 中增量写入了，这里不需要再次保存。
        """
        logging.info("save_to_csv called. Data has already been saved incrementally.")
        print("✅ [兼容性提示] 数据已在处理过程中增量保存完毕。")
        
        # 返回 None 或者一个空的 DataFrame 以防止外部接收返回值后报错
        # 如果外部代码非常依赖返回的 df 来做后续统计，这里可能需要重新 read_csv，
        # 但考虑到内存优化，通常建议直接返回 None。
        return None


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    converter = WeiboToOASISConverter()
    
    converter.process_all_files()
    converter.save_report()
    
    print(f"✅ 任务结束")

if __name__ == "__main__":
    main()