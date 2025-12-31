import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import sys

# --- ⚙️ 配置区域 ---
# 项目根目录 (根据你的实际结构调整)
PROJECT_ROOT = Path("MARS") 

# 原始数据根目录 (存放分日期的文件夹)
INPUT_ROOT = PROJECT_ROOT / "data" / "output"

# 最终合并的大表路径
FINAL_OUTPUT_FILE = PROJECT_ROOT / "data" / "merged" / "user_profiles_merged.csv"

# 📅 [关键] 指定要合并的日期范围
# 只有在这个范围内的文件夹会被读取并合并进主表
START_DATE = "2025-06-19"
END_DATE   = "2025-06-19"

# --------------------

def get_target_files(input_root, start_date_str, end_date_str):
    """
    根据日期范围筛选需要处理的 CSV 文件
    """
    csv_files = []
    print(f"🔍 正在扫描 {input_root} 下 {start_date_str} 至 {end_date_str} 的数据...")
    
    if not input_root.exists():
        print(f"❌ 目录不存在: {input_root}")
        return []

    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"❌ 日期格式配置错误: {e}")
        return []

    # 遍历 output 下的子文件夹
    for folder in input_root.iterdir():
        if folder.is_dir():
            try:
                # 1. 解析文件夹日期
                folder_date = datetime.strptime(folder.name, "%Y-%m-%d").date()
                
                # 2. 只有在范围内的才处理
                if start_dt <= folder_date <= end_dt:
                    target_file = folder / "user_profiles.csv"
                    if target_file.exists():
                        csv_files.append((folder_date, target_file))
            except ValueError:
                continue # 跳过非日期命名的文件夹
    
    # 按日期排序 (旧 -> 新)，确保后续处理顺序正确
    csv_files.sort(key=lambda x: x[0])
    return csv_files

def merge_profiles():
    start_time = time.time()
    
    # 1. 准备新数据列表
    new_files = get_target_files(INPUT_ROOT, START_DATE, END_DATE)
    if not new_files:
        print("⚠️ 指定范围内没有找到任何 user_profiles.csv 文件。")
        return

    all_dfs = []
    
    # 2. 读取现有的主表 (Base)

    if FINAL_OUTPUT_FILE.exists():
        print(f"📖 读取现有主数据库: {FINAL_OUTPUT_FILE.name}")
        try:
            # dtype={'user_id': str} 防止 ID 变科学计数法
            existing_df = pd.read_csv(FINAL_OUTPUT_FILE, dtype={'user_id': str})
            
            # 给现有数据一个"极小"的日期，保证它能被任何新导入的日期覆盖
            existing_df['__merge_date__'] = pd.Timestamp.min
            
            all_dfs.append(existing_df)
            print(f"   -> 当前库中有: {len(existing_df)} 用户")
        except Exception as e:
            print(f"❌ 读取主库失败 (将重新创建): {e}")

    # 3. 读取指定日期范围内的新文件 (Updates)
    print(f"🚀 开始加载 {len(new_files)} 个新文件...")
    
    new_records_count = 0
    for file_date, file_path in new_files:
        try:
            df = pd.read_csv(file_path, dtype={'user_id': str})
            
            # 关键：将文件夹日期赋予数据，用于排序去重
            df['__merge_date__'] = pd.to_datetime(file_date)
            
            all_dfs.append(df)
            new_records_count += len(df)
            print(f"   + [{file_date}] 导入 {len(df)} 条数据")
        except Exception as e:
            print(f"⚠️ 读取错误 {file_path}: {e}")

    if not all_dfs:
        return

    # 4. 合并 (Merge)
    print("\n🔄 正在合并与去重...")
    # 拼接所有数据
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    total_raw = len(full_df)

    # 5. 核心去重逻辑
    # A. 按 __merge_date__ 升序排列 (旧数据在前，新数据在后)
    #    Existing DB (min date) -> 2025-06-14 -> 2025-06-15
    full_df.sort_values(by='__merge_date__', ascending=True, inplace=True)
    
    # B. 根据 user_id 去重，保留最后一条 (keep='last')
    #    因为我们刚刚排过序，"最后一条"就是日期最新的那条
    final_df = full_df.drop_duplicates(subset=['user_id'], keep='last')
    
    # C. 移除辅助列
    final_df = final_df.drop(columns=['__merge_date__'])

    # 6. 统计与保存
    total_final = len(final_df)
    overwritten = total_raw - total_final
    
    print("\n📊 合并报告:")
    print(f"   --------------------------------")
    print(f"   📥 总处理行数: {total_raw}")
    print(f"   🆕 新增/更新源: {new_records_count}")
    print(f"   👤 最终用户数: {total_final}")
    print(f"   ♻️ 覆盖旧数据: {overwritten}")
    print(f"   --------------------------------")

    print(f"💾 正在写入: {FINAL_OUTPUT_FILE}")
    FINAL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # utf-8-sig 保证 Excel 打开中文不乱码
    final_df.to_csv(FINAL_OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"✅ Merge 完成! 耗时: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    merge_profiles()