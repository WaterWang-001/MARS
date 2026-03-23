import pandas as pd
import matplotlib.pyplot as plt
import os

# === 配置 ===
# 你的 CSV 文件路径 (请确保这是你刚刚生成的那个统计文件)
CSV_PATH = 'MARS_result/data/output/2025-06-14/sampled_tag_stats_2025-06-14.csv'
OUTPUT_DIR = '/remote-home/JuelinW/oasis_project/MARS_result/data/stats/plots'

# 建议的阈值线
THRESHOLDS = [100, 50]

def plot_distribution():
    if not os.path.exists(CSV_PATH):
        print(f"❌ File not found: {CSV_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📖 Loading data from {CSV_PATH}...")
    try:
        # 读取数据 (假设列名为 Rank, Tag, Total_Freq, Unique_Users, ...)
        # 如果你的 CSV 列名不一样，请在这里调整
        df = pd.read_csv(CSV_PATH)
        
        # 确保按频次降序
        if 'Total_Freq' in df.columns:
            target_col = 'Total_Freq'
        elif 'Frequency' in df.columns:
            target_col = 'Frequency'
        else:
            print("❌ Cannot find Frequency column!")
            return

        df = df.sort_values(by=target_col, ascending=False).reset_index(drop=True)
        
        frequencies = df[target_col].values
        ranks = range(1, len(frequencies) + 1)
        
        print(f"📊 Total Tags: {len(frequencies)}")
        print(f"   Max Freq: {frequencies.max()}")
        print(f"   Min Freq: {frequencies.min()}")

        # ==========================================
        # 图 1: 整体长尾分布 (Log-Log Scale)
        # ==========================================
        plt.figure(figsize=(12, 6))
        plt.plot(ranks, frequencies, color='#1f77b4', linewidth=1.5, label='Tag Frequency')
        
        # 画阈值线
        colors = ['red', 'orange']
        for i, th in enumerate(THRESHOLDS):
            plt.axhline(y=th, color=colors[i], linestyle='--', label=f'Threshold = {th}')
            
            # 计算切掉的比例
            cut_count = (df[target_col] > th).sum()
            cut_ratio = cut_count / len(df) * 100
            print(f"   ✂️ Threshold {th}: cuts top {cut_count} tags ({cut_ratio:.2f}%)")
            
            plt.text(len(df)*0.05, th + (th*0.1), f'Threshold: {th}\n(Top {cut_count} tags)', color=colors[i])

        plt.xscale('log') # 对数坐标轴能更好展示长尾
        plt.yscale('log')
        plt.title(f'Tag Frequency Distribution (Log-Log) - {os.path.basename(CSV_PATH)}')
        plt.xlabel('Rank (Log)')
        plt.ylabel('Frequency (Log)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        save_path_1 = os.path.join(OUTPUT_DIR, 'tag_distribution_log.png')
        plt.savefig(save_path_1, dpi=150)
        print(f"✅ Saved global plot to: {save_path_1}")
        plt.close()

        # ==========================================
        # 图 2: 阈值决策放大图 (Linear Scale, Zoomed)
        # ==========================================
        # 只看前 2000 名 或者 频次 > 20 的部分，关注腰部截断点
        zoom_df = df[df[target_col] > 20]
        if len(zoom_df) > 2000:
            zoom_df = zoom_df.head(2000)
            
        zoom_freqs = zoom_df[target_col].values
        zoom_ranks = range(1, len(zoom_freqs) + 1)

        plt.figure(figsize=(12, 6))
        plt.plot(zoom_ranks, zoom_freqs, color='#2ca02c', linewidth=2)
        
        for i, th in enumerate(THRESHOLDS):
            plt.axhline(y=th, color=colors[i], linestyle='--', linewidth=2, label=f'Threshold = {th}')

        plt.title('Threshold Decision View (Linear Scale, Top 2000)')
        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path_2 = os.path.join(OUTPUT_DIR, 'tag_distribution_linear_zoom.png')
        plt.savefig(save_path_2, dpi=150)
        print(f"✅ Saved zoom plot to: {save_path_2}")
        plt.close()

    except Exception as e:
        print(f"❌ Plotting Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_distribution()