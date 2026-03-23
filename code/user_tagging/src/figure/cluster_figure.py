import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import os

# === 配置 ===
DB_PATH = 'MARS_result/data/benchmark_coldstart_hash/bench_state_buffer.db'
OUTPUT_PNG = 'MARS_result/data/stats/plots/cluster_visualization_mpl.png'
MAX_SAMPLES = 5000 
TOP_N_CLUSTERS = 1000  # 只给前 N 个大类上色，其他的变灰

def deserialize_vec(blob):
    if not blob: return None
    return np.frombuffer(blob, dtype=np.float32)

def visualize_clusters():
    # 1. 检查数据库
    if not os.path.exists(DB_PATH):
        print(f"❌ DB not found: {DB_PATH}")
        return

    print("🚀 Loading data from DB...")
    conn = sqlite3.connect(DB_PATH)
    
    # 2. 读取数据
    query = """
    SELECT tag_text, mapped_tag, embedding
    FROM global_tag_registry 
    WHERE status = 1 AND embedding IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("⚠️ No clustered data found.")
        return

    print(f"📊 Total tags loaded: {len(df)}")

    # 3. 预处理
    df['vec'] = df['embedding'].apply(deserialize_vec)
    
    # 采样
    if len(df) > MAX_SAMPLES:
        print(f"✂️ Sampling {MAX_SAMPLES} data points...")
        df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

    # 4. t-SNE 降维
    print("🧠 Running t-SNE (High Dim -> 2D)...")
    matrix = np.stack(df['vec'].values)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    projections = tsne.fit_transform(matrix)
    
    df['x'] = projections[:, 0]
    df['y'] = projections[:, 1]

    # 5. 准备绘图数据 (区分 Top 类簇 和 Others)
    # 统计前 N 个大类
    top_clusters = df['mapped_tag'].value_counts().head(TOP_N_CLUSTERS).index.tolist()
    
    # 创建画布
    print("🎨 Generating Matplotlib Plot...")
    plt.figure(figsize=(16, 10), dpi=300) # 高分辨率
    
    # 设置样式 (类似 Seaborn 的简洁风格)
    plt.style.use('fast') 
    
    # A. 先画背景层：Others (灰色)
    others_df = df[~df['mapped_tag'].isin(top_clusters)]
    plt.scatter(
        others_df['x'], 
        others_df['y'], 
        c='#E0E0E0',  # 浅灰色
        s=10, 
        alpha=0.5, 
        label='Others',
        edgecolors='none',
        zorder=1 # 放在最底层
    )

    # B. 再画前景层：Top Clusters (彩色)
    # 获取一个颜色映射表 (Tab20 适合分类较多的情况)
    cmap = plt.get_cmap('tab20')
    
    for i, cluster_name in enumerate(top_clusters):
        cluster_data = df[df['mapped_tag'] == cluster_name]
        color = cmap(i % 20) # 循环取色
        
        plt.scatter(
            cluster_data['x'], 
            cluster_data['y'], 
            color=color,
            s=20, # 稍微大一点
            alpha=0.9,
            label=cluster_name, # 用于图例
            edgecolors='white', # 加个白边让点更清晰
            linewidth=0.3,
            zorder=2 # 放在上层
        )

    # 6. 图表修饰
    plt.title(f"Tag Semantic Clusters (t-SNE) - Top {TOP_N_CLUSTERS} Topics", fontsize=16, fontweight='bold', pad=20)
    
    # 移除坐标轴刻度 (t-SNE 坐标数值无意义)
    plt.xticks([])
    plt.yticks([])
    
    # 移除边框 spines
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # # 添加图例 (放在图外侧，以免遮挡数据)
    # plt.legend(
    #     bbox_to_anchor=(1.02, 1), 
    #     loc='upper left', 
    #     borderaxespad=0, 
    #     title="Topics",
    #     frameon=False,
    #     fontsize=10
    # )

    # 7. 保存
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, bbox_inches='tight') # bbox_inches='tight' 防止图例被裁切
    print(f"✅ Visualization saved to: {os.path.abspath(OUTPUT_PNG)}")
    
    # 关闭画布释放内存
    plt.close()

if __name__ == "__main__":
    visualize_clusters()