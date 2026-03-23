import sqlite3
import numpy as np
import json
import logging
import concurrent.futures
from collections import defaultdict
from sklearn.cluster import HDBSCAN 
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

class TagClusterManager:
    def __init__(self, db_path, api_client):
        self.db_path = db_path
        self.api_client = api_client
        
        # [归并参数]
        self.MERGE_THRESHOLD = 0.8
        self.EMBED_BATCH_SIZE = 64
        
        # [HDBSCAN 参数 - 保持高粘性]
        # 建议：如果还是太碎，可以将 min_cluster_size 调至 3
        self.MIN_CLUSTER_SIZE = 3
        self.MIN_SAMPLES = 2
        self.CLUSTER_SELECTION_EPSILON = 0.1
        
        # [新增：噪声回收阈值]
        # 只要噪声点离某个簇中心的相似度 > 0.55，就强行归类
        self.NOISE_RECOVERY_THRESHOLD = 0.55
        
        # [PCA 参数]
        self.PCA_ENABLE_THRESHOLD = 1000 
        self.PCA_TARGET_DIM = 64

        # [并发参数]
        self.MAX_WORKERS = 20
        
        # [L1领域分类列表]
        self.L1_CATEGORY_LIST = [
            "游戏", "电竞", "动漫", "体育", "运动", "科技", "AI", "数码", 
            "汽车", "美妆", "时尚", "母婴", "女性成长", "美食", "旅游", 
            "摄影", "艺术", "房产", "家居", "健康", "科普", "历史", 
            "校园", "教育", "星座", "萌宠", "幽默", "情感", "颜值", 
            "财经", "军事", "法律", "时事", "公益", "同城", "电影", 
            "电视剧", "综艺", "音乐", "娱评", "读书", "其他"
        ]
        
    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _deserialize_vec(self, blob):
        if not blob: return None
        return np.frombuffer(blob, dtype=np.float32)

    def _serialize_vec(self, vec):
        return vec.astype(np.float32).tobytes()

    def _ensure_category_schema(self, cursor, conn):
        for col in ["mapped_tag", "category_l1"]:
            try:
                cursor.execute(f"ALTER TABLE global_tag_registry ADD COLUMN {col} TEXT")
                print(f"✅ [Schema] Added column: {col}")
            except sqlite3.OperationalError:
                pass
        conn.commit()

    def _perform_core_clustering_only(self, cursor, conn):
        """
        Phase 1: 纯数学聚类 (Embedding -> Merge -> HDBSCAN -> Noise Recovery)
        """
        # ================= A. 加载数据 =================
        cursor.execute("SELECT tag_id, tag_text, embedding FROM global_tag_registry WHERE status = 0")
        new_rows = cursor.fetchall()
        
        pending_tags = []
        for r in new_rows:
            vec = self._deserialize_vec(r['embedding'])
            if vec is not None:
                pending_tags.append({'tag_id': r['tag_id'], 'text': r['tag_text'], 'vec': vec})
        
        if not pending_tags:
            print("✅ [Cluster] No new tags to process.")
            return

        print(f"📊 [Cluster] Processing {len(pending_tags)} tags...")

        # 加载现有簇中心 (用于归并)
        cursor.execute("SELECT mapped_tag, embedding, category_l1 FROM global_tag_registry WHERE status = 1 AND mapped_tag IS NOT NULL")
        existing_rows = cursor.fetchall()
        
        cluster_centroids = {} 
        cluster_meta = {} # 记录现有簇的 L1 分类
        cluster_groups = defaultdict(list)
        
        for r in existing_rows:
            vec = self._deserialize_vec(r['embedding'])
            if vec is not None:
                cluster_groups[r['mapped_tag']].append(vec)
                if r['mapped_tag'] not in cluster_meta and r['category_l1']:
                     cluster_meta[r['mapped_tag']] = r['category_l1']
        
        # 计算旧簇的质心
        for label, vecs in cluster_groups.items():
            cluster_centroids[label] = np.mean(vecs, axis=0)

        # ================= B. 归并逻辑 (Merge) =================
        
        merge_updates = []
        unmatched_tags = [] 
        
        if cluster_centroids:
            centroid_labels = list(cluster_centroids.keys())
            centroid_matrix = np.array([cluster_centroids[k] for k in centroid_labels])
            
            # 批量计算相似度 (优化：如果pending太多，可以分批，这里假设内存够)
            pending_vecs = np.array([t['vec'] for t in pending_tags])
            
            # Matrix: (N_pending, N_centroids)
            if len(pending_vecs) > 0 and len(centroid_matrix) > 0:
                sims_matrix = cosine_similarity(pending_vecs, centroid_matrix)
                
                for i, tag in enumerate(pending_tags):
                    sims = sims_matrix[i]
                    best_idx = np.argmax(sims)
                    
                    if sims[best_idx] >= self.MERGE_THRESHOLD:
                        target_label = centroid_labels[best_idx]
                        target_l1 = cluster_meta.get(target_label, None)
                        merge_updates.append((target_label, target_l1, 1, tag['tag_id']))
                    else:
                        unmatched_tags.append(tag)
            else:
                 unmatched_tags = pending_tags
        else:
            unmatched_tags = pending_tags

        if merge_updates:
            cursor.executemany("""
                UPDATE global_tag_registry 
                SET mapped_tag=?, category_l1=?, status=? 
                WHERE tag_id=?
            """, merge_updates)
            conn.commit()
            print(f"🔗 [Merge] Merged {len(merge_updates)} tags into existing clusters.")

        # ================= C. 新聚类逻辑 (HDBSCAN) =================
        if not unmatched_tags:
            return

        print(f"🧩 [Cluster] Clustering {len(unmatched_tags)} remaining tags with HDBSCAN...")
        
        X = np.array([t['vec'] for t in unmatched_tags]).astype('float64')
        
        if len(unmatched_tags) > self.PCA_ENABLE_THRESHOLD and X.shape[1] > self.PCA_TARGET_DIM:
            pca = PCA(n_components=self.PCA_TARGET_DIM, random_state=42)
            X_cluster = pca.fit_transform(X)
        else:
            X_cluster = X
            
        clusterer = HDBSCAN(
            min_cluster_size=self.MIN_CLUSTER_SIZE,
            min_samples=self.MIN_SAMPLES,
            cluster_selection_epsilon=self.CLUSTER_SELECTION_EPSILON,
            metric='euclidean',
            cluster_selection_method='eom' # 或者试试 'leaf'
        )
        labels = clusterer.fit_predict(X_cluster)
        
        new_updates = []
        
        # 临时存储新簇的 Tag 对象，用于后续计算新簇质心
        temp_clusters = defaultdict(list) 
        noise_candidates = [] # 待回收的噪声点

        for i, label in enumerate(labels):
            tag_obj = unmatched_tags[i]
            if label == -1:
                noise_candidates.append(tag_obj)
            else:
                temp_clusters[label].append(tag_obj)
        
        # 1. 确定新簇的名称，并准备写入 DB
        # 同时计算新簇的质心，加入到 "可选目标池" 中
        new_cluster_centroids = {}
        
        for label_id, group in temp_clusters.items():
            # 临时命名
            temp_label = min([t['text'] for t in group], key=len)
            
            # 计算该新簇的质心
            group_vecs = [t['vec'] for t in group]
            new_cluster_centroids[temp_label] = np.mean(group_vecs, axis=0)
            
            for t_obj in group:
                new_updates.append((temp_label, None, 1, t_obj['tag_id']))

        # ================= D. 噪声回收 (Noise Recovery) =================
        # 逻辑：将 noise_candidates 与 (Old Clusters + New Clusters) 的质心进行对比
        
        recovered_count = 0
        if noise_candidates and (cluster_centroids or new_cluster_centroids):
            print(f"🚑 [Recovery] Attempting to recover {len(noise_candidates)} noise tags...")
            
            # 合并所有可用的簇中心 (Old + New)
            # map: label_name -> centroid_vector
            all_centroids_map = {**cluster_centroids, **new_cluster_centroids}
            all_labels = list(all_centroids_map.keys())
            all_matrix = np.array([all_centroids_map[k] for k in all_labels])
            
            noise_vecs = np.array([t['vec'] for t in noise_candidates])
            
            # 计算相似度矩阵: (N_noise, N_all_clusters)
            noise_sims = cosine_similarity(noise_vecs, all_matrix)
            
            for i, tag_obj in enumerate(noise_candidates):
                sims = noise_sims[i]
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                
                if best_score >= self.NOISE_RECOVERY_THRESHOLD:
                    target_label = all_labels[best_idx]
                    # 如果归属到旧簇，继承L1；如果是新簇，L1为None等待LLM
                    target_l1 = cluster_meta.get(target_label, None)
                    
                    new_updates.append((target_label, target_l1, 1, tag_obj['tag_id']))
                    recovered_count += 1
                else:
                    # 彻底没救了，保持 status=0 (下次再试) 或者 status=1 但 mapped_tag=NULL (标记为孤儿)
                    # 这里保持不更新 (status=0)，或者你可以选择标记为 'Noise'
                    pass

        # 执行批量更新 (新簇成员 + 回收的噪声)
        if new_updates:
            cursor.executemany("""
                UPDATE global_tag_registry 
                SET mapped_tag=?, category_l1=?, status=? 
                WHERE tag_id=?
            """, new_updates)
            conn.commit()
            
            total_clustered = len(temp_clusters)
            total_tags_saved = len(new_updates)
            real_noise = len(noise_candidates) - recovered_count
            
            print(f"💾 [Cluster] Created {total_clustered} new clusters.")
            print(f"✨ [Recovery] Recovered {recovered_count} tags from noise.")
            print(f"📉 [Result] Total saved: {total_tags_saved}. Remaining Noise: {real_noise}")

    def _perform_llm_labeling_only(self, cursor, conn):
        """
        Phase 2: LLM 标注 (保持不变)
        """
        print("🤖 [Label] Scanning for clusters needing L1 categorization...")
        
        cursor.execute("""
            SELECT mapped_tag 
            FROM global_tag_registry 
            WHERE status=1 
              AND mapped_tag IS NOT NULL 
              AND (category_l1 IS NULL OR category_l1 = '')
            GROUP BY mapped_tag
        """)
        
        clusters_to_process = [row[0] for row in cursor.fetchall()]
        if not clusters_to_process:
            print("✅ [Label] All clusters are already categorized.")
            return

        print(f"📊 [Label] Found {len(clusters_to_process)} clusters pending LLM processing.")
        
        tasks = []
        for cluster_name in clusters_to_process:
            cursor.execute("SELECT tag_text FROM global_tag_registry WHERE mapped_tag = ? LIMIT 20", (cluster_name,))
            sample_tags = [r[0] for r in cursor.fetchall()]
            if len(sample_tags) < 1: continue 
            tasks.append((cluster_name, sample_tags))
        
        if not tasks: return

        print(f"🚀 [Label] Starting concurrent LLM inference for {len(tasks)} clusters...")
        
        processed_count = 0
        total_count = len(tasks)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_to_name = {
                executor.submit(self._generate_cluster_info_with_llm, tags): name 
                for name, tags in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_name):
                cluster_name = future_to_name[future]
                try:
                    info = future.result()
                    new_label = info['label']
                    l1_category_str = info['l1_category']
                    
                    cursor.execute("""
                        UPDATE global_tag_registry 
                        SET mapped_tag = ?, category_l1 = ?
                        WHERE mapped_tag = ?
                    """, (new_label, l1_category_str, cluster_name))
                    conn.commit()
                    processed_count += 1
                    print(f"   [{processed_count}/{total_count}] ✨ Saved: [{cluster_name}] -> 🏷️ [{new_label}] ({l1_category_str})")
                except Exception as e:
                    print(f"❌ [Error] Failed to process/save cluster {cluster_name}: {e}")
        print("✅ [Label] Done.")

    def _generate_cluster_info_with_llm(self, tags):
        # 保持不变
        if len(tags) == 1: 
            return {"label": tags[0], "l1_category": "其他"}
        
        tags_str = json.dumps(tags[:20], ensure_ascii=False)
        cat_list_str = ", ".join(self.L1_CATEGORY_LIST)
        
        prompt = f"""# Role
        Social Media Taxonomy Expert.
        
        # Input
        Tags: {tags_str}
        Allowed Categories: [{cat_list_str}]

        # Task
        1. **Label**: Name this cluster with a concise Entity/Concept (1-3 words).
        2. **L1 Category**: Choose the most relevant category from the list.

        # Output (Strict JSON)
        {{
          "label": "String",
          "l1_category": "String"
        }}
        """
        default_result = {"label": min(tags, key=len), "l1_category": "其他"}
        try:
            response = self.api_client.call_api(prompt, temperature=0.0)
            if isinstance(response, str):
                clean_str = response.replace("```json", "").replace("```", "").strip()
                response = json.loads(clean_str)
            label = response.get("label", "").strip()
            raw_cats = response.get("l1_category", "")
            if isinstance(raw_cats, list):
                l1_category_str = ",".join([str(c) for c in raw_cats])
            else:
                l1_category_str = str(raw_cats)

            if not label: return default_result
            return {"label": label, "l1_category": l1_category_str}
        except Exception as e:
            return default_result

    def _fill_missing_embeddings(self, cursor, conn):
        # 保持不变
        print("🔌 [Embed] Checking for missing embeddings...")
        cursor.execute("SELECT tag_id, tag_text FROM global_tag_registry WHERE status = 0 AND embedding IS NULL")
        rows = cursor.fetchall()
        if not rows: return
        total_missing = len(rows)
        print(f"   -> Found {total_missing} tags needing embedding.")
        updates = []
        batch_texts = []
        batch_ids = []
        for i, row in enumerate(rows):
            batch_texts.append(row['tag_text'])
            batch_ids.append(row['tag_id']) 
            if len(batch_texts) >= self.EMBED_BATCH_SIZE or i == total_missing - 1:
                try:
                    embeddings = self.api_client.get_embeddings(batch_texts)
                    for tid, vec_list in zip(batch_ids, embeddings):
                        vec_np = np.array(vec_list)
                        blob = self._serialize_vec(vec_np)
                        updates.append((blob, tid))
                    batch_texts = []
                    batch_ids = []
                except Exception as e:
                    print(f"⚠️ [Embed] Batch failed: {e}")
                    batch_texts = []
                    batch_ids = []
        if updates:
            cursor.executemany("UPDATE global_tag_registry SET embedding = ? WHERE tag_id = ?", updates)
            conn.commit()

    def _reset_all_status(self, cursor):
        # 保持不变
        print("⚠️ [Reset] FLAG DETECTED: Resetting all tags to status=0...")
        cursor.execute("SELECT COUNT(*) FROM global_tag_registry WHERE status = 1")
        count = cursor.fetchone()[0]
        if count > 0:
            cursor.execute("UPDATE global_tag_registry SET status = 0, mapped_tag = NULL, category_l1 = NULL")
            print(f"   -> {count} tags have been reset.")
        else:
            print("   -> Table is already clean.")

    def _print_cluster_statistics(self, cursor):
        # 保持不变
        print("\n📈 [Stats] Calculating cluster statistics...")
        cursor.execute("""
            SELECT mapped_tag, COUNT(*) as cnt, category_l1
            FROM global_tag_registry 
            WHERE status = 1 AND mapped_tag IS NOT NULL
            GROUP BY mapped_tag
            HAVING cnt > 1
            ORDER BY cnt DESC
        """)
        rows = cursor.fetchall()
        if not rows:
            print("   -> No active clusters found.")
            return
        total_clusters = len(rows)
        counts = [r['cnt'] for r in rows]
        total_tags = sum(counts)
        avg_size = total_tags / total_clusters if total_clusters > 0 else 0
        print(f"   -> 🟢 Total Clusters: {total_clusters}")
        print(f"   -> 🔵 Total Tags (Clustered): {total_tags}")
        print(f"   -> 🟡 Avg Tags/Cluster: {avg_size:.2f}")
        for r in rows[:5]:
             print(f"      - {r['mapped_tag']} ({r['cnt']}) [{r['category_l1']}]")
        
    def run_clustering(self, reset_all=False, enable_clustering=True, enable_labeling=True): 
        # 保持不变
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        print(f"🚀 [Job Start] Reset={reset_all} | Cluster={enable_clustering} | Label={enable_labeling}")
        try:
            self._ensure_category_schema(cursor, conn)
            if reset_all:
                self._reset_all_status(cursor)
                conn.commit()
            if enable_clustering:
                self._fill_missing_embeddings(cursor, conn)
                print("\n=== Phase 1: Clustering (Vector Calculation) ===")
                self._perform_core_clustering_only(cursor, conn)
            else:
                print("\n=== Phase 1: Skipped (Clustering) ===")
            if enable_labeling:
                print("\n=== Phase 2: Labeling (LLM Naming & L1 Category) ===")
                self._perform_llm_labeling_only(cursor, conn)
            else:
                print("\n=== Phase 2: Skipped (Labeling) ===")
            self._print_cluster_statistics(cursor)
        except Exception as e:
            print(f"❌ [Error] Job failed: {e}")
            import traceback
            traceback.print_exc()
            conn.rollback()
        finally:
            conn.close()