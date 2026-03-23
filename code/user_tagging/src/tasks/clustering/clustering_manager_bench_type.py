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
        self.MIN_CLUSTER_SIZE = 5
        self.MIN_SAMPLES = 2
        self.CLUSTER_SELECTION_EPSILON = 0.1
        
        # [PCA 参数]
        self.PCA_ENABLE_THRESHOLD = 1000 
        self.PCA_TARGET_DIM = 64

        # [并发参数]
        self.MAX_WORKERS = 20  # 根据API限流情况调整，通常10-20均可
        
        # [L1领域分类列表]
        self.L1_CATEGORY_LIST = [
            "游戏", "电竞", "动漫", "体育", "运动", "科技", "AI", "数码", 
            "汽车", "美妆", "时尚", "母婴", "女性成长", "美食", "旅游", 
            "摄影", "艺术", "房产", "家居", "健康", "科普", "历史", 
            "校园", "教育", "星座", "萌宠", "幽默", "情感", "颜值", 
            "财经", "军事", "法律", "时事", "公益", "同城", "电影", 
            "电视剧", "综艺", "音乐", "娱评", "读书"
        ]
        
    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _deserialize_vec(self, blob):
        if not blob: return None
        return np.frombuffer(blob, dtype=np.float32)

    def _serialize_vec(self, vec):
        return vec.astype(np.float32).tobytes()

    def _ensure_category_schema(self, cursor, conn):
        for col in ["category", "category_l1", "type"]:
            try:
                cursor.execute(f"ALTER TABLE global_tag_registry ADD COLUMN {col} TEXT")
                print(f"✅ [Schema] Added column: {col}")
            except sqlite3.OperationalError:
                pass
        conn.commit()

    def _perform_core_clustering_only(self, cursor, conn):
        # ================= A. 加载待处理的新数据 (Pending Tags) =================
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

        # ================= A.5 [补全] 加载现有簇 (Existing Clusters) =================
        cursor.execute("SELECT mapped_tag, embedding, category FROM global_tag_registry WHERE status = 1 AND mapped_tag IS NOT NULL")
        existing_rows = cursor.fetchall()
        
        cluster_centroids = {} 
        cluster_meta = {} 
        cluster_groups = defaultdict(list)
        
        for r in existing_rows:
            vec = self._deserialize_vec(r['embedding'])
            if vec is not None:
                cluster_groups[r['mapped_tag']].append(vec)
                if r['mapped_tag'] not in cluster_meta and r['category']:
                     cluster_meta[r['mapped_tag']] = r['category']
        
        for label, vecs in cluster_groups.items():
            cluster_centroids[label] = np.mean(vecs, axis=0)

        # ================= B. 尝试归并到现有簇 (Merge Logic) =================
        
        merge_groups = defaultdict(list)
        unmatched_tags = [] 
        
        if cluster_centroids:
            centroid_labels = list(cluster_centroids.keys())
            centroid_matrix = np.array([cluster_centroids[k] for k in centroid_labels])
            
            # 计算相似度并分组
            for tag in pending_tags:
                sims = cosine_similarity([tag['vec']], centroid_matrix)[0]
                best_idx = np.argmax(sims)
                
                if sims[best_idx] >= self.MERGE_THRESHOLD:
                    target_label = centroid_labels[best_idx]
                    merge_groups[target_label].append(tag)
                else:
                    unmatched_tags.append(tag)
        else:
            unmatched_tags = pending_tags

        # --- [并发优化] 批量处理 Merge 分组 (LLM Type Check) ---
        updates = []
        if merge_groups:
            print(f"🔄 [Merge] Processing {len(merge_groups)} target clusters for merging (Concurrent)...")
            
            # 1. 提交所有任务到线程池
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                # 建立 future -> target_label 的映射
                future_to_label = {}
                for target_label, tags_in_group in merge_groups.items():
                    tag_texts = [t['text'] for t in tags_in_group]
                    # 提交并发任务
                    future = executor.submit(self._batch_classify_tag_types, target_label, tag_texts)
                    future_to_label[future] = target_label
                
                # 2. 获取结果
                for future in concurrent.futures.as_completed(future_to_label):
                    target_label = future_to_label[future]
                    tags_in_group = merge_groups[target_label]
                    
                    try:
                        type_map = future.result()
                    except Exception as e:
                        print(f"⚠️ [Merge Error] Type check failed for {target_label}: {e}")
                        type_map = {}

                    # 3. 准备更新数据 (主线程)
                    target_category = cluster_meta.get(target_label, None)
                    for tag_obj in tags_in_group:
                        t_text = tag_obj['text']
                        t_id = tag_obj['tag_id']
                        determined_type = type_map.get(t_text, 'topic')
                        updates.append((target_label, target_category, determined_type, 1, t_id))

            # 4. 批量执行更新
            if updates:
                cursor.executemany("""
                    UPDATE global_tag_registry 
                    SET mapped_tag=?, category=?, type=?, status=? 
                    WHERE tag_id=?
                """, updates)
                conn.commit()
                print(f"🔗 [Merge] Merged {len(updates)} tags with refined types.")

        # ================= C. HDBSCAN 新聚类 (New Cluster Logic) =================
        if not unmatched_tags:
            return

        print(f"🧩 [Cluster] Clustering {len(unmatched_tags)} remaining tags with HDBSCAN...")
        
        X = np.array([t['vec'] for t in unmatched_tags]).astype('float64')
        
        if len(unmatched_tags) > self.PCA_ENABLE_THRESHOLD and X.shape[1] > self.PCA_TARGET_DIM:
            print(f"   ⚡ [PCA] Reducing dimensions...")
            pca = PCA(n_components=self.PCA_TARGET_DIM, random_state=42)
            X_cluster = pca.fit_transform(X)
        else:
            X_cluster = X
            
        clusterer = HDBSCAN(
            min_cluster_size=self.MIN_CLUSTER_SIZE,
            min_samples=self.MIN_SAMPLES,
            cluster_selection_epsilon=self.CLUSTER_SELECTION_EPSILON,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(X_cluster)
        
        new_clusters = defaultdict(list)
        noise_tags = []

        for i, label in enumerate(labels):
            tag_obj = unmatched_tags[i]
            if label == -1:
                noise_tags.append(tag_obj)
            else:
                new_clusters[label].append(tag_obj)
        
        print(f"   -> HDBSCAN Result: {len(new_clusters)} clusters, {len(noise_tags)} noise points.")
        
        # --- [并发优化] 批量处理 新簇 (LLM Type Check) ---
        new_updates = []
        if new_clusters:
            print(f"✨ [New] Analyzing types for {len(new_clusters)} new clusters (Concurrent)...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                future_to_cluster_id = {}
                for label_id, group in new_clusters.items():
                    tag_texts = [t['text'] for t in group]
                    temp_label = min(tag_texts, key=len)
                    
                    # 提交并发任务
                    future = executor.submit(self._batch_classify_tag_types, temp_label, tag_texts)
                    # 存储必要元数据以回溯
                    future_to_cluster_id[future] = (label_id, temp_label)
                
                for future in concurrent.futures.as_completed(future_to_cluster_id):
                    label_id, temp_label = future_to_cluster_id[future]
                    group = new_clusters[label_id]
                    
                    try:
                        type_map = future.result()
                    except Exception as e:
                        type_map = {}
                    
                    for t_obj in group:
                        t_text = t_obj['text']
                        t_id = t_obj['tag_id']
                        determined_type = type_map.get(t_text, 'topic')
                        # 新簇 category 暂时为 None
                        new_updates.append((temp_label, None, determined_type, 1, t_id))

        if new_updates:
            cursor.executemany("""
                UPDATE global_tag_registry 
                SET mapped_tag=?, category=?, type=?, status=? 
                WHERE tag_id=?
            """, new_updates)
            conn.commit()
            print(f"💾 [Cluster] Saved {len(new_updates)} new clustered tags with types.")

    def _perform_llm_labeling_only(self, cursor, conn):
        """
        Phase 2: LLM 标注 (并发优化版)
        """
        print("🤖 [Label] Scanning for clusters needing categorization...")
        
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
        
        cursor.execute("""
            SELECT mapped_tag, COUNT(*) as cnt 
            FROM global_tag_registry 
            WHERE status=1 AND mapped_tag IS NOT NULL
            GROUP BY mapped_tag
        """)
        cluster_sizes = {row[0]: row[1] for row in cursor.fetchall()}

        # 1. 准备任务列表 (Pre-fetch data in Main Thread)
        tasks = []
        for cluster_name in clusters_to_process:
            if cluster_sizes.get(cluster_name, 0) < 2:
                continue
            
            cursor.execute("SELECT tag_text FROM global_tag_registry WHERE mapped_tag = ? LIMIT 20", (cluster_name,))
            sample_tags = [r[0] for r in cursor.fetchall()]
            
            if len(sample_tags) < 2: continue 
            
            tasks.append((cluster_name, sample_tags))
        
        if not tasks:
            print("✅ [Label] No valid clusters to process (size check).")
            return

        print(f"🚀 [Label] Starting concurrent LLM inference for {len(tasks)} clusters...")
        
        # 2. 并发执行 LLM 任务
        processed_count = 0
        total_count = len(tasks)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_name = {
                executor.submit(self._generate_cluster_info_with_llm, tags): name 
                for name, tags in tasks
            }
            
            # 3. 获取结果并实时写入 DB (主线程写入，确保线程安全)
            for future in concurrent.futures.as_completed(future_to_name):
                cluster_name = future_to_name[future]
                try:
                    info = future.result()
                    
                    new_label = info['label']
                    l1_category_str = info['l1_category']
                    tag_types_map = info.get('tag_types', {})
                    
                    # 写入 Cluster 元数据
                    cursor.execute("""
                        UPDATE global_tag_registry 
                        SET mapped_tag = ?, category_l1 = ?
                        WHERE mapped_tag = ?
                    """, (new_label, l1_category_str, cluster_name))
                    
                    # 写入 Tag Types
                    type_updates = []
                    for tag_text, t_type in tag_types_map.items():
                        type_updates.append((t_type, tag_text))
                    
                    if type_updates:
                        cursor.executemany("UPDATE global_tag_registry SET type = ? WHERE tag_text = ?", type_updates)

                    conn.commit()
                    processed_count += 1
                    print(f"   [{processed_count}/{total_count}] ✨ Saved: [{cluster_name}] -> 🏷️ [{new_label}]")
                    
                except Exception as e:
                    print(f"❌ [Error] Failed to process/save cluster {cluster_name}: {e}")
                    # 不回滚整个事务，仅跳过当前失败项
                    # conn.rollback() # 可以选择回滚或者忽略

        print("✅ [Label] Done.")

    def _generate_cluster_info_with_llm(self, tags):
        """Generate metadata for a cluster (English Prompt)"""
        if len(tags) == 1: 
            return {
                "label": tags[0], 
                "l1_category": "", 
                "tag_types": {tags[0]: "topic"}
            }
        
        tags_str = json.dumps(tags[:20], ensure_ascii=False)
        cat_list_str = ", ".join(self.L1_CATEGORY_LIST)
        
        prompt = f"""# Role
            You are a Social Media Knowledge Graph Expert. Your task is to structure a group of semantically similar Hashtags into standard metadata.
            
            # Input Data
            - Tag List: {tags_str}
            - Allowed Categories: [{cat_list_str}]
    
            # Tasks
            
            ## 1. Extract Label
            - Extract the core entity or the smallest common concept shared by these tags.
            - Constraint: Keep it concise, accurate, and free of punctuation/redundancy.
            
            ## 2. L1 Category Classification
            - Select **1-3** most relevant categories from the "Allowed Categories" list.
            - Sort by relevance descending.
            
            ## 3. Type Classification
            Classify each tag strictly following these mutually exclusive standards (**Event takes precedence**):
            
            ### **event** (Dynamic)
            > **Definition**: Describes specific actions, state changes, breaking news, or time-sensitive activities.
            > **Features**:
            - **Temporality**: Has a clear time anchor or sense of process.
            - **Action**: Usually contains a predicate (verb) describing "who did what".
            - **Special Case**: Short phrases referring to matches, festivals, or shows (e.g., "WorldCup", "Gala") are Events because they imply a schedule.
            - **Test**: Answers "What happened?" or "When?".

            ### **topic** (Static)
            > **Definition**: Objectively existing nouns, concepts, people, or works.
            > **Features**:
            - **Static Nature**: Describes essential attributes; existence does not change with time.
            - **Structure**: Strictly Nouns or Noun Phrases. **MUST NOT** contain specific actions or news descriptions.
            - **Test**: Answers "What is it?" or "Who is it?".

            ### **others**
            - Meaningless characters or pure emotional particles.

            # Output
            Strict JSON format:
            {{
              "label": "String (Core Entity)",
              "l1_category": ["Cat1", "Cat2"],
              "tag_types": {{
                "tag1": "type",
                "tag2": "type"
              }}
            }}
            """
        
        default_result = {
            "label": min(tags, key=len), 
            "l1_category": "", 
            "tag_types": {t: "topic" for t in tags}
        }
        
        try:
            response = self.api_client.call_api(prompt)
            if isinstance(response, str):
                clean_str = response.replace("```json", "").replace("```", "").strip()
                response = json.loads(clean_str)
            
            label = response.get("label", "").strip()
            
            # 处理 L1 Category
            raw_cats = response.get("l1_category", [])
            if isinstance(raw_cats, str):
                l1_category_str = raw_cats
            elif isinstance(raw_cats, list):
                l1_category_str = ",".join([str(c) for c in raw_cats])
            else:
                l1_category_str = ""

            tag_types = response.get("tag_types", {})
            
            if not label: return default_result
            
            # 补全缺失的 type
            for t in tags:
                if t not in tag_types:
                    tag_types[t] = "topic"
            
            return {
                "label": label, 
                "l1_category": l1_category_str, 
                "tag_types": tag_types
            }
        except Exception as e:
            print(f"⚠️ [LLM] Info generation failed: {e}")
            return default_result
    
    def _batch_classify_tag_types(self, cluster_name, tag_texts):
        """Batch classify tag types using LLM (English Prompt)"""
        if not tag_texts: return {}
        
        tags_str = json.dumps(tag_texts, ensure_ascii=False)
        
        prompt = f"""
        # Context
        The following tags belong to the semantic cluster: "{cluster_name}".
        
        # Definitions (Strict & Mutually Exclusive)
        
        1. **Event** (Dynamic/Time-Sensitive)
           - **Core**: Represents an occurrence, action, state change, or scheduled activity.
           - **Key Features**: 
             - Implies a specific timestamp or timeline (past/present/future).
             - Often contains a predicate (verb) indicating action (e.g., release, attack, marry).
             - Includes recurring competitions, festivals, or shows (viewed as proceedings).
           - **Litmus Test**: Can you ask "When did it happen?" or "What happened?"
        
        2. **Topic** (Static/Entity)
           - **Core**: Represents an object, person, concept, work, or location.
           - **Key Features**:
             - Timeless existence; exists rather than occurs.
             - Strictly consists of Nouns/Noun Phrases. 
             - **MUST NOT** contain verbs or action descriptions.
             - Refers to the *subject* itself (e.g., a game), not a specific *instance* (e.g., a match).
           - **Litmus Test**: Can you ask "What is it?" or "Who is it?"

        3. **Others**
           - Meaningless text, pure emotional interjections, or incomplete fragments.

        # Task
        Classify each tag based on the cluster context. **Event takes precedence over Topic** if a specific action is implied.
        
        # Input Tags
        {tags_str}

        # Output
        Strict JSON format only: {{"tag_text": "type", ...}}
        """

        try:
            response = self.api_client.call_api(prompt, temperature=0.0)
            
            if isinstance(response, str):
                clean_str = response.replace("```json", "").replace("```", "").strip()
                result_map = json.loads(clean_str)
            else:
                result_map = response

            final_map = {}
            for t in tag_texts:
                final_map[t] = result_map.get(t, "topic")
            
            return final_map

        except Exception as e:
            print(f"⚠️ [TypeCheck] Batch failed for cluster '{cluster_name}': {e}")
            return {t: "topic" for t in tag_texts}

    def _fill_missing_embeddings(self, cursor, conn):
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
        print("⚠️ [Reset] FLAG DETECTED: Resetting all tags to status=0...")
        cursor.execute("SELECT COUNT(*) FROM global_tag_registry WHERE status = 1")
        count = cursor.fetchone()[0]
        if count > 0:
            cursor.execute("UPDATE global_tag_registry SET status = 0, mapped_tag = NULL, category = NULL, category_l1 = NULL, type = NULL")
            print(f"   -> {count} tags have been reset. They will be re-clustered now.")
        else:
            print("   -> Table is already clean.")

    def _print_cluster_statistics(self, cursor):
        print("\n📈 [Stats] Calculating cluster statistics...")
        cursor.execute("""
            SELECT mapped_tag, COUNT(*) as cnt 
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
        
    def run_clustering(self, reset_all=False, enable_clustering=True, enable_labeling=True): 
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
            if enable_clustering:
                print("\n=== Phase 1: Clustering (Vector Calculation) ===")
                self._perform_core_clustering_only(cursor, conn)
            else:
                print("\n=== Phase 1: Skipped (Clustering) ===")
            if enable_labeling:
                print("\n=== Phase 2: Labeling (LLM Naming & Categorization) ===")
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