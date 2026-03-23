import hashlib
import json
import logging
import threading
import numpy as np
import sqlite3
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

class ClusteringManager:
    def __init__(self, state_manager, api_client=None, prompt_manager=None):
        self.state_manager = state_manager
        self.api_client = api_client
        self.prompt_manager = prompt_manager
        
        # [Online] 内存缓存：仅缓存 Text -> Raw ID，减少 DB SELECT
        self.tag_id_cache = {} 
        self._lock = threading.Lock()

        # [Offline] 聚类参数
        self.SIMILARITY_THRESHOLD = 0.92  # 直接映射阈值
        self.DBSCAN_EPS = 0.15            # 聚类距离阈值
        self.DBSCAN_MIN_SAMPLES = 2
    
    def _get_thread_safe_conn(self):
        """[新增] 获取线程安全的短连接"""
        return sqlite3.connect(self.db_path, timeout=30.0)

    def _get_conn(self):
        return self.state_manager._get_conn()

    def _compute_hash(self, l1, l2_id, l3_id=None):
        """生成路径指纹 (基于 Raw ID)"""
        raw = f"{l1}-{l2_id}-{l3_id or 'None'}"
        return hashlib.md5(raw.encode()).hexdigest()

    # ==========================================================
    #  Part 1: 在线逻辑 (Online Process) - 极速写入模式
    # ==========================================================

    def save_tagging_result(self, interest_tree):
        """
        在线写入：线程安全版本
        """
        if not interest_tree: return

        # 兼容处理
        tree_data = interest_tree
        if isinstance(interest_tree, dict) and 'interest_tree' in interest_tree:
            tree_data = interest_tree['interest_tree']
        if not isinstance(tree_data, list): return

        # [修改] 使用独立的短连接，避免多线程事务冲突
        conn = self._get_thread_safe_conn()
        conn.row_factory = sqlite3.Row # 方便读取
        cursor = conn.cursor()

        try:
            for l1_node in tree_data:
                l1_label = l1_node.get('interest_L1', 'Unknown')
                
                # 处理 L2
                if 'children' in l1_node:
                    for l2_node in l1_node['children']:
                        l2_text = l2_node.get('interest_L2')
                        if not l2_text: continue

                        # 1. 获取 Raw L2 ID
                        l2_raw_id = self._get_or_create_raw_tag(cursor, l2_text, 'L2')

                        if 'children' not in l2_node or not l2_node['children']:
                            self._upsert_raw_graph_edge(cursor, l1_label, l2_raw_id, None)
                        else:
                            # 处理 L3
                            for l3_node in l2_node['children']:
                                l3_text = l3_node.get('interest_L3')
                                if not l3_text: continue
                                
                                # 2. 获取 Raw L3 ID
                                l3_raw_id = self._get_or_create_raw_tag(cursor, l3_text, 'L3')
                                
                                # 3. 写入 Raw Edge
                                self._upsert_raw_graph_edge(cursor, l1_label, l2_raw_id, l3_raw_id)
            
            conn.commit()
        except Exception as e:
            logging.error(f"[ClusteringManager] Online Save Error: {e}")
            conn.rollback()
        finally:
            conn.close()

    def _get_or_create_raw_tag(self, cursor, text, level):
        """查找或创建标签"""
        text = str(text).strip()
        if not text: return None

        # 1. 查内存缓存 (极快)
        with self._lock: # 读取缓存也要加锁，防止读写竞争
            if text in self.tag_id_cache:
                return self.tag_id_cache[text]

        # 2. 查 DB
        # 注意：这里不能用 self._lock 包裹 execute，因为 cursor 是外部传入的，属于外部事务的一部分
        cursor.execute("SELECT tag_id FROM global_tag_registry WHERE tag_text = ?", (text,))
        row = cursor.fetchone()

        if row:
            tag_id = row['tag_id']
            with self._lock:
                self.tag_id_cache[text] = tag_id
            return tag_id
        
        # 3. 插入新词 (Status=0)
        try:
            cursor.execute("INSERT INTO global_tag_registry (tag_text, tag_level, status) VALUES (?, ?, 0)", (text, level))
            new_tag_id = cursor.lastrowid
            with self._lock:
                self.tag_id_cache[text] = new_tag_id
            return new_tag_id
        except sqlite3.IntegrityError:
            # 并发写入时可能已经被别的线程插进去了，重查一次
            cursor.execute("SELECT tag_id FROM global_tag_registry WHERE tag_text = ?", (text,))
            row = cursor.fetchone()
            if row:
                tag_id = row['tag_id']
                with self._lock:
                    self.tag_id_cache[text] = tag_id
                return tag_id
            return None

    def _upsert_raw_graph_edge(self, cursor, l1, l2_id, l3_id):
        """写入图谱边 (Raw Mode)"""
        if not l2_id: return # 防御

        edge_hash = self._compute_hash(l1, l2_id, l3_id)
        cursor.execute("""
            INSERT INTO global_taxonomy_graph (l1_label, l2_tag_id, l3_tag_id, edge_hash, frequency, is_active)
            VALUES (?, ?, ?, ?, 1, 1)
            ON CONFLICT(edge_hash) DO UPDATE SET 
                frequency = frequency + 1,
                last_seen_at = CURRENT_TIMESTAMP
        """, (l1, l2_id, l3_id, edge_hash))


    # ==========================================================
    #  Part 2: 离线定期聚类 (Offline Periodic Process)
    # ==========================================================

    def run_clustering_job(self):
        """
        [离线任务入口]
        1. 对 L2 进行 Embedding + 映射 + 聚类 + 命名。
        2. 对 L3 (在 L2 上下文下) 进行同样操作。
        3. 根据新的映射关系，合并 Graph 表中的边。
        """
        if not self.embedding_client or not self.api_client:
            logging.error("[Clustering] Clients not initialized.")
            return

        logging.info("[Clustering] Step 1: Processing L2 Tags...")
        self._process_l2_clustering()

        logging.info("[Clustering] Step 2: Processing L3 Tags...")
        self._process_l3_clustering()

        logging.info("[Clustering] Step 3: Consolidating Graph...")
        self._consolidate_graph()
        
        logging.info("[Clustering] Job Finished.")

    # --- 辅助：向量处理 ---
    def _serialize_vec(self, vec): return vec.astype(np.float32).tobytes()
    def _deserialize_vec(self, blob): return np.frombuffer(blob, dtype=np.float32)
    def _get_vecs(self, texts): return self.api_client.get_embeddings(texts)

    # --- L2 处理逻辑 ---
    def _process_l2_clustering(self):
        conn = self._get_thread_safe_conn()
        cursor = conn.cursor()
        
        # 1. 获取 Anchors (Status=1)
        cursor.execute("SELECT tag_id, tag_text, embedding FROM global_tag_registry WHERE tag_level='L2' AND status=1")
        anchors = [{'id': r['tag_id'], 'text': r['tag_text'], 'vec': self._deserialize_vec(r['embedding'])} for r in cursor.fetchall()]

        # 2. 获取 Pendings (Status=0)
        cursor.execute("SELECT tag_id, tag_text FROM global_tag_registry WHERE tag_level='L2' AND status=0")
        pendings = [{'id': r['tag_id'], 'text': r['tag_text']} for r in cursor.fetchall()]
        
        if not pendings: return

        # 3. 补全 Embedding
        texts = [p['text'] for p in pendings]
        vecs = self._get_vecs(texts)
        for i, vec in enumerate(vecs):
            pendings[i]['vec'] = np.array(vec)
            # 回写 DB 防止丢失
            cursor.execute("UPDATE global_tag_registry SET embedding=? WHERE tag_id=?", 
                           (self._serialize_vec(pendings[i]['vec']), pendings[i]['id']))
        conn.commit()

        # 4. 执行聚类
        self._execute_clustering_logic(cursor, pendings, anchors, level='L2')
        conn.commit()
        conn.close()

    # --- L3 处理逻辑 ---
    def _process_l3_clustering(self):
        conn = self._get_thread_safe_conn()
        cursor = conn.cursor()
        
        # 遍历所有标准 L2，处理其下的 L3
        cursor.execute("SELECT tag_id, tag_text FROM global_tag_registry WHERE tag_level='L2' AND status=1")
        l2_anchors = cursor.fetchall()

        for l2_row in l2_anchors:
            l2_id = l2_row['tag_id']
            l2_text = l2_row['tag_text']

            # 查找该 L2 上下文下的 L3 Pendings 和 Anchors
            # (逻辑简化：通过 Graph 表关联)
            
            # 找 Anchors
            cursor.execute("""
                SELECT DISTINCT r3.tag_id, r3.tag_text, r3.embedding
                FROM global_taxonomy_graph g
                JOIN global_tag_registry r2 ON g.l2_tag_id = r2.tag_id
                JOIN global_tag_registry r3 ON g.l3_tag_id = r3.tag_id
                WHERE (r2.tag_id = ? OR r2.canonical_id = ?) AND r3.status = 1
            """, (l2_id, l2_id))
            anchors = [{'id': r['tag_id'], 'text': r['tag_text'], 'vec': self._deserialize_vec(r['embedding'])} for r in cursor.fetchall()]

            # 找 Pendings
            cursor.execute("""
                SELECT DISTINCT r3.tag_id, r3.tag_text, r3.embedding
                FROM global_taxonomy_graph g
                JOIN global_tag_registry r2 ON g.l2_tag_id = r2.tag_id
                JOIN global_tag_registry r3 ON g.l3_tag_id = r3.tag_id
                WHERE (r2.tag_id = ? OR r2.canonical_id = ?) AND r3.status = 0
            """, (l2_id, l2_id))
            pendings = []
            rows = cursor.fetchall()
            
            # 补全 L3 Embedding
            texts_to_embed = []
            indices_to_embed = []
            for i, row in enumerate(rows):
                p = {'id': row['tag_id'], 'text': row['tag_text']}
                if row['embedding']:
                    p['vec'] = self._deserialize_vec(row['embedding'])
                else:
                    texts_to_embed.append(row['tag_text'])
                    indices_to_embed.append(i)
                pendings.append(p)
            
            if texts_to_embed:
                new_vecs = self._get_vecs(texts_to_embed)
                for idx, vec in zip(indices_to_embed, new_vecs):
                    pendings[idx]['vec'] = np.array(vec)
                    cursor.execute("UPDATE global_tag_registry SET embedding=? WHERE tag_id=?", 
                                   (self._serialize_vec(pendings[idx]['vec']), pendings[idx]['id']))
                conn.commit()

            if pendings:
                self._execute_clustering_logic(cursor, pendings, anchors, level='L3', context=l2_text)
                conn.commit()
        conn.close()

    # --- 核心聚类算法 (DBSCAN + LLM) ---
    def _execute_clustering_logic(self, cursor, pendings, anchors, level, context=""):
        updates = []
        unmatched = []

        # 1. Direct Mapping (Cosine Similarity)
        if anchors:
            anchor_vecs = np.array([a['vec'] for a in anchors])
            for p in pendings:
                if 'vec' not in p: continue
                sims = cosine_similarity([p['vec']], anchor_vecs)[0]
                max_idx = np.argmax(sims)
                if sims[max_idx] >= self.SIMILARITY_THRESHOLD:
                    updates.append((anchors[max_idx]['id'], 2, p['id'])) # Merged
                else:
                    unmatched.append(p)
        else:
            unmatched = pendings
        
        # 提交直接映射
        if updates:
            cursor.executemany("UPDATE global_tag_registry SET canonical_id=?, status=? WHERE tag_id=?", updates)
            updates = []

        if not unmatched: return

        # 2. DBSCAN Clustering
        if len(unmatched) < 2:
            # 数据太少不聚类，直接转正
            for p in unmatched:
                updates.append((p['id'], 1, p['id']))
            cursor.executemany("UPDATE global_tag_registry SET canonical_id=?, status=? WHERE tag_id=?", updates)
            return

        X = np.array([p['vec'] for p in unmatched])
        db = DBSCAN(eps=self.DBSCAN_EPS, min_samples=self.DBSCAN_MIN_SAMPLES, metric='cosine').fit(X)
        
        clusters = defaultdict(list)
        for idx, label in enumerate(db.labels_):
            if label == -1: # Noise -> Anchor
                updates.append((unmatched[idx]['id'], 1, unmatched[idx]['id']))
            else:
                clusters[label].append(unmatched[idx])
        
        if updates:
            cursor.executemany("UPDATE global_tag_registry SET canonical_id=?, status=? WHERE tag_id=?", updates)
            updates = []

        # 3. LLM Naming for Clusters
        for _, group in clusters.items():
            tags_text = [p['text'] for p in group]
            
            # [LLM Call]
            new_label = self._generate_cluster_label(tags_text, level, context)
            
            # Check exist
            cursor.execute("SELECT tag_id FROM global_tag_registry WHERE tag_text = ?", (new_label,))
            row = cursor.fetchone()
            
            if row:
                canon_id = row['tag_id']
                cursor.execute("UPDATE global_tag_registry SET status=1 WHERE tag_id=?", (canon_id,))
            else:
                mean_vec = np.mean([p['vec'] for p in group], axis=0)
                cursor.execute("INSERT INTO global_tag_registry (tag_text, tag_level, embedding, status, canonical_id) VALUES (?, ?, ?, 1, NULL)", 
                               (new_label, level, self._serialize_vec(mean_vec)))
                canon_id = cursor.lastrowid
                cursor.execute("UPDATE global_tag_registry SET canonical_id=? WHERE tag_id=?", (canon_id, canon_id))
            
            # Map group to canon_id
            group_updates = [(canon_id, 2, p['id']) for p in group]
            cursor.executemany("UPDATE global_tag_registry SET canonical_id=?, status=? WHERE tag_id=?", group_updates)

    def _generate_cluster_label(self, tags, level, context):
        """生成类名"""
        prompt = f"任务：为以下语义相似的{level}标签生成一个标准中文名。\n上下文：{context}\n标签：{json.dumps(tags, ensure_ascii=False)}\n要求：仅返回一个词，不要解释。"
        try:
            return self.api_client.call_api(prompt, temperature=0.1).strip().replace('"', '')
        except:
            return sorted(tags, key=len)[0] # Fallback

    # ==========================================================
    #  Part 3: 图谱合并 (Consolidate Graph) - [核心修改]
    # ==========================================================

    def _consolidate_graph(self):
        """
        依照新的映射关系更新 Graph 表。
        策略：Soft Merge
        1. 找出所有包含 '已归并(Status=2)' 标签的边。
        2. 计算这些边应该指向的新标准边 (New Hash)。
        3. 如果新标准边已存在，将旧边的 Frequency 累加过去，并标记旧边为 Inactive。
        4. 如果新标准边不存在，直接更新旧边的 ID 和 Hash 为新标准。
        """
        conn = self._get_thread_safe_conn()
        cursor = conn.cursor()
        
        # 1. 获取映射字典 (Raw ID -> Canonical ID)
        cursor.execute("SELECT tag_id, canonical_id FROM global_tag_registry WHERE status=2")
        mapping = {row['tag_id']: row['canonical_id'] for row in cursor.fetchall()}
        
        if not mapping:
            return

        # 2. 找出需要处理的边 (Active 且 包含 mapping 中的 ID)
        # 为避免全表扫描，可依赖定期任务频率较高，增量不多
        cursor.execute("SELECT * FROM global_taxonomy_graph WHERE is_active=1")
        edges = cursor.fetchall()
        
        updates_to_inactive = [] # [(edge_id), ...]
        upserts_new = []         # [(l1, new_l2, new_l3, new_hash, added_freq), ...]
        direct_updates = []      # [(new_l2, new_l3, new_hash, edge_id), ...]

        # 缓存当前 DB 中已有的 edges 以判断是 Update 还是 Merge
        # (edge_hash) -> edge_id
        existing_hashes = {row['edge_hash']: row['edge_id'] for row in edges}

        for edge in edges:
            eid = edge['edge_id']
            l1 = edge['l1_label']
            l2 = edge['l2_tag_id']
            l3 = edge['l3_tag_id']
            freq = edge['frequency']
            old_hash = edge['edge_hash']

            new_l2 = mapping.get(l2, l2)
            new_l3 = mapping.get(l3, l3) if l3 else None

            # 如果没有变化，跳过
            if new_l2 == l2 and new_l3 == l3:
                continue

            # 计算新 Hash
            new_hash = self._compute_hash(l1, new_l2, new_l3)

            # 策略分支
            if new_hash in existing_hashes:
                # Case A: 目标边已存在 -> 合并频率
                # 1. 废弃当前边
                updates_to_inactive.append((eid,))
                # 2. 将当前边的 freq 累加到目标边 (目标边 ID = existing_hashes[new_hash])
                # 这里我们收集增量，最后执行 Upsert (ON CONFLICT DO UPDATE SET freq = freq + excluded.freq)
                upserts_new.append((l1, new_l2, new_l3, new_hash, freq))
            else:
                # Case B: 目标边不存在 -> 直接原地变身
                # 这样保留了原有的 frequency，只是 ID 变了
                # 注意：变身后的 hash 需要加入 existing_hashes，防止后续循环冲突
                direct_updates.append((new_l2, new_l3, new_hash, eid))
                existing_hashes[new_hash] = eid # 占位，防止后续重复

        # 执行数据库操作
        if updates_to_inactive:
            cursor.executemany("UPDATE global_taxonomy_graph SET is_active=0 WHERE edge_id=?", updates_to_inactive)
        
        if direct_updates:
            cursor.executemany("UPDATE global_taxonomy_graph SET l2_tag_id=?, l3_tag_id=?, edge_hash=? WHERE edge_id=?", direct_updates)
        
        if upserts_new:
            # 这里的逻辑是：将旧边的 freq 加到新边上
            # SQL: INSERT ... ON CONFLICT(edge_hash) DO UPDATE SET frequency = frequency + excluded.frequency
            cursor.executemany("""
                INSERT INTO global_taxonomy_graph (l1_label, l2_tag_id, l3_tag_id, edge_hash, frequency, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(edge_hash) DO UPDATE SET 
                    frequency = frequency + excluded.frequency,
                    last_seen_at = CURRENT_TIMESTAMP
            """, upserts_new)

        conn.commit()
        logging.info(f"[Graph] Consolidated: {len(updates_to_inactive)} disabled, {len(direct_updates)} mutated, {len(upserts_new)} merged.")
        conn.close()