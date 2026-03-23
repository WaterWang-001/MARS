"""
offline_tag_cluster.py
=====================
Incremental tag clustering with SQLite-backed state management.

Pipeline modes:
  full        – ingest all dates → base clustering → incremental assignment → export
  ingest      – ingest specific dates only (no clustering)
  recluster   – re-run full clustering on all accumulated tags

DB Schema (per platform):
  tag_registry       – tag text, frequency, dates, embedding, cluster assignment
  cluster_centroids  – centroid vectors and metadata
  process_log        – date-level processing history

Usage:
    # Full pipeline
    python -m src.offline_tag_cluster --platforms weibo --mode full

    # Ingest specific dates
    python -m src.offline_tag_cluster --platforms weibo --mode ingest --dates 2025-07-01 2025-07-02

    # Recluster all tags
    python -m src.offline_tag_cluster --platforms weibo --mode recluster

Output: MARS_result/data/benchmark/data/tag_clusters/{platform}/
    tag_cluster.db       – core SQLite DB
    cluster_index.json   – {cluster_id: [tags]}
    cluster_stats.csv    – per-cluster statistics
    tag_clusters.jsonl   – one line per tag (backward-compatible)
"""

import argparse
import csv
import glob
import json
import os
import sqlite3
import sys
import threading
import time
import logging
from collections import defaultdict
from datetime import datetime

import re

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ======================== Config ========================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "MARS_result", "data", "output")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "MARS_result", "data", "benchmark", "data", "tag_clusters")
DEFAULT_EMBEDDING_MODEL = os.path.join(PROJECT_ROOT, "MARS_result/model/embedding/BAAI/bge-small-zh-v1.5")

ALL_PLATFORMS = ["weibo", "xiaohongshu", "toutiao", "zhihu", "douban"]

PLATFORM_PARAMS = {
    "weibo": {
        "min_freq": 3,
        "max_coverage": 1.0,
        "n_clusters": 500,
    },
    "xiaohongshu": {
        "min_freq": 3,
        "max_coverage": 1.0,
        "n_clusters": 800,
    },
    "toutiao": {
        "min_freq": 3,
        "max_coverage": 1.0,
        "n_clusters": 400,
    },
    "zhihu": {
        "min_freq": 2,
        "max_coverage": 1.0,
        "n_clusters": 300,
    },
    "douban": {
        "min_freq": 2,
        "max_coverage": 1.0,
        "n_clusters": 200,
    },
}

DEFAULT_WARMUP_DAYS = 10
DEFAULT_OUTLIER_THRESHOLD = 0.85  # cosine distance threshold for outlier detection


# ======================== TagClusterDB ========================

class TagClusterDB:
    """SQLite-backed tag cluster state manager (modeled after BenchStateManager)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, timeout=60.0)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA synchronous=OFF;")
            self._local.conn.execute("PRAGMA busy_timeout=60000;")
            self._local.conn.execute("PRAGMA temp_store=MEMORY;")
            self._local.conn.execute("PRAGMA wal_autocheckpoint=1000;")
        return self._local.conn

    def _execute_write(self, operation):
        last_exc = None
        for attempt in range(15):
            try:
                return operation()
            except sqlite3.OperationalError as exc:
                if "database is locked" not in str(exc).lower():
                    raise
                last_exc = exc
                time.sleep(0.05 * (attempt + 1))
        raise last_exc

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS tag_registry (
                tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag_text TEXT NOT NULL UNIQUE,
                frequency INTEGER DEFAULT 0,
                first_seen_date TEXT,
                last_seen_date TEXT,
                n_dates INTEGER DEFAULT 0,
                embedding BLOB,
                cluster_id INTEGER DEFAULT -1,
                centroid_dist REAL DEFAULT -1,
                status INTEGER DEFAULT 0
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tag_text ON tag_registry (tag_text)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cluster_id ON tag_registry (cluster_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_status ON tag_registry (status)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS cluster_centroids (
                cluster_id INTEGER PRIMARY KEY,
                centroid BLOB,
                size INTEGER DEFAULT 0,
                created_date TEXT,
                updated_date TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS process_log (
                date TEXT NOT NULL,
                action TEXT NOT NULL,
                n_tags_new INTEGER,
                n_tags_total INTEGER,
                timestamp TEXT,
                PRIMARY KEY (date, action)
            )
        """)

        conn.commit()
        conn.close()

    # ---------- Read helpers ----------

    def get_processed_dates(self, action: str = "ingest") -> set[str]:
        conn = self._get_conn()
        rows = conn.execute("SELECT date FROM process_log WHERE action = ?", (action,)).fetchall()
        return {r["date"] for r in rows}

    def get_tag_count(self, min_freq: int = 0) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM tag_registry WHERE frequency >= ?", (min_freq,)
        ).fetchone()
        return row["cnt"]

    def get_tags_for_clustering(self, min_freq: int, cutoff_date: str | None = None) -> list[dict]:
        conn = self._get_conn()
        if cutoff_date:
            rows = conn.execute(
                "SELECT tag_id, tag_text, frequency FROM tag_registry "
                "WHERE frequency >= ? AND first_seen_date <= ? ORDER BY frequency DESC",
                (min_freq, cutoff_date),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT tag_id, tag_text, frequency FROM tag_registry "
                "WHERE frequency >= ? ORDER BY frequency DESC",
                (min_freq,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_pending_tags(self, min_freq: int = 0) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tag_id, tag_text, frequency FROM tag_registry "
            "WHERE status = 0 AND embedding IS NOT NULL AND frequency >= ?",
            (min_freq,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_new_tags_without_embedding(self, min_freq: int = 0) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tag_id, tag_text FROM tag_registry "
            "WHERE status = 0 AND embedding IS NULL AND frequency >= ?",
            (min_freq,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_clustered_count(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) AS cnt FROM tag_registry WHERE status = 1").fetchone()
        return row["cnt"]

    def get_outlier_count(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) AS cnt FROM tag_registry WHERE status = 2").fetchone()
        return row["cnt"]

    def load_centroids(self) -> dict[int, np.ndarray]:
        conn = self._get_conn()
        rows = conn.execute("SELECT cluster_id, centroid FROM cluster_centroids").fetchall()
        result = {}
        for r in rows:
            result[r["cluster_id"]] = np.frombuffer(r["centroid"], dtype=np.float32).copy()
        return result

    def get_all_clustered_embeddings(self) -> tuple[list[int], np.ndarray]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tag_id, embedding FROM tag_registry WHERE status IN (1, 2) AND embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return [], np.array([])
        tag_ids = [r["tag_id"] for r in rows]
        embeddings = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        return tag_ids, embeddings

    # ---------- Write helpers ----------

    def upsert_tags_from_csv(self, date: str, csv_path: str,
                             tag_transform=None) -> int:
        rows_data = []
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = row.get("Tag", "").strip()
                if not tag:
                    continue
                freq = int(row.get("Frequency", 0))
                if freq <= 0:
                    continue
                # Apply platform-specific transform (e.g. strip douban prefixes)
                if tag_transform:
                    tag = tag_transform(tag)
                if len(tag) < 2:
                    continue
                if all(c in "0123456789ABCDEFabcdef" for c in tag):
                    continue
                rows_data.append((tag, freq, date, date))

        if not rows_data:
            return 0

        with self._write_lock:
            conn = self._get_conn()

            def _op():
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM tag_registry")
                before = cur.fetchone()[0]
                # Batch upsert: INSERT on new tag, UPDATE freq/date on conflict
                BATCH = 5000
                for i in range(0, len(rows_data), BATCH):
                    batch = rows_data[i:i + BATCH]
                    cur.executemany(
                        """
                        INSERT INTO tag_registry (tag_text, frequency, first_seen_date, last_seen_date, n_dates)
                        VALUES (?, ?, ?, ?, 1)
                        ON CONFLICT(tag_text) DO UPDATE SET
                            frequency = frequency + excluded.frequency,
                            last_seen_date = excluded.last_seen_date,
                            n_dates = n_dates + 1
                        """,
                        batch,
                    )
                conn.commit()
                cur.execute("SELECT COUNT(*) FROM tag_registry")
                after = cur.fetchone()[0]
                return after - before

            n_new = self._execute_write(_op)

        return n_new

    def save_embeddings(self, tag_ids: list[int], embeddings: np.ndarray):
        data = [(embeddings[i].astype(np.float32).tobytes(), tag_ids[i])
                for i in range(len(tag_ids))]
        with self._write_lock:
            conn = self._get_conn()

            def _op():
                BATCH = 5000
                for i in range(0, len(data), BATCH):
                    conn.executemany(
                        "UPDATE tag_registry SET embedding = ? WHERE tag_id = ?",
                        data[i:i + BATCH],
                    )
                conn.commit()

            self._execute_write(_op)

    def save_cluster_assignments(self, tag_ids: list[int], labels: np.ndarray,
                                  centroid_dists: np.ndarray | None = None):
        data = []
        for i, tid in enumerate(tag_ids):
            cid = int(labels[i])
            dist = float(centroid_dists[i]) if centroid_dists is not None else -1.0
            data.append((cid, dist, 1, tid))  # status=1 (clustered)

        with self._write_lock:
            conn = self._get_conn()

            def _op():
                BATCH = 5000
                for i in range(0, len(data), BATCH):
                    conn.executemany(
                        "UPDATE tag_registry SET cluster_id = ?, centroid_dist = ?, status = ? WHERE tag_id = ?",
                        data[i:i + BATCH],
                    )
                conn.commit()

            self._execute_write(_op)

    def save_centroids(self, centroids: np.ndarray, cluster_sizes: dict[int, int], date: str):
        with self._write_lock:
            conn = self._get_conn()

            def _op():
                cur = conn.cursor()
                cur.execute("DELETE FROM cluster_centroids")
                for cid in range(len(centroids)):
                    cur.execute(
                        "INSERT INTO cluster_centroids (cluster_id, centroid, size, created_date, updated_date) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (cid, centroids[cid].astype(np.float32).tobytes(),
                         cluster_sizes.get(cid, 0), date, date),
                    )
                conn.commit()

            self._execute_write(_op)

    def mark_outliers(self, tag_ids: list[int]):
        if not tag_ids:
            return
        with self._write_lock:
            conn = self._get_conn()

            def _op():
                cur = conn.cursor()
                for tid in tag_ids:
                    cur.execute("UPDATE tag_registry SET status = 2 WHERE tag_id = ?", (tid,))
                conn.commit()

            self._execute_write(_op)

    def reset_all_clusters(self):
        with self._write_lock:
            conn = self._get_conn()

            def _op():
                cur = conn.cursor()
                cur.execute("UPDATE tag_registry SET cluster_id = -1, centroid_dist = -1, status = 0")
                cur.execute("DELETE FROM cluster_centroids")
                conn.commit()

            self._execute_write(_op)

    def log_process(self, date: str, n_tags_new: int, n_tags_total: int, action: str):
        with self._write_lock:
            conn = self._get_conn()

            def _op():
                conn.execute(
                    "INSERT OR REPLACE INTO process_log (date, action, n_tags_new, n_tags_total, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (date, action, n_tags_new, n_tags_total, datetime.now().isoformat()),
                )
                conn.commit()

            self._execute_write(_op)


# ======================== Tag Preprocessing ========================

# Douban prefix pattern: "想读图书:", "看过影视:", "图书:", etc.
_DOUBAN_PREFIX_RE = re.compile(
    r'^(?:想读|在读|读过|想看|在看|看过|想听|在听|听过)?'
    r'(?:图书|影视|音乐)[:\uff1a]'
)


def strip_douban_prefix(tag: str) -> str:
    """Remove action+type prefix from a douban tag. E.g. '想读图书:红楼梦' -> '红楼梦'."""
    result = _DOUBAN_PREFIX_RE.sub('', tag).strip()
    return result if result else tag


def preprocess_tags_for_embedding(tags: list[str], platform: str) -> list[str]:
    """
    Platform-specific tag preprocessing before embedding computation.
    - zhihu: extract TF-IDF keywords (jieba) and join with spaces
    - douban: strip action+type prefixes
    - others: no-op
    """
    if platform == "zhihu":
        import jieba.analyse
        processed = []
        for tag in tags:
            keywords = jieba.analyse.extract_tags(tag, topK=5)
            processed.append(" ".join(keywords) if keywords else tag)
        logger.info(f"  [zhihu] Extracted keywords for {len(tags)} tags via jieba TF-IDF")
        return processed
    elif platform == "douban":
        processed = [strip_douban_prefix(t) for t in tags]
        logger.info(f"  [douban] Stripped prefixes for {len(tags)} tags")
        return processed
    return tags


# ======================== Embedding (reused) ========================

def compute_embeddings(
    tags: list[str],
    model_path: str,
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """Compute normalized embeddings for a list of tag strings."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        logger.info(f"  GPU free memory: {free_mem:.1f} GB")
        if free_mem < 1.0:
            logger.warning(f"  GPU memory low ({free_mem:.1f} GB), falling back to CPU")
            device = "cpu"

    logger.info(f"  Loading embedding model from {model_path} (device={device})")
    model = SentenceTransformer(model_path, device=device)

    logger.info(f"  Encoding {len(tags):,} tags (batch_size={batch_size}, device={device})...")
    t0 = time.time()
    embeddings = model.encode(
        tags,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    logger.info(f"  Encoding done in {elapsed:.1f}s ({len(tags)/elapsed:.0f} tags/sec)")
    return embeddings


# ======================== Clustering (reused) ========================

def cluster_tags(
    embeddings: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster embeddings using Mini-Batch K-Means.
    Returns (labels, cluster_centers).
    """
    from sklearn.cluster import MiniBatchKMeans

    actual_n = min(n_clusters, len(embeddings))
    logger.info(f"  Clustering {len(embeddings):,} tags into {actual_n} clusters (MiniBatchKMeans)...")
    t0 = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=actual_n,
        batch_size=4096,
        max_iter=300,
        random_state=42,
        n_init=3,
    )
    labels = kmeans.fit_predict(embeddings)
    elapsed = time.time() - t0
    logger.info(f"  Clustering done in {elapsed:.1f}s, inertia={kmeans.inertia_:.2f}")
    return labels, kmeans.cluster_centers_


# ======================== Phase Functions ========================

def discover_dates(platform: str) -> list[str]:
    """Find all available dates with sampled_tag_stats CSV for a platform."""
    pattern = os.path.join(OUTPUT_BASE, "*/blacklist", platform, "sampled_tag_stats_*.csv")
    files = sorted(glob.glob(pattern))
    dates = []
    for f in files:
        basename = os.path.basename(f)
        # sampled_tag_stats_2025-06-04.csv → 2025-06-04
        date_str = basename.replace("sampled_tag_stats_", "").replace(".csv", "")
        dates.append(date_str)
    return sorted(set(dates))


def find_csv_for_date(platform: str, date: str) -> str | None:
    """Find the sampled_tag_stats CSV path for a given date and platform."""
    pattern = os.path.join(OUTPUT_BASE, date, "blacklist", platform, f"sampled_tag_stats_{date}.csv")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    # Also check date_old directories
    pattern2 = os.path.join(OUTPUT_BASE, f"{date}_old", "blacklist", platform, f"sampled_tag_stats_{date}.csv")
    matches2 = glob.glob(pattern2)
    return matches2[0] if matches2 else None


def ingest_dates(db: TagClusterDB, platform: str, dates: list[str]) -> None:
    """Phase 1: Ingest tag stats from CSV files into the DB."""
    processed = db.get_processed_dates()
    tag_transform = strip_douban_prefix if platform == "douban" else None

    for date in dates:
        if date in processed:
            logger.info(f"  [{platform}] Date {date} already ingested, skipping")
            continue

        csv_path = find_csv_for_date(platform, date)
        if csv_path is None:
            logger.warning(f"  [{platform}] No CSV found for date {date}, skipping")
            continue

        n_new = db.upsert_tags_from_csv(date, csv_path, tag_transform=tag_transform)
        total = db.get_tag_count()
        db.log_process(date, n_new, total, "ingest")
        logger.info(f"  [{platform}] Ingested {date}: {n_new} new tags, {total} total in DB")


def run_base_clustering(
    db: TagClusterDB,
    platform: str,
    params: dict,
    cutoff_date: str,
    embedding_model_path: str,
    device: str = "cuda",
) -> None:
    """Phase 2: Full clustering on tags up to cutoff_date."""
    min_freq = params["min_freq"]
    n_clusters = params["n_clusters"]

    tags_data = db.get_tags_for_clustering(min_freq, cutoff_date)
    if not tags_data:
        logger.warning(f"  [{platform}] No tags meeting freq threshold for base clustering")
        return

    logger.info(f"  [{platform}] Base clustering: {len(tags_data)} tags (freq >= {min_freq}, date <= {cutoff_date})")

    tag_ids = [t["tag_id"] for t in tags_data]
    tag_texts = [t["tag_text"] for t in tags_data]

    # Compute embeddings (with platform-specific preprocessing)
    embedding_texts = preprocess_tags_for_embedding(tag_texts, platform)
    embeddings = compute_embeddings(embedding_texts, embedding_model_path, device=device)

    # Save embeddings to DB
    logger.info(f"  [{platform}] Saving {len(tag_ids)} embeddings to DB...")
    db.save_embeddings(tag_ids, embeddings)

    # Cluster
    labels, centroids = cluster_tags(embeddings, n_clusters)

    # Compute distances to assigned centroids
    dists = np.linalg.norm(embeddings - centroids[labels], axis=1)

    # Save assignments
    db.save_cluster_assignments(tag_ids, labels, dists)

    # Compute cluster sizes
    cluster_sizes = defaultdict(int)
    for label in labels:
        cluster_sizes[int(label)] += 1

    # Save centroids
    db.save_centroids(centroids, cluster_sizes, cutoff_date)

    db.log_process(cutoff_date, 0, len(tags_data), "cluster")
    logger.info(f"  [{platform}] Base clustering complete: {len(set(labels))} clusters, "
                f"avg dist={dists.mean():.4f}")


def assign_incremental(
    db: TagClusterDB,
    platform: str,
    params: dict,
    embedding_model_path: str,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    device: str = "cuda",
) -> None:
    """Phase 3: Assign new (pending) tags to nearest existing centroid."""
    min_freq = params["min_freq"]

    # First compute embeddings for tags that don't have them yet
    tags_no_emb = db.get_new_tags_without_embedding(min_freq)
    if tags_no_emb:
        logger.info(f"  [{platform}] Computing embeddings for {len(tags_no_emb)} new tags (freq >= {min_freq})...")
        tag_ids = [t["tag_id"] for t in tags_no_emb]
        tag_texts = [t["tag_text"] for t in tags_no_emb]
        embedding_texts = preprocess_tags_for_embedding(tag_texts, platform)
        embeddings = compute_embeddings(embedding_texts, embedding_model_path, device=device)
        db.save_embeddings(tag_ids, embeddings)

    # Get pending tags (status=0, have embeddings, freq >= min_freq)
    pending = db.get_pending_tags(min_freq)
    if not pending:
        logger.info(f"  [{platform}] No pending tags to assign incrementally")
        return

    # Load centroids
    centroids_dict = db.load_centroids()
    if not centroids_dict:
        logger.warning(f"  [{platform}] No centroids found — run base clustering first")
        return

    centroid_ids = sorted(centroids_dict.keys())
    centroid_matrix = np.stack([centroids_dict[cid] for cid in centroid_ids])

    # Load pending embeddings
    conn = db._get_conn()
    tag_ids = []
    emb_list = []
    for t in pending:
        row = conn.execute("SELECT embedding FROM tag_registry WHERE tag_id = ?", (t["tag_id"],)).fetchone()
        if row and row["embedding"]:
            tag_ids.append(t["tag_id"])
            emb_list.append(np.frombuffer(row["embedding"], dtype=np.float32).copy())

    if not emb_list:
        return

    embeddings = np.stack(emb_list)
    logger.info(f"  [{platform}] Assigning {len(tag_ids)} pending tags to nearest centroid...")

    # Compute distances using matrix multiplication (memory-efficient)
    # Since embeddings are L2-normalized: ||a-b||^2 = 2 - 2*(a·b)
    # similarity: (N, K), avoids (N, K, D) intermediate array
    similarity = embeddings @ centroid_matrix.T
    dists = np.sqrt(np.maximum(2.0 - 2.0 * similarity, 0.0))
    nearest_idx = dists.argmin(axis=1)
    nearest_dist = dists[np.arange(len(dists)), nearest_idx]
    nearest_cid = np.array([centroid_ids[i] for i in nearest_idx])

    # Separate clustered vs outliers
    outlier_mask = nearest_dist > outlier_threshold
    n_outlier = outlier_mask.sum()
    n_assigned = len(tag_ids) - n_outlier

    # Save all assignments (cluster_id set for all, but status differs)
    labels = nearest_cid
    db.save_cluster_assignments(tag_ids, labels, nearest_dist)

    # Mark outliers
    outlier_ids = [tag_ids[i] for i in range(len(tag_ids)) if outlier_mask[i]]
    db.mark_outliers(outlier_ids)

    logger.info(f"  [{platform}] Incremental assignment: {n_assigned} clustered, {n_outlier} outliers "
                f"(threshold={outlier_threshold:.2f})")


def recluster_all(
    db: TagClusterDB,
    platform: str,
    params: dict,
    embedding_model_path: str,
    device: str = "cuda",
) -> None:
    """Phase 4: Full recluster of all tags that have embeddings."""
    min_freq = params["min_freq"]
    n_clusters = params["n_clusters"]

    # Reset cluster state
    db.reset_all_clusters()

    # Get all tags meeting frequency threshold
    tags_data = db.get_tags_for_clustering(min_freq)
    if not tags_data:
        logger.warning(f"  [{platform}] No tags for reclustering")
        return

    logger.info(f"  [{platform}] Reclustering {len(tags_data)} tags...")

    tag_ids = [t["tag_id"] for t in tags_data]
    tag_texts = [t["tag_text"] for t in tags_data]

    # Check which already have embeddings
    conn = db._get_conn()
    ids_need_emb = []
    texts_need_emb = []
    existing_emb = {}
    for tid, text in zip(tag_ids, tag_texts):
        row = conn.execute("SELECT embedding FROM tag_registry WHERE tag_id = ?", (tid,)).fetchone()
        if row and row["embedding"]:
            existing_emb[tid] = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        else:
            ids_need_emb.append(tid)
            texts_need_emb.append(text)

    # Compute missing embeddings
    if texts_need_emb:
        logger.info(f"  [{platform}] Computing embeddings for {len(texts_need_emb)} tags without embeddings...")
        embedding_texts = preprocess_tags_for_embedding(texts_need_emb, platform)
        new_embs = compute_embeddings(embedding_texts, embedding_model_path, device=device)
        db.save_embeddings(ids_need_emb, new_embs)
        for tid, emb in zip(ids_need_emb, new_embs):
            existing_emb[tid] = emb

    # Build embedding matrix in tag_ids order
    embeddings = np.stack([existing_emb[tid] for tid in tag_ids])

    # Cluster
    labels, centroids = cluster_tags(embeddings, n_clusters)
    dists = np.linalg.norm(embeddings - centroids[labels], axis=1)

    # Save
    db.save_cluster_assignments(tag_ids, labels, dists)
    cluster_sizes = defaultdict(int)
    for label in labels:
        cluster_sizes[int(label)] += 1
    today = datetime.now().strftime("%Y-%m-%d")
    db.save_centroids(centroids, cluster_sizes, today)

    db.log_process(today, 0, len(tags_data), "recluster")
    logger.info(f"  [{platform}] Recluster complete: {len(set(labels))} clusters, avg dist={dists.mean():.4f}")


# ======================== Export ========================

def export_results(db: TagClusterDB, platform: str, output_dir: str) -> None:
    """Export DB state to backward-compatible file formats."""
    plat_dir = os.path.join(output_dir, platform)
    os.makedirs(plat_dir, exist_ok=True)

    conn = db._get_conn()
    rows = conn.execute(
        "SELECT tag_text, frequency, cluster_id, centroid_dist, status FROM tag_registry "
        "WHERE status IN (1, 2) ORDER BY cluster_id, frequency DESC"
    ).fetchall()

    if not rows:
        logger.warning(f"  [{platform}] No clustered tags to export")
        return

    # 1. tag_clusters.jsonl
    jsonl_path = os.path.join(plat_dir, "tag_clusters.jsonl")
    cluster_index = defaultdict(list)
    tag_freq_map = {}
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            rec = {
                "tag": r["tag_text"],
                "freq": r["frequency"],
                "cluster_id": r["cluster_id"],
                "centroid_dist": round(r["centroid_dist"], 4),
                "status": "clustered" if r["status"] == 1 else "outlier",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cluster_index[r["cluster_id"]].append(r["tag_text"])
            tag_freq_map[r["tag_text"]] = r["frequency"]
    logger.info(f"  Saved {jsonl_path} ({len(rows)} tags)")

    # 2. cluster_index.json
    index_path = os.path.join(plat_dir, "cluster_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(cluster_index.items())}, f, ensure_ascii=False, indent=1)
    logger.info(f"  Saved {index_path}")

    # 3. cluster_stats.csv
    stats_path = os.path.join(plat_dir, "cluster_stats.csv")
    with open(stats_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "size", "avg_freq", "top_tags"])
        for cid in sorted(cluster_index.keys()):
            members = cluster_index[cid]
            avg_f = sum(tag_freq_map.get(t, 0) for t in members) / max(len(members), 1)
            top = sorted(members, key=lambda t: -tag_freq_map.get(t, 0))[:5]
            writer.writerow([cid, len(members), f"{avg_f:.1f}", " | ".join(top)])
    logger.info(f"  Saved {stats_path}")

    n_clusters = len(cluster_index)
    sizes = [len(v) for v in cluster_index.values()]
    logger.info(f"  [{platform}] Export: {n_clusters} clusters, {len(rows)} tags, "
                f"size: min={min(sizes)} median={sorted(sizes)[len(sizes)//2]} max={max(sizes)}")


# ======================== Main Entry Points ========================

def process_platform_incremental(
    platform: str,
    output_dir: str,
    embedding_model_path: str,
    mode: str = "full",
    dates: list[str] | None = None,
    warmup_days: int = DEFAULT_WARMUP_DAYS,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    device: str = "cuda",
):
    """Main entry: run the incremental clustering pipeline for one platform."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {platform} (mode={mode})")
    logger.info(f"{'='*60}")

    params = PLATFORM_PARAMS.get(platform, PLATFORM_PARAMS["weibo"])
    plat_dir = os.path.join(output_dir, platform)
    os.makedirs(plat_dir, exist_ok=True)
    db_path = os.path.join(plat_dir, "tag_cluster.db")
    db = TagClusterDB(db_path)
    logger.info(f"  DB: {db_path}")

    if mode == "full":
        # Discover all dates
        all_dates = dates if dates else discover_dates(platform)
        if not all_dates:
            logger.warning(f"  [{platform}] No dates found")
            return
        logger.info(f"  [{platform}] Found {len(all_dates)} dates: {all_dates[0]} ~ {all_dates[-1]}")

        # Phase 1: Ingest all dates
        logger.info(f"\n  --- Phase 1: Ingesting {len(all_dates)} dates ---")
        ingest_dates(db, platform, all_dates)

        # Phase 2: Base clustering on warmup period
        warmup_end_idx = min(warmup_days, len(all_dates)) - 1
        cutoff_date = all_dates[warmup_end_idx]
        logger.info(f"\n  --- Phase 2: Base clustering (warmup period, cutoff={cutoff_date}) ---")
        run_base_clustering(db, platform, params, cutoff_date, embedding_model_path, device)

        # Phase 3: Incremental assignment for remaining tags
        logger.info(f"\n  --- Phase 3: Incremental assignment ---")
        assign_incremental(db, platform, params, embedding_model_path, outlier_threshold, device)

        # Export
        logger.info(f"\n  --- Exporting results ---")
        export_results(db, platform, output_dir)

        # Summary
        n_clustered = db.get_clustered_count()
        n_outlier = db.get_outlier_count()
        total = db.get_tag_count()
        denom = n_clustered + n_outlier
        outlier_pct = (n_outlier / denom * 100) if denom > 0 else 0.0
        logger.info(f"  [{platform}] Summary: {total} total tags, {n_clustered} clustered, "
                    f"{n_outlier} outliers ({outlier_pct:.1f}% outlier rate)")

    elif mode == "ingest":
        if not dates:
            dates = discover_dates(platform)
        logger.info(f"  [{platform}] Ingesting {len(dates)} dates...")
        ingest_dates(db, platform, dates)

    elif mode == "recluster":
        logger.info(f"  [{platform}] Running full recluster...")
        recluster_all(db, platform, params, embedding_model_path, device)
        export_results(db, platform, output_dir)

    else:
        logger.error(f"Unknown mode: {mode}")
        sys.exit(1)


# ======================== Legacy batch interface (preserved) ========================

def load_and_merge_tag_stats(platform: str) -> dict[str, dict]:
    """Merge all sampled_tag_stats CSVs for a platform across all dates."""
    pattern = os.path.join(OUTPUT_BASE, "*/blacklist", platform, "sampled_tag_stats_*.csv")
    files = sorted(glob.glob(pattern))
    logger.info(f"[{platform}] Found {len(files)} sampled_tag_stats files")

    tag_data = defaultdict(lambda: {"freq": 0, "max_coverage": 0.0, "n_dates": 0})
    for fpath in files:
        with open(fpath, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = row.get("Tag", "").strip()
                if not tag:
                    continue
                freq = int(row.get("Frequency", 0))
                coverage_str = row.get("Coverage (%)", "0").replace("%", "")
                try:
                    coverage = float(coverage_str)
                except ValueError:
                    coverage = 0.0

                tag_data[tag]["freq"] += freq
                tag_data[tag]["max_coverage"] = max(tag_data[tag]["max_coverage"], coverage)
                tag_data[tag]["n_dates"] += 1

    logger.info(f"[{platform}] Total unique tags after merge: {len(tag_data):,}")
    return dict(tag_data)


def filter_tags(tag_data: dict[str, dict], params: dict) -> list[tuple[str, int]]:
    """Filter tags by frequency and coverage thresholds."""
    min_freq = params["min_freq"]
    max_cov = params["max_coverage"]

    filtered = []
    for tag, info in tag_data.items():
        if info["freq"] < min_freq:
            continue
        if info["max_coverage"] > max_cov:
            continue
        if len(tag) < 2:
            continue
        if all(c in "0123456789ABCDEFabcdef" for c in tag):
            continue
        filtered.append((tag, info["freq"]))

    filtered.sort(key=lambda x: -x[1])
    logger.info(f"  After filtering (min_freq={min_freq}, max_cov={max_cov}): {len(filtered):,} tags")
    return filtered


def save_results(
    platform: str,
    output_dir: str,
    tags: list[str],
    freqs: list[int],
    labels: np.ndarray,
    embeddings: np.ndarray | None = None,
    save_embeddings: bool = False,
):
    """Save cluster results in multiple formats (legacy batch interface)."""
    plat_dir = os.path.join(output_dir, platform)
    os.makedirs(plat_dir, exist_ok=True)

    jsonl_path = os.path.join(plat_dir, "tag_clusters.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for tag, freq, label in zip(tags, freqs, labels):
            f.write(json.dumps({
                "tag": tag,
                "freq": freq,
                "cluster_id": int(label),
            }, ensure_ascii=False) + "\n")
    logger.info(f"  Saved {jsonl_path}")

    cluster_index = defaultdict(list)
    for tag, label in zip(tags, labels):
        cluster_index[int(label)].append(tag)

    index_path = os.path.join(plat_dir, "cluster_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(cluster_index, f, ensure_ascii=False, indent=1)
    logger.info(f"  Saved {index_path}")

    stats_path = os.path.join(plat_dir, "cluster_stats.csv")
    tag_freq_map = dict(zip(tags, freqs))
    with open(stats_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "size", "avg_freq", "top_tags"])
        for cid in sorted(cluster_index.keys()):
            members = cluster_index[cid]
            avg_f = sum(tag_freq_map.get(t, 0) for t in members) / len(members)
            top = sorted(members, key=lambda t: -tag_freq_map.get(t, 0))[:5]
            writer.writerow([cid, len(members), f"{avg_f:.1f}", " | ".join(top)])
    logger.info(f"  Saved {stats_path}")

    if save_embeddings and embeddings is not None:
        emb_path = os.path.join(plat_dir, "embeddings.npy")
        np.save(emb_path, embeddings)
        logger.info(f"  Saved embeddings: {emb_path}")

    n_clusters = len(cluster_index)
    sizes = [len(v) for v in cluster_index.values()]
    logger.info(f"  [{platform}] {n_clusters} clusters, "
                f"size: min={min(sizes)} median={sorted(sizes)[len(sizes)//2]} max={max(sizes)}")


def process_platform(
    platform: str,
    output_dir: str,
    embedding_model_path: str,
    device: str = "cuda",
    save_embeddings: bool = False,
):
    """Legacy batch pipeline for one platform."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing (batch mode): {platform}")
    logger.info(f"{'='*60}")

    params = PLATFORM_PARAMS.get(platform, PLATFORM_PARAMS["weibo"])
    tag_data = load_and_merge_tag_stats(platform)
    filtered = filter_tags(tag_data, params)

    if not filtered:
        logger.warning(f"[{platform}] No tags after filtering, skipping.")
        return

    tags = [t for t, _ in filtered]
    freqs = [f for _, f in filtered]

    embeddings = compute_embeddings(tags, embedding_model_path, device=device)
    labels, _centroids = cluster_tags(embeddings, params["n_clusters"])
    save_results(platform, output_dir, tags, freqs, labels, embeddings, save_embeddings)


# ======================== CLI ========================

def main():
    parser = argparse.ArgumentParser(description="Incremental tag clustering with SQLite state")
    parser.add_argument("--platforms", nargs="+", default=ALL_PLATFORMS,
                        choices=ALL_PLATFORMS, help="Platforms to process")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for DB and export files")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                        help="Path to SentenceTransformer model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for embedding computation")

    # Mode selection
    parser.add_argument("--mode", default="full",
                        choices=["full", "ingest", "recluster", "batch"],
                        help="Pipeline mode: full (ingest+cluster+incremental), "
                             "ingest (add new dates only), recluster (redo all clusters), "
                             "batch (legacy one-shot pipeline)")
    parser.add_argument("--dates", nargs="+", default=None,
                        help="Specific dates to process (YYYY-MM-DD). If not set, auto-discover all.")
    parser.add_argument("--warmup-days", type=int, default=DEFAULT_WARMUP_DAYS,
                        help="Number of initial dates for base clustering warmup")
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_THRESHOLD,
                        help="Cosine distance threshold for outlier detection")

    # Legacy options
    parser.add_argument("--save-embeddings", action="store_true",
                        help="(batch mode only) Save embeddings as .npy files")

    args = parser.parse_args()

    t_start = time.time()

    if args.mode == "batch":
        # Legacy batch pipeline
        for platform in args.platforms:
            process_platform(
                platform=platform,
                output_dir=args.output_dir,
                embedding_model_path=args.embedding_model,
                device=args.device,
                save_embeddings=args.save_embeddings,
            )
    else:
        # Incremental pipeline
        for platform in args.platforms:
            process_platform_incremental(
                platform=platform,
                output_dir=args.output_dir,
                embedding_model_path=args.embedding_model,
                mode=args.mode,
                dates=args.dates,
                warmup_days=args.warmup_days,
                outlier_threshold=args.outlier_threshold,
                device=args.device,
            )

    logger.info(f"\nAll done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
