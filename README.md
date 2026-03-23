# MARS: Massive Agent-Based Real-World Simulation

MARS is a next-generation engine designed for building high-fidelity, data-driven digital twins of complex social systems.

This project provides a **complete end-to-end pipeline** — from raw social media data ingestion, through LLM-powered user profiling and tagging, to large-scale agent-based simulation and evaluation — all compatible with the [OASIS multi-agent simulation platform](https://github.com/camel-ai/oasis.git). By bridging real-world data with Large Language Models (LLMs), MARS enables sociologists and researchers to simulate information propagation, public opinion evolution, and social interventions with unprecedented realism.

> **Note on Data:** While the raw datasets used in our research are not publicly available due to licensing restrictions, MARS provides a fully functional end-to-end implementation. Users can supply their own social media data conforming to the [Input Data Requirements](#input-data-requirements) below, and the entire pipeline — data processing, user profiling, tagging, simulation, and evaluation — will run out of the box.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Input Data Requirements](#input-data-requirements)
- [Directory Structure](#directory-structure)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1: Data Processing](#stage-1-data-processing-codedata_process)
  - [Stage 2: User Tagging](#stage-2-user-tagging-codeuser_tagging)
  - [Stage 3: User Selection](#stage-3-user-selection-codeuser_selection)
  - [Stage 4: Simulation & Evaluation](#stage-4-simulation--evaluation-codesimulation_process)
- [OASIS Simulation Engine](#oasis-simulation-engine-oasis)
- [Key Features](#key-features)
- [Quick Start](#quick-start)

---

## Architecture Overview

MARS consists of **four pipeline stages** and an **underlying simulation engine**. Each stage's output feeds into the next:

```
data/raw/YYYY-MM-DD/*.txt (Raw JSON)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: data_process/                                  │
│  data_process.py (orchestrator)                          │
│  ├── user_profile.py  → per-platform CSV user profiles  │
│  └── user_post.py     → per-platform SQLite post DBs    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
  MARS_result/data/output/YYYY-MM-DD/
  ├── user_profile_{platform}.csv
  └── posts_{platform}.db
        │
        ├──────────────────────────────┐
        ▼                              ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│ Stage 2: user_tagging│  │ Stage 3: user_selection       │
│                      │  │                                │
│ run_sft_tagging.py  │  │ selection_process.py           │
│ → interest tree +   │  │ → oasis_agent_init.csv        │
│   user profiles     │  │ → oasis_database.db           │
│                      │  └──────────────┬────────────────┘
│ run_benchmark.py    │                 │
│ → benchmark dataset │                 ▼
│                      │  ┌───────────────────────────────┐
│ run_blacklist.py    │  │ Stage 4: simulation_process/   │
│ → tag blacklist     │  │                                │
│                      │  │ oasis_attitude.py → annotation│
│ run_clustering.py   │  │ oasis_evaluation.py→ metrics  │
│ → tag clustering    │  └───────────────────────────────┘
└──────────────────────┘
```

---

## Input Data Requirements

MARS is designed as a **data-agnostic end-to-end system**. While we developed it using Chinese social media data, the pipeline can work with any social media dataset that provides the required attributes. Users have **two entry points** depending on the format of their data:

```
                  ┌──────────────────────────────────┐
                  │  Option A: Raw JSON               │
                  │  (start from Stage 1)             │
                  │                                    │
                  │  Provide per-record JSON with      │
                  │  user + post attributes             │
                  └──────────┬───────────────────────┘
                             │
                             ▼
                  ┌──────────────────────────────────┐
                  │  Stage 1: data_process            │
                  │  Internally converts JSON to:     │
                  │  • SQLite post databases           │
                  │  • CSV user profile files          │
                  └──────────┬───────────────────────┘
                             │
                             ▼
                  ┌──────────────────────────────────┐
                  │  Option B: Pre-structured Data    │
                  │  (start directly from Stage 2)    │
                  │                                    │
                  │  Provide SQLite DBs + CSV files    │
                  │  matching the schemas below        │
                  └──────────┬───────────────────────┘
                             │
                             ▼
                    Stage 2 → 3 → 4 ...
```

### Option A: Raw JSON Input (Full Pipeline from Stage 1)

Place your data in `data/raw/YYYY-MM-DD/` directories (one directory per date). Each directory should contain `.txt` files (or `.zip` archives of `.txt` files) where **each line is a valid JSON object** representing one social media record.

Each JSON record must include the following **logical attributes**. The internal field naming is handled by the data_process layer — you only need to ensure these attributes are present:

#### Post Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | Unique identifier of the post author |
| `content` | string | Yes | Post text content |
| `title` | string | No | Post title (e.g., for articles or threads) |
| `publish_time` | string | Yes | Publish timestamp (ISO 8601 recommended, e.g., `2025-06-15T14:30:00`) |
| `platform` | string | Yes | Platform identifier (e.g., `"weibo"`, `"twitter"`, `"reddit"`) |
| `quote_content` | string | No | Content of the quoted/reposted original post |

#### User Profile Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | Unique user identifier (must match post author IDs) |
| `nickname` | string | Yes | Display name |
| `followers_count` | int | Recommended | Number of followers (used for user grouping) |
| `following_count` | int | No | Number of accounts followed |
| `gender` | string | No | Gender (`"male"` / `"female"` / `"unknown"`) |
| `location` | string | No | User-reported location |
| `verified` | bool | No | Whether the account is officially verified |
| `description` | string | No | User bio / self-description |
| `platform` | string | Yes | Platform identifier |

#### Engagement Attributes (Optional)

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `mentions` | list[string] | No | List of @mentioned user IDs |
| `like_count` | int | No | Number of likes |
| `comment_count` | int | No | Number of comments |
| `share_count` | int | No | Number of shares/reposts |

> **Adapting your JSON schema:** If your raw data uses different field names, you can write a lightweight adapter in `code/data_process/user_post.py` (add a new `parse_yourplatform()` function and register it in `PLATFORM_MAP`). The rest of the pipeline remains unchanged.

### Option B: Pre-structured Data (Skip Stage 1, Start from Stage 2)

If you already have structured data (e.g., from a database export or another ETL tool), you can skip Stage 1 entirely by providing files that match the following schemas.

#### Post Database — SQLite (`posts_{platform}.db`)

A SQLite database with a `posts` table:

```sql
CREATE TABLE posts (
    user_id       TEXT NOT NULL,
    title         TEXT DEFAULT '',
    content       TEXT NOT NULL,
    quote_content TEXT DEFAULT '',
    created_at    TEXT NOT NULL    -- ISO 8601 timestamp
);
```

You can have one DB per platform (e.g., `posts_twitter.db`, `posts_reddit.db`) or a single combined DB.

#### User Profile — CSV (`user_profile_{platform}.csv`)

A CSV file with at minimum these columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `user_id` | string | Yes | Must match `user_id` in the post database |
| `nickname` | string | Yes | Display name |
| `platform` | string | Yes | Platform identifier |
| `followers_count` | int | Recommended | Used for quantile-based user grouping |
| `following_count` | int | No | |
| `gender` | string | No | |
| `location` | string | No | |
| `verified` | bool | No | |
| `description` | string | No | User bio |

Place these files in `MARS_result/data/output/YYYY-MM-DD/` and proceed directly to Stage 2 (user tagging) or Stage 3 (user selection).

### Minimum Viable Dataset

| Component | Minimum Requirement |
|-----------|-------------------|
| Users | ~100 users with posts |
| Posts per user | ≥5 posts (needed by the buffer threshold for LLM tagging) |
| Time span | ≥1 day (for temporal calibration / ground-truth split) |
| Core fields | `user_id`, `content`, `publish_time`, `platform` |
| LLM access | A running vLLM server or OpenAI-compatible API endpoint |
| Embedding model | `BAAI/bge-small-zh-v1.5` (or equivalent sentence embedding model) |

### End-to-End Pipeline Overview

```
Option A: Your Raw JSON
    → [Stage 1] data_process   → structured DBs + CSVs
    → [Stage 2] user_tagging   → interest trees + demographic profiles
    → [Stage 3] user_selection → OASIS simulation inputs
    → [Stage 4] simulation     → attitude annotation → evaluation metrics

Option B: Your SQLite DBs + CSVs
    → [Stage 2] user_tagging   → interest trees + demographic profiles
    → [Stage 3] user_selection → OASIS simulation inputs
    → [Stage 4] simulation     → attitude annotation → evaluation metrics
```

Each stage is independently runnable, so you can also use individual components (e.g., just the tagging pipeline) without running the full simulation.

---

## Directory Structure

```
MARS/
├── code/
│   ├── data_process/                  # Stage 1: Raw JSON → Structured DB + CSV
│   │   ├── data_process.py            #   Pipeline orchestrator (entry point)
│   │   ├── user_post.py               #   Post ETL: JSON → per-platform SQLite DB
│   │   ├── user_profile.py            #   Profile extraction: JSON → per-platform CSV
│   │   ├── user_relationship.py       #   Social graph extraction: JSON → edge CSV
│   │   └── test_single_file.py        #   Debug script for single file testing
│   │
│   ├── user_tagging/                  # Stage 2: LLM-based user profiling
│   │   ├── src/
│   │   │   ├── core/                  #   Infrastructure (API/DB/NLP/Prompt)
│   │   │   │   ├── api_client.py      #     Unified LLM + Embedding client
│   │   │   │   ├── db_client.py       #     Thread-safe SQLite reader
│   │   │   │   ├── nlp_processor.py   #     spaCy NER & candidate extraction
│   │   │   │   ├── prompt_manager.py  #     Template loading & variable substitution
│   │   │   │   ├── io_helpers.py      #     Config/CSV/log utilities
│   │   │   │   └── merge_util.py      #     Interest tree & profile merging
│   │   │   │
│   │   │   ├── platforms/             #   Platform adapters (5 platforms x 2 files)
│   │   │   │   ├── base_adapter.py    #     ABC: extract_tags, clean_text, normalize
│   │   │   │   ├── base_filter.py     #     ABC: 3-layer filtering framework
│   │   │   │   ├── weibo_*.py         #     Weibo: hashtag #tag#, fandom detection
│   │   │   │   ├── zhihu_*.py         #     Zhihu: Q&A format handling
│   │   │   │   ├── xiaohongshu_*.py   #     Xiaohongshu: note format handling
│   │   │   │   ├── toutiao_*.py       #     Toutiao: news format handling
│   │   │   │   └── douban_*.py        #     Douban: review format handling
│   │   │   │
│   │   │   ├── tasks/
│   │   │   │   ├── sft/               #   SFT dataset generation
│   │   │   │   │   ├── tagging_service.py   # Core incremental tagging orchestrator
│   │   │   │   │   ├── state_manager.py     # User state & tag registry persistence
│   │   │   │   │   └── dataset_manager.py   # SFT quality control & classification
│   │   │   │   ├── benchmark/         #   Benchmark evaluation dataset
│   │   │   │   │   ├── bench_service.py
│   │   │   │   │   ├── state_manager_bench.py
│   │   │   │   │   └── benchmark_selector.py
│   │   │   │   └── clustering/        #   Offline tag clustering
│   │   │   │       └── clustering_manager.py
│   │   │   │
│   │   │   ├── scripts/               #   Entry point scripts
│   │   │   │   ├── run_sft_tagging.py #     ★ Main SFT tagging pipeline
│   │   │   │   ├── run_benchmark.py   #     Single-date benchmark generation
│   │   │   │   ├── run_benchmark_all.py #   Multi-date benchmark orchestrator
│   │   │   │   ├── run_blacklist.py   #     Tag frequency stats & blacklist
│   │   │   │   └── run_clustering.py  #     Offline tag clustering job
│   │   │   │
│   │   │   ├── prompts/               #   LLM prompt templates
│   │   │   │   ├── interest_prompt.md #     L1→L2→L3 interest tree extraction
│   │   │   │   ├── demographic_prompt.md  # Individual user demographics
│   │   │   │   ├── firmographic_prompt.md # Organization profiling
│   │   │   │   └── domain_description.json # 41 L1 domain definitions
│   │   │   │
│   │   │   └── figure/                #   Visualization utilities
│   │   │
│   │   └── configs/                   #   Platform YAML configs (10 files)
│   │       ├── {platform}_tagging.yaml  # Tagging pipeline config
│   │       └── {platform}_bench.yaml    # Benchmark pipeline config
│   │
│   ├── user_selection/                # Stage 3: User filtering + OASIS format conversion
│   │   ├── selection_process.py       #   CLI entry point (select/convert/all)
│   │   ├── user_selection.py          #   Criteria-based user filtering
│   │   ├── oasis_user.py             #   Profile → oasis_agent_init.csv builder
│   │   ├── oasis_post_from_processed.py # Platform DB → oasis_database.db
│   │   ├── oasis_relaitonship.py     #   Relationship edge normalization
│   │   └── data_selection.py          #   Multi-date subset collector
│   │
│   ├── simulation_process/            # Stage 4: Simulation evaluation
│   │   ├── oasis_attitude.py          #   Multi-dimension attitude annotator
│   │   ├── attitude_annotator.py      #   LLM backends (OpenAI/vLLM/Local)
│   │   ├── oasis_evaluation_overall.py #  Bias/Diversity/Pearson metrics
│   │   ├── oasis_test_grouping.py     #   Agent tier assignment & activation
│   │   ├── db_manager.py             #   DB connection retry & table reset
│   │   └── intervention_processor.py  #   Broadcast/bribery/registration system
│   │
│   ├── data_process.sh                # Shell wrapper for batch date processing
│   └── user_selection.sh              # Shell wrapper for user selection
│
└── oasis/                             # OASIS simulation engine (core platform)
    ├── social_agent/                  #   LLM-powered agent implementation
    ├── social_platform/               #   Platform simulation (DB/recsys/schema)
    ├── environment/                   #   OpenAI Gym-style environment
    └── clock/                         #   Simulation clock
```

---

## Pipeline Stages

### Stage 1: Data Processing (`code/data_process/`)

Cleans and transforms raw JSON data into structured SQLite databases and CSV user profiles.

#### `data_process.py` — Pipeline Orchestrator

Command-line entry point supporting four execution modes:

| Mode | Behavior |
|------|----------|
| `all` | Unzip → extract profiles → ingest posts (full pipeline) |
| `posts` | Unzip + post ingestion only |
| `profiles` | Unzip + profile extraction only |
| `relationships` | Unzip + relationship extraction only |

- Auto-extracts `.zip` files; tracks completed zips in `.processed_zips.log` to avoid re-extraction
- In `all` mode, executes sequentially: `profiles → posts` (`relationships` is currently commented out)
- Configures sub-module paths by injecting module-level global variables

#### `user_post.py` — Post ETL (`MultiDBProcessor`)

Reads raw JSON (one record per line), detects the platform, parses post fields, and **writes to separate SQLite databases per platform**.

**Processing flow:**
1. Scans `.txt` files, skipping those already recorded in `processed_post.log`
2. Parallel processing via `multiprocessing`, one worker per file
3. `dispatch_parse()` detects platform via `PLATFORM_MAP`, routes to `parse_weibo()` / `parse_toutiao()` etc.
4. Extracts fields: `user_id`, `title`, `content`, `quote_content`, `created_at`
5. `_build_quote_chain()` constructs repost/quote chains
6. Batch INSERT every 100K records; uses `orjson` for fast JSON parsing
7. Writes to temp dir `/root/sztemp/` first, then moves to final directory (avoids NFS write bottleneck)

**SQLite write optimizations:** `PRAGMA journal_mode=OFF`, `synchronous=OFF`, large `cache_size`

**Output:** `posts_weibo.db`, `posts_toutiao.db`, `posts_xiaohongshu.db`, `posts_zhihu.db`, `posts_douban.db`

#### `user_profile.py` — Profile Extraction (`ParallelProcessor`)

Extracts user metadata from multiple pojo types in raw JSON, deduplicates, and outputs per-platform CSV files.

**Processing flow:**
1. On startup, loads existing user_id sets from CSV for cross-run deduplication
2. Parallel workers process each file, extracting from `authorContentPojo` (post author), `authorCommentPojo` (comment author), `authorContentRootPojo` (original post author)
3. `FIELD_MAPPING` maps raw field names to standardized columns (e.g., `sjcjNickName` → `nickname`)
4. `PLATFORM_SCHEMA` filters valid fields per platform
5. **Core user selection logic:** `verified=True` OR `followers>10000` OR (`followers>1000` AND `random<0.3`)
6. Flushes to CSV every 5000 users or 50 files

**Output:** `user_profile_weibo.csv`, `user_profile_toutiao.csv`, etc.

#### `user_relationship.py` — Social Graph Extraction (`FastRelationshipPipeline`)

Extracts inter-user relationships from @mention, repost, and follow fields, building weighted edge lists.

**Two-phase MapReduce:**
- **Map:** Each worker independently parses files, locally aggregates edges, writes to temp CSV
- **Reduce:** Merges all temp CSVs with global GroupBy weight aggregation

**Edge weights:** mention/@ = 3, repost/follow = 2, other = 1

**Output:** `merged_edges.csv`, `normalized_edges.csv`, `follow_list.csv`

> Note: Currently commented out in `data_process.py:run_all()` (line 160).

---

### Stage 2: User Tagging (`code/user_tagging/`)

Uses LLMs to automatically profile users, generating **3-level interest taxonomy trees** (L1→L2→L3) and **demographic/firmographic profiles**. This is the most architecturally complex module in the system.

#### Core Design: Incremental User Profiling

The system uses an **incremental accumulation + buffer-triggered** architecture rather than processing all posts at once:

```
User post stream (daily increments)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Multi-Layer Filter Pipeline                              │
│                                                          │
│ Layer 1 — User-level (is_user_qualified)                │
│   Validates basic user attributes (real user check)      │
│                                                          │
│ Layer 2 — Batch-level (validate_batch_behavior)         │
│   ├─ Exact repetition > 20%       → bot suspect        │
│   ├─ Fuzzy similarity > 75%       → marketing suspect  │
│   ├─ Personality ratio < 10%      → repost-only        │
│   └─ Embedding coherence ∉ [0.15, 0.85] → anomaly     │
│                                                          │
│ Layer 3 — Post-level (per post)                         │
│   ├─ fatal_risk: sensitive content detection            │
│   ├─ cleanability: symbol spam / garbled text           │
│   └─ meaningfulness: substantive content check          │
└───────────────────────────┬─────────────────────────────┘
                            │ valid posts
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Buffer Accumulation                                      │
│                                                          │
│ Posts → user_pending_buffer table                        │
│ count < trigger_threshold (5) → return "buffered"       │
│ count ≥ trigger_threshold     → trigger LLM inference   │
│ max_posts_per_call = 50                                  │
└───────────────────────────┬─────────────────────────────┘
                            │ threshold reached
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Parallel Dual-Task LLM Inference                         │
│                                                          │
│ Task 1: Interest Tree (interest_prompt.md)              │
│ ├─ Input: posts + NLP candidates + persona_summary      │
│ ├─ Output: L1→L2→L3 tree + updated persona_summary     │
│ └─ L1: 41 domains | L2: [WORKS]/[PERSON]/[PRODUCT]     │
│    L3: concrete entities                                 │
│                                                          │
│ Task 2: Profile Inference                                │
│ ├─ Individual → demographic_prompt.md                   │
│ │   (gender/age/education/residence/job/income)          │
│ └─ Organization → firmographic_prompt.md                │
│     (org_type/industry/function)                         │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Merge & Quality Control                                  │
│                                                          │
│ merge_interest_trees(): confidence-based tree fusion     │
│ merge_profiles(): high-confidence overwrites low         │
│ persona_summary: cognitive context for next batch        │
│                                                          │
│ SFT Quality Gate (6 checks):                             │
│ ├─ Content richness ≥ 50 chars                          │
│ ├─ Complete L1→L2→L3 paths exist                        │
│ ├─ Focus: dominant L1 ≥ 40% of leaf nodes               │
│ ├─ Grounding: ≥ 20% L3 tags appear in source posts     │
│ ├─ Confidence quota: High:Mid:NA = 7:2:1               │
│ └─ L1 balance: no domain > 1.5x average                │
│                                                          │
│ Pass → sft_training.jsonl | Fail → state DB (normal)    │
└─────────────────────────────────────────────────────────┘
```

**Interest Tree Example Output:**
```json
{
  "interest_tree": [
    {
      "L1": "Gaming",
      "confidence": "High",
      "reasoning": "Multiple posts discuss mobile games...",
      "sub_domains": [
        {
          "L2": "Mobile Games",
          "meta_type": "PRODUCT",
          "entities": [
            {"L3": "Honor of Kings", "confidence": "High"},
            {"L3": "PUBG Mobile", "confidence": "Medium"}
          ]
        }
      ]
    }
  ],
  "persona_summary": "Core gamer, focused on competitive mobile games..."
}
```

#### Core Components (`src/core/`)

| File | Class | Description |
|------|-------|-------------|
| `api_client.py` | `APIClient` | Unified LLM + Embedding interface. Embedding always uses local BGE model (CPU). LLM supports remote (vLLM/OpenAI) and local (Transformers). Built-in JSON parsing with Markdown cleanup. Default temperature=0.1 |
| `db_client.py` | `DBClient` | Thread-safe SQLite reader. `get_batch_posts()` filters spam users (>max_count posts), then fetches details. `get_user_posts_incremental()` supports cursor-based incremental fetch. UTC→GMT+8 conversion |
| `nlp_processor.py` | `NLPProcessor` | spaCy-based (`zh_core_web_sm`) NER and candidate extraction. Priority: named entities > proper nouns > dep-filtered nouns. Outputs up to 100 `"entity (LABEL)"` candidates. Pre-filters URLs/HTML/hex strings |
| `prompt_manager.py` | `PromptManager` | Loads `.md` templates from `src/prompts/`, supports `{username}/{posts}/{candidate_entities}` variable substitution. Three main templates: interest / demographic / firmographic |
| `io_helpers.py` | Utilities | YAML config loading, CSV reading, processed ID list management |
| `merge_util.py` | Merge functions | `merge_interest_trees()`: confidence-based tree fusion; `merge_profiles()`: high-confidence overwrite |

#### Platform Adapters (`src/platforms/`)

Each platform has an **Adapter** (data normalization) and **Filter** (quality filtering):

| Files | Platform-Specific Behavior |
|-------|---------------------------|
| `base_adapter.py` | ABC defining `extract_tags()`, `normalize_user()`, `clean_text()`, `normalize_post()` |
| `base_filter.py` | ABC defining 3-layer filtering: user-level → batch-level (repetition, personality ratio, embedding coherence) → post-level (fatal risk, cleanability, meaningfulness) |
| `weibo_*.py` | Weibo: `#tag#` hashtag extraction, fandom text detection |
| `zhihu_*.py` | Zhihu: Q&A format handling |
| `xiaohongshu_*.py` | Xiaohongshu: visual note format handling |
| `toutiao_*.py` | Toutiao: news/article format handling |
| `douban_*.py` | Douban: review/short comment format handling |

#### Task Implementations (`src/tasks/`)

| Module | Core Class | Description |
|--------|-----------|-------------|
| `sft/tagging_service.py` | `TaggingService` | **Core orchestrator.** `process_user_incremental()` chains: static filter → fetch & clean posts → buffer management → dual-task LLM inference → history merge → SFT classification → state persistence |
| `sft/state_manager.py` | `StateManager` | SQLite persistence with 4 tables: `user_full_state` (tree/profile/persona snapshots), `user_pending_buffer` (post accumulation), `global_tag_registry` (L2/L3 tags + embeddings + cluster IDs), `global_taxonomy_graph` (L1→L2→L3 edges + frequencies). Thread-local connections, WAL mode |
| `sft/dataset_manager.py` | `DatasetManager` | SFT quality gate: 6-check classification (structure/focus/grounding/confidence quota/L1 balance). Qualified users → `sft_training.jsonl` |
| `benchmark/bench_service.py` | `BenchService` | Benchmark variant of TaggingService for evaluation dataset generation |
| `benchmark/state_manager_bench.py` | `BenchStateManager` | Separate state DB for benchmark data |
| `benchmark/benchmark_selector.py` | `BenchmarkSelector` | Sample selection strategy for benchmark users |
| `clustering/clustering_manager.py` | `ClusteringManager` | Offline tag clustering: read all tags → BGE embedding → semantic clustering → dedup & canonicalize → update `cluster_id` |

#### Prompt Templates (`src/prompts/`)

| File | Purpose | Key Content |
|------|---------|-------------|
| `interest_prompt.md` | Interest tree extraction | 41 official L1 domains; L2 containers [WORKS]/[PERSON]/[PRODUCT]; 5-level confidence; historical persona as Internal Memory State S_{n-1} |
| `demographic_prompt.md` | Individual demographics | 8 dimensions: gender / age(6 bands) / education(7 levels) / residence(5 types) / marriage(5 states) / job(17 categories) / personal income(5 bands) / family income(5 bands) |
| `firmographic_prompt.md` | Organization profiling | 3 dimensions: org_type(4) / industry(12) / function(6); maps from verified_type codes |
| `domain_description.json` | L1 domain definitions | Detailed text descriptions of 41 L1 domains to guide LLM classification |

#### Configuration (`configs/`)

Each platform has `{platform}_tagging.yaml` and `{platform}_bench.yaml`:

```yaml
data_source:
  user_pool_filename: "{date}_users.csv"     # Daily user list filename template

fetch_limits:
  max_count: 10                              # Users with >N posts treated as spam

buffer_rules:
  trigger_threshold: 5                       # Accumulate N posts before LLM call
  max_posts_per_call: 50                     # Max posts per LLM inference

quality_rules:
  behavior_risk:
    min_personality_ratio: 0.1               # ≥10% posts must show personal opinion
    repetition_threshold: 0.20               # Exact duplicate threshold
    fuzzy_repetition_threshold: 0.75         # Near-duplicate similarity threshold
    coherence:
      low_threshold: 0.15                    # Reject if avg embedding sim < 15%
      high_threshold: 0.85                   # Reject if avg embedding sim > 85%

blacklist:
  enable: true/false                         # Enable tag blacklist generation
  sample_size: 500000                        # Posts to sample for frequency analysis
  threshold_percent: 0.02                    # Top 2% frequent tags → blacklist

llm_service:
  mode: remote                               # "remote" (vLLM/OpenAI) or "local"
  api_key: "${API_KEY}"
  base_url: "http://localhost:8000/v1"       # vLLM endpoint
  model_name: "deepseek-chat"
  embedding_model_path: "MARS_result/model/embedding/BAAI/bge-small-zh-v1.5"
  max_workers: 10                            # Thread pool concurrency
  timeout: 120                               # API timeout in seconds
```

#### Entry Scripts (`src/scripts/`)

**`run_sft_tagging.py`** — Main SFT tagging pipeline (most frequently used):

```bash
python MARS/code/user_tagging/src/scripts/run_sft_tagging.py \
    --date 2025-06-15 \
    --platform weibo \
    --db-base-dir MARS_result/data/output \
    --output-dir MARS_result/data/online_cold_start
```

Flow: load YAML config → copy DB to temp dir → load user CSV (skip processed) → Macro Batch (100 users) → parallel `TaggingService.process_user_incremental()` → export master JSONL → cleanup temp DB.

**`run_benchmark.py`** — Benchmark dataset generation:

```bash
python MARS/code/user_tagging/src/scripts/run_benchmark.py \
    --date 2025-06-15 \
    --platform weibo \
    --db-base-dir MARS_result/data/output \
    --output-dir MARS_result/data/benchmark
```

Similar to SFT but uses `BenchService` + `BenchStateManager`, generates tag stats + blacklist first (if enabled), larger Macro Batch (500), outputs anchor statistics CSV.

**`run_benchmark_all.py`** — Multi-date benchmark orchestrator:

```bash
python MARS/code/user_tagging/src/scripts/run_benchmark_all.py \
    --start-date 2025-06-04 \
    --end-date 2025-06-15 \
    --platforms weibo zhihu toutiao xiaohongshu douban
```

Iterates date range × platform list, spawns `run_benchmark.py` subprocess for each combo.

**`run_blacklist.py`** — Tag frequency analysis & blacklist generation:

```bash
python MARS/code/user_tagging/src/scripts/run_blacklist.py \
    --date 2025-06-15 \
    --platform weibo \
    --db-base-dir MARS_result/data/output \
    --output-dir MARS_result/data/output/2025-06-15/blacklist/weibo
```

Flow: open DB (read-only) → random sample N posts → extract anchors via platform filter → frequency stats → output CSV `[rank, tag, freq, coverage%, status]` → write blacklist TXT (tags above threshold).

**`run_clustering.py`** — Offline tag clustering & normalization:

```bash
python MARS/code/user_tagging/src/scripts/run_clustering.py \
    --platform weibo \
    --output-dir MARS_result/data/online_cold_start
```

Flow: load all tags from `global_tag_registry` → BGE embedding → semantic clustering → dedup → update `cluster_id` and `canonical_id`.

---

### Stage 3: User Selection (`code/user_selection/`)

Filters user subsets from Stage 1 outputs and converts them into OASIS simulation engine input format.

#### `selection_process.py` — CLI Entry Point

Supports 4 subcommands:

| Subcommand | Function | Core Class |
|------------|----------|------------|
| `select` | Filter users by criteria → `selected.txt` | `UserSelector` |
| `convert_profiles` | Profile CSV → `oasis_agent_init.csv` | `OasisUserBuilder` |
| `convert_posts` | Platform DB → `oasis_database.db` | `ProcessedPostCollector` |
| `all` | Run all three steps sequentially | All of the above |

#### `user_selection.py` — Criteria-Based Filtering (`UserSelector`)

Loads profile CSV and supports dynamic query syntax:

| Syntax | Meaning | Example |
|--------|---------|---------|
| `min_COLUMN=N` | Numeric lower bound | `min_followers_count=10000` |
| `max_COLUMN=N` | Numeric upper bound | `max_posts_count=500` |
| `COLUMN_in=A,B` | Enum membership | `province_in=Beijing,Shanghai` |
| `COLUMN=VALUE` | Exact match | `gender=female`, `verified=True` |

Uses pandas `query()` for efficient filtering. Outputs one `user_id` per line.

#### `oasis_user.py` — Agent Initialization (`OasisUserBuilder`)

Transforms profile CSV + relationship CSV into `oasis_agent_init.csv`:

1. Assigns sequential `agent_id` (0 to N)
2. **Quantile-based user grouping:**

| Group | Criteria | Simulation Tier |
|-------|----------|-----------------|
| Authority Media / Top Influencer | followers > Q99 | Tier 1 (LLM) |
| Active KOL | followers > Q90 AND posts > Q90 | Tier 1 (LLM) |
| Active Creator | followers ≤ Q90 AND posts > Q80 | Tier 1 (LLM) |
| Regular User | Default | Tier 1 (LLM) |
| Lurker | followers ≤ Q50 AND posts ≤ Q50 | Tier 2 (ABM) |

3. Maps following relationships from user_ids to agent_ids
4. Output columns: `user_id, name, username, following_agentid_list, user_char, description, group`

#### `oasis_post_from_processed.py` — Simulation DB (`ProcessedPostCollector`)

Collects posts from `posts_{platform}.db` into unified `oasis_database.db` with temporal split:

| Time Range | Target Table | Purpose |
|-----------|-------------|---------|
| `created_at ≤ calibration_end` | `post` | Calibration set (agent history) |
| `calibration_end < created_at ≤ ground_truth_end` | `ground_truth_post` | Ground truth (evaluation) |
| `created_at > ground_truth_end` | Discarded | — |

Auto-discovers all `posts_*.db` files, merges title+content, establishes quote relationships.

#### Other Files

| File | Description |
|------|-------------|
| `oasis_relaitonship.py` | Merges weight CSVs → normalizes (max/sum method) → quantile threshold filtering → `follow_list.csv` |
| `data_selection.py` | `SubsetCollector`: collects profiles/edges/posts subsets across multiple date directories for a given user set |

---

### Stage 4: Simulation & Evaluation (`code/simulation_process/`)

Post-simulation attitude annotation and effectiveness evaluation.

#### `oasis_attitude.py` — Attitude Processor (`OasisAttitudeProcessor`)

Annotates posts in the simulation database with multi-dimensional attitude scores (-1.0 to 1.0), computing each user's initial and final attitude scores.

**Steps:**
1. Annotate `post` table (calibration set) via LLM
2. Annotate `ground_truth_post` table via LLM
3. Generate user scores CSV: `initial_{dim}` (avg from post), `final_{dim}` (avg from post + ground_truth_post), plus overall `attitude_avg`

**Attitude dimensions** are configurable:
```python
attitude_config = {
    'attitude_lifestyle_culture': "Evaluate sentiment towards lifestyle and cultural topics.",
    'attitude_sport_ent': "Evaluate sentiment towards sports and entertainment.",
    'attitude_sci_health': "Evaluate trust in science and health information.",
    'attitude_politics_econ': "Evaluate stance on political and economic stability."
}
```

#### `attitude_annotator.py` — LLM Annotation Backends

| Class | Backend | Use Case |
|-------|---------|----------|
| `OpenAIAttitudeAnnotator` | OpenAI API (gpt-4o-mini) | Requires API key |
| `_VLLMAttitudeAnnotator` | Remote vLLM server | Self-hosted inference |
| `VLLMAttitudeAnnotator` | Local Transformers | Direct model loading, forced concurrency=1 to avoid OOM |

Flow: fetch unannotated posts → format prompt `[User Comment] / [Forwarded Original]` → call LLM → parse JSON scores → update DB with `attitude_annotated = 1`.

#### `oasis_evaluation_overall.py` — Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **Bias** | Mean attitude value per time step per dimension |
| **Diversity** | Std deviation per time step per dimension |
| **Pearson** | Correlation between simulation and ground truth |

Separates LLM Agent vs ABM Agent metrics. Maps simulation time steps to real timestamps (`TIME_STEP_MINUTES=3`). Outputs `attitude_timeseries_chart.png`.

#### `oasis_test_grouping.py` — Agent Tier Assignment

| Tier | Groups | AI Strategy | Activation Rate |
|------|--------|------------|-----------------|
| Tier 1 (LLM) | Authority Media / Top Influencer | LLMAction | 0.8 |
| Tier 1 (LLM) | Active KOL | LLMAction | 0.7 |
| Tier 1 (LLM) | Active Creator | LLMAction | 0.6 |
| Tier 1 (LLM) | Regular User | LLMAction | 0.3 |
| Tier 2 (ABM) | Lurker | HeuristicAction | 0.1 |

Stochastically samples active agents per step to reduce computation.

#### `db_manager.py` — Database Management

- `_connect_with_retry()`: 12 retries on SQLite lock (1s intervals)
- `reset_simulation_tables()`: Clears simulation artifacts, preserves `post` and `ground_truth_post`
- `DB_BUSY_TIMEOUT_MS = 15000`

#### `intervention_processor.py` — Intervention System (`InterventionProcessor`)

Reads intervention CSV, supports 3 intervention types:

| Type | Description | Target |
|------|-------------|--------|
| `broadcast` | Global system message | All agents |
| `bribery` | Directed attitude/instruction override | Specific group (by ratio) or agent ID |
| `register_user` | Dynamic agent injection | New agent with specified profile |

CSV columns: `time_step, intervention_type, content, target_group, target_id, ratio, attitude_target, user_profile`

---

## OASIS Simulation Engine (`oasis/`)

The underlying simulation platform built on the CAMEL-AI framework. Typically does not require direct modification.

```
oasis/
├── social_agent/
│   ├── agent.py               # Core SocialAgent (CAMEL-AI based)
│   ├── agent_action.py        # Action execution (91 action types)
│   ├── agent_attitude.py      # Attitude state management
│   ├── agent_graph.py         # Social relationship graph
│   └── agents_generator.py    # Batch agent instantiation
│
├── social_platform/
│   ├── platform.py            # ★ Core platform class (73KB) — main simulation loop
│   ├── database.py            # SQLite CRUD operations
│   ├── recsys.py              # Recommendation algorithms (Twitter/TWHIN/Reddit/Random)
│   ├── typing.py              # ActionType enum (91 actions) + RecsysType enum
│   └── schema/                # 18 SQL schema files
│       ├── user.sql, post.sql, comment.sql
│       ├── like.sql, dislike.sql, follow.sql
│       ├── rec.sql, trace.sql, intervention_message.sql
│       └── ...
│
├── environment/
│   ├── env.py                 # OpenAI Gym-style environment wrapper
│   └── make.py                # Environment factory function
│
└── clock/
    └── clock.py               # Simulation clock management
```

---

## Key Features

### 1. Hybrid Agent Architecture

Dual-tier agent system balancing fidelity and efficiency:

- **Tier 1 (LLM Agents / SocialAgent):** Authority Media, KOLs, Active Creators. Powered by LLMs (Qwen2.5, GPT-4) with complex personas, memory, and reasoning for high-quality content generation.
- **Tier 2 (ABM Agents / LurkerAgent):** Silent Majority / passive users. Powered by heuristic models (e.g., Bounded Confidence Model) for efficient state evolution without LLM inference, enabling scale to hundreds of thousands of agents.

### 2. User Grouping & Dynamic Activation

- **Automatic Categorization:** Users classified by follower counts and posting frequency into 5 groups
- **Probabilistic Activation:** High-influence agents (Tier 1) have 30-80% activation per step; Tier 2 agents activate at 10%, simulating realistic activity patterns

### 3. Customizable Evaluation Metrics

- Define attitude metrics as a `{column_name: description}` dictionary
- Engine auto-generates DB schema, SQL queries, and LLM system prompts
- No code changes needed for new research dimensions

### 4. Internal State & Tool-Calling Mechanism

"Mind-Action" consistency loop using Parallel Function Calling:
- **Internal Update:** `update_internal_attitude` adjusts psychological state
- **External Action:** Social tools (create_post, like, repost) reflect updated state
- All state changes logged for temporal analysis

### 5. Multi-Modal Intervention System

Three intervention types via unified CSV:
- **Broadcast:** Global messages visible to all agents
- **Bribery (Coercion):** Targeted instructions overriding agent persona
- **Register User (Astroturfing):** Dynamic agent injection with pre-defined profiles

---

## Quick Start

```bash
# 0. Activate environment
source /remote-home/JuelinW/anaconda3/bin/activate oasis
cd /remote-home/JuelinW/oasis_project

# 1. Data Processing — raw JSON → structured DB + CSV
bash MARS/code/data_process.sh 2025-06-15 2025-06-15

# 2a. User Tagging — LLM-based profiling
python MARS/code/user_tagging/src/scripts/run_sft_tagging.py \
    --date 2025-06-15 --platform weibo \
    --db-base-dir MARS_result/data/output \
    --output-dir MARS_result/data/online_cold_start

# 2b. Benchmark generation (optional)
python MARS/code/user_tagging/src/scripts/run_benchmark.py \
    --date 2025-06-15 --platform weibo \
    --db-base-dir MARS_result/data/output \
    --output-dir MARS_result/data/benchmark

# 3. User Selection — filter + convert to OASIS format
bash MARS/code/user_selection.sh all \
    --input-csv MARS_result/data/output/2025-06-15/user_profiles.csv \
    --posts-input-dir MARS_result/data/output/2025-06-15 \
    --oasis-out-dir data/oasis \
    --calibration-end "2025-06-15 16:00:00" \
    --ground-truth-end "2025-06-16 00:00:00"

# 4. Attitude Annotation & Evaluation (in Python)
python -c "
from MARS.code.simulation_process.oasis_attitude import OasisAttitudeProcessor
import asyncio
config = {
    'attitude_lifestyle_culture': 'Evaluate sentiment towards lifestyle and cultural topics.',
    'attitude_sport_ent': 'Evaluate sentiment towards sports and entertainment.',
}
proc = OasisAttitudeProcessor(
    oasis_db_path='data/oasis/oasis_database.db',
    user_csv_path='data/oasis/oasis_agent_init.csv',
    user_csv_output_path='data/oasis/oasis_agent_init_scored.csv',
    attitude_config=config,
)
asyncio.run(proc.run())
"
```

---

We are actively developing MARS. Stay Tuned!
