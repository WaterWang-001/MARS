

<h1 align="center">MARS: Massive Agent-Based Real-World Simulation</h1>

<p align="center">
  <b>Data-Driven Social Digital Twins Powered by Large Language Models</b>
  <br/>
  <em>From Raw Social Data to Large-Scale Agent Simulation — End to End</em>
  <br/><br/>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square" alt="License"/></a>
  <a href="README_CN.md"><img src="https://img.shields.io/badge/lang-中文-red?style=flat-square" alt="中文"/></a>
</p>

---

## :zap: Try It Now — Interactive Console

MARS ships with a ready-to-use **service layer** at [`code/marketing_simulation/`](code/marketing_simulation/) so you can drive the full simulation pipeline without writing any Python. Two options:

### Option 1 · Streamlit UI (zero-code, easiest)

A visual console for designing interventions, picking attitude metrics, and running simulations with one click.

```bash
# 0. Install dependencies (one-time)
conda create -n oasis python=3.11 -y && conda activate oasis
pip install -e oasis
pip install fastmcp streamlit pandas python-dotenv

# 1. Set up your LLM credentials
cp code/marketing_simulation/.env.example code/marketing_simulation/data/.env
# edit data/.env and fill in MARS_MODEL_BASE_URL + MARS_MODEL_API_KEY

# 2. Launch the console
bash code/marketing_simulation/run.sh
```

Then open the URL Streamlit prints (usually `http://localhost:8501`) and use the **Simulation Console** tab to design interventions (broadcast / bribery / register_user), pick attitude dimensions, and hit **Run**.

### Option 2 · Claude Desktop via MCP (conversational)

Let Claude design and run marketing experiments for you end-to-end. MARS exposes **11 MCP tools** via a FastMCP server.

1. **Install MARS** as shown above (steps 0 and 1).

2. **Register the MCP server** in Claude Desktop config:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

   ```json
   {
     "mcpServers": {
       "mars-marketing": {
         "command": "/opt/anaconda3/envs/oasis/bin/python",
         "args": ["/absolute/path/to/MARS/code/marketing_simulation/mcp_server.py"],
         "env": {
           "MARS_MODEL_BASE_URL": "https://api.openai.com/v1",
           "MARS_MODEL_API_KEY": "sk-..."
         }
       }
     }
   }
   ```

3. **Restart Claude Desktop**, open a new chat, and paste the contents of [`code/marketing_simulation/skill.md`](code/marketing_simulation/skill.md) — this teaches Claude the strict 4-step experimental SOP (Data Ingestion → Configuration → Intervention Design → Simulation Execution).

4. **Start experimenting**, e.g.:

   > *"Import the sample user profiles at `code/marketing_simulation/data/oasis_agent_init.csv`, set up 5 simulation steps with an `attitude_brand` metric, broadcast a positive product announcement at step 2, then run the simulation and show me the final attitude distribution."*

Claude will call `import_user_profiles` → `set_simulation_config` → `build_intervention_csv` → `run_marketing_simulation` → `query_db_table` in sequence, and return the analysis.

> :bulb: For CLI smoke-testing the MCP stack without Claude Desktop, see `python code/marketing_simulation/my_client.py`.

For full service-layer documentation (all 11 tools, environment variables, architecture), see [`code/marketing_simulation/README.md`](code/marketing_simulation/README.md).

---

## :telescope: Overview

**MARS** is a next-generation engine for building high-fidelity, data-driven digital twins of complex social systems, built on top of the [OASIS multi-agent simulation platform](https://github.com/camel-ai/oasis.git).

Starting from raw social media data (e.g., posts, user profiles, social graphs), MARS provides a **complete end-to-end pipeline** that automatically:

1. **Ingests and structures** raw JSON data into per-platform databases and user profile tables;
2. **Profiles users with LLMs**, generating hierarchical interest taxonomy trees (41 domains, 3 levels) and demographic/firmographic attributes through incremental, quality-controlled inference;
3. **Constructs simulation-ready agent populations**, assigning personas, social relationships, and behavioral tiers (LLM-powered influencers vs. heuristic-driven silent majority);
4. **Runs large-scale social simulations** on a fully functional simulated social platform — agents autonomously post, comment, like, repost, follow, and search, driven by personalized recommendation feeds (Twitter/TWHIN/Reddit algorithms). A **hybrid dual-tier agent architecture** (LLM-powered influencers + heuristic-driven silent majority) enables scaling to hundreds of thousands of agents without sacrificing behavioral fidelity. Each agent maintains **internal attitude states** (multi-dimensional, ranging from -1.0 to 1.0) that evolve dynamically through content exposure and social interaction, creating a "Mind-Action" consistency loop where private beliefs drive public expressions;
5. **Supports multi-modal intervention experiments** — researchers can inject **broadcast messages** (global announcements), **targeted bribery** (overriding specific agents' or groups' instructions/attitudes), and **astroturfing** (dynamically registering new agents mid-simulation with pre-defined profiles), all configured via a simple CSV file;
6. **Evaluates simulation fidelity** against real-world ground truth using configurable attitude dimensions and statistical metrics (Bias, Diversity, Pearson correlation), with automatic temporal alignment and separate analysis for LLM vs. ABM agents.

MARS enables researchers to study information propagation, public opinion evolution, and the impact of social interventions in a controlled, reproducible environment — bridging the gap between real-world social data and actionable simulation insights.

> :information_source: **Note on Data:** Raw datasets used in our research are not publicly available due to licensing restrictions. MARS provides a fully functional implementation — supply your own data conforming to the [Input Data Requirements](#clipboard-input-data-requirements) and the entire pipeline runs out of the box.


---

## :star2: Highlights

<table>
<tr>
<td width="50%" valign="top">

### :link: End-to-End Pipeline

A **complete, production-ready pipeline** transforms raw social media data into large-scale agent-based simulations — no manual data wrangling required.

```
Raw JSON → Data Processing → LLM Profiling
         → Simulation → Evaluation
```

</td>
<td width="50%" valign="top">

### :brain: LLM-Powered User Profiling

Uses LLMs to build **3-level interest taxonomy trees** (L1→L2→L3) covering 41 domains, with rich demographic/firmographic profiles. Incremental buffer-triggered architecture with confidence-based merging.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### :busts_in_silhouette: Hybrid Agent Architecture

Dual-tier system scales to **hundreds of thousands of agents**:

| Tier | Strategy |
|------|----------|
| **Tier 1** (Authority, KOL, Creator, User) | LLM-powered reasoning |
| **Tier 2** (Lurkers / Silent Majority) | Heuristic models (BCM) |

</td>
<td width="50%" valign="top">

### :syringe: Multi-Modal Intervention System

Test real-world intervention strategies in simulation:
- **Broadcast** — Global messages to all agents
- **Bribery** — Targeted attitude/instruction override
- **Astroturfing** — Dynamic agent injection mid-simulation

</td>
</tr>
<tr>
<td width="50%" valign="top">

### :shield: Intelligent Quality Control

**3-layer filtering pipeline** with automatic bot detection:
- **User-level** — Attribute validation
- **Batch-level** — Repetition, fuzzy similarity, embedding coherence
- **Post-level** — Sensitive content, spam, meaningfulness

</td>
<td width="50%" valign="top">

### :bar_chart: Customizable Evaluation

Define attitude dimensions as `{column: description}` dicts — the engine auto-generates DB schema, SQL, and LLM prompts. Evaluate with **Bias**, **Diversity**, and **Pearson** metrics against ground truth.

</td>
</tr>
<tr>
<td colspan="2" align="center">

### :globe_with_meridians: Data-Agnostic Design

MARS works with **any social media dataset**. Built-in support for 5 platforms (Weibo, Zhihu, Xiaohongshu, Toutiao, Douban), with a clean adapter pattern for adding new platforms.

</td>
</tr>
</table>

---

## :arrows_counterclockwise: Workflow

<table>
<tr>
<td align="center" colspan="3">

**Raw Social Media Data** (JSON / ZIP)

:arrow_down:

</td>
</tr>
<tr>
<td align="center" colspan="3">

:one: **Data Processing**
<br/>
Raw JSON :arrow_right: per-platform SQLite DBs + CSV user profiles

</td>
</tr>
<tr>
<td align="center" width="50%">

:arrow_down:

:two: **User Tagging (LLM)**
<br/>
3-level interest taxonomy trees
<br/>
Demographic & firmographic profiles

</td>
<td align="center" width="50%">

:arrow_down:

:three: **User Selection**
<br/>
Criteria-based filtering & grouping
<br/>
:arrow_right: `oasis_agent_init.csv` + `oasis_database.db`

</td>
</tr>
<tr>
<td align="center" colspan="3">

:arrow_down:

:four: **Simulation & Evaluation**
<br/>
Hybrid LLM + ABM agent simulation :arrow_right: Attitude annotation :arrow_right: Bias / Diversity / Pearson metrics

</td>
</tr>
</table>

---

## ⚖️MARS vs. Original OASIS

MARS is built on the [OASIS](https://github.com/camel-ai/oasis.git) open-source framework and extends it substantially. The table below summarizes the key differences:

| Capability | Original OASIS | MARS |
|------------|---------------|------|
| **Agent Types** | 1 generic `SocialAgent` (LLM-only) | 6 specialized types across 2 tiers: 4 LLM agents (AuthorityAgent, KOLAgent, ActiveCreatorAgent, NormalUserAgent) + 2 heuristic agents (HeuristicAgent, LurkerAgent) |
| **Agent Scaling** | All agents require LLM inference, limiting scale | Hybrid architecture: only high-influence agents (Tier 1) use LLMs; the silent majority (Tier 2) uses lightweight heuristic models, enabling 10x+ agent counts |
| **Activation Model** | All agents act every step | Probabilistic activation by group (10%-80%), simulating realistic activity distributions |
| **Attitude / Opinion Dynamics** | No attitude tracking | Multi-dimensional attitude system (-1.0 to 1.0) with dynamic evolution: LLM agents update via tool calls; ABM agents converge via Bounded Confidence Model |
| **Mind-Action Consistency** | Actions only | Dual expression model — agents maintain private internal attitudes that drive public social actions, logged per time step for temporal analysis |
| **Intervention System** | None | 3 intervention types via CSV: broadcast (global), bribery (targeted attitude/instruction override), register_user (mid-simulation agent injection) |
| **Data Pipeline** | Manual data preparation | End-to-end automated pipeline: raw JSON → structured DB → LLM profiling → simulation-ready agents |
| **User Profiling** | Static persona strings | LLM-generated hierarchical interest trees (41 domains x 3 levels) + demographic/firmographic profiles with confidence scores |
| **Quality Control** | None | 3-layer filtering pipeline (user/batch/post level) with bot detection, spam filtering, and content quality checks |
| **Database Schema** | 15 static tables | +3 intervention tables + dynamically generated attitude log tables per metric |
| **Evaluation** | Basic interview-based testing | Quantitative metrics (Bias, Diversity, Pearson) against real-world ground truth, with LLM vs. ABM separation and temporal alignment |
| **Reproducibility** | -- | Deterministic intervention targeting via content-based seeding; calibration/ground-truth temporal splits for rigorous evaluation |
---

## :bookmark_tabs: Table of Contents

- [:clipboard: Input Data Requirements](#clipboard-input-data-requirements)
- [:file_folder: Directory Structure](#file_folder-directory-structure)
- [:gear: Pipeline Stages](#gear-pipeline-stages)
- [:joystick: OASIS Simulation Engine](#joystick-oasis-simulation-engine)
- [:rocket: Quick Start](#rocket-quick-start)

---

## :clipboard: Input Data Requirements

MARS is **data-agnostic** with two entry points:

- **Option A: Raw JSON** — Start from Stage 1. Place JSON files in `data/raw/YYYY-MM-DD/` (one JSON object per line).
- **Option B: Pre-structured Data** — Skip to Stage 2. Provide SQLite DBs + CSV files matching the schemas below.

### Required Attributes (Option A)

#### Post Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | Unique identifier of the post author |
| `content` | string | Yes | Post text content |
| `publish_time` | string | Yes | Publish timestamp (ISO 8601) |
| `platform` | string | Yes | Platform identifier (e.g., `"weibo"`, `"twitter"`) |
| `title` | string | No | Post title |
| `quote_content` | string | No | Quoted/reposted content |

#### User Profile Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | Must match post author IDs |
| `nickname` | string | Yes | Display name |
| `platform` | string | Yes | Platform identifier |
| `followers_count` | int | Recommended | For user grouping |
| `following_count`, `gender`, `location`, `verified`, `description` | various | No | Optional enrichment |

#### Engagement Attributes (Optional)

`mentions` (list), `like_count`, `comment_count`, `share_count`

> **Custom schemas:** Write a lightweight adapter in `code/data_process/user_post.py` and register it in `PLATFORM_MAP`. The rest of the pipeline remains unchanged.

### Pre-structured Data Schemas (Option B)

**SQLite** (`posts_{platform}.db`): Table `posts` with columns `user_id TEXT, title TEXT, content TEXT, quote_content TEXT, created_at TEXT`.

**CSV** (`user_profile_{platform}.csv`): Columns `user_id, nickname, platform` (required) + `followers_count, following_count, gender, location, verified, description` (optional).

Place files in `MARS_result/data/output/YYYY-MM-DD/`.

### Minimum Viable Dataset

| Component | Minimum Requirement |
|-----------|-------------------|
| Users | ~100 users with posts |
| Posts per user | >=5 (buffer threshold for LLM tagging) |
| Time span | >=1 day |
| Core fields | `user_id`, `content`, `publish_time`, `platform` |
| LLM access | vLLM server or OpenAI-compatible API |
| Embedding model | `BAAI/bge-small-zh-v1.5` (or equivalent) |

---

## :file_folder: Directory Structure

```
MARS/
├── code/
│   ├── data_process/          # Stage 1: Raw JSON → Structured DB + CSV
│   ├── user_tagging/          # Stage 2: LLM-based user profiling
│   │   ├── src/
│   │   │   ├── core/          #   API client, DB client, NLP, prompts, merging
│   │   │   ├── platforms/     #   Platform adapters & filters (5 platforms)
│   │   │   ├── tasks/         #   SFT tagging, benchmark, clustering
│   │   │   ├── scripts/       #   Entry point scripts
│   │   │   └── prompts/       #   LLM prompt templates
│   │   └── configs/           #   Per-platform YAML configs
│   ├── user_selection/        # Stage 3: User filtering + OASIS format conversion
│   ├── simulation_process/    # Stage 4: Simulation, attitude annotation, evaluation
│   └── marketing_simulation/  # Stage 5 (optional): Interactive console
│       ├── mcp_server.py      #   FastMCP server exposing 11 tools to LLM agents
│       ├── streamlit_app.py   #   Streamlit UI for intervention design & run
│       ├── skill.md           #   Claude Code / Agent SOP for experiment orchestration
│       ├── my_client.py       #   Python smoke-test client
│       └── data/              #   Sample CSVs; runtime db/logs (gitignored)
│
└── oasis/                     # OASIS simulation engine (extended)
    ├── social_agent/          #   LLM-powered agent implementation
    ├── social_platform/       #   Platform simulation (DB/recsys/schema)
    ├── environment/           #   OpenAI Gym-style environment
    └── clock/                 #   Simulation clock
```

---

## :gear: Pipeline Stages

### Stage 1: Data Processing (`code/data_process/`)

Transforms raw JSON into structured SQLite databases and CSV user profiles.

- **`data_process.py`** — Orchestrator supporting modes: `all`, `posts`, `profiles`, `relationships`
- **`user_post.py`** — Parallel post ETL: JSON → per-platform SQLite DBs (with SQLite write optimizations)
- **`user_profile.py`** — Profile extraction with cross-run deduplication and sampling (verified users, high-follower users, random sampling)
- **`user_relationship.py`** — Social graph extraction via two-phase MapReduce with weighted edges

### Stage 2: User Tagging (`code/user_tagging/`)

The most architecturally complex module. Uses LLMs to build **3-level interest taxonomy trees** and **demographic/firmographic profiles** through an incremental accumulation + buffer-triggered architecture.

**Core workflow:**

```
User posts → 3-Layer Quality Filter → Buffer Accumulation (threshold=5)
    → Parallel Dual-Task LLM Inference:
        Task 1: Interest Tree (41 L1 domains → L2 sub-domains → L3 entities)
        Task 2: Demographic/Firmographic Profile
    → Confidence-Based Merge → SFT Quality Gate (6 checks) → Output
```

**Interest Tree Example:**
```json
{
  "interest_tree": [
    {
      "L1": "Gaming", "confidence": "High",
      "sub_domains": [
        { "L2": "Mobile Games", "meta_type": "PRODUCT",
          "entities": [{"L3": "Honor of Kings", "confidence": "High"}] }
      ]
    }
  ],
  "persona_summary": "Core gamer, focused on competitive mobile games..."
}
```

**Key components:**
- **Core** (`src/core/`): API client (LLM + embedding), DB client, spaCy NLP processor, prompt manager, merge utilities
- **Platform adapters** (`src/platforms/`): Adapter + Filter per platform, implementing platform-specific data normalization and quality filtering
- **Tasks** (`src/tasks/`): SFT tagging orchestrator, state management (SQLite with 4 tables), SFT quality control, benchmark generation, offline tag clustering

**Configuration:** Per-platform YAML files control buffer thresholds, quality rules, LLM endpoints, and concurrency settings.

### Stage 3: User Selection (`code/user_selection/`)

Filters user subsets and converts them into OASIS simulation input format.

- **User filtering** with dynamic query syntax (`min_`, `max_`, `_in=`, exact match)
- **Quantile-based user grouping** into 5 tiers (Authority Media → Lurker) based on followers and posting frequency
- **Temporal data splitting** into calibration set and ground truth for evaluation
- **Output:** `oasis_agent_init.csv` (agent profiles) + `oasis_database.db` (posts with temporal split)

### Stage 4: Simulation & Evaluation (`code/simulation_process/`)

Orchestrates the simulation execution, intervention delivery, and post-simulation evaluation.

**Simulation Execution Flow:**

```
1. Database Reset (preserve calibration & ground truth posts)
2. Platform Init (Twitter/Reddit mode) + Agent Generation
   → Load profiles → Create Tier 1 (LLM) + Tier 2 (Heuristic) agents
   → Preload memory from historical posts → Batch register
3. Intervention Preprocessing (parse CSV → write to DB tables)
4. Per Time Step:
   a. Dynamic agent injection (if register_user interventions exist)
   b. Activation pool sampling (probabilistic, by group tier)
   c. Action execution (LLMAction for Tier 1, HeuristicAction for Tier 2)
   d. Attitude state logging (all agents, all dimensions)
   e. Recommendation system update
5. Post-simulation attitude annotation + evaluation
```

**Key components:**

- **Agent tier assignment** (`oasis_test_grouping.py`): Maps user groups to agent classes and probabilistic activation rates (Authority Media 80%, KOL 70%, Active Creator 60%, Regular User 30%, Lurker 10%)
- **Intervention system** (`intervention_processor.py`): Parses intervention CSV, supports group-based targeting with sampling ratios, deterministic seeding for reproducibility
- **Attitude annotation** (`oasis_attitude.py`): LLM-based multi-dimensional attitude scoring (-1.0 to 1.0) with configurable dimensions; computes per-user initial and final attitude scores
- **Evaluation metrics** (`oasis_evaluation_overall.py`): Bias (mean), Diversity (std), Pearson correlation — separated by LLM vs ABM agents, with temporal alignment to real timestamps

### Stage 5 (Optional): Interactive Console (`code/marketing_simulation/`)

An end-user **service layer** that lets non-developers drive the simulation
pipeline without writing Python. It shares the same `oasis/` engine and the
same `code/simulation_process/` pipeline — no duplicated source code.

Three interchangeable entry points:

| Mode | Command | Use case |
|---|---|---|
| **Streamlit UI** | `bash code/marketing_simulation/run.sh` | Interactive GUI with intervention editor and one-click run |
| **Claude Desktop via MCP** | Configure `mcp_server.py` in `claude_desktop_config.json` and paste `skill.md` into chat | Let Claude design and run experiments conversationally |
| **Python smoke test** | `python code/marketing_simulation/my_client.py` | Verify the full 4-step pipeline end-to-end from the CLI |

The MCP server exposes 11 tools (`get_runtime_defaults`, `import_user_profiles`,
`build_intervention_csv`, `run_marketing_simulation`, `query_db_table`, …)
that Claude invokes by following the strict 4-step SOP defined in
`code/marketing_simulation/skill.md`. Runtime artifacts (database, logs,
generated intervention CSVs) are all written to
`code/marketing_simulation/data/` and are gitignored.

See [`code/marketing_simulation/README.md`](code/marketing_simulation/README.md) for full setup instructions.

---

## :joystick: OASIS Simulation Engine (`oasis/`)

The underlying simulation platform built on the CAMEL-AI framework. Key components:

| Module | Description |
|--------|-------------|
| `social_agent/` | Core SocialAgent with 91 action types, attitude state, and social graph |
| `social_platform/` | Platform simulation with SQLite CRUD and recommendation algorithms (Twitter/TWHIN/Reddit/Random) |
| `environment/` | OpenAI Gym-style environment wrapper |
| `clock/` | Simulation clock management |

---

### :wrench: Key Architectural Changes to OASIS Core

MARS modifies the following OASIS core components:

- **`social_agent/agent_custom.py`** (new): Defines 6 specialized agent subclasses with differentiated personas and behavioral profiles
- **`social_agent/agent_attitude.py`** (new): Attitude state management and LLM-callable `update_internal_attitude()` tool, enabling the Mind-Action consistency loop
- **`social_agent/agents_generator.py`** (modified): Extended to support tier-based agent instantiation, attitude score loading from profiles, and memory preloading from historical posts
- **`social_platform/platform.py`** (modified): `refresh()` now returns intervention instructions (broadcast messages + targeted bribery with attitude targets) alongside recommended posts; accepts `attitude_metrics` parameter
- **`social_platform/database.py`** (modified): Dynamic table creation for attitude metrics (`log_attitude_{metric}`) and intervention tables (`intervention_message`, `agent_intervention`, `pending_registrations`)
- **`environment/env.py`** (modified): Passes attitude metrics configuration through to platform initialization

---

## :rocket: Quick Start

```bash
# 0. Activate environment
source /remote-home/JuelinW/anaconda3/bin/activate oasis
cd /remote-home/JuelinW/oasis_project

# 1. Data Processing — raw JSON → structured DB + CSV
bash MARS/code/data_process.sh 2025-06-15 2025-06-15

# 2. User Tagging — LLM-based profiling
python MARS/code/user_tagging/src/scripts/run_sft_tagging.py \
    --date 2025-06-15 --platform weibo \
    --db-base-dir MARS_result/data/output \
    --output-dir MARS_result/data/online_cold_start

# 3. User Selection — filter + convert to OASIS format
bash MARS/code/user_selection.sh all \
    --input-csv MARS_result/data/output/2025-06-15/user_profiles.csv \
    --posts-input-dir MARS_result/data/output/2025-06-15 \
    --oasis-out-dir data/oasis \
    --calibration-end "2025-06-15 16:00:00" \
    --ground-truth-end "2025-06-16 00:00:00"

# 4. Attitude Annotation & Evaluation
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

## :handshake: Acknowledgments

MARS's simulation engine is powered by **[OASIS (Open Agent Social Interaction Simulations)](https://github.com/camel-ai/oasis)**. We sincerely thank the CAMEL-AI team for their open-source contributions!

---

<p align="center"><em>We are actively developing MARS. Stay Tuned!</em></p>

