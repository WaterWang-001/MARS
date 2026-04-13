


<h1 align="center">MARS: 大规模智能体社会仿真系统</h1>

<p align="center">
  <b>基于大语言模型的数据驱动社会数字孪生</b>
  <br/>
  <em>从原始社交数据到大规模智能体仿真 — 端到端全流程</em>
  <br/><br/>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square" alt="License"/></a>
  <a href="README.md"><img src="https://img.shields.io/badge/lang-English-blue?style=flat-square" alt="English"/></a>
</p>

---

## :zap: 快速体验 — 交互式控制台

MARS 提供了开箱即用的**服务层**，位于 [`code/marketing_simulation/`](code/marketing_simulation/)，无需编写任何 Python 代码即可驱动完整的仿真流程。两种方式：

### 方式一 · Streamlit 图形界面（零代码，最简单）

可视化控制台，用于设计干预策略、选择态度指标、一键运行仿真。

```bash
# 0. 安装依赖（仅首次）
conda create -n oasis python=3.11 -y && conda activate oasis
pip install -e oasis
pip install fastmcp streamlit pandas python-dotenv

# 1. 配置 LLM 凭证
cp code/marketing_simulation/.env.example code/marketing_simulation/data/.env
# 编辑 data/.env，填写 MARS_MODEL_BASE_URL 和 MARS_MODEL_API_KEY

# 2. 启动控制台
bash code/marketing_simulation/run.sh
```

打开 Streamlit 输出的 URL（通常为 `http://localhost:8501`），在 **Simulation Console** 标签页中设计干预措施（broadcast / bribery / register_user），选择态度维度，点击 **Run** 即可。

### 方式二 · 通过 MCP 接入 Claude Desktop（对话式）

让 Claude 为您端到端地设计和运行营销实验。MARS 通过 FastMCP 服务器暴露了 **11 个 MCP 工具**。

1. **安装 MARS**（参见上方步骤 0 和 1）。

2. **注册 MCP 服务器**到 Claude Desktop 配置文件：
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

3. **重启 Claude Desktop**，新建对话，粘贴 [`code/marketing_simulation/skill.md`](code/marketing_simulation/skill.md) 的内容 — 这会教会 Claude 严格的四步实验流程（数据导入 → 配置 → 干预设计 → 仿真执行）。

4. **开始实验**，例如：

   > *"导入 `code/marketing_simulation/data/oasis_agent_init.csv` 中的用户画像，配置 5 步仿真并使用 `attitude_brand` 指标，在第 2 步广播一条正面产品公告，然后运行仿真并展示最终的态度分布。"*

Claude 会依次调用 `import_user_profiles` → `set_simulation_config` → `build_intervention_csv` → `run_marketing_simulation` → `query_db_table`，并返回分析结果。

> :bulb: 如需在不使用 Claude Desktop 的情况下测试 MCP 栈，可运行 `python code/marketing_simulation/my_client.py`。

完整的服务层文档（全部 11 个工具、环境变量、架构说明）请参见 [`code/marketing_simulation/README.md`](code/marketing_simulation/README.md)。

---

## :telescope: 概述

**MARS** 是新一代高保真、数据驱动的复杂社会系统数字孪生引擎，基于 [OASIS 多智能体仿真平台](https://github.com/camel-ai/oasis.git) 构建。

从原始社交媒体数据（如帖子、用户画像、社交关系图）出发，MARS 提供了一条**完整的端到端流水线**，自动完成以下步骤：

1. **数据摄入与结构化** — 将原始 JSON 数据转换为按平台划分的数据库和用户画像表；
2. **LLM 驱动的用户画像生成** — 构建层级兴趣分类树（41 个领域，3 个层级），并通过增量式、质量可控的推理生成人口统计/企业画像属性；
3. **仿真智能体群体构建** — 分配人设、社交关系和行为层级（LLM 驱动的意见领袖 vs. 启发式驱动的沉默多数）；
4. **大规模社会仿真** — 在功能完备的模拟社交平台上运行仿真。智能体自主发帖、评论、点赞、转发、关注和搜索，并受个性化推荐算法（Twitter/TWHIN/Reddit）驱动。**混合双层智能体架构**（LLM 驱动的意见领袖 + 启发式驱动的沉默多数）可在不牺牲行为保真度的前提下扩展至数十万智能体。每个智能体维护**内部态度状态**（多维度，范围 -1.0 至 1.0），通过内容暴露和社交互动动态演化，形成"心智-行为"一致性闭环，即私人信念驱动公开表达；
5. **多模态干预实验** — 研究者可注入**广播消息**（全局公告）、**定向贿赂**（覆盖特定智能体或群组的指令/态度）和**水军注入**（仿真中途动态注册预设画像的新智能体），所有配置通过简单的 CSV 文件完成；
6. **仿真保真度评估** — 使用可配置的态度维度和统计指标（Bias、Diversity、Pearson 相关系数）对比真实世界基准数据，支持自动时间对齐以及 LLM 与 ABM 智能体的分别分析。

MARS 使研究者能够在可控、可复现的环境中研究信息传播、舆论演化以及社会干预的影响 — 弥合真实社交数据与可操作仿真洞察之间的鸿沟。

> :information_source: **数据说明：** 由于许可限制，研究中使用的原始数据集不公开。MARS 提供了完整可用的实现 — 只需提供符合[输入数据要求](#clipboard-输入数据要求)的数据，整条流水线即可开箱即用。

---

## :star2: 核心亮点

<table>
<tr>
<td width="50%" valign="top">

### :link: 端到端流水线

**完整的生产级流水线**，将原始社交媒体数据转化为大规模智能体仿真 — 无需手动数据清洗。

```
原始 JSON → 数据处理 → LLM 画像生成
         → 仿真运行 → 效果评估
```

</td>
<td width="50%" valign="top">

### :brain: LLM 驱动的用户画像

使用 LLM 构建 **3 层兴趣分类树**（L1→L2→L3），覆盖 41 个领域，生成丰富的人口统计/企业画像。采用增量式缓冲触发架构，支持基于置信度的合并。

</td>
</tr>
<tr>
<td width="50%" valign="top">

### :busts_in_silhouette: 混合智能体架构

双层系统可扩展至**数十万智能体**：

| 层级 | 策略 |
|------|------|
| **第一层**（权威媒体、KOL、活跃创作者、普通用户） | LLM 驱动推理 |
| **第二层**（潜水者 / 沉默多数） | 启发式模型 (BCM) |

</td>
<td width="50%" valign="top">

### :syringe: 多模态干预系统

在仿真中测试真实世界的干预策略：
- **广播 (Broadcast)** — 向所有智能体发送全局消息
- **贿赂 (Bribery)** — 定向覆盖态度/指令
- **水军注入 (Astroturfing)** — 仿真中途动态注入智能体

</td>
</tr>
<tr>
<td width="50%" valign="top">

### :shield: 智能质量控制

**3 层过滤流水线**，自带机器人检测：
- **用户级** — 属性验证
- **批次级** — 重复检测、模糊相似度、嵌入一致性
- **帖子级** — 敏感内容、垃圾信息、意义性检查

</td>
<td width="50%" valign="top">

### :bar_chart: 可定制评估

将态度维度定义为 `{column: description}` 字典 — 引擎自动生成数据库 Schema、SQL 和 LLM 提示词。使用 **Bias**、**Diversity** 和 **Pearson** 指标对比真实数据评估。

</td>
</tr>
<tr>
<td colspan="2" align="center">

### :globe_with_meridians: 数据无关设计

MARS 适用于**任何社交媒体数据集**。内置支持 5 个平台（微博、知乎、小红书、头条、豆瓣），并提供简洁的适配器模式以添加新平台。

</td>
</tr>
</table>

---

## :arrows_counterclockwise: 工作流程

<table>
<tr>
<td align="center" colspan="3">

**原始社交媒体数据**（JSON / ZIP）

:arrow_down:

</td>
</tr>
<tr>
<td align="center" colspan="3">

:one: **数据处理**
<br/>
原始 JSON :arrow_right: 按平台的 SQLite 数据库 + CSV 用户画像

</td>
</tr>
<tr>
<td align="center" width="50%">

:arrow_down:

:two: **用户标签（LLM）**
<br/>
3 层兴趣分类树
<br/>
人口统计与企业画像

</td>
<td align="center" width="50%">

:arrow_down:

:three: **用户筛选**
<br/>
条件过滤与分组
<br/>
:arrow_right: `oasis_agent_init.csv` + `oasis_database.db`

</td>
</tr>
<tr>
<td align="center" colspan="3">

:arrow_down:

:four: **仿真与评估**
<br/>
混合 LLM + ABM 智能体仿真 :arrow_right: 态度标注 :arrow_right: Bias / Diversity / Pearson 指标

</td>
</tr>
</table>

---

## :balance_scale: MARS 与原版 OASIS 对比

MARS 基于 [OASIS](https://github.com/camel-ai/oasis.git) 开源框架构建并进行了大幅扩展。下表总结了主要差异：

| 能力 | 原版 OASIS | MARS |
|------|-----------|------|
| **智能体类型** | 1 种通用 `SocialAgent`（仅 LLM） | 跨 2 个层级的 6 种专用类型：4 种 LLM 智能体（AuthorityAgent, KOLAgent, ActiveCreatorAgent, NormalUserAgent）+ 2 种启发式智能体（HeuristicAgent, LurkerAgent） |
| **智能体扩展性** | 所有智能体均需 LLM 推理，限制规模 | 混合架构：仅高影响力智能体（第一层）使用 LLM；沉默多数（第二层）使用轻量级启发式模型，智能体数量可提升 10 倍以上 |
| **激活模型** | 所有智能体每步都行动 | 按组的概率激活（10%-80%），模拟真实活跃度分布 |
| **态度/舆论动力学** | 无态度追踪 | 多维态度系统（-1.0 至 1.0），动态演化：LLM 智能体通过工具调用更新；ABM 智能体通过有界信心模型（BCM）收敛 |
| **心智-行为一致性** | 仅有行为 | 双重表达模型 — 智能体维护驱动公开社交行为的内部私人态度，每个时间步记录以供时序分析 |
| **干预系统** | 无 | 3 种干预类型（CSV 配置）：broadcast（全局）、bribery（定向态度/指令覆盖）、register_user（仿真中途注入智能体） |
| **数据流水线** | 手动数据准备 | 端到端自动化流水线：原始 JSON → 结构化数据库 → LLM 画像 → 仿真就绪智能体 |
| **用户画像** | 静态人设字符串 | LLM 生成的层级兴趣树（41 领域 x 3 层级）+ 带置信度评分的人口统计/企业画像 |
| **质量控制** | 无 | 3 层过滤流水线（用户/批次/帖子级别），含机器人检测、垃圾过滤和内容质量检查 |
| **数据库 Schema** | 15 张静态表 | +3 张干预表 + 按指标动态生成的态度日志表 |
| **评估** | 基础访谈式测试 | 量化指标（Bias, Diversity, Pearson）对比真实基准数据，支持 LLM/ABM 分离分析和时间对齐 |
| **可复现性** | -- | 基于内容的确定性干预定位；校准/基准数据时间分割，确保严格评估 |

---

## :bookmark_tabs: 目录

- [:clipboard: 输入数据要求](#clipboard-输入数据要求)
- [:file_folder: 目录结构](#file_folder-目录结构)
- [:gear: 流水线阶段](#gear-流水线阶段)
- [:joystick: OASIS 仿真引擎](#joystick-oasis-仿真引擎)
- [:rocket: 快速开始](#rocket-快速开始)

---

## :clipboard: 输入数据要求

MARS 是**数据无关**的，提供两个入口点：

- **选项 A：原始 JSON** — 从阶段 1 开始。将 JSON 文件放入 `data/raw/YYYY-MM-DD/`（每行一个 JSON 对象）。
- **选项 B：预结构化数据** — 跳至阶段 2。提供符合以下 Schema 的 SQLite 数据库 + CSV 文件。

### 必需属性（选项 A）

#### 帖子属性

| 属性 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `user_id` | string | 是 | 帖子作者的唯一标识符 |
| `content` | string | 是 | 帖子文本内容 |
| `publish_time` | string | 是 | 发布时间戳（ISO 8601） |
| `platform` | string | 是 | 平台标识符（如 `"weibo"`, `"twitter"`） |
| `title` | string | 否 | 帖子标题 |
| `quote_content` | string | 否 | 引用/转发的内容 |

#### 用户画像属性

| 属性 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `user_id` | string | 是 | 须与帖子作者 ID 匹配 |
| `nickname` | string | 是 | 显示名称 |
| `platform` | string | 是 | 平台标识符 |
| `followers_count` | int | 建议 | 用于用户分组 |
| `following_count`, `gender`, `location`, `verified`, `description` | 各类 | 否 | 可选补充信息 |

#### 互动属性（可选）

`mentions`（列表）、`like_count`、`comment_count`、`share_count`

> **自定义 Schema：** 在 `code/data_process/user_post.py` 中编写轻量级适配器并注册到 `PLATFORM_MAP`。流水线其余部分无需修改。

### 预结构化数据 Schema（选项 B）

**SQLite**（`posts_{platform}.db`）：表 `posts`，列为 `user_id TEXT, title TEXT, content TEXT, quote_content TEXT, created_at TEXT`。

**CSV**（`user_profile_{platform}.csv`）：列 `user_id, nickname, platform`（必需）+ `followers_count, following_count, gender, location, verified, description`（可选）。

文件放入 `MARS_result/data/output/YYYY-MM-DD/`。

### 最小可用数据集

| 组件 | 最低要求 |
|------|---------|
| 用户数 | ~100 个有帖子的用户 |
| 每用户帖子数 | >=5（LLM 标签的缓冲阈值） |
| 时间跨度 | >=1 天 |
| 核心字段 | `user_id`, `content`, `publish_time`, `platform` |
| LLM 接入 | vLLM 服务或 OpenAI 兼容 API |
| 嵌入模型 | `BAAI/bge-small-zh-v1.5`（或同等模型） |

---

## :file_folder: 目录结构

```
MARS/
├── code/
│   ├── data_process/          # 阶段 1：原始 JSON → 结构化数据库 + CSV
│   ├── user_tagging/          # 阶段 2：LLM 驱动的用户画像
│   │   ├── src/
│   │   │   ├── core/          #   API 客户端、数据库客户端、NLP、提示词、合并工具
│   │   │   ├── platforms/     #   平台适配器与过滤器（5 个平台）
│   │   │   ├── tasks/         #   SFT 标签、基准测试、聚类
│   │   │   ├── scripts/       #   入口脚本
│   │   │   └── prompts/       #   LLM 提示词模板
│   │   └── configs/           #   按平台的 YAML 配置
│   ├── user_selection/        # 阶段 3：用户过滤 + OASIS 格式转换
│   ├── simulation_process/    # 阶段 4：仿真、态度标注、评估
│   └── marketing_simulation/  # 阶段 5（可选）：交互式控制台
│       ├── mcp_server.py      #   FastMCP 服务器，向 LLM 智能体暴露 11 个工具
│       ├── streamlit_app.py   #   Streamlit 界面，用于干预设计与运行
│       ├── skill.md           #   Claude Code / Agent 实验编排 SOP
│       ├── my_client.py       #   Python 冒烟测试客户端
│       └── data/              #   示例 CSV；运行时数据库/日志（已 gitignore）
│
└── oasis/                     # OASIS 仿真引擎（扩展版）
    ├── social_agent/          #   LLM 驱动的智能体实现
    ├── social_platform/       #   平台仿真（数据库/推荐系统/Schema）
    ├── environment/           #   OpenAI Gym 风格的环境封装
    └── clock/                 #   仿真时钟
```

---

## :gear: 流水线阶段

### 阶段 1：数据处理（`code/data_process/`）

将原始 JSON 转换为结构化的 SQLite 数据库和 CSV 用户画像。

- **`data_process.py`** — 编排器，支持模式：`all`、`posts`、`profiles`、`relationships`
- **`user_post.py`** — 并行帖子 ETL：JSON → 按平台的 SQLite 数据库（含 SQLite 写入优化）
- **`user_profile.py`** — 画像提取，支持跨运行去重和抽样（认证用户、高粉用户、随机抽样）
- **`user_relationship.py`** — 社交图谱提取，采用两阶段 MapReduce 加权边方式

### 阶段 2：用户标签（`code/user_tagging/`）

架构最为复杂的模块。使用 LLM 通过增量积累 + 缓冲触发架构构建 **3 层兴趣分类树**和**人口统计/企业画像**。

**核心工作流：**

```
用户帖子 → 3 层质量过滤 → 缓冲积累（阈值=5）
    → 并行双任务 LLM 推理：
        任务 1：兴趣树（41 个 L1 领域 → L2 子领域 → L3 实体）
        任务 2：人口统计/企业画像
    → 基于置信度的合并 → SFT 质量门控（6 项检查） → 输出
```

**兴趣树示例：**
```json
{
  "interest_tree": [
    {
      "L1": "游戏", "confidence": "High",
      "sub_domains": [
        { "L2": "手机游戏", "meta_type": "PRODUCT",
          "entities": [{"L3": "王者荣耀", "confidence": "High"}] }
      ]
    }
  ],
  "persona_summary": "核心玩家，专注于竞技手游..."
}
```

**关键组件：**
- **核心模块**（`src/core/`）：API 客户端（LLM + 嵌入）、数据库客户端、spaCy NLP 处理器、提示词管理器、合并工具
- **平台适配器**（`src/platforms/`）：每个平台的适配器 + 过滤器，实现平台特定的数据标准化和质量过滤
- **任务模块**（`src/tasks/`）：SFT 标签编排器、状态管理（SQLite，4 张表）、SFT 质量控制、基准生成、离线标签聚类

**配置：** 按平台的 YAML 文件控制缓冲阈值、质量规则、LLM 端点和并发设置。

### 阶段 3：用户筛选（`code/user_selection/`）

筛选用户子集并转换为 OASIS 仿真输入格式。

- **用户过滤** — 动态查询语法（`min_`、`max_`、`_in=`、精确匹配）
- **基于分位数的用户分组** — 按粉丝数和发帖频率分为 5 个层级（权威媒体 → 潜水者）
- **时序数据分割** — 分为校准集和基准真值，用于评估
- **输出：** `oasis_agent_init.csv`（智能体画像）+ `oasis_database.db`（含时序分割的帖子）

### 阶段 4：仿真与评估（`code/simulation_process/`）

编排仿真执行、干预投放和仿真后评估。

**仿真执行流程：**

```
1. 数据库重置（保留校准和基准真值帖子）
2. 平台初始化（Twitter/Reddit 模式）+ 智能体生成
   → 加载画像 → 创建第一层（LLM）+ 第二层（启发式）智能体
   → 从历史帖子预加载记忆 → 批量注册
3. 干预预处理（解析 CSV → 写入数据库表）
4. 每个时间步：
   a. 动态智能体注入（如有 register_user 干预）
   b. 激活池抽样（按组层级的概率）
   c. 行为执行（第一层用 LLMAction，第二层用 HeuristicAction）
   d. 态度状态记录（所有智能体、所有维度）
   e. 推荐系统更新
5. 仿真后态度标注 + 评估
```

**关键组件：**

- **智能体层级分配**（`oasis_test_grouping.py`）：将用户组映射到智能体类和概率激活率（权威媒体 80%、KOL 70%、活跃创作者 60%、普通用户 30%、潜水者 10%）
- **干预系统**（`intervention_processor.py`）：解析干预 CSV，支持基于组的定向投放和抽样比例，确定性种子确保可复现性
- **态度标注**（`oasis_attitude.py`）：LLM 驱动的多维态度评分（-1.0 至 1.0），支持可配置维度；计算每个用户的初始和最终态度分数
- **评估指标**（`oasis_evaluation_overall.py`）：Bias（均值）、Diversity（标准差）、Pearson 相关系数 — 分别分析 LLM 与 ABM 智能体，支持与真实时间戳的时间对齐

### 阶段 5（可选）：交互式控制台（`code/marketing_simulation/`）

面向终端用户的**服务层**，让非开发者无需编写 Python 即可驱动仿真流水线。它共享相同的 `oasis/` 引擎和 `code/simulation_process/` 流水线 — 无重复源代码。

三种可互换的入口：

| 模式 | 命令 | 使用场景 |
|------|------|---------|
| **Streamlit 界面** | `bash code/marketing_simulation/run.sh` | 交互式 GUI，含干预编辑器和一键运行 |
| **Claude Desktop (MCP)** | 在 `claude_desktop_config.json` 中配置 `mcp_server.py`，在对话中粘贴 `skill.md` | 让 Claude 以对话方式设计和运行实验 |
| **Python 冒烟测试** | `python code/marketing_simulation/my_client.py` | 从命令行端到端验证完整的 4 步流水线 |

MCP 服务器暴露 11 个工具（`get_runtime_defaults`、`import_user_profiles`、`build_intervention_csv`、`run_marketing_simulation`、`query_db_table` 等），Claude 遵循 `code/marketing_simulation/skill.md` 中定义的严格 4 步 SOP 调用这些工具。运行时产物（数据库、日志、生成的干预 CSV）均写入 `code/marketing_simulation/data/` 并已被 gitignore。

详见 [`code/marketing_simulation/README.md`](code/marketing_simulation/README.md)。

---

## :joystick: OASIS 仿真引擎（`oasis/`）

基于 CAMEL-AI 框架构建的底层仿真平台。核心组件：

| 模块 | 说明 |
|------|------|
| `social_agent/` | 核心 SocialAgent，含 91 种行为类型、态度状态和社交图谱 |
| `social_platform/` | 平台仿真，含 SQLite CRUD 和推荐算法（Twitter/TWHIN/Reddit/Random） |
| `environment/` | OpenAI Gym 风格的环境封装 |
| `clock/` | 仿真时钟管理 |

---

### :wrench: OASIS 核心架构变更

MARS 修改了以下 OASIS 核心组件：

- **`social_agent/agent_custom.py`**（新增）：定义 6 种专用智能体子类，具有差异化人设和行为画像
- **`social_agent/agent_attitude.py`**（新增）：态度状态管理和 LLM 可调用的 `update_internal_attitude()` 工具，实现心智-行为一致性闭环
- **`social_agent/agents_generator.py`**（修改）：扩展支持基于层级的智能体实例化、从画像加载态度分数以及从历史帖子预加载记忆
- **`social_platform/platform.py`**（修改）：`refresh()` 现在在推荐帖子之外还返回干预指令（广播消息 + 含态度目标的定向贿赂）；接受 `attitude_metrics` 参数
- **`social_platform/database.py`**（修改）：动态创建态度指标表（`log_attitude_{metric}`）和干预表（`intervention_message`、`agent_intervention`、`pending_registrations`）
- **`environment/env.py`**（修改）：将态度指标配置传递至平台初始化

---

## :rocket: 快速开始

```bash
# 0. 激活环境
source /remote-home/JuelinW/anaconda3/bin/activate oasis
cd /remote-home/JuelinW/oasis_project

# 1. 数据处理 — 原始 JSON → 结构化数据库 + CSV
bash MARS/code/data_process.sh 2025-06-15 2025-06-15

# 2. 用户标签 — LLM 驱动的画像生成
python MARS/code/user_tagging/src/scripts/run_sft_tagging.py \
    --date 2025-06-15 --platform weibo \
    --db-base-dir MARS_result/data/output \
    --output-dir MARS_result/data/online_cold_start

# 3. 用户筛选 — 过滤 + 转换为 OASIS 格式
bash MARS/code/user_selection.sh all \
    --input-csv MARS_result/data/output/2025-06-15/user_profiles.csv \
    --posts-input-dir MARS_result/data/output/2025-06-15 \
    --oasis-out-dir data/oasis \
    --calibration-end "2025-06-15 16:00:00" \
    --ground-truth-end "2025-06-16 00:00:00"

# 4. 态度标注与评估
python -c "
from MARS.code.simulation_process.oasis_attitude import OasisAttitudeProcessor
import asyncio
config = {
    'attitude_lifestyle_culture': '评估对生活方式和文化话题的情感倾向。',
    'attitude_sport_ent': '评估对体育和娱乐的情感倾向。',
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

## :handshake: 致谢

MARS 的仿真引擎由 **[OASIS (Open Agent Social Interaction Simulations)](https://github.com/camel-ai/oasis)** 提供支持。衷心感谢 CAMEL-AI 团队的开源贡献！

---

<p align="center"><em>MARS 正在积极开发中，敬请期待！</em></p>
