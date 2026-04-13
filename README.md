<h1 align="center">MARS：Massive Agent-Based Real-World Simulation</h1>


<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square" alt="License"/></a>
</p>

<p align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</p>

---

<a name="english"></a>

## What is MARS?

MARS transforms **raw social media data** into **large-scale agent simulations** — end to end, fully automated. Built on [OASIS](https://github.com/camel-ai/oasis.git), it creates high-fidelity digital twins of complex social systems where hundreds of thousands of AI agents autonomously post, comment, like, and interact on a simulated social platform.

```
Raw Social Data ──→ Data Processing ──→ LLM User Profiling ──→ Agent Simulation ──→ Evaluation
    (JSON)            (SQLite + CSV)      (Interest Trees)      (LLM + Heuristic)    (Bias/Diversity/Pearson)
```

## Key Features

:zap: **End-to-End Pipeline** — From raw JSON to running simulation, zero manual data wrangling

:busts_in_silhouette: **Hybrid Agent Architecture** — LLM-powered influencers + heuristic-driven silent majority, scales to 100K+ agents

:brain: **Attitude Dynamics** — Multi-dimensional attitude states (-1.0 to 1.0) that evolve through social interaction

:syringe: **Intervention Experiments** — Inject broadcasts, targeted bribery, or astroturfing agents mid-simulation via CSV

:bar_chart: **Quantitative Evaluation** — Bias, Diversity, and Pearson metrics against real-world ground truth

:globe_with_meridians: **Multi-Platform** — Built-in support for Weibo, Zhihu, Xiaohongshu, Toutiao, Douban + extensible adapters

## Quick Start

```bash
# Install
conda create -n oasis python=3.11 -y && conda activate oasis
pip install -e oasis
pip install fastmcp streamlit pandas python-dotenv

# Configure LLM credentials
cp code/marketing_simulation/.env.example code/marketing_simulation/data/.env
# Edit data/.env → fill in MARS_MODEL_BASE_URL + MARS_MODEL_API_KEY

# Launch the interactive console
bash code/marketing_simulation/run.sh
# Open http://localhost:8501 → design interventions → click Run
```

## Two Ways to Use

| Mode | How | Best For |
|------|-----|----------|
| **Streamlit UI** | `bash code/marketing_simulation/run.sh` | Visual intervention design, one-click simulation |
| **Claude Desktop (MCP)** | [Setup guide](docs/README_FULL_EN.md#option-2--claude-desktop-via-mcp-conversational) | Conversational experiment design with AI |

## Full Documentation

:point_right: **[Complete English Documentation](docs/README_FULL_EN.md)** — Pipeline stages, architecture, input schemas, OASIS engine details, and more.

---

<a name="中文"></a>

## MARS 是什么？

MARS 将**原始社交媒体数据**转化为**大规模智能体仿真** — 端到端，全自动。基于 [OASIS](https://github.com/camel-ai/oasis.git) 构建，它创建复杂社会系统的高保真数字孪生，数十万 AI 智能体在模拟社交平台上自主发帖、评论、点赞和互动。

```
原始社交数据 ──→ 数据处理 ──→ LLM 用户画像 ──→ 智能体仿真 ──→ 评估
   (JSON)       (SQLite + CSV)  (兴趣分类树)    (LLM + 启发式)  (Bias/Diversity/Pearson)
```

## 核心特性

:zap: **端到端流水线** — 从原始 JSON 到运行仿真，零手动数据清洗

:busts_in_silhouette: **混合智能体架构** — LLM 驱动的意见领袖 + 启发式驱动的沉默多数，可扩展至 10 万+ 智能体

:brain: **态度动力学** — 多维态度状态（-1.0 至 1.0），通过社交互动动态演化

:syringe: **干预实验** — 通过 CSV 配置，在仿真中途注入广播消息、定向贿赂或水军智能体

:bar_chart: **量化评估** — Bias、Diversity、Pearson 指标对比真实世界基准数据

:globe_with_meridians: **多平台支持** — 内置微博、知乎、小红书、头条、豆瓣适配器，可扩展

## 快速开始

```bash
# 安装
conda create -n oasis python=3.11 -y && conda activate oasis
pip install -e oasis
pip install fastmcp streamlit pandas python-dotenv

# 配置 LLM 凭证
cp code/marketing_simulation/.env.example code/marketing_simulation/data/.env
# 编辑 data/.env → 填写 MARS_MODEL_BASE_URL 和 MARS_MODEL_API_KEY

# 启动交互式控制台
bash code/marketing_simulation/run.sh
# 打开 http://localhost:8501 → 设计干预策略 → 点击 Run
```

## 两种使用方式

| 模式 | 方法 | 适用场景 |
|------|------|---------|
| **Streamlit 图形界面** | `bash code/marketing_simulation/run.sh` | 可视化干预设计，一键运行仿真 |
| **Claude Desktop (MCP)** | [配置指南](docs/README_FULL_CN.md#方式二--通过-mcp-接入-claude-desktop对话式) | AI 对话式实验设计 |

## 完整文档

:point_right: **[完整中文文档](docs/README_FULL_CN.md)** — 流水线阶段、架构设计、输入数据格式、OASIS 引擎详情等。

---

## Acknowledgments / 致谢

MARS is powered by **[OASIS](https://github.com/camel-ai/oasis)** from the CAMEL-AI team. / MARS 的仿真引擎由 CAMEL-AI 团队的 **[OASIS](https://github.com/camel-ai/oasis)** 提供支持。

<p align="center"><em>MARS is under active development. Stay tuned! / MARS 正在积极开发中，敬请期待！</em></p>
