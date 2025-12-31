MARS: Massive Agent-Based Real-World Simulation

MARS is a next-generation engine designed for building high-fidelity, data-driven digital twins of complex social systems.

This project provides a complete end-to-end data processing pipeline to ingest raw social media data (e.g., from platforms like Weibo) and transform it into a runnable, large-scale agent-based model compatible with the OASIS multi-agent simulation platform(The OASIS project: https://github.com/camel-ai/oasis.git). By bridging real-world data with Large Language Models (LLMs), MARS enables sociologists and researchers to simulate information propagation, public opinion evolution, and social interventions with unprecedented realism.

Following Function are currently supported(lively updated):

1. Hybrid Agent Architecture 
To balance simulation fidelity with computational efficiency, MARS implements a dual-tier agent system:

Tier 1: LLM Agents (SocialAgent)

Role: Represents Authority Media, KOLs (Key Opinion Leaders), and Active Creators.

Mechanism: Powered by LLMs (e.g., Qwen2.5, GPT-4). They utilize complex personas, memory, and reasoning to generate high-quality content and perform nuanced social actions.

Tier 2: ABM Agents (LurkerAgent)

Role: Represents the "Silent Majority" or passive users.

Mechanism: Powered by Heuristic/Rule-based models (e.g., Bounded Confidence Model). They efficiently evolve their internal states based on observed content without invoking heavy LLM inference, allowing the simulation to scale to hundreds of thousands of agents.

2. User Grouping and Dynamic Activation
Simulations are optimized through smart scheduling:

Automatic Categorization: Users are automatically classified into groups based on their raw data profiles (follower counts, posting frequency).

Probabilistic Activation: Not all agents run at every time step. High-influence agents (Tier 1) have higher activation probabilities, while Tier 2 agents are activated sparsely, simulating realistic online activity patterns.

3. Customized Evaluation Metrics
MARS allows researchers to define dynamic attitude metrics without changing the codebase.

Configuration: Simply define a dictionary of metrics and their natural language definitions.

Dynamic Schema: The engine automatically generates the necessary database tables and SQL queries.

LLM Alignment: These definitions are injected into the LLM's system prompt, ensuring the model evaluates content and updates its internal state based on your specific research criteria.

4. Internal State & Tool-Calling Mechanism
We implement a "Mind-Action" consistency loop using Parallel Function Calling:

Dual-Action Mandate: In every simulation step, an LLM Agent must perform two parallel actions:

Internal Update: Call update_internal_attitude to adjust its own psychological state based on the environment.

External Action: Call a social tool (e.g., create_post, like, repost) that reflects this updated state.

Persistence: All internal state changes are logged to the database for temporal analysis.

5. Multi-Modal Intervention System
The engine supports a sophisticated intervention system to simulate external forces on the social network. These are configured via a unified CSV file:

📢 Broadcast: Global system messages visible to all agents (e.g., "Breaking News").

💰 Bribery (Coercion): Targeted instructions sent to specific Agents (by ID) or Groups (by ratio). These act as "Mandatory Special Instructions" in the prompt, overriding the agent's default persona to simulate bought influence or PR campaigns.

🤖 Register User (Astroturfing): Dynamically inject new Agents ("Water Army" or "PR Bots") into the simulation at specific time steps with pre-defined profiles, attitudes, and immediate missions.

We are keep developing our system. Stay Tuned! 

