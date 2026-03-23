# =========== attitude_tool.py ===========
from typing import Dict, Any
import logging
from camel.toolkits import FunctionTool

class AttitudeToolHandler:
    """
    专门用于管理 Agent 态度更新工具的处理器。
    """
    TOOL_NAME = "update_internal_attitude"

    def __init__(self, agent_instance):
        """
        Args:
            agent_instance: Agent 实例 (需要有 self.attitude_scores 属性)
        """
        self.agent = agent_instance
        self.logger = logging.getLogger("social.agent")

    def _get_description(self) -> str:
        """根据 Agent 当前的指标动态生成工具描述"""
        # 获取当前所有指标的 key
        keys = list(self.agent.attitude_scores.keys())
        return (
            f"Updates the agent's internal attitude scores based on recent events. "
            f"Arguments should be a dictionary where keys are a subset of {keys} "
            f"and values are floats between -1.0 (Negative) and 1.0 (Positive). "
            f"Example: {{'{keys[0] if keys else 'attitude_example'}': 0.5}}"
        )

    def update_attitude_func(self, new_scores: Dict[str, float] | None = None, **direct_scores: float) -> str:
        """
        这是实际会被 LLM 调用的函数。
        它直接修改 agent 实例中的 attitude_scores 字典。
        """
        updated_keys = []
        merged_scores: Dict[str, float] = {}
        if isinstance(new_scores, dict):
            merged_scores.update(new_scores)
        for k, v in direct_scores.items():
            merged_scores[k] = v

        for k, v in merged_scores.items():
            # 安全检查：只更新存在的 key，防止 LLM 幻觉创造新指标
            if k in self.agent.attitude_scores:
                try:
                    # 限制范围 [-1, 1]
                    safe_val = max(-1.0, min(1.0, float(v)))
                    self.agent.attitude_scores[k] = safe_val
                    updated_keys.append(f"{k}={safe_val:.2f}")
                except ValueError:
                    continue
        
        if updated_keys:
            msg = f"Attitude Updated: {', '.join(updated_keys)}"
            # self.logger.info(f"Agent {self.agent.agent_id} internal thought: {msg}")
            return msg
        return "No attitude changes made."

    def create_tool(self) -> FunctionTool:
        """
        创建并返回 CAMEL FunctionTool 对象
        """
        openai_tool_schema = {
            "type": "function",
            "function": {
                "name": self.TOOL_NAME,
                "description": self._get_description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_scores": {
                            "type": "object",
                            "description": (
                                "Mapping of attitude metric names to float scores between -1.0 and 1.0."
                            ),
                            "additionalProperties": {"type": "number"}
                        }
                    },
                    "additionalProperties": {
                        "type": "number",
                        "description": "Optional shortcut: specify attitude dimensions directly without nesting inside new_scores."
                    },
                }
            }
        }

        return FunctionTool(self.update_attitude_func, openai_tool_schema=openai_tool_schema)