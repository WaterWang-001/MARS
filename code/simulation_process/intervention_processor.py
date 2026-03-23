import csv
import json
import ast
import random
import sqlite3
import sys
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

# 优先使用本地 MARS/oasis 包而非 pip 安装的外部包
_MARS_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _MARS_ROOT not in sys.path:
    sys.path.insert(0, _MARS_ROOT)

from oasis.social_agent.agents_generator import create_and_register_single_agent


logger = logging.getLogger("intervention_processor")

class InterventionProcessor:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _parse_dict_field(self, raw_str: str) -> dict:
        """
        辅助函数：将 CSV 中的字符串解析为字典。
        支持标准 JSON (双引号) 和 Python Dict (单引号)。
        """
        if not raw_str:
            return {}
        
        # 1. 尝试标准 JSON
        try:
            return json.loads(raw_str)
        except json.JSONDecodeError:
            pass
            
        # 2. 尝试 Python 字面量 (支持 {'a': 1})
        try:
            val = ast.literal_eval(raw_str)
            if isinstance(val, dict):
                return val
        except (ValueError, SyntaxError):
            pass
            
        # 3. 解析失败，返回空字典并警告
        # logger.warning(f"无法解析字典字段: {raw_str}，将使用空字典。")
        return {}

    def process_and_distribute(self, csv_path: str, agent_list: List[Any]):
        """
        读取 CSV，处理并分发干预指令到数据库。
        """
        logger.info(f"开始处理干预文件: {csv_path}")
        
        # 1. 读取 CSV
        raw_rows = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                for row in reader:
                    raw_rows.append(row)
        except Exception as e:
            logger.error(f"读取干预 CSV 失败: {e}")
            return

        # 2. 建立 Agent 索引
        agents_by_group = defaultdict(list)
        agents_by_id = {}
        
        for agent in agent_list:
            # 获取 Group
            group = agent.user_info.profile["other_info"].get("group", "default")
            aid = agent.agent_id
            
            agents_by_group[group].append(agent)
            agents_by_group['ALL'].append(agent)
            agents_by_id[str(aid)] = agent 

        # 3. 分类处理数据
        broadcast_records = [] 
        bribery_records = []   
        registration_records = []
        
        for row in raw_rows:
            try:
                # --- 解析基础字段 ---
                time_step = int(row['time_step'])
                i_type = row.get('intervention_type', 'broadcast').strip()
                content = row.get('content', '')
                
                # --- [关键修改] 解析字典字段 ---
                # 无论 CSV 里写的是 "{""a"":1}" 还是 "{'a':1}"，这里都转成 Python Dict
                att_dict = self._parse_dict_field(row.get('attitude_target', '{}'))
                profile_dict = self._parse_dict_field(row.get('user_profile', '{}'))
                
                # 转回标准 JSON 字符串存 DB，确保 Platform 读取时不出错
                att_json_str = json.dumps(att_dict, ensure_ascii=False)
                profile_json_str = json.dumps(profile_dict, ensure_ascii=False)

                # --- 分支 1: 注册新用户 ---
                if i_type == 'register_user':
                    registration_records.append((
                        time_step,
                        profile_json_str, # 存标准 JSON
                        att_json_str,     # 存标准 JSON
                        content
                    ))
                    continue

                # --- 分支 2: 全局广播 ---
                if i_type == 'broadcast':
                    broadcast_records.append((time_step, content))
                    continue

                # --- 分支 3: 定向干预 ---
                target_group = row.get('target_group', '')
                target_id_str = row.get('target_id', '')
                ratio_str = row.get('ratio', '1.0')

                target_agents = []

                # 策略 A: 指定 ID
                if target_id_str:
                    if target_id_str in agents_by_id:
                        target_agents.append(agents_by_id[target_id_str])
                
                # 策略 B: 指定群体 + 比例
                elif target_group:
                    candidates = agents_by_group.get(target_group, [])
                    if candidates:
                        try:
                            ratio = float(ratio_str)
                        except:
                            ratio = 1.0
                        count = int(len(candidates) * ratio)
                        if count > 0:
                            rng = random.Random(f"{time_step}_{target_group}_{content}")
                            target_agents = rng.sample(candidates, count)

                for target in target_agents:
                    target_numeric_id = target.agent_id_int if hasattr(target, "agent_id_int") else int(target.agent_id)
                    bribery_records.append((
                        time_step,
                        target_numeric_id,
                        content,
                        att_json_str, # 存标准 JSON
                        i_type
                    ))
                    
            except Exception as e:
                logger.error(f"处理干预行失败: {row} -> {e}")

        # 4. 写入数据库
        self._write_to_db(broadcast_records, bribery_records, registration_records)

    def _write_to_db(self, broadcast_records, bribery_records, registration_records):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            self._create_tables_if_missing(cursor)

            # 1. Broadcast
            if broadcast_records:
                sql_bc = "INSERT INTO intervention_message (time_step, content) VALUES (?, ?)"
                cursor.executemany(sql_bc, broadcast_records)
                logger.info(f"✅ 写入 {len(broadcast_records)} 条全局广播。")

            # 2. Bribery
            if bribery_records:
                sql_br = """
                INSERT INTO agent_intervention 
                (time_step, agent_id, content, attitude_target, intervention_type) 
                VALUES (?, ?, ?, ?, ?)
                """
                cursor.executemany(sql_br, bribery_records)
                logger.info(f"✅ 写入 {len(bribery_records)} 条定向干预。")

            # 3. Registration
            if registration_records:
                sql_reg = """
                INSERT INTO pending_registrations 
                (time_step, user_profile, initial_attitude, instruction_content) 
                VALUES (?, ?, ?, ?)
                """
                cursor.executemany(sql_reg, registration_records)
                logger.info(f"✅ 写入 {len(registration_records)} 条注册指令。")
            
            conn.commit()
        except Exception as e:
            logger.error(f"写入数据库失败: {e}")
        finally:
            conn.close()

    def _create_tables_if_missing(self, cursor):
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS intervention_message (
            time_step INTEGER,
            content TEXT
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_intervention (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time_step INTEGER NOT NULL,
            agent_id INTEGER NOT NULL,
            content TEXT,
            attitude_target TEXT,
            intervention_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS pending_registrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time_step INTEGER NOT NULL,
            user_profile TEXT,
            initial_attitude TEXT,
            instruction_content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

    def _load_pending_registrations(self, cursor, current_step: int) -> List[Dict[str, Any]]:
        cursor.execute(
            """
            SELECT id, user_profile, initial_attitude, instruction_content
            FROM pending_registrations
            WHERE time_step = ?
            """,
            (current_step,)
        )
        rows = cursor.fetchall()
        registrations = []
        for reg_id, profile_json, attitude_json, instruction in rows:
            profile_data = self._parse_dict_field(profile_json) if profile_json else {}
            attitude_data = self._parse_dict_field(attitude_json) if attitude_json else {}
            registrations.append({
                "db_id": reg_id,
                "user_profile": profile_data,
                "initial_attitude": attitude_data,
                "instruction": instruction or ""
            })
        return registrations

    def _clear_processed_registrations(self, cursor, processed_ids: List[int]):
        if not processed_ids:
            return
        placeholders = ",".join(["?"] * len(processed_ids))
        cursor.execute(
            f"DELETE FROM pending_registrations WHERE id IN ({placeholders})",
            processed_ids
        )

    # 复用之前的动态注册逻辑 (保持不变)
    def execute_dynamic_registrations(self, env, current_step, current_max_agent_id, model, available_actions, attitude_metrics, agent_list):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            new_registrations = self._load_pending_registrations(cursor, current_step)
            if not new_registrations:
                return [], current_max_agent_id

            logger.info(f"检测到 {len(new_registrations)} 个新 Agent 注册指令，开始处理...")
            new_agents_created = []
            temp_max_id = current_max_agent_id

            for reg_data in new_registrations:
                temp_max_id += 1
                new_agent_id = temp_max_id
                
                profile_data = reg_data['user_profile'].copy()
                profile_data['user_id'] = str(new_agent_id)
                if 'user_char' not in profile_data:
                    profile_data['user_char'] = "You are an influential media account."
                
                target_group = profile_data.get('group', "权威媒体/大V")
                profile_data['group'] = target_group
                
                # 合并态度
                profile_data.update(reg_data['initial_attitude'])
                
                # 创建
                new_agent = create_and_register_single_agent(
                    agent_id=new_agent_id,
                    user_profile=profile_data,
                    platform=env.platform,
                    db_path=self.db_path,
                    model=model, 
                    available_actions=available_actions,
                    current_time_step=current_step, 
                    attitude_metrics=attitude_metrics,
                    is_dynamic_injection=True 
                )
                
                # 插入任务
                instruction = reg_data['instruction']
                if instruction:
                    cursor.execute(
                        """
                        INSERT INTO agent_intervention 
                        (time_step, agent_id, content, attitude_target, intervention_type) 
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            current_step, 
                            new_agent_id, 
                            instruction, 
                            json.dumps(reg_data['initial_attitude'], ensure_ascii=False), 
                            'bribery'
                        )
                    )

                new_agents_created.append(new_agent)
                agent_list.append(new_agent)
                if isinstance(env.agent_graph, list):
                    env.agent_graph.append(new_agent)
                
                logger.info(f"✅ 新 Agent {new_agent_id} ({new_agent.user_info.name}) 已上线并接受指令。")

            processed_ids = [item["db_id"] for item in new_registrations]
            self._clear_processed_registrations(cursor, processed_ids)
            conn.commit()
            return new_agents_created, temp_max_id
        except Exception as e:
            logger.error(f"动态注册过程中发生错误: {e}", exc_info=True)
        finally:
            conn.close()
        return [], current_max_agent_id