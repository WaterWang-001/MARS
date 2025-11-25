CREATE TABLE IF NOT EXISTS log_attitude_average (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_step INTEGER,
    user_id TEXT,       
    agent_id INTEGER,   
    agent_type TEXT,
    metric_type TEXT,
    attitude_score REAL
);