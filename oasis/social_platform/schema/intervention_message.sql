CREATE TABLE IF NOT EXISTS intervention_message (
            intervention_id INTEGER PRIMARY KEY AUTOINCREMENT,
            time_step INTEGER NOT NULL,
            content TEXT
        )