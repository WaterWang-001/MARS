-- This is the schema definition for the user table
CREATE TABLE user (
    user_id TEXT PRIMARY KEY,
    agent_id INTEGER,
    user_name TEXT,
    name TEXT,
    bio TEXT,
    created_at DATETIME,
    num_followings INTEGER DEFAULT 0,
    num_followers INTEGER DEFAULT 0,
    attitude_lifestyle_culture REAL DEFAULT 0.0,
    attitude_sport_ent REAL DEFAULT 0.0,
    attitude_sci_health REAL DEFAULT 0.0,
    attitude_politics_econ REAL DEFAULT 0.0,
    initial_attitude_avg REAL DEFAULT 0.0
);