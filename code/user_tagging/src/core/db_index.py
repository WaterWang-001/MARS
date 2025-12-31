import sqlite3

# 连接你的数据库
conn = sqlite3.connect('MARS/data/output/2025-06-14/user_post_database.db')
cursor = conn.cursor()

# 创建索引（如果不存在）
print("Creating index on user_id...")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_user_id ON post (user_id);")
conn.commit()
print("✅ Index created successfully!")
conn.close()