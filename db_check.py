import sqlite3

DB_PATH = "/data/user_data.db"

def print_table(table):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        print(f"--- {table} ---")
        for row in rows:
            print(row)
    except Exception as e:
        print(f"Error reading {table}: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    for table in ["users", "stripe_customers", "messages"]:
        print_table(table)