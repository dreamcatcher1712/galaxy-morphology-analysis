import os
import sqlite3
import pandas as pd

print("CURRENT WORKING DIR =", os.getcwd())
print("DB PATH =", os.path.abspath("results/sql/galaxy_morphology.db"))

# 🔥 Change this to YOUR absolute path
DB_PATH = "outputs/sql/galaxy_morphology.db"

def query(sql, params=None):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

def list_tables():
    return query("SELECT name FROM sqlite_master WHERE type='table'")
