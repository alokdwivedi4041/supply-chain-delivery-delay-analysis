"""Run every query in sql/queries.sql against the SQLite DB and print results."""
import sqlite3
import pandas as pd

con = sqlite3.connect("data/processed/supply_chain.db")
text = open("sql/queries.sql").read()
blocks = [b.strip() for b in text.split(";") if "SELECT" in b.upper() or "WITH" in b.upper()]
for b in blocks:
    title = next((l for l in b.splitlines() if l.startswith("-- Q")), "-- query")
    print(f"\n{title}")
    print(pd.read_sql_query(b, con).to_string(index=False))
con.close()
