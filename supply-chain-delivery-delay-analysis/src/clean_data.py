"""Clean raw orders data, engineer features, and build a SQLite database."""
import sqlite3
import numpy as np
import pandas as pd

raw = pd.read_csv("data/raw/orders_raw.csv")
log = {"raw_rows": len(raw)}

# 1. Remove exact duplicates
raw = raw.drop_duplicates()
log["after_dedup"] = len(raw)

# 2. Standardise text columns (whitespace + casing)
fix_map = {"swiftship": "SwiftShip", "rapidx": "RapidX", "transindia": "TransIndia",
           "ecomove": "EcoMove", "quickkart logistics": "QuickKart Logistics"}
raw["carrier"] = raw["carrier"].str.strip().str.lower().map(fix_map)
raw["customer_city"] = raw["customer_city"].str.strip().str.title()

# 3. Remove invalid values (delivery time can't be < 1 day)
bad_days = raw["actual_days"] < 1
log["invalid_actual_days_removed"] = int(bad_days.sum())
raw = raw[~bad_days].copy()

# 4. Missing values
log["missing_carrier_labelled_Unknown"] = int(raw["carrier"].isna().sum())
raw["carrier"] = raw["carrier"].fillna("Unknown")
med_cost = raw.groupby("shipping_mode")["shipping_cost"].transform("median")
log["missing_shipping_cost_imputed"] = int(raw["shipping_cost"].isna().sum())
raw["shipping_cost"] = raw["shipping_cost"].fillna(med_cost)
# ratings stay NaN on purpose: imputing a satisfaction score would invent data

# 5. Outliers: cap order_value at the 99th percentile (winsorise, keep the row)
cap = raw["order_value"].quantile(0.99)
log["order_value_capped"] = int((raw["order_value"] > cap).sum())
raw["order_value"] = raw["order_value"].clip(upper=cap)

# 6. Feature engineering
df = raw.copy()
df["order_date"] = pd.to_datetime(df["order_date"])
df["delay_days"] = (df["actual_days"] - df["promised_days"]).clip(lower=0)
df["is_delayed"] = (df["actual_days"] > df["promised_days"]).astype(int)
df["month"] = df["order_date"].dt.month
df["month_name"] = df["order_date"].dt.strftime("%b")
df["season"] = np.select(
    [df["month"].isin([10, 11]), df["month"].isin([7, 8, 9]), df["month"].isin([12, 1, 2])],
    ["Festive", "Monsoon", "Winter"], default="Summer")
df["distance_band"] = pd.cut(df["distance_km"], [0, 300, 1000, 3000],
                             labels=["Local (<300km)", "Regional (300-1000km)", "National (>1000km)"])
df["delay_severity"] = pd.cut(df["delay_days"], [-1, 0, 2, 100],
                              labels=["On time", "Minor (1-2d)", "Major (3d+)"])
df["delay_reason"] = df["delay_reason"].fillna("None")
df.loc[df["is_delayed"] == 0, "delay_reason"] = "None"
df["revenue_at_risk"] = df["order_value"] * df["is_delayed"]

df = df.sort_values("order_date").reset_index(drop=True)
df.to_csv("data/processed/orders_clean.csv", index=False)

# 7. SQLite DB so the SQL queries can run
con = sqlite3.connect("data/processed/supply_chain.db")
d = df.copy()
d["order_date"] = d["order_date"].dt.strftime("%Y-%m-%d")
d["distance_band"] = d["distance_band"].astype(str)
d["delay_severity"] = d["delay_severity"].astype(str)
d.to_sql("orders", con, if_exists="replace", index=False)
con.close()

log["clean_rows"] = len(df)
print("Cleaning log:")
for k, v in log.items():
    print(f"  {k:<34} {v}")
