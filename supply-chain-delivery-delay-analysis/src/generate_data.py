"""Generate a realistic, intentionally messy synthetic supply-chain dataset.

Why synthetic? It lets the project run anywhere with zero downloads, and the
'dirty' rows give the cleaning step real work to do. To use real data instead
(e.g. the Kaggle 'DataCo Smart Supply Chain' dataset), drop it in data/raw/
and adapt the column mapping in src/clean_data.py.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 6000

REGIONS = {
    "North": ["Delhi", "Dehradun", "Lucknow", "Chandigarh", "Jaipur"],
    "West": ["Mumbai", "Pune", "Ahmedabad", "Surat"],
    "South": ["Bengaluru", "Chennai", "Hyderabad", "Kochi"],
    "East": ["Kolkata", "Patna", "Bhubaneswar", "Guwahati"],
}
city_list = [(c, r) for r, cs in REGIONS.items() for c in cs]
region_weights = {"North": 0.30, "West": 0.28, "South": 0.27, "East": 0.15}
w = np.array([region_weights[r] / len(REGIONS[r]) for _, r in city_list])
w /= w.sum()

CARRIERS = ["SwiftShip", "RapidX", "TransIndia", "EcoMove", "QuickKart Logistics"]
carrier_effect = {"SwiftShip": -0.5, "RapidX": -0.2, "TransIndia": 0.1,
                  "EcoMove": 0.5, "QuickKart Logistics": 0.9}
WAREHOUSES = ["Delhi-WH", "Mumbai-WH", "Bengaluru-WH", "Kolkata-WH"]
CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Grocery", "Beauty", "Books"]
cat_value = {"Electronics": 9000, "Apparel": 1800, "Home & Kitchen": 3200,
             "Grocery": 700, "Beauty": 1100, "Books": 450}

idx = rng.choice(len(city_list), N, p=w)
cities = np.array([city_list[i][0] for i in idx])
regions = np.array([city_list[i][1] for i in idx])

order_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 366, N), unit="D")
carrier = rng.choice(CARRIERS, N, p=[0.25, 0.22, 0.23, 0.17, 0.13])
mode = rng.choice(["Standard", "Express", "Same-Day"], N, p=[0.68, 0.27, 0.05])
warehouse = rng.choice(WAREHOUSES, N, p=[0.32, 0.28, 0.25, 0.15])
category = rng.choice(CATEGORIES, N, p=[0.2, 0.25, 0.18, 0.17, 0.12, 0.08])

zone = rng.choice(["Local", "Regional", "National"], N, p=[0.2, 0.45, 0.35])
distance = np.where(zone == "Local", rng.uniform(40, 300, N),
           np.where(zone == "Regional", rng.uniform(300, 1000, N),
                    rng.uniform(1000, 2500, N))).round(0)

value = np.array([rng.lognormal(np.log(cat_value[c]), 0.6) for c in category]).round(2)
qty = rng.integers(1, 6, N)

month = order_date.month.to_numpy()
festive = np.isin(month, [10, 11])          # Diwali / festive rush
monsoon = np.isin(month, [7, 8, 9])         # monsoon disruption
warehouse_backlog = rng.random(N) < 0.08    # stock-out / dispatch backlog

promised = np.where(mode == "Same-Day", 1,
           np.where(mode == "Express", 2 + (distance > 1000).astype(int),
                    3 + (distance // 600).astype(int)))

logit = (-1.35
         + np.array([carrier_effect[c] for c in carrier])
         + np.where(mode == "Express", 0.25, 0)
         + np.where(mode == "Same-Day", 0.6, 0)
         + 0.55 * (distance / 1000)
         + 0.55 * festive + 0.40 * monsoon
         + 0.7 * warehouse_backlog
         + np.where(regions == "East", 0.25, 0)
         + np.where(cities == "Guwahati", 0.5, 0))
p_delay = 1 / (1 + np.exp(-logit))
delayed = rng.random(N) < p_delay

extra = np.where(delayed, rng.geometric(0.45, N) + (rng.random(N) < 0.15) * 3, 0)
early = (~delayed) & (rng.random(N) < 0.18) & (promised > 1)
actual = promised + extra - early.astype(int)

reasons = ["Weather", "Carrier Capacity", "Warehouse Dispatch Delay", "Address Issue",
           "Customer Unavailable", "Traffic / Strike", "Stock-out"]
delay_reason = []
for i in range(N):
    if not delayed[i]:
        delay_reason.append(None)
        continue
    pr = np.array([0.10, 0.20, 0.17, 0.12, 0.13, 0.10, 0.08])
    if monsoon[i]:
        pr[0] += 0.30
    if festive[i]:
        pr[1] += 0.28
    if warehouse_backlog[i]:
        pr[2] += 0.5
        pr[6] += 0.2
    pr /= pr.sum()
    delay_reason.append(rng.choice(reasons, p=pr))

mode_cost = {"Standard": 0.05, "Express": 0.11, "Same-Day": 0.25}
ship_cost = (40 + distance * np.array([mode_cost[m] for m in mode])
             * rng.uniform(0.8, 1.2, N)).round(2)

rating = np.clip(np.round(4.6 - 0.45 * np.minimum(extra, 5) + rng.normal(0, 0.7, N)), 1, 5)

df = pd.DataFrame({
    "order_id": [f"ORD{100000 + i}" for i in range(N)],
    "order_date": order_date.strftime("%Y-%m-%d"),
    "customer_city": cities, "region": regions, "warehouse": warehouse,
    "carrier": carrier, "shipping_mode": mode, "product_category": category,
    "quantity": qty, "order_value": value, "shipping_cost": ship_cost,
    "distance_km": distance, "promised_days": promised, "actual_days": actual,
    "delay_reason": delay_reason, "customer_rating": rating,
})

# ---- inject realistic mess ----
df.loc[df.sample(frac=0.03, random_state=1).index, "customer_rating"] = np.nan
df.loc[df.sample(frac=0.015, random_state=2).index, "carrier"] = np.nan
df.loc[df.sample(frac=0.02, random_state=3).index, "shipping_cost"] = np.nan
messy = df.sample(frac=0.06, random_state=4).index
df.loc[messy, "carrier"] = df.loc[messy, "carrier"].map(
    lambda x: x.upper() + " " if isinstance(x, str) else x)
lc = df.sample(frac=0.03, random_state=5).index
df.loc[lc, "customer_city"] = df.loc[lc, "customer_city"].str.lower() + "  "
df.loc[df.sample(n=12, random_state=6).index, "order_value"] *= 25   # outliers
df.loc[df.sample(n=8, random_state=7).index, "actual_days"] = -1      # invalid
df = pd.concat([df, df.sample(frac=0.02, random_state=8)], ignore_index=True)  # duplicates
df = df.sample(frac=1, random_state=9).reset_index(drop=True)

df.to_csv("data/raw/orders_raw.csv", index=False)
print(f"Saved data/raw/orders_raw.csv  ->  {df.shape[0]} rows, {df.shape[1]} columns")
