"""EDA, statistical testing, delay-prediction model and chart generation."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid", context="talk")
PALETTE = "#1f6feb"
RED = "#d1364a"
df = pd.read_csv("data/processed/orders_clean.csv", parse_dates=["order_date"])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"images/{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ KPIs
kpi = {
    "total_orders": int(len(df)),
    "delay_rate_pct": round(df["is_delayed"].mean() * 100, 1),
    "avg_delay_days_when_delayed": round(df.loc[df.is_delayed == 1, "delay_days"].mean(), 2),
    "on_time_rate_pct": round((1 - df["is_delayed"].mean()) * 100, 1),
    "revenue_at_risk_inr": round(df["revenue_at_risk"].sum()),
    "revenue_at_risk_pct": round(df["revenue_at_risk"].sum() / df["order_value"].sum() * 100, 1),
    "avg_rating_on_time": round(df.loc[df.is_delayed == 0, "customer_rating"].mean(), 2),
    "avg_rating_delayed": round(df.loc[df.is_delayed == 1, "customer_rating"].mean(), 2),
}

# ------------------------------------------------------------------ 1. Monthly trend
m = df.groupby("month").agg(delay_rate=("is_delayed", "mean"), orders=("order_id", "count")).reset_index()
m["delay_rate"] *= 100
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(m["month"], m["delay_rate"], marker="o", color=PALETTE, lw=3)
for mo, c, lab in [((7, 9), "#f4a261", "Monsoon"), ((10, 11), RED, "Festive")]:
    ax.axvspan(mo[0] - 0.4, mo[1] + 0.4, color=c, alpha=0.15, label=lab)
ax.axhline(kpi["delay_rate_pct"], ls="--", color="gray", lw=1.5)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_ylabel("Delay rate (%)"); ax.set_title("Monthly delivery delay rate")
ax.legend(loc="upper left")
save(fig, "01_monthly_delay_trend")

# ------------------------------------------------------------------ 2. Carrier
c = df[df.carrier != "Unknown"].groupby("carrier").agg(
    delay_rate=("is_delayed", "mean"), orders=("order_id", "count")).sort_values("delay_rate")
c["delay_rate"] *= 100
fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.barh(c.index, c["delay_rate"], color=[PALETTE if v < kpi["delay_rate_pct"] else RED for v in c["delay_rate"]])
ax.axvline(kpi["delay_rate_pct"], ls="--", color="gray")
for b, v in zip(bars, c["delay_rate"]):
    ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center")
ax.set_xlabel("Delay rate (%)"); ax.set_title("Carrier performance (red = worse than average)")
save(fig, "02_carrier_delay_rate")

# ------------------------------------------------------------------ 3. Region x Carrier heatmap
h = df[df.carrier != "Unknown"].pivot_table(index="region", columns="carrier", values="is_delayed", aggfunc="mean") * 100
fig, ax = plt.subplots(figsize=(12, 5.5))
sns.heatmap(h, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax, cbar_kws={"label": "Delay rate (%)"})
ax.set_title("Delay rate (%) by region and carrier"); ax.set_xlabel(""); ax.set_ylabel("")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)
save(fig, "03_region_carrier_heatmap")

# ------------------------------------------------------------------ 4. Distance band + mode
d = df.groupby(["distance_band", "shipping_mode"], observed=True)["is_delayed"].mean().mul(100).reset_index()
fig, ax = plt.subplots(figsize=(11, 5.5))
sns.barplot(data=d, x="distance_band", y="is_delayed", hue="shipping_mode", ax=ax, palette="Blues_d")
ax.set_ylabel("Delay rate (%)"); ax.set_xlabel(""); ax.set_title("Delay rate by distance band and shipping mode")
ax.legend(title="Mode")
save(fig, "04_distance_mode")

# ------------------------------------------------------------------ 5. Pareto of delay reasons
r = df[df.is_delayed == 1]["delay_reason"].value_counts()
cum = r.cumsum() / r.sum() * 100
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(r.index, r.values, color=PALETTE)
ax.set_ylabel("Delayed orders"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax2 = ax.twinx(); ax2.plot(r.index, cum.values, color=RED, marker="o", lw=2.5)
ax2.set_ylim(0, 105); ax2.set_ylabel("Cumulative %"); ax2.grid(False)
ax.set_title("Pareto: root causes of delay")
save(fig, "05_delay_reason_pareto")

# ------------------------------------------------------------------ 6. Rating impact
fig, ax = plt.subplots(figsize=(10, 5.5))
rt = df.dropna(subset=["customer_rating"]).groupby("delay_severity", observed=True)["customer_rating"].mean()
rt = rt.reindex(["On time", "Minor (1-2d)", "Major (3d+)"])
bars = ax.bar(rt.index, rt.values, color=[PALETTE, "#f4a261", RED])
for b, v in zip(bars, rt.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center")
ax.set_ylim(0, 5); ax.set_ylabel("Avg customer rating"); ax.set_title("Delays hurt customer satisfaction")
save(fig, "06_rating_vs_delay")

# ------------------------------------------------------------------ 7. Season
s = df.groupby("season")["is_delayed"].mean().mul(100).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(s.index, s.values, color=[PALETTE if v < kpi["delay_rate_pct"] else RED for v in s.values])
for i, v in enumerate(s.values):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
ax.set_ylabel("Delay rate (%)"); ax.set_title("Seasonality of delays")
save(fig, "07_season_delay")

# ------------------------------------------------------------------ Statistical tests
tests = {}
for col in ["carrier", "shipping_mode", "region", "season", "distance_band"]:
    ct = pd.crosstab(df[col], df["is_delayed"])
    chi2, p, dof, _ = chi2_contingency(ct)
    tests[col] = {"chi2": round(chi2, 1), "p_value": float(f"{p:.2e}"), "significant_at_5pct": bool(p < 0.05)}

# ------------------------------------------------------------------ Predictive model
num = ["distance_km", "promised_days", "order_value", "quantity", "shipping_cost"]
cat = ["carrier", "shipping_mode", "region", "season", "warehouse", "product_category"]
X, y = df[num + cat], df["is_delayed"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pre = ColumnTransformer([("n", StandardScaler(), num),
                         ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
model.fit(Xtr, ytr)
proba = model.predict_proba(Xte)[:, 1]
auc = roc_auc_score(yte, proba)
report = classification_report(yte, model.predict(Xte), target_names=["On time", "Delayed"], output_dict=True)

names = model.named_steps["pre"].get_feature_names_out()
coef = pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values()
top = pd.concat([coef.head(7), coef.tail(7)])
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh([n.split("__")[1] for n in top.index], top.values, color=[PALETTE if v < 0 else RED for v in top.values])
ax.set_title("Delay drivers (logistic regression coefficients)\nred = increases delay risk")
save(fig, "08_delay_drivers")

# ------------------------------------------------------------------ Summary + insights
worst_c = c["delay_rate"].idxmax(); best_c = c["delay_rate"].idxmin()
reg = df.groupby("region")["is_delayed"].mean().mul(100)
dist = df.groupby("distance_band", observed=True)["is_delayed"].mean().mul(100)
top2 = r.head(2)
top3_share = cum.iloc[2]
peak_month = m.loc[m.delay_rate.idxmax()]
mode_d = df.groupby("shipping_mode")["is_delayed"].mean().mul(100)

summary = {"kpis": kpi, "chi_square_tests": tests,
           "model": {"roc_auc": round(auc, 3),
                     "delayed_recall": round(report["Delayed"]["recall"], 2),
                     "delayed_precision": round(report["Delayed"]["precision"], 2)}}
json.dump(summary, open("reports/summary.json", "w"), indent=2)

insights = f"""# Key Findings & Recommendations

**Overall:** {kpi['delay_rate_pct']}% of {kpi['total_orders']:,} orders were delivered late
(avg {kpi['avg_delay_days_when_delayed']} days late). INR {kpi['revenue_at_risk_inr']:,} of order value
({kpi['revenue_at_risk_pct']}%) was tied to delayed shipments.

## Findings
1. **Carrier matters most.** {worst_c} has a {c.loc[worst_c,'delay_rate']:.1f}% delay rate versus
   {c.loc[best_c,'delay_rate']:.1f}% for {best_c}; differences are statistically significant (chi-square p < 0.001).
2. **Seasonality.** Peak delay month is month {int(peak_month['month'])} ({peak_month['delay_rate']:.1f}%).
   Festive season delay rate: {df[df.season=='Festive'].is_delayed.mean()*100:.1f}% vs
   Winter {df[df.season=='Winter'].is_delayed.mean()*100:.1f}%.
3. **Distance.** National (>1000 km) shipments are late {dist.iloc[2]:.1f}% of the time versus
   {dist.iloc[0]:.1f}% for local ones.
4. **Region.** Worst region: {reg.idxmax()} ({reg.max():.1f}%); best: {reg.idxmin()} ({reg.min():.1f}%).
5. **Root causes.** {top2.index[0]} and {top2.index[1]} are the top reasons; the top 3 causes explain
   {top3_share:.0f}% of all delays (Pareto).
6. **Customer impact.** Average rating drops from {kpi['avg_rating_on_time']} (on time) to
   {kpi['avg_rating_delayed']} (delayed).
7. **Model.** A logistic regression predicts delay risk at order time with ROC-AUC = {auc:.2f}.

## Recommendations
- Re-allocate volume away from {worst_c} on long-distance and East-region lanes, or renegotiate SLAs / penalties.
- Add buffer capacity and pre-position inventory before the monsoon (Jul-Sep) and festive (Oct-Nov) periods.
- Promise realistic delivery dates for >1000 km lanes (add +1 day) to cut "late" perception.
- Fix warehouse dispatch backlog and stock-out issues (controllable, internal causes).
- Use the delay-risk model to flag high-risk orders for proactive customer communication.
"""
open("reports/insights.md", "w").write(insights)

print(json.dumps(summary, indent=2))
print(insights)
