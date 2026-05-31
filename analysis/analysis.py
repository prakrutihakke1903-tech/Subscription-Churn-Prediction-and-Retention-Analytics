"""
analysis.py  -  Parts C, D, E, F
===================================

WHAT THIS FILE DOES:
--------------------
Part C  - KPI calculations, cohort retention, segment deep-dives, churn investigation table
Part D  - Churn prediction model (Logistic Regression + Random Forest)
Part E  - Retention strategy (printed as structured recommendations)
Part F  - 30-day financial impact estimation

HOW TO RUN (after running etl_pipeline.py first):
  python analysis/analysis.py

Charts saved to: analysis/charts/
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # needed for saving charts without a display window
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (roc_auc_score, classification_report,
                                     confusion_matrix, roc_curve)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
CHARTS_DIR = os.path.join(BASE_DIR, "analysis", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

AS_OF = pd.Timestamp("2026-01-16")

# Style
sns.set_style("whitegrid")
C = {"churn": "#E74C3C", "retain": "#2ECC71", "blue": "#3498DB", "warn": "#F39C12", "purple": "#9B59B6"}

def save_chart(fig, name):
    path = os.path.join(CHARTS_DIR, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Chart saved: {name}.png")

print("=" * 60)
print("  CHURN ANALYSIS  —  Parts C, D, E, F")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
dim    = pd.read_csv(os.path.join(DATA_DIR, "dim_users_enriched.csv"),
                     parse_dates=["signup_date", "sub_end", "last_active_date"])
weekly = pd.read_csv(os.path.join(DATA_DIR, "fact_user_weekly.csv"),
                     parse_dates=["week_start"])
model  = pd.read_csv(os.path.join(DATA_DIR, "model_churn_dataset.csv"))

subs   = pd.read_csv(os.path.join(RAW_DIR, "subscriptions.csv"),
                     parse_dates=["start_date", "end_date", "cancel_date"])
subs["status"] = subs["status"].str.lower().str.strip()

pays   = pd.read_csv(os.path.join(RAW_DIR, "payments.csv"),
                     parse_dates=["payment_date"])
pays["payment_status"] = pays["payment_status"].str.upper().str.strip()

usage  = pd.read_csv(os.path.join(RAW_DIR, "usage_daily.csv"),
                     parse_dates=["date"])

print(f"\nLoaded: dim={len(dim):,}  weekly={len(weekly):,}  model={len(model):,}")

# ══════════════════════════════════════════════════════════════
# PART C-1  MONTHLY CHURN & RETENTION RATES
# ══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("PART C-1  Monthly Churn & Retention Rates")
print("─" * 60)

# Count subscriptions that ended (churned) in each month
churned = subs[subs["status"].isin(["cancelled", "expired"])].copy()
churned["churn_month"] = churned["end_date"].dt.to_period("M")
monthly_churn = (churned.groupby("churn_month").size()
                         .reset_index(name="churned_count"))

# Count subs that were active at any point during each month
all_months = pd.period_range(start="2025-07", end="2026-01", freq="M")
active_rows = []
for m in all_months:
    ms     = m.to_timestamp()
    me     = (m + 1).to_timestamp()
    active = ((subs["start_date"] <= me) & (subs["end_date"] >= ms)).sum()
    active_rows.append({"month": m, "active_count": active})

active_df = pd.DataFrame(active_rows)
monthly = (active_df
           .merge(monthly_churn.rename(columns={"churn_month": "month"}),
                  on="month", how="left")
           .fillna(0))
monthly["churn_rate"]     = monthly["churned_count"] / monthly["active_count"] * 100
monthly["retention_rate"] = 100 - monthly["churn_rate"]
monthly["month_str"]      = monthly["month"].astype(str)

print(monthly[["month_str", "active_count", "churned_count",
               "churn_rate", "retention_rate"]].to_string(index=False))

# Chart 1: Churn & Retention trend
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(monthly["month_str"], monthly["churn_rate"], color=C["churn"], alpha=0.85)
ax1.set_title("Monthly Churn Rate (%)", fontsize=13, fontweight="bold")
ax1.set_ylabel("Churn Rate (%)")
ax1.tick_params(axis="x", rotation=30)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())

ax2.plot(monthly["month_str"], monthly["retention_rate"],
         marker="o", linewidth=2.5, color=C["retain"])
ax2.set_title("Monthly Retention Rate (%)", fontsize=13, fontweight="bold")
ax2.set_ylabel("Retention Rate (%)")
ax2.tick_params(axis="x", rotation=30)
ax2.set_ylim(50, 100)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
save_chart(fig, "C1_monthly_churn_retention")

# Payment failure rate by month
pays["pay_month"] = pays["payment_date"].dt.to_period("M")
mpay = (pays.groupby("pay_month")
             .agg(attempts=("payment_id", "count"),
                  failures=("payment_status", lambda x: (x == "FAILED").sum()))
             .reset_index())
mpay["fail_rate"] = mpay["failures"] / mpay["attempts"] * 100
mpay["month_str"] = mpay["pay_month"].astype(str)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(mpay["month_str"], mpay["fail_rate"], color=C["warn"], alpha=0.85)
ax.set_title("Monthly Payment Failure Rate (%)", fontsize=13, fontweight="bold")
ax.set_ylabel("Failure Rate (%)")
ax.tick_params(axis="x", rotation=30)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
save_chart(fig, "C1_payment_failure_rate")

# ── Cohort Retention Heatmap ──────────────────────────────────
print("\n  Building cohort retention table ...")
users_raw = pd.read_csv(os.path.join(RAW_DIR, "users.csv"),
                        parse_dates=["signup_date"])
subs2 = subs.merge(users_raw[["user_id", "signup_date"]], on="user_id", how="left")
subs2["cohort_month"]        = subs2["signup_date"].dt.to_period("M")
subs2["sub_month"]           = subs2["start_date"].dt.to_period("M")
subs2["months_since_signup"] = (subs2["sub_month"] - subs2["cohort_month"]).apply(
    lambda x: x.n if hasattr(x, "n") else 0)

cohort_size = subs2.groupby("cohort_month")["user_id"].nunique()
cohort_ret  = (subs2.groupby(["cohort_month", "months_since_signup"])["user_id"]
                    .nunique().reset_index())
cohort_ret  = cohort_ret.merge(cohort_size.rename("cohort_size"), on="cohort_month")
cohort_ret["retention_pct"] = cohort_ret["user_id"] / cohort_ret["cohort_size"] * 100

pivot = cohort_ret.pivot_table(
    index="cohort_month", columns="months_since_signup",
    values="retention_pct").round(1)
pivot.index = pivot.index.astype(str)
print(pivot.to_string())

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Retention %"},
            vmin=0, vmax=100)
ax.set_title("Cohort Retention Heatmap  (% still active after N months)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Months Since First Subscription")
ax.set_ylabel("Cohort (Signup Month)")
plt.tight_layout()
save_chart(fig, "C1_cohort_retention_heatmap")

# ══════════════════════════════════════════════════════════════
# PART C-2  WEEKLY KPI TRENDS
# ══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("PART C-2  Weekly KPI Trends")
print("─" * 60)

kpi_wk = (weekly.groupby("week_start")
                 .agg(
                     avg_active_days = ("active_days_week",      "mean"),
                     avg_minutes     = ("total_minutes_week",    "mean"),
                     avg_sessions    = ("sessions_week",         "mean"),
                     pay_failures    = ("payment_failures_week", "sum"),
                     pay_attempts    = ("payment_attempts_week", "sum"),
                     renewal_flags   = ("renewal_due_flag",      "sum"),
                 )
                 .reset_index())
kpi_wk["pay_fail_rate"] = (kpi_wk["pay_failures"]
                            / kpi_wk["pay_attempts"].replace(0, 1) * 100)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes[0, 0].plot(kpi_wk["week_start"], kpi_wk["avg_active_days"],
                color=C["retain"], linewidth=2, marker="o", markersize=3)
axes[0, 0].set_title("Avg Active Days/Week per User", fontweight="bold")
axes[0, 0].set_ylabel("Days")

axes[0, 1].plot(kpi_wk["week_start"], kpi_wk["avg_minutes"],
                color=C["blue"], linewidth=2, marker="o", markersize=3)
axes[0, 1].set_title("Avg Minutes/Week per User", fontweight="bold")
axes[0, 1].set_ylabel("Minutes")

axes[1, 0].plot(kpi_wk["week_start"], kpi_wk["avg_sessions"],
                color=C["warn"], linewidth=2, marker="o", markersize=3)
axes[1, 0].set_title("Avg Sessions/Week per User", fontweight="bold")
axes[1, 0].set_ylabel("Sessions")

axes[1, 1].plot(kpi_wk["week_start"], kpi_wk["pay_fail_rate"],
                color=C["churn"], linewidth=2, marker="o", markersize=3)
axes[1, 1].set_title("Weekly Payment Failure Rate (%)", fontweight="bold")
axes[1, 1].set_ylabel("Failure Rate (%)")

for ax in axes.flat:
    ax.tick_params(axis="x", rotation=30)
plt.suptitle("Weekly KPI Trends", fontsize=15, fontweight="bold")
plt.tight_layout()
save_chart(fig, "C2_weekly_kpi_trends")

# ══════════════════════════════════════════════════════════════
# PART C-3  SEGMENT DEEP DIVE
# ══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("PART C-3  Segment Deep Dive")
print("─" * 60)

def churn_by_segment(col, label):
    """Compute churn rate and revenue at risk for each value in a column."""
    g = (dim.groupby(col)
            .agg(
                total          = ("user_id",              "count"),
                churned        = ("is_churned",           "sum"),
                avg_plan_price = ("current_plan_price",   "mean"),
            )
            .reset_index())
    g["churn_rate"]     = g["churned"] / g["total"] * 100
    g["revenue_at_risk"] = g["churned"] * g["avg_plan_price"]
    g = g.sort_values("churn_rate", ascending=False)
    print(f"\n  Churn by {label}:")
    print(g[[col, "total", "churned", "churn_rate", "revenue_at_risk"]].to_string(index=False))
    return g

seg_plan   = churn_by_segment("current_plan_id", "Plan")
seg_seg    = churn_by_segment("segment",         "User Segment")
seg_tier   = churn_by_segment("city_tier",       "City Tier")
seg_eng    = churn_by_segment("engagement_band", "Engagement Band")

dim["tenure_band"] = pd.cut(
    dim["tenure_days"],
    bins=[0, 30, 90, 180, 9999],
    labels=["New (0-30d)", "Early (31-90d)", "Mid (91-180d)", "Loyal (180d+)"]
)
seg_tenure = churn_by_segment("tenure_band", "Tenure Band")

# Multi-panel segment chart
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

def bar_segment(ax, df, col, title):
    ax.bar(df[col].astype(str), df["churn_rate"],
           color=C["churn"], alpha=0.85, edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for i, v in enumerate(df["churn_rate"]):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

bar_segment(axes[0, 0], seg_plan,   "current_plan_id", "Churn Rate by Plan")
bar_segment(axes[0, 1], seg_seg,    "segment",         "Churn Rate by User Segment")
bar_segment(axes[0, 2], seg_tier,   "city_tier",       "Churn Rate by City Tier")
bar_segment(axes[1, 0], seg_eng,    "engagement_band", "Churn Rate by Engagement")
bar_segment(axes[1, 1], seg_tenure, "tenure_band",     "Churn Rate by Tenure Band")

# Revenue at risk bar
rev_plan = seg_plan.sort_values("revenue_at_risk", ascending=False)
axes[1, 2].bar(rev_plan["current_plan_id"], rev_plan["revenue_at_risk"],
               color=C["warn"], alpha=0.85, edgecolor="white")
axes[1, 2].set_title("Revenue at Risk by Plan (Rs)", fontweight="bold")
axes[1, 2].set_ylabel("Revenue at Risk (Rs)")

plt.suptitle("Churn Segment Deep Dive", fontsize=15, fontweight="bold")
plt.tight_layout()
save_chart(fig, "C3_segment_deep_dive")

# Print top segments
print("\n  TOP 3 — Highest Churn Rate:")
top3_churn = pd.concat([
    seg_plan.nlargest(1, "churn_rate").assign(dimension="Plan"),
    seg_seg.nlargest(1, "churn_rate").assign(dimension="Segment"),
    seg_eng.nlargest(1, "churn_rate").assign(dimension="Engagement"),
])
print(top3_churn[["dimension", "churn_rate", "churned"]].to_string(index=False))

print("\n  TOP 3 — Highest Revenue at Risk (Rs):")
top3_rev = pd.concat([
    seg_plan.nlargest(1, "revenue_at_risk").assign(dimension="Plan"),
    seg_seg.nlargest(1, "revenue_at_risk").assign(dimension="Segment"),
    seg_tenure.nlargest(1, "revenue_at_risk").assign(dimension="Tenure"),
])
print(top3_rev[["dimension", "revenue_at_risk", "churned"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════
# PART C-4  CHURN INVESTIGATION TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("PART C-4  Churn Investigation Table")
print("─" * 60)

total_users   = len(dim)
total_churned = dim["is_churned"].sum()
overall_rate  = total_churned / total_users * 100

low_eng   = dim[dim["engagement_band"] == "low"]
d1_rate   = low_eng["is_churned"].mean() * 100
d1_n      = low_eng["is_churned"].sum()
d1_contrib = d1_n / total_churned * 100

med_low  = dim[dim["engagement_band"] == "low"]["avg_active_days"].median()
med_high = dim[dim["engagement_band"] == "high"]["avg_active_days"].median()

model["has_failure"] = (model["pay_failures"] > 0).astype(int)
d2_rates  = model.groupby("has_failure")["will_churn_14d"].mean() * 100
d2_rate   = d2_rates.get(1, d2_rates.iloc[-1])
d2_n      = model[(model["has_failure"] == 1) & (model["will_churn_14d"] == 1)].shape[0]
d2_contrib = d2_n / total_churned * 100 if total_churned > 0 else 0

print(f"""
OVERALL  Churn Rate : {overall_rate:.1f}%  ({total_churned} / {total_users} users)

DRIVER 1 : Low Engagement (< 2 active days / week)
  Churn rate in segment : {d1_rate:.1f}%
  Contribution to total : {d1_contrib:.1f}% of all churns ({d1_n} users)
  Median active days/wk : LOW={med_low:.1f}  vs  HIGH={med_high:.1f}
  Top contributing sub-segments : Basic plan, New users (0-30d tenure)
  Hypotheses:
    H1 - Poor onboarding; users do not discover core value in first 14 days
    H2 - Content/feature recommendation quality is low for new users
    H3 - No habit loop formed before first renewal date
  Validation experiments:
    H1 -> A/B test: enhanced onboarding checklist (treated) vs none (control)
    H2 -> Analyse feature_events by days_since_signup; check if users reaching
          3 features in D7 have higher 30-day retention
    H3 -> Track D7 and D14 active days as leading indicators of M1 renewal
  Evidence: segment deep dive chart C3, cohort heatmap C1

DRIVER 2 : Payment Failures
  Churn rate (>= 1 failure) : {d2_rate:.1f}%
  Contribution to total     : {d2_contrib:.1f}% of all churns (~{d2_n} users)
  Top sub-segments          : NetBanking users, Basic plan holders
  Hypotheses:
    H1 - Involuntary churn: card expiry / insufficient funds go unaddressed
    H2 - No automated retry or dunning sequence triggered after failure
    H3 - Certain payment methods (NetBanking) have higher failure rates
  Validation experiments:
    H1 -> Add a "payment expiry alert" email 7 days before renewal; measure retry rate
    H2 -> A/B test 3-step SMS dunning vs no follow-up after failed payment
    H3 -> Compare churn rate by payment_method using payments.csv cross-tab
  Evidence: payment failure chart C1, pay_failure_rate feature in model
""")

# ══════════════════════════════════════════════════════════════
# PART D  CHURN PREDICTION MODEL
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("PART D  Churn Prediction Model")
print("=" * 60)

df = model.copy()

# Encode text columns as numbers so sklearn can use them
cat_cols = ["current_plan_id", "tier", "segment", "engagement_band"]
for c in cat_cols:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

FEATURES = [
    "total_active_days", "total_minutes", "total_sessions", "total_feature_events",
    "usage_trend", "days_last_week", "pay_attempts", "pay_failures", "pay_failure_rate",
    "days_since_last_activity", "touch_count", "touch_clicked", "touch_redeemed",
    "current_plan_price", "tenure_days", "current_plan_id", "tier", "segment",
    "engagement_band",
]
FEATURES = [f for f in FEATURES if f in df.columns]
TARGET   = "will_churn_14d"

X = df[FEATURES].fillna(0)
y = df[TARGET]

print(f"\n  Samples: {len(X):,}   Features: {len(FEATURES)}   Churn rate: {y.mean()*100:.1f}%")

# TIME-BASED SPLIT:
# Users with higher tenure are "older" and represent historical data (TRAIN).
# Users with lower tenure are more recent (TEST).
# This mimics how you would train on past data to predict future churn.
split_thresh = df["tenure_days"].quantile(0.75)
train_mask   = df["tenure_days"] >= split_thresh
test_mask    = df["tenure_days"] <  split_thresh

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
print(f"  Train (long-tenure users) : {len(X_train):,}   Test (short-tenure): {len(X_test):,}")

# Scale features — important for Logistic Regression (not for RF but doesn't hurt)
scaler  = StandardScaler()
Xtr_sc  = scaler.fit_transform(X_train)
Xte_sc  = scaler.transform(X_test)

# ── Model 1: Logistic Regression (baseline) ───────────────────
print("\n  [Model 1] Logistic Regression ...")
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(Xtr_sc, y_train)
lr_probs = lr.predict_proba(Xte_sc)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_probs)
print(f"  LR  ROC-AUC : {lr_auc:.3f}")

# ── Model 2: Random Forest ────────────────────────────────────
print("  [Model 2] Random Forest ...")
rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                             class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_probs)
print(f"  RF  ROC-AUC : {rf_auc:.3f}")

# ── Threshold selection ───────────────────────────────────────
# Business context: It is cheaper to send a retention offer to a user who
# would NOT have churned (false positive cost ~ Rs 50 discount) than to MISS
# a churner (false negative cost ~ Rs 299-799 lost revenue).
# So we lower the threshold to 0.35 to catch more true churners (high recall).
THRESHOLD = 0.35
rf_preds  = (rf_probs >= THRESHOLD).astype(int)

print(f"\n  === Results @ threshold = {THRESHOLD} (chosen for high recall) ===")
print(classification_report(y_test, rf_preds, target_names=["Retained", "Churned"]))

cm = confusion_matrix(y_test, rf_preds)
print(f"  Confusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

# ── ROC curves ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
axes[0].plot(fpr_lr, tpr_lr, label=f"Logistic Reg (AUC={lr_auc:.2f})",
             color=C["blue"], linewidth=2)
axes[0].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_auc:.2f})",
             color=C["retain"], linewidth=2)
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[0].set_title("ROC Curves", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# Feature importance
feat_imp = pd.DataFrame({
    "feature":   FEATURES,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False).head(12)

axes[1].barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1],
             color=C["purple"], alpha=0.85)
axes[1].set_title("Top Feature Importances (Random Forest)", fontweight="bold")
axes[1].set_xlabel("Importance Score")
plt.tight_layout()
save_chart(fig, "D_model_performance")

# Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"], ax=ax)
ax.set_title(f"Confusion Matrix @ threshold={THRESHOLD}", fontweight="bold")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
save_chart(fig, "D_confusion_matrix")

print("\n  Top 10 churn-predictive features:")
print(feat_imp[["feature", "importance"]].head(10).to_string(index=False))

print("""
  Early Warning Signals (from feature importance + business logic):
    1. days_since_last_activity   - users going quiet = strong churn precursor
    2. usage_trend (negative)     - dropping week-over-week engagement
    3. pay_failures               - even 1 failed payment raises churn risk sharply
    4. total_active_days (low)    - overall low engagement over the 4-week window
    5. days_last_week             - near-zero activity in final week before renewal
""")

# ── Risk score & risk band ─────────────────────────────────────
# Apply model to ALL users to get churn probability scores
df_all = model.copy()
for c in cat_cols:
    df_all[c] = LabelEncoder().fit_transform(df_all[c].astype(str))
X_all = df_all[FEATURES].fillna(0)
df_all["churn_prob"] = rf.predict_proba(X_all)[:, 1]
df_all["risk_band"]  = pd.cut(df_all["churn_prob"],
                               bins=[0, 0.35, 0.60, 1.0],
                               labels=["Low", "Medium", "High"])
df_all["user_id"]    = model["user_id"].values

# Save targeting list
target_list = df_all[df_all["risk_band"] == "High"][
    ["user_id", "churn_prob", "risk_band", "current_plan_price"]
].sort_values("churn_prob", ascending=False)
target_path = os.path.join(DATA_DIR, "high_risk_targets.csv")
target_list.to_csv(target_path, index=False)
print(f"  High-risk users identified: {len(target_list):,}  (saved to high_risk_targets.csv)")

risk_dist = df_all["risk_band"].value_counts()
print(f"  Risk band distribution:\n{risk_dist.to_string()}")

# Risk distribution chart
fig, ax = plt.subplots(figsize=(7, 5))
risk_dist.plot(kind="bar", ax=ax,
               color=[C["retain"], C["warn"], C["churn"]], edgecolor="white", alpha=0.9)
ax.set_title("User Risk Band Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Risk Band")
ax.set_ylabel("Number of Users")
ax.tick_params(axis="x", rotation=0)
for i, v in enumerate(risk_dist):
    ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
plt.tight_layout()
save_chart(fig, "D_risk_band_distribution")

# ══════════════════════════════════════════════════════════════
# PART E  RETENTION STRATEGY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART E  Retention Strategy + Experiment Design")
print("=" * 60)

high_risk_count = (df_all["risk_band"] == "High").sum()
med_risk_count  = (df_all["risk_band"] == "Medium").sum()

print(f"""
TARGETING LOGIC
  Who  : High-risk users (churn_prob >= 0.60) = {high_risk_count:,} users
          Medium-risk users (0.35-0.60)        = {med_risk_count:,} users (secondary wave)
  When : Trigger campaign 7-10 days before subscription end_date
         OR immediately when usage_trend < -2 days/week for 2 consecutive weeks

INTERVENTION 1 — Renewal Reminder (Low cost, broad reach)
  Target    : ALL high-risk users
  Action    : Personalised push notification + email 7 days before renewal
  Message   : "Your subscription renews in 7 days — here is what you have been watching"
  Expected  : 8-12% of targeted users re-engage and renew without discount

INTERVENTION 2 — Limited-Time Discount (Medium cost, medium-high engagement)
  Target    : High-risk users on Premium / Standard plans (higher LTV)
  Action    : 20% off next 3 months if they renew within 48 hours
  Message   : "We want you to stay — here is a special offer just for you"
  Expected  : 15-20% conversion of targeted users
  Cost      : Rs 100 per user (20% of Rs 499 avg plan price)

INTERVENTION 3 — Payment Retry Nudge (Low cost, addresses involuntary churn)
  Target    : Users with pay_failures >= 1 in last 4 weeks
  Action    : SMS + app notification with "Update payment method" deep link
  Message   : "Your last payment did not go through — tap here to update your card"
  Expected  : 30-40% of affected users complete payment retry within 3 days
  Cost      : Near zero (automated SMS ~Rs 0.50/user)

INTERVENTION 4 — Personalised Content Pack (No-cost, engagement play)
  Target    : Low-engagement users (active < 2 days/week, not yet paying)
  Action    : Curated "top picks for you" in-app notification + email digest
  Message   : "You have not watched in a while — we picked 5 things you will love"
  Expected  : 10-15% of targeted users return within 7 days

A/B TEST PLAN
  Test      : Intervention 2 (20% Discount)
  Control   : No discount, only renewal reminder (Intervention 1)
  Treatment : Renewal reminder + 20% discount offer
  Primary metric   : 30-day renewal rate
  Guardrail metrics:
    - Discount cost per retained user (target: < Rs 300)
    - Net revenue per user over 3 months (must stay positive vs CAC)
    - Support complaints / cancellation requests (should not rise)
  Duration  : 30 days (1 full billing cycle)
  Sample size (rough):
    - Baseline renewal rate = ~73% (1 - 27.2% churn)
    - Target uplift         = +10 percentage points -> 83%
    - Alpha=0.05, Power=0.80 -> ~400 users per arm (~800 total)
    - We have {high_risk_count:,} high-risk users — sufficient for the test
""")

# ══════════════════════════════════════════════════════════════
# PART F  30-DAY IMPACT ESTIMATION
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("PART F  30-Day Impact Estimation")
print("=" * 60)

# Key numbers from the data
total_active     = (dim["sub_status"] == "active").sum()
avg_plan_price   = dim.loc[dim["sub_status"] == "active", "current_plan_price"].mean()
baseline_churn_r = model["will_churn_14d"].mean()
monthly_churners = int(total_active * baseline_churn_r)
revenue_at_risk  = monthly_churners * avg_plan_price

# Intervention assumptions
reach_pct          = 0.70   # we reach 70% of high-risk users with campaign
conversion_pct_b   = 0.12   # base case:  12% of reached users retain
conversion_pct_bst = 0.20   # best case:  20%
conversion_pct_wst = 0.06   # worst case: 6%
discount_pct       = 0.20   # 20% discount applied to 50% of retained users
targeted_users     = int(monthly_churners * reach_pct)
discount_cost_per  = avg_plan_price * discount_pct * 0.50  # half get a discount

def scenario(label, conv_pct):
    retained       = int(targeted_users * conv_pct)
    rev_retained   = retained * avg_plan_price
    discount_cost  = retained * discount_cost_per
    net_impact     = rev_retained - discount_cost
    print(f"  {label:12s} | Users retained: {retained:>4}  | "
          f"Revenue saved: Rs {rev_retained:>8,.0f}  | "
          f"Discount cost: Rs {discount_cost:>6,.0f}  | "
          f"Net impact: Rs {net_impact:>8,.0f}")
    return net_impact

print(f"""
  Active subscribers    : {total_active:,}
  Avg plan price        : Rs {avg_plan_price:.0f}/month
  Baseline churn rate   : {baseline_churn_r*100:.1f}%
  Expected churners/mo  : {monthly_churners:,}
  Revenue at risk/mo    : Rs {revenue_at_risk:,.0f}
  Users targeted (70%)  : {targeted_users:,}
""")
print(f"  {'Scenario':<12} | {'Users Retained':>16} | {'Revenue Saved':>20} | "
      f"{'Discount Cost':>17} | {'Net Impact':>17}")
print("  " + "-" * 90)
scenario("WORST CASE",  conversion_pct_wst)
scenario("BASE CASE",   conversion_pct_b)
scenario("BEST CASE",   conversion_pct_bst)

print(f"""
  KEY ASSUMPTIONS:
    1. Baseline churn rate remains {baseline_churn_r*100:.1f}% without intervention
    2. Campaign reaches 70% of high-risk users (email + push deliverability)
    3. 50% of retained users receive the 20% discount; 50% retained by reminder only
    4. Retained users stay for at least 1 additional billing cycle
    5. No significant cannibalization of organic renewals
""")

print("=" * 60)
print("  ANALYSIS COMPLETE — charts saved to analysis/charts/")
print("=" * 60)
