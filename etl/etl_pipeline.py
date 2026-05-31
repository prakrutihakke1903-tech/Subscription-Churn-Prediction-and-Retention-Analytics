import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS  — os.path.join works on Windows AND Mac/Linux
# ─────────────────────────────────────────────────────────────
# __file__ = this script's location  →  etl/etl_pipeline.py
# dirname(__file__)  = etl/
# dirname(dirname()) = project root  = Capstone_SubscriptionChurn/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")       # data/raw/
OUT_DIR  = os.path.join(BASE_DIR, "data")              # data/
os.makedirs(OUT_DIR, exist_ok=True)

# Key dates
AS_OF_DATE = pd.Timestamp("2026-01-16")   # our prediction reference date
OBS_WEEKS  = 4                             # look-back window for features

print("=" * 60)
print("  SUBSCRIPTION CHURN ETL PIPELINE")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD RAW FILES
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading raw files from:", RAW_DIR)

users   = pd.read_csv(os.path.join(RAW_DIR, "users.csv"))
plans   = pd.read_csv(os.path.join(RAW_DIR, "plans.csv"))
subs    = pd.read_csv(os.path.join(RAW_DIR, "subscriptions.csv"))
pays    = pd.read_csv(os.path.join(RAW_DIR, "payments.csv"))
usage   = pd.read_csv(os.path.join(RAW_DIR, "usage_daily.csv"))
touches = pd.read_csv(os.path.join(RAW_DIR, "campaign_touchpoints.csv"))

print(f"  users       : {len(users):>7,} rows")
print(f"  plans       : {len(plans):>7,} rows")
print(f"  subs        : {len(subs):>7,} rows")
print(f"  payments    : {len(pays):>7,} rows")
print(f"  usage_daily : {len(usage):>7,} rows")
print(f"  touchpoints : {len(touches):>7,} rows")

# ─────────────────────────────────────────────────────────────
# STEP 2 — CLEAN & STANDARDISE EACH TABLE
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Cleaning & standardising ...")

# ── users ────────────────────────────────────────────────────
# drop_duplicates: keeps the first occurrence of each user_id, removes repeats
before = len(users)
users  = users.drop_duplicates(subset="user_id", keep="first")
print(f"  users   : removed {before - len(users)} duplicate rows")

# .str.lower().str.strip() → fixes mixed casing ("Mobile" → "mobile") and spaces
users["segment"]             = users["segment"].str.lower().str.strip()
users["preferred_device"]    = users["preferred_device"].str.lower().str.strip()
users["acquisition_channel"] = users["acquisition_channel"].str.lower().str.strip()
users["preferred_device"]    = users["preferred_device"].fillna("unknown")   # fill missing
users["signup_date"]         = pd.to_datetime(users["signup_date"])

# ── subscriptions ────────────────────────────────────────────
before = len(subs)
subs   = subs.drop_duplicates(subset="subscription_id", keep="first")
print(f"  subs    : removed {before - len(subs)} duplicate rows")

subs["status"]      = subs["status"].str.lower().str.strip()
subs["start_date"]  = pd.to_datetime(subs["start_date"])
subs["end_date"]    = pd.to_datetime(subs["end_date"])
subs["cancel_date"] = pd.to_datetime(subs["cancel_date"], errors="coerce")  # NaT if blank

# ── payments ─────────────────────────────────────────────────
before = len(pays)
pays   = pays.drop_duplicates(subset="payment_id", keep="first")
print(f"  pays    : removed {before - len(pays)} duplicate rows")

# Standardise to ALL CAPS so "success", "SUCCESS", "Success" all become "SUCCESS"
pays["payment_status"] = pays["payment_status"].str.upper().str.strip()
pays["payment_method"] = pays["payment_method"].str.lower().str.strip()
pays["payment_date"]   = pd.to_datetime(pays["payment_date"])

# Flag outliers: amounts more than 3 standard deviations above the mean
mean_a = pays["amount"].mean()
std_a  = pays["amount"].std()
pays["amount_outlier_flag"] = (pays["amount"] > mean_a + 3 * std_a).astype(int)
n_outliers = pays["amount_outlier_flag"].sum()
print(f"  pays    : flagged {n_outliers} payment amount outliers (>{mean_a + 3*std_a:.0f})")

# ── usage_daily ───────────────────────────────────────────────
before = len(usage)
# A user can only have 1 usage record per day — remove any duplicates
usage  = usage.drop_duplicates(subset=["user_id", "date"], keep="first")
print(f"  usage   : removed {before - len(usage)} duplicate rows")

usage["date"]           = pd.to_datetime(usage["date"])
usage["minutes_used"]   = pd.to_numeric(usage["minutes_used"],   errors="coerce").fillna(0)
usage["sessions_count"] = pd.to_numeric(usage["sessions_count"], errors="coerce").fillna(0)
usage["feature_events"] = pd.to_numeric(usage["feature_events"], errors="coerce").fillna(0)

# Flag minute outliers at the 99th percentile (extreme values, not errors per se)
p99 = usage["minutes_used"].quantile(0.99)
usage["minutes_outlier_flag"] = (usage["minutes_used"] >= p99).astype(int)
print(f"  usage   : flagged {usage['minutes_outlier_flag'].sum()} minute outliers (>={p99:.0f} min/day)")

# ── campaign_touchpoints ──────────────────────────────────────
before  = len(touches)
touches = touches.drop_duplicates(subset="touch_id", keep="first")
print(f"  touches : removed {before - len(touches)} duplicate rows")

touches["channel"]       = touches["channel"].str.lower().str.strip()
touches["campaign_type"] = touches["campaign_type"].str.lower().str.strip().fillna("unknown")
touches["touch_date"]    = pd.to_datetime(touches["touch_date"])

# ─────────────────────────────────────────────────────────────
# STEP 3 — OUTPUT 1: dim_users_enriched.csv
# One row per user with profile info + derived features
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Building dim_users_enriched.csv ...")

# Get each user's most recent subscription (sort newest first, keep first per user)
latest_sub = (
    subs.sort_values("start_date", ascending=False)
        .drop_duplicates("user_id", keep="first")
    [["user_id", "plan_id", "plan_price", "start_date", "end_date", "status"]]
    .rename(columns={
        "plan_id":    "current_plan_id",
        "plan_price": "current_plan_price",
        "start_date": "sub_start",
        "end_date":   "sub_end",
        "status":     "sub_status",
    })
)

# Count subscription cycles per user → proxy for "lifetime paid months"
paid_months = (
    subs.groupby("user_id")
        .agg(lifetime_paid_months=("subscription_id", "count"))
        .reset_index()
)

# Most recent day a user appeared in the usage data
last_active = (
    usage.groupby("user_id")["date"]
         .max()
         .reset_index()
         .rename(columns={"date": "last_active_date"})
)

# Average usage over the last 4 weeks (observation window)
obs_start = AS_OF_DATE - pd.Timedelta(weeks=OBS_WEEKS)
recent    = usage[usage["date"] >= obs_start]
avg_wk    = (
    recent.groupby("user_id")
          .agg(
              total_days_obs=("date",         "count"),
              total_min_obs =("minutes_used", "sum"),
          )
          .reset_index()
)
avg_wk["avg_active_days"]  = avg_wk["total_days_obs"] / OBS_WEEKS
avg_wk["avg_minutes_week"] = avg_wk["total_min_obs"]  / OBS_WEEKS

# Engagement band: how many days/week did the user actively use the product?
# low = < 2 days/week,  mid = 2-5 days/week,  high = > 5 days/week
avg_wk["engagement_band"] = pd.cut(
    avg_wk["avg_active_days"],
    bins=[-1, 2, 5, 100],
    labels=["low", "mid", "high"]
)

# Join all pieces onto the main users table
dim = (
    users
    .merge(latest_sub,  on="user_id", how="left")
    .merge(paid_months, on="user_id", how="left")
    .merge(last_active, on="user_id", how="left")
    .merge(
        avg_wk[["user_id", "avg_active_days", "avg_minutes_week", "engagement_band"]],
        on="user_id", how="left"
    )
    .merge(
        plans[["plan_id", "plan_name", "tier"]].rename(columns={"plan_id": "current_plan_id"}),
        on="current_plan_id", how="left"
    )
)

# tenure_days = how many days since the user signed up (as of AS_OF_DATE)
dim["tenure_days"]          = (AS_OF_DATE - dim["signup_date"]).dt.days
dim["lifetime_paid_months"] = dim["lifetime_paid_months"].fillna(0)
dim["engagement_band"]      = dim["engagement_band"].astype(str).replace("nan", "low").fillna("low")

# is_churned = 1 if their latest sub is cancelled or expired
dim["is_churned"] = dim["sub_status"].isin(["cancelled", "expired"]).astype(int)

# Select & order columns for the final output
dim_out = dim[[
    "user_id", "signup_date", "city_tier", "segment",
    "preferred_device", "acquisition_channel",
    "tenure_days", "current_plan_id", "plan_name", "tier",
    "current_plan_price", "lifetime_paid_months",
    "last_active_date", "engagement_band",
    "avg_active_days", "avg_minutes_week",
    "sub_status", "sub_end", "is_churned",
]]

out_path = os.path.join(OUT_DIR, "dim_users_enriched.csv")
dim_out.to_csv(out_path, index=False)
print(f"  -> dim_users_enriched.csv  ({len(dim_out):,} rows)  saved to: {out_path}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — OUTPUT 2: fact_user_weekly.csv
# One row per user per week with weekly aggregated metrics
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Building fact_user_weekly.csv ...")

# week_start = the Monday of each week that date falls in
# pd.to_timedelta(dayofweek, 'd') subtracts days back to Monday
usage["week_start"] = (
    usage["date"] - pd.to_timedelta(usage["date"].dt.dayofweek, unit="d")
)

# Sum/count usage per user per week
wk_usage = (
    usage.groupby(["user_id", "week_start"])
         .agg(
             active_days_week         = ("date",           "nunique"),
             total_minutes_week       = ("minutes_used",   "sum"),
             sessions_week            = ("sessions_count", "sum"),
             feature_usage_count_week = ("feature_events", "sum"),
         )
         .reset_index()
)

# Payment attempts and failures per user per week
pays["week_start"] = (
    pays["payment_date"] - pd.to_timedelta(pays["payment_date"].dt.dayofweek, unit="d")
)
wk_pays = (
    pays.groupby(["user_id", "week_start"])
        .agg(
            payment_attempts_week = ("payment_id",     "count"),
            payment_failures_week = ("payment_status", lambda x: (x == "FAILED").sum()),
        )
        .reset_index()
)

# Merge usage and payments
fact = wk_usage.merge(wk_pays, on=["user_id", "week_start"], how="left")
fact["payment_attempts_week"] = fact["payment_attempts_week"].fillna(0)
fact["payment_failures_week"] = fact["payment_failures_week"].fillna(0)

# renewal_due_flag = 1 if any subscription end_date falls in the next 14 days from week_start
# Pre-build a dict: user_id -> list of all their subscription end dates
sub_end_map = subs.groupby("user_id")["end_date"].apply(list).to_dict()

def compute_renewal_flag(user_id, week_start):
    """Return 1 if the user has a subscription expiring within 14 days of this week."""
    ends       = sub_end_map.get(user_id, [])
    window_end = week_start + pd.Timedelta(days=13)
    return int(any(week_start <= e <= window_end for e in ends))

fact["renewal_due_flag"] = [
    compute_renewal_flag(row.user_id, row.week_start)
    for row in fact[["user_id", "week_start"]].itertuples()
]

out_path = os.path.join(OUT_DIR, "fact_user_weekly.csv")
fact.to_csv(out_path, index=False)
print(f"  -> fact_user_weekly.csv    ({len(fact):,} rows)  saved to: {out_path}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — OUTPUT 3: model_churn_dataset.csv
# One row per user, feature matrix + churn label for ML
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Building model_churn_dataset.csv ...")

obs_start = AS_OF_DATE - pd.Timedelta(weeks=OBS_WEEKS)
usage_obs  = usage[(usage["date"]          >= obs_start) & (usage["date"]          < AS_OF_DATE)]
pays_obs   = pays[ (pays["payment_date"]   >= obs_start) & (pays["payment_date"]   < AS_OF_DATE)]
touch_obs  = touches[(touches["touch_date"] >= obs_start) & (touches["touch_date"] < AS_OF_DATE)]

# ── Feature group 1: total usage over 4-week window ───────────
f_usage = (
    usage_obs.groupby("user_id")
             .agg(
                 total_active_days    = ("date",           "nunique"),
                 total_minutes        = ("minutes_used",   "sum"),
                 total_sessions       = ("sessions_count", "sum"),
                 total_feature_events = ("feature_events", "sum"),
             )
             .reset_index()
)

# ── Feature group 2: usage TREND (last 1 wk vs prior 3 wks) ───
# This captures "is engagement dropping?" — a key early warning signal
usage_obs = usage_obs.copy()
usage_obs["week_start"] = (
    usage_obs["date"] - pd.to_timedelta(usage_obs["date"].dt.dayofweek, unit="d")
)
last_ws = AS_OF_DATE - pd.Timedelta(weeks=1)
last_ws = last_ws - pd.Timedelta(days=last_ws.dayofweek)

wk_days = (
    usage_obs.groupby(["user_id", "week_start"])["date"]
             .nunique()
             .reset_index()
             .rename(columns={"date": "active_days"})
)
wk1  = (wk_days[wk_days["week_start"] >= last_ws]
            .groupby("user_id")["active_days"].mean()
            .reset_index().rename(columns={"active_days": "days_last_week"}))
wk14 = (wk_days[wk_days["week_start"] < last_ws]
            .groupby("user_id")["active_days"].mean()
            .reset_index().rename(columns={"active_days": "days_prior_avg"}))
trend = wk1.merge(wk14, on="user_id", how="outer").fillna(0)
# Positive trend = improving,  Negative trend = declining (churn signal!)
trend["usage_trend"] = trend["days_last_week"] - trend["days_prior_avg"]

# ── Feature group 3: payment behaviour ────────────────────────
f_pays = (
    pays_obs.groupby("user_id")
            .agg(
                pay_attempts = ("payment_id",     "count"),
                pay_failures = ("payment_status", lambda x: (x == "FAILED").sum()),
            )
            .reset_index()
)
f_pays["pay_failure_rate"] = (
    f_pays["pay_failures"] / f_pays["pay_attempts"].replace(0, 1)
)

# ── Feature group 4: recency (days since last usage) ──────────
f_rec = (
    usage_obs.groupby("user_id")["date"]
             .max()
             .reset_index()
             .rename(columns={"date": "last_activity_date"})
)
f_rec["days_since_last_activity"] = (AS_OF_DATE - f_rec["last_activity_date"]).dt.days

# ── Feature group 5: campaign engagement ──────────────────────
f_touch = (
    touch_obs.groupby("user_id")
             .agg(
                 touch_count    = ("touch_id", "count"),
                 touch_clicked  = ("clicked",  "sum"),
                 touch_redeemed = ("redeemed", "sum"),
             )
             .reset_index()
)

# ── Base: user + plan profile features ────────────────────────
plan_feat = dim_out[[
    "user_id", "current_plan_id", "current_plan_price", "tier",
    "tenure_days", "city_tier", "segment", "engagement_band"
]].copy()

# ── Merge all feature groups ───────────────────────────────────
model_df = (
    plan_feat
    .merge(f_usage,  on="user_id", how="left")
    .merge(trend[["user_id", "usage_trend", "days_last_week"]], on="user_id", how="left")
    .merge(f_pays,   on="user_id", how="left")
    .merge(f_rec,    on="user_id", how="left")
    .merge(f_touch,  on="user_id", how="left")
)

# Fill nulls with 0 (user had no activity/payments in the window)
num_cols = [
    "total_active_days", "total_minutes", "total_sessions", "total_feature_events",
    "usage_trend", "days_last_week", "pay_attempts", "pay_failures", "pay_failure_rate",
    "days_since_last_activity", "touch_count", "touch_clicked", "touch_redeemed",
]
model_df[num_cols] = model_df[num_cols].fillna(0)

# ── CHURN LABEL: will_churn_14d ────────────────────────────────
# Label = 1 if user's latest subscription is cancelled or expired
latest_s = (
    subs.sort_values("start_date", ascending=False)
        .drop_duplicates("user_id", keep="first")
    [["user_id", "status", "end_date"]]
)
latest_s["will_churn_14d"] = latest_s["status"].isin(["cancelled", "expired"]).astype(int)

model_df = model_df.merge(latest_s[["user_id", "will_churn_14d"]], on="user_id", how="left")
model_df["will_churn_14d"] = model_df["will_churn_14d"].fillna(0).astype(int)
model_df["as_of_date"]     = str(AS_OF_DATE.date())

out_path = os.path.join(OUT_DIR, "model_churn_dataset.csv")
model_df.to_csv(out_path, index=False)

churn_pct = model_df["will_churn_14d"].mean() * 100
print(f"  -> model_churn_dataset.csv ({len(model_df):,} rows | churn rate: {churn_pct:.1f}%)")
print(f"     saved to: {out_path}")

print("\n" + "=" * 60)
print("  ETL COMPLETE - 3 files saved to data/ folder")
print("=" * 60)
