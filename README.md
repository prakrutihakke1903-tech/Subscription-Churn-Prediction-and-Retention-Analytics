# Subscription Churn Early-Warning & Retention Strategy

## 👤 Author

**Prakruti Hakke**
Business Analytics Associate

---

# 📌 PROJECT OVERVIEW

This project develops an end-to-end analytics solution to diagnose, predict, and reduce subscription churn in a digital subscription business (SaaS / Learning / OTT model).

The solution includes:

- Business framing & KPI design
- Churn definition engineering
- Data cleaning & curated analytics tables
- KPI, cohort & segment diagnostics
- Supervised churn prediction model
- Risk-based retention strategy
- A/B experiment design
- 30-day financial impact estimation
- Executive dashboard & final memo

The objective is to build a **data-driven early-warning system** that predicts churn 7–14 days in advance and enables targeted retention intervention.

---

# 📌 PART A — BUSINESS FRAMING

---

## 1️⃣ Business Objective

The company is experiencing rising churn in its subscription-based digital product.

The primary objective of this project is:

> Reduce subscription churn and increase short-term retained revenue by building a predictive early-warning system that identifies high-risk users 7–14 days before churn.

This initiative aims to:

- Identify behavioral and financial drivers of churn
- Predict high-risk users before renewal
- Prioritize revenue-at-risk segments
- Launch measurable retention experiments
- Estimate 30-day financial impact

---

## 2️⃣ North Star Metric

### 🎯 Monthly Retention Rate

[
Monthly\ Retention\ Rate = 1 - Monthly\ Churn\ Rate
]

This reflects subscription stability and recurring revenue sustainability.

### Revenue Lens (Business Proxy)

**Revenue Retained (30-day forward projection)**
Used to quantify financial impact.

---

## 3️⃣ Supporting KPIs

### 🔹 Churn Rate (Weekly & Monthly)

[
Churn = \frac{Users\ Churned}{Active\ Users\ at\ Start}
]

---

### 🔹 Retention Rate

[
Retention = 1 - Churn
]

---

### 🔹 Cohort Retention

Retention by:

- Signup month
- First paid month

---

### 🔹 Renewal Success Rate

[
Renewal\ Success = \frac{Successful\ Renewals}{Renewal\ Due}
]

---

### 🔹 Payment Failure Rate

[
Payment\ Failure\ Rate = \frac{Failed\ Payments}{Total\ Attempts}
]

---

### 🔹 Engagement Metrics

- Active days per week
- Total minutes per week
- Sessions per week

---

### 🔹 Revenue Lost (Proxy)

Churned Users × Plan Price

---

### 🔹 Revenue-at-Risk

Revenue from users predicted to churn in next 14 days.

---

### 🔹 Campaign Response Rate

[
Response\ Rate = \frac{Renewed\ After\ Campaign}{Users\ Targeted}
]

---

### 🔹 Engagement Band Distribution

% users in:

- Low engagement
- Medium engagement
- High engagement

---

## 4️⃣ Explicit Churn Definition

A user is classified as churned if:

1. Subscription is explicitly cancelled
   **OR**
2. Subscription is not renewed within 14 days after expiry

### Prediction Target:

Will the user churn in the next 14 days?

Binary label:

- 1 = Will churn
- 0 = Will not churn

---

## 5️⃣ Time Windows

### Observation Window (Feature Window)

Last **4 weeks** of behavioral and payment data.

### Prediction Window

Next **14 days**.

---

## 6️⃣ Stakeholder Questions Addressed

- What behavioral signals precede churn?
- Are payment failures leading indicators?
- Which plans or price bands churn most?
- Which segments drive highest revenue-at-risk?
- Can churn be predicted 7–14 days early?
- Who should be targeted first?
- What is the 30-day revenue impact?

---

## 7️⃣ Analytical Strategy

1. Curated analytics tables
2. KPI & cohort diagnosis
3. Supervised ML churn model
4. Risk-based targeting
5. Retention experiment design
6. Financial impact simulation
7. Executive dashboard

---

## 8️⃣ Assumptions & Limitations

- Revenue approximated using average plan price
- Campaign uplift based on industry benchmarks
- Data quality impacts prediction performance

---

# 🏗 DATA ARCHITECTURE

Raw data is transformed into:

---

## 1️⃣ `dim_users_enriched.csv`

(1 row per user)

Includes:

- user_id
- signup_date
- tenure_days
- city_tier
- segment
- lifetime_paid_months
- last_active_date
- engagement_band

---

## 2️⃣ `fact_user_weekly.csv`

(1 row per user per week)

Includes:

- user_id
- week_start
- active_days_week
- total_minutes_week
- sessions_week
- payment_attempts_week
- payment_failures_week

---

## 3️⃣ `model_churn_dataset.csv`

(ML-ready)

Includes:

- Feature set (last 4 weeks)
- Behavioral trends
- Payment signals
- Engagement band
- will_churn_14d (label)

---

# ⚙️ HOW TO RUN

1️⃣ Place raw files in:

```
/raw_data/
```

2️⃣ Run ETL:

```bash
python etl_pipeline.py
```

3️⃣ Outputs generated in:

```
/data/
```

---

# 🤖 MODELING APPROACH

Models used:

- Logistic Regression (baseline)
- Random Forest (final model)

Evaluation metrics:

- ROC-AUC
- Precision / Recall
- Confusion Matrix
- Threshold optimization

Random Forest selected due to stronger predictive power.

---

# 🎯 RETENTION STRATEGY

Risk segmentation:

- High Risk
- Medium Risk
- Low Risk

Primary focus:

High-risk + high revenue users

Interventions:

- Renewal reminders
- Payment retry automation
- Limited-time discounts
- Engagement nudges

Experiment design:

- Control vs Treatment
- Measure 14-day renewal uplift
- Monitor margin impact

---

# 💰 30-DAY IMPACT ESTIMATION

Scenarios modeled:

- 2% uplift (worst case)
- 5% uplift (base case)
- 8% uplift (best case)

Result: Positive net revenue impact under realistic assumptions.

---

# 📊 DASHBOARD

Tool: Tableau

Includes:

1. Executive Summary
2. Cohort & Segment View
3. Driver Diagnostics
4. Model & Targeting

---

# 📂 FOLDER STRUCTURE

```
PrakrutiHakke_Capstone_SubscriptionChurn/

│── README.md
│── Part_A_Problem_Framing
│
├── analysis/
├── dashboard/
├── data/
├── etl/
└── final_story/

```

---

# 🚀 KEY OUTCOMES

- Built production-ready churn definition
- Engineered leakage-safe labels
- Identified churn drivers
- Developed predictive model
- Designed retention framework
- Estimated financial impact
- Delivered executive-ready dashboard & memo

---

# 🏁 FINAL NOTE

This project demonstrates the ability to:

- Translate business problems into analytical systems
- Build structured ETL pipelines
- Engineer time-aware churn labels
- Apply supervised ML responsibly
- Convert predictions into business strategy
- Quantify financial ROI

---

# ✅ END OF README

---
