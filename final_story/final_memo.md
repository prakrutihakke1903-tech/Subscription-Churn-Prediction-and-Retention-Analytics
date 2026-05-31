# Final Decision Memo

## Author

Name: Prakruti Hakke

## Subscription Churn Early-Warning & Retention Strategy

---

## 1. Goal, KPIs & Churn Definition

**Business Objective:** Reduce monthly subscriber churn by identifying at-risk users
14 days before their renewal date and deploying targeted retention interventions.

**North Star Metric:** Monthly Retention Rate (target ≥ 85%)

**Churn Definition:** A user is churned if their subscription `status` is `cancelled`
or `expired`. For ML prediction: `will_churn_14d = 1` if the user's latest subscription
ends within the 14-day prediction window.

**Supporting KPIs:**

| KPI                       | Current Value | Target   |
| ------------------------- | ------------- | -------- |
| Monthly Churn Rate        | ~27%          | < 15%    |
| Monthly Retention Rate    | ~73%          | > 85%    |
| Payment Failure Rate      | ~8-10%        | < 5%     |
| Avg Active Days/Week      | ~3.2 days     | > 4 days |
| Renewal Success Rate      | ~72%          | > 85%    |
| Campaign Response Rate    | ~8%           | > 12%    |
| Revenue at Risk (monthly) | Rs 2.3 Lakh   | Minimize |

---

## 2. Data & Methodology Summary

**Data:** 2,500 users, 2,510 subscriptions, 10,200 payments, 344,932 daily usage
records, 5,248 campaign touches — covering July 2025 to January 2026.

**ETL Cleaning Actions:**

- Removed 10 duplicate subscriptions, 30 duplicate payments, 688 duplicate usage rows
- Standardised payment_status casing (SUCCESS/FAILED consistently)
- Flagged 3,444 minute outliers in usage (≥ 96 min/day)
- Filled 0% missing preferred_device with 'unknown'

**Three curated outputs built:**

1. `dim_users_enriched.csv` — 1 row/user with tenure, engagement band, plan info
2. `fact_user_weekly.csv` — 51,546 rows of weekly usage + payment signals
3. `model_churn_dataset.csv` — 2,500 rows, 19 features + `will_churn_14d` label

**Feature Engineering Highlights:**

- `usage_trend` = last week's active days minus prior 3-week average
- `days_since_last_activity` = recency signal
- `pay_failure_rate` = failures / attempts in last 4 weeks
- `engagement_band` = low/mid/high based on avg active days/week

**ML Approach:** Time-based split (long-tenure = train, short-tenure = test) to
simulate predicting future churners from historical patterns.

---

## 3. Key Insights (7)

1. **27.2% baseline churn rate** — nearly 1 in 3 active users churns each billing cycle,
   well above the industry benchmark of 5-7% for subscription products.

2. **Low engagement is the #1 churn driver.** Users in the "low" engagement band
   (< 2 active days/week) churn at over 40%, contributing ~60% of all churns.
   They average only 1.3 active days/week vs 5.8 for high-engagement users.

3. **Cohort retention drops sharply after Month 1.** Month 0 cohorts show 100%
   retention, but by Month 2 many cohorts have lost 40-50% of users — suggesting
   the critical retention window is the first 30 days.

4. **Payment failures are highly predictive of churn.** Users with at least one
   payment failure in the past 4 weeks churn at nearly 2x the rate of users with
   clean payment histories. This represents significant "involuntary churn" that
   can be fixed with a simple retry nudge.

5. **Basic plan users churn most; Premium plan users have highest revenue at risk.**
   Basic plan (Rs 299) has the highest churn rate, but Premium plan (Rs 799) carries
   the most revenue risk due to higher price per churned user.

6. **City Tier 3 users churn more than Tier 1.** This may reflect lower content
   relevance, connectivity issues, or lower willingness to pay for digital subscriptions.

7. **Campaign response rate is low (~8%).** Current retention touches are not
   converting well — suggesting timing, message personalisation, or channel choice
   needs improvement.

---

## 4. Model Performance & Targeting

**Random Forest outperformed Logistic Regression:**

| Model               | ROC-AUC | Precision (Churn) | Recall (Churn) |
| ------------------- | ------- | ----------------- | -------------- |
| Logistic Regression | ~0.72   | ~0.65             | ~0.68          |
| Random Forest       | ~0.82   | ~0.72             | ~0.74          |

**Threshold chosen: 0.35** (lower than 0.50 default) — because the cost of missing
a churner (lost Rs 299-799 revenue) far outweighs the cost of incorrectly targeting
a non-churner (Rs 50-100 discount offered unnecessarily).

**Top early warning signals (feature importance):**

1. `days_since_last_activity` — silence = strongest churn signal
2. `usage_trend` — declining week-over-week engagement
3. `pay_failures` — any payment failure in last 4 weeks
4. `total_active_days` — low overall engagement in observation window
5. `days_last_week` — near-zero activity in the week before renewal

**Risk segmentation result:**

- High risk (prob ≥ 0.60): **436 users** — immediate campaign targets
- Medium risk (0.35-0.60): **~680 users** — secondary wave
- Low risk (< 0.35): remaining users — no action needed

---

## 5. Recommended Retention Plan

**Three-pronged intervention, triggered 7-10 days before renewal:**

| #   | Intervention                      | Target                      | Expected Conversion | Cost/User |
| --- | --------------------------------- | --------------------------- | ------------------- | --------- |
| 1   | Renewal reminder (push + email)   | All high-risk               | 8-12%               | ~Rs 0     |
| 2   | 20% discount offer (48-hr window) | High-risk Premium/Standard  | 15-20%              | ~Rs 100   |
| 3   | Payment retry nudge (SMS)         | Users with payment failures | 30-40% retry        | ~Rs 0.50  |
| 4   | Personalised content digest       | Low-engagement users        | 10-15% re-engage    | ~Rs 0     |

**A/B Test Design for Intervention 2:**

- Control: Renewal reminder only
- Treatment: Renewal reminder + 20% discount
- Primary metric: 30-day renewal rate
- Duration: 30 days | Sample: ~400 users per arm
- Success threshold: +10pp renewal rate improvement

---

## 6. 30-Day Impact Estimate

| Scenario                   | Users Retained | Revenue Saved | Discount Cost | Net Impact    |
| -------------------------- | -------------- | ------------- | ------------- | ------------- |
| Worst case (6% conversion) | 20             | Rs 9,321      | Rs 932        | **Rs 8,389**  |
| Base case (12% conversion) | 41             | Rs 19,108     | Rs 1,911      | **Rs 17,197** |
| Best case (20% conversion) | 69             | Rs 32,158     | Rs 3,216      | **Rs 28,942** |

**Key assumptions:**

- 27.2% baseline churn rate, 1,821 active subscribers
- Campaign reaches 70% of 494 at-risk users (345 targeted)
- 50% of retained users receive the discount; 50% renew from reminder alone
- Retained users stay at least 1 additional billing cycle

---

## 7. Risks, Limitations & Next Steps

**Risks:**

- Discount cannibalization: users who would have renewed anyway may claim the offer
- Short data window (7 months): seasonal patterns may not be fully captured
- No A/B test yet — impact estimates are modelled projections, not measured results

**Limitations:**

- No support ticket data to identify dissatisfied users (noted in brief)
- Payment method data exists but payment gateway-level failure codes are missing
- City-level granularity limited to tier (1/2/3), not individual cities

**Immediate Next Steps (Week 1-2):**

1. Deploy payment retry nudge immediately — zero cost, high potential ROI
2. Launch A/B test for discount intervention with 800-user sample
3. Integrate model churn scores into CRM for daily automated targeting

**Medium-term (Month 1-3):** 4. Build real-time churn score pipeline (daily score refresh) 5. Improve onboarding flow for new users (target: D14 active days ≥ 3) 6. Investigate Tier 3 city churn — could warrant regional pricing or content

---
