# Part A — Problem Framing Document

## Author

Name: Prakruti Hakke

## Business Objective

Reduce monthly subscriber churn by identifying at-risk users 7–14 days before their
subscription renewal, enabling targeted retention actions that preserve revenue.

---

## North Star Metric

**Monthly Retention Rate**
= Subscribers at end of month / Subscribers at start of month × 100
**Target: ≥ 85%** | Current: ~73%

---

## Supporting KPIs (8)

| #   | KPI                      | Formula                                      | Current | Target   |
| --- | ------------------------ | -------------------------------------------- | ------- | -------- |
| 1   | Monthly Churn Rate       | Churned / Active at month start              | ~27%    | < 15%    |
| 2   | Weekly Churn Rate        | Churned in week / Active at week start       | ~7%     | < 4%     |
| 3   | Cohort Retention (M3)    | % of Month-0 cohort still active at Month 3  | ~55%    | > 70%    |
| 4   | Renewal Success Rate     | Successful renewals / Total renewal attempts | ~72%    | > 85%    |
| 5   | Payment Failure Rate     | Failed payments / All payment attempts       | ~9%     | < 5%     |
| 6   | Avg Active Days/Week     | Active days / (users × weeks)                | ~3.2    | > 4.0    |
| 7   | Revenue Retained (proxy) | Active users × avg plan price                | Rs 8.5L | Maximize |
| 8   | Campaign Response Rate   | Clicked or redeemed / Delivered              | ~8%     | > 12%    |

---

## Churn Definition (Explicit)

> **A user is CHURNED if their subscription `status` = 'cancelled' OR 'expired'.**
>
> For machine learning prediction:
> `will_churn_14d = 1` if the user's latest subscription is cancelled or expired
> at the time of prediction (as_of_date = 2026-01-16).

---

## Scope & Windows

| Dimension          | Value                                                    |
| ------------------ | -------------------------------------------------------- |
| Data period        | 2025-07-04 to 2026-01-30                                 |
| Observation window | Last **4 weeks** of usage/payment data before as_of_date |
| Prediction window  | Will user churn in the **next 14 days**?                 |
| as_of_date         | **2026-01-16** (14 days before data end)                 |

---

## Stakeholder Questions Answered by Data

1. **What % of users churn each month, and is it trending up or down?**
   → Monthly churn rate & retention trend chart (Part C-1)

2. **Which plan type has the highest churn? Are expensive plans stickier?**
   → Churn rate by plan_id (Part C-3)

3. **Do users who fail a payment churn at a higher rate?**
   → Cross-tab: payment failure vs churn rate (Part C-4)

4. **How does usage behaviour differ in the 4 weeks before churn vs retention?**
   → usage_trend feature + churn investigation table (Part C-4)

5. **Which city tier or user segment has the most revenue at risk?**
   → Revenue-at-risk segment table (Part C-3)

6. **Can we predict who will churn 14 days ahead with actionable accuracy?**
   → ML model ROC-AUC and precision/recall at chosen threshold (Part D)

7. **What is the expected revenue saved if we run a retention campaign next month?**
   → 30-day scenario model: worst/base/best case (Part F)
