"""
churn_analysis.py
-------------------
Customer Retention & Churn Analysis for a subscription-based business.

Covers:
 1. Data loading & cleaning
 2. Churn rate overview
 3. Churn by plan, contract type, payment method
 4. Cohort analysis (tenure-based retention curve)
 5. Customer Lifetime Value (CLV) estimate
 6. Key retention drivers (correlation + satisfaction/support call analysis)
 7. Dashboard of charts saved as PNG files
 8. Written insights + recommendations printed to console and saved to a report file

Run this AFTER generate_dataset.py has created customer_data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ----------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ----------------------------------------------------------------
df = pd.read_csv("customer_data.csv")

print("=" * 60)
print("STEP 1: DATA OVERVIEW")
print("=" * 60)
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Clean: drop duplicates, fix types
df = df.drop_duplicates(subset="CustomerID")
df["Churn_Flag"] = df["Churn"].map({"Yes": 1, "No": 0})

# ----------------------------------------------------------------
# 2. OVERALL CHURN RATE
# ----------------------------------------------------------------
overall_churn_rate = df["Churn_Flag"].mean() * 100
print("\n" + "=" * 60)
print("STEP 2: OVERALL CHURN RATE")
print("=" * 60)
print(f"Overall churn rate: {overall_churn_rate:.2f}%")

# ----------------------------------------------------------------
# 3. CHURN BY SEGMENT
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: CHURN RATE BY SEGMENT")
print("=" * 60)

churn_by_plan = df.groupby("Plan")["Churn_Flag"].mean().sort_values(ascending=False) * 100
churn_by_contract = df.groupby("Contract")["Churn_Flag"].mean().sort_values(ascending=False) * 100
churn_by_payment = df.groupby("PaymentMethod")["Churn_Flag"].mean().sort_values(ascending=False) * 100

print("\nChurn % by Plan:\n", churn_by_plan.round(2))
print("\nChurn % by Contract:\n", churn_by_contract.round(2))
print("\nChurn % by Payment Method:\n", churn_by_payment.round(2))

# ----------------------------------------------------------------
# 4. COHORT / TENURE-BASED RETENTION ANALYSIS
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: COHORT (TENURE) RETENTION ANALYSIS")
print("=" * 60)

# bucket tenure into cohorts (0-6, 6-12, 12-24, 24-36, 36+ months)
bins = [0, 6, 12, 24, 36, 60]
labels = ["0-6 mo", "6-12 mo", "12-24 mo", "24-36 mo", "36+ mo"]
df["TenureCohort"] = pd.cut(df["TenureMonths"], bins=bins, labels=labels)

cohort_retention = df.groupby("TenureCohort")["Churn_Flag"].apply(
    lambda x: 100 - x.mean() * 100
)
print("\nRetention % by Tenure Cohort:\n", cohort_retention.round(2))

# ----------------------------------------------------------------
# 5. CUSTOMER LIFETIME VALUE (CLV) ESTIMATE
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: CUSTOMER LIFETIME VALUE (CLV)")
print("=" * 60)

# Simple CLV = Avg Monthly Revenue * Avg Tenure (months) / (1 - churn rate as proxy for longevity)
avg_monthly_revenue = df["MonthlyCharges"].mean()
avg_tenure = df["TenureMonths"].mean()
df["CLV_Estimate"] = df["MonthlyCharges"] * df["TenureMonths"]

clv_by_plan = df.groupby("Plan")["CLV_Estimate"].mean().sort_values(ascending=False)
print(f"\nAverage Monthly Revenue per customer: ${avg_monthly_revenue:.2f}")
print(f"Average Tenure: {avg_tenure:.1f} months")
print("\nAverage CLV by Plan:\n", clv_by_plan.round(2))

# ----------------------------------------------------------------
# 6. KEY RETENTION DRIVERS
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: KEY RETENTION DRIVERS")
print("=" * 60)

churn_by_satisfaction = df.groupby("SatisfactionScore")["Churn_Flag"].mean() * 100
churn_by_support_calls = df.groupby("SupportCalls")["Churn_Flag"].mean() * 100

print("\nChurn % by Satisfaction Score:\n", churn_by_satisfaction.round(2))
print("\nChurn % by Number of Support Calls:\n", churn_by_support_calls.round(2))

numeric_cols = ["Age", "TenureMonths", "MonthlyCharges", "TotalCharges",
                "SupportCalls", "SatisfactionScore", "Churn_Flag"]
corr = df[numeric_cols].corr()["Churn_Flag"].sort_values(ascending=False)
print("\nCorrelation of numeric features with Churn:\n", corr.round(3))

# ----------------------------------------------------------------
# 7. DASHBOARD: SAVE CHARTS
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: GENERATING DASHBOARD CHARTS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Customer Retention & Churn Analysis Dashboard", fontsize=18, fontweight="bold")

# Chart 1: Overall churn pie
axes[0, 0].pie(
    df["Churn"].value_counts(),
    labels=df["Churn"].value_counts().index,
    autopct="%1.1f%%",
    colors=["#4CAF50", "#E53935"],
    startangle=90,
)
axes[0, 0].set_title("Overall Churn vs Retained")

# Chart 2: Churn by plan
sns.barplot(x=churn_by_plan.index, y=churn_by_plan.values, hue=churn_by_plan.index, ax=axes[0, 1], palette="Oranges_r", legend=False)
axes[0, 1].set_title("Churn Rate (%) by Plan")
axes[0, 1].set_ylabel("Churn %")

# Chart 3: Churn by contract
sns.barplot(x=churn_by_contract.index, y=churn_by_contract.values, hue=churn_by_contract.index, ax=axes[0, 2], palette="Blues_r", legend=False)
axes[0, 2].set_title("Churn Rate (%) by Contract Type")
axes[0, 2].set_ylabel("Churn %")
axes[0, 2].tick_params(axis="x", rotation=15)

# Chart 4: Retention by tenure cohort
sns.barplot(x=cohort_retention.index, y=cohort_retention.values, hue=cohort_retention.index, ax=axes[1, 0], palette="Greens_r", legend=False)
axes[1, 0].set_title("Retention Rate (%) by Tenure Cohort")
axes[1, 0].set_ylabel("Retention %")

# Chart 5: Churn by satisfaction score
sns.lineplot(x=churn_by_satisfaction.index, y=churn_by_satisfaction.values, marker="o", ax=axes[1, 1], color="crimson")
axes[1, 1].set_title("Churn Rate (%) by Satisfaction Score")
axes[1, 1].set_xlabel("Satisfaction Score (1-5)")
axes[1, 1].set_ylabel("Churn %")

# Chart 6: Churn by support calls
sns.lineplot(x=churn_by_support_calls.index, y=churn_by_support_calls.values, marker="o", ax=axes[1, 2], color="purple")
axes[1, 2].set_title("Churn Rate (%) by Support Calls")
axes[1, 2].set_xlabel("Number of Support Calls")
axes[1, 2].set_ylabel("Churn %")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("churn_dashboard.png", dpi=150)
print("Dashboard saved as churn_dashboard.png")
plt.close()

# ----------------------------------------------------------------
# 8. WRITTEN INSIGHTS & RECOMMENDATIONS
# ----------------------------------------------------------------
top_churn_plan = churn_by_plan.index[0]
top_churn_contract = churn_by_contract.index[0]
worst_cohort = cohort_retention.idxmin()

insights = f"""
CUSTOMER RETENTION & CHURN ANALYSIS - SUMMARY REPORT
=====================================================

OVERVIEW
--------
Total Customers Analyzed : {len(df)}
Overall Churn Rate        : {overall_churn_rate:.2f}%
Average Monthly Revenue   : ${avg_monthly_revenue:.2f}
Average Customer Tenure   : {avg_tenure:.1f} months

KEY CHURN PATTERNS
------------------
1. Plan with highest churn        : {top_churn_plan} ({churn_by_plan.iloc[0]:.1f}%)
2. Contract type with highest churn: {top_churn_contract} ({churn_by_contract.iloc[0]:.1f}%)
3. Weakest tenure cohort (lowest retention): {worst_cohort} ({cohort_retention.min():.1f}% retained)
4. Customers with satisfaction score <= 2 churn at {churn_by_satisfaction.loc[1:2].mean():.1f}% on average
5. Customers with 4+ support calls churn at {churn_by_support_calls.loc[churn_by_support_calls.index >= 4].mean():.1f}% on average

CUSTOMER LIFETIME VALUE
------------------------
{clv_by_plan.round(2).to_string()}

ACTIONABLE RECOMMENDATIONS
---------------------------
1. Target Month-to-Month customers with incentives to switch to Annual/Two-Year
   contracts (e.g., discounts for longer commitments) since they show the highest churn.
2. Set up proactive outreach for customers with satisfaction scores of 1-2 and/or
   4+ support tickets in the last month — this segment is at highest churn risk.
3. Strengthen onboarding for the 0-6 month tenure cohort, since early-tenure customers
   show the lowest retention. Consider a structured 90-day onboarding journey.
4. Reassess pricing or value proposition for the "{top_churn_plan}" plan, which has the
   highest churn rate among all plans.
5. Use CLV by plan to prioritize retention spend — protect high-CLV segments first.
"""

print(insights)

with open("churn_report.txt", "w") as f:
    f.write(insights)

print("\nFull written report saved as churn_report.txt")
print("\nAnalysis complete. Deliverables: churn_dashboard.png, churn_report.txt")
