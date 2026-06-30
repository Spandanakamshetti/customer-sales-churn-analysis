"""
generate_dataset.py
--------------------
Creates a realistic synthetic dataset for a subscription-based business
and saves it as customer_data.csv.

Run this FIRST, before running churn_analysis.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000  # number of customers

customer_id = [f"CUST{i:05d}" for i in range(1, N + 1)]

genders = np.random.choice(["Male", "Female"], size=N)
age = np.random.randint(18, 70, size=N)

plans = np.random.choice(
    ["Basic", "Standard", "Premium"], size=N, p=[0.45, 0.35, 0.20]
)

plan_price = {"Basic": 9.99, "Standard": 19.99, "Premium": 39.99}
monthly_charges = np.array([plan_price[p] for p in plans]) + np.random.normal(0, 1.5, N).round(2)
monthly_charges = monthly_charges.clip(5, None).round(2)

contract = np.random.choice(
    ["Month-to-Month", "Annual", "Two-Year"], size=N, p=[0.55, 0.30, 0.15]
)

tenure_months = np.random.randint(1, 60, size=N)

payment_method = np.random.choice(
    ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"], size=N
)

support_calls = np.random.poisson(1.5, size=N)

# Simulate satisfaction score 1-5 (lower satisfaction -> more likely to churn)
satisfaction_score = np.random.randint(1, 6, size=N)

# Total charges = monthly_charges * tenure roughly, with noise
total_charges = (monthly_charges * tenure_months * np.random.uniform(0.9, 1.1, N)).round(2)

# --- Build churn probability based on realistic business logic ---
churn_prob = 0.10  # base churn rate
churn_prob = churn_prob + (contract == "Month-to-Month") * 0.25
churn_prob = churn_prob + (satisfaction_score <= 2) * 0.30
churn_prob = churn_prob + (support_calls >= 4) * 0.20
churn_prob = churn_prob + (tenure_months < 6) * 0.15
churn_prob = churn_prob - (contract == "Two-Year") * 0.15
churn_prob = churn_prob - (satisfaction_score >= 4) * 0.15
churn_prob = np.clip(churn_prob, 0.02, 0.95)

churn = np.random.binomial(1, churn_prob)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame({
    "CustomerID": customer_id,
    "Gender": genders,
    "Age": age,
    "Plan": plans,
    "Contract": contract,
    "TenureMonths": tenure_months,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "PaymentMethod": payment_method,
    "SupportCalls": support_calls,
    "SatisfactionScore": satisfaction_score,
    "Churn": churn_label,
})

df.to_csv("customer_data.csv", index=False)

print(f"Dataset created: customer_data.csv  ({len(df)} rows)")
print(df["Churn"].value_counts(normalize=True).round(3))
