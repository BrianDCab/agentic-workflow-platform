import pandas as pd
import numpy as np

# Mode B sample data: business accounts rather than individual players. This is a
# separate file because companies have a different shape than people, debt and
# revenue and equity instead of visits and wagers. Keeping it separate keeps both
# modes honest and easy to reason about.
np.random.seed(7)

NUM = 150

names = [f"Account {i:03d}" for i in range(1, NUM + 1)]
industries = np.random.choice(
    ["Manufacturing", "Retail", "Healthcare", "Tech", "Energy", "Logistics"],
    size=NUM,
)

# Annual revenue, skewed so a few large accounts dominate, like real portfolios.
revenue = np.round(np.random.exponential(scale=4_000_000, size=NUM) + 200_000, 2)

# Debt and equity sized loosely off revenue so the balance sheet stays plausible.
debt = np.round(revenue * np.random.uniform(0.1, 0.9, size=NUM), 2)
equity = np.round(revenue * np.random.uniform(0.2, 1.2, size=NUM), 2)
liabilities = np.round(debt * np.random.uniform(1.0, 1.6, size=NUM), 2)

# Common credit and risk fields.
credit_score = np.random.randint(40, 100, size=NUM)  # a 0 to 100 internal score
risk_tier = np.where(credit_score >= 75, "Low Risk",
             np.where(credit_score >= 55, "Medium Risk", "High Risk"))

# Relationship signals.
years_as_client = np.random.randint(0, 20, size=NUM)
months_since_contact = np.random.randint(0, 24, size=NUM)
products_held = np.random.randint(1, 8, size=NUM)

df = pd.DataFrame({
    "AccountID": [f"A{1000 + i}" for i in range(NUM)],
    "AccountName": names,
    "Industry": industries,
    "AnnualRevenue": revenue,
    "Debt": debt,
    "Liabilities": liabilities,
    "Equity": equity,
    "CreditScore": credit_score,
    "RiskTier": risk_tier,
    "YearsAsClient": years_as_client,
    "MonthsSinceContact": months_since_contact,
    "ProductsHeld": products_held,
})

# Add a little realistic mess for the cleaner to handle.
df.loc[df.sample(frac=0.05).index, "AnnualRevenue"] = np.nan
df.loc[df.sample(frac=0.04).index, "Industry"] = ""

df.to_csv("sample_companies.csv", index=False)
print(f"Created sample_companies.csv with {len(df)} accounts and {len(df.columns)} columns.")