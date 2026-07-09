import pandas as pd
import numpy as np

# Seeding so the sample is reproducible. Anyone running this gets the same file.
np.random.seed(7)

NUM = 150

account_ids = [f"A{200000 + i}" for i in range(NUM)]

# Company names built from parts so they read like real businesses, and so the
# company mode has a FullName column to match the player mode.
prefixes = ["Summit", "Apex", "Northwind", "Blue Harbor", "Ironclad", "Evergreen", "Crestline",
            "Pioneer", "Granite", "Silverpeak", "Cobalt", "Meridian", "Vanguard", "Lakeshore",
            "Redwood", "Stonebridge", "Atlas", "Beacon", "Cardinal", "Frontier"]
suffixes = ["Holdings", "Industries", "Logistics", "Capital", "Manufacturing", "Group", "Partners",
            "Systems", "Trading", "Enterprises", "Solutions", "Foods", "Materials", "Freight"]
company_names = [f"{np.random.choice(prefixes)} {np.random.choice(suffixes)}" for _ in range(NUM)]

industries = np.random.choice(
    ["Manufacturing", "Retail", "Logistics", "Technology", "Healthcare", "Construction",
     "Hospitality", "Agriculture", "Energy", "Finance"], size=NUM)

annual_revenue = np.round(np.random.exponential(scale=2_000_000, size=NUM) + 200_000, 2)

equity = np.round(annual_revenue * np.random.uniform(0.2, 0.8, size=NUM), 2)
debt = np.round(equity * np.random.uniform(0.3, 2.5, size=NUM), 2)
liabilities = np.round(debt * np.random.uniform(1.0, 1.6, size=NUM), 2)

credit_score = np.random.randint(500, 820, size=NUM)
risk_tier = np.where(credit_score >= 720, "Low Risk",
             np.where(credit_score >= 600, "Medium Risk", "High Risk"))

years_as_client = np.random.randint(1, 25, size=NUM)
months_since_contact = np.random.randint(0, 26, size=NUM)
products_held = np.random.randint(1, 9, size=NUM)

df = pd.DataFrame({
    "AccountID": account_ids,
    "FullName": company_names,
    "Industry": industries,
    "AnnualRevenue": annual_revenue,
    "Debt": debt,
    "Liabilities": liabilities,
    "Equity": equity,
    "CreditScore": credit_score,
    "RiskTier": risk_tier,
    "YearsAsClient": years_as_client,
    "MonthsSinceContact": months_since_contact,
    "ProductsHeld": products_held,
})

# Intentionally dirty a slice so the cleanup step has real work.
df.loc[df.sample(frac=0.04).index, "AnnualRevenue"] = np.nan
df.loc[df.sample(frac=0.03).index, "Industry"] = ""

df.to_csv("sample_companies.csv", index=False)
# Blank template: same headers, no rows, so users can fill in their own accounts.
df.head(0).to_csv("company_template.csv", index=False)

print(f"Created sample_companies.csv with {len(df)} accounts and {len(df.columns)} columns.")
print("Created company_template.csv (headers only, for users to fill in).")