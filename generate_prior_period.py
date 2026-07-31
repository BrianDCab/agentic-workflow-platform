import pandas as pd
import numpy as np

# Builds a "last period" snapshot from the current player sample so the trend
# feature has demo data. Seeded so every run produces the same file.
np.random.seed(11)

df = pd.read_csv("sample_players.csv")
prev = df.copy()

vals = pd.to_numeric(
    prev["NetADT"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
    errors="coerce")

# Most players looked similar last period. A slice were much higher then (so
# they read as declining now) and a slice were lower (rising now).
factor = np.random.normal(loc=1.0, scale=0.12, size=len(prev))
decliners = np.random.rand(len(prev)) < 0.18
factor[decliners] = np.random.uniform(1.35, 2.2, size=decliners.sum())
risers = np.random.rand(len(prev)) < 0.12
factor[risers] = np.random.uniform(0.45, 0.7, size=risers.sum())

prev["NetADT"] = (vals * factor).round(2)
prev.to_csv("sample_players_prev.csv", index=False)
print(f"Created sample_players_prev.csv with {len(prev)} players.")