import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Seeding randomness so the sample data is reproducible. Anyone running this gets
# the same file, which keeps the demo and screenshots consistent.
np.random.seed(42)

NUM_PLAYERS = 200
TODAY = datetime(2026, 6, 1)  # fixed "today" so recency math is stable across runs

# ---------- Core identifiers ----------
player_ids = [f"P{100000 + i}" for i in range(NUM_PLAYERS)]
universal_ids = [f"U{np.random.randint(10000000, 99999999)}" for _ in range(NUM_PLAYERS)]

first_names = np.random.choice(
    ["James","Maria","Robert","Linda","Michael","Patricia","David","Jennifer",
     "John","Elizabeth","William","Susan","Richard","Jessica","Joseph","Karen",
     "Thomas","Nancy","Carlos","Sofia","Wei","Mei","Raj","Priya","Ahmed","Fatima"],
    size=NUM_PLAYERS)
last_names = np.random.choice(
    ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
     "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
     "Thomas","Lee","Patel","Nguyen","Kim","Chen","Singh","Khan","Ali"],
    size=NUM_PLAYERS)
# A combined full name column, handy for display and for searching by whole name.
full_names = [f"{f} {l}" for f, l in zip(first_names, last_names)]

# ---------- Core value metric ----------
net_adt = np.round(np.random.exponential(scale=45, size=NUM_PLAYERS), 2)
value_factor = (net_adt - net_adt.min()) / (net_adt.max() - net_adt.min())

# ---------- Dates and recency ----------
signup_dates_dt = [TODAY - timedelta(days=int(np.random.randint(180, 2200))) for _ in range(NUM_PLAYERS)]
last_visit_dates_dt = [TODAY - timedelta(days=int(np.random.randint(1, 540))) for _ in range(NUM_PLAYERS)]
signup_dates = [d.strftime("%Y-%m-%d") for d in signup_dates_dt]
last_visit_dates = [d.strftime("%Y-%m-%d") for d in last_visit_dates_dt]
days_since_last_visit = [(TODAY - d).days for d in last_visit_dates_dt]

# ---------- Visit frequency ----------
total_visits_month = np.random.poisson(lam=3, size=NUM_PLAYERS)
total_visits_year = total_visits_month * np.random.randint(8, 14, size=NUM_PLAYERS)

# ---------- Spend and value ----------
avg_wager = np.round(np.random.exponential(scale=80, size=NUM_PLAYERS), 2)
lifetime_value = np.round(net_adt * total_visits_year * np.random.uniform(0.8, 1.4, size=NUM_PLAYERS), 2)
account_balance = np.round(np.random.exponential(scale=500, size=NUM_PLAYERS), 2)

# ---------- Cash flow behavior ----------
deposits_month = np.round(np.random.exponential(scale=300, size=NUM_PLAYERS), 2)
withdrawals_month = np.round(deposits_month * np.random.uniform(0.3, 1.1, size=NUM_PLAYERS), 2)

# ---------- Offers (comps) ----------
def tiered_offer(step, cap):
    ladder = np.arange(0, cap + step, step)
    offers = []
    for vf in value_factor:
        pos = np.clip(vf + np.random.uniform(-0.15, 0.15), 0, 1)
        idx = int(round(pos * (len(ladder) - 1)))
        offers.append(ladder[idx])
    return np.array(offers)

offer_slots = tiered_offer(20, 1000)
offer_tables = tiered_offer(30, 1500)
offer_food = tiered_offer(45, 1300)
offer_hotel = tiered_offer(200, 2000)
offer_total = offer_slots + offer_tables + offer_food + offer_hotel

# ---------- Coin-in ----------
def scaled_coin_in(base):
    return np.round(base * (0.2 + value_factor) * np.random.uniform(0.5, 1.5, size=NUM_PLAYERS), 2)

coin_in_slots = scaled_coin_in(4000)
coin_in_tables = scaled_coin_in(3000)
coin_in_food = scaled_coin_in(600)
coin_in_hotel = scaled_coin_in(1200)
coin_in_total = np.round(coin_in_slots + coin_in_tables + coin_in_food + coin_in_hotel, 2)

# ---------- Risk / credit ----------
risk_score = np.random.randint(300, 850, size=NUM_PLAYERS)
credit_tiers = np.where(risk_score >= 720, "Low Risk",
                np.where(risk_score >= 580, "Medium Risk", "High Risk"))
household_income = np.random.choice(
    ["<50k", "50k-100k", "100k-150k", "150k-250k", "250k+"],
    size=NUM_PLAYERS, p=[0.35, 0.30, 0.20, 0.10, 0.05])

# ---------- Consent flags ----------
can_email = np.random.choice(["Yes", "No"], size=NUM_PLAYERS, p=[0.7, 0.3])
can_call = np.random.choice(["Yes", "No"], size=NUM_PLAYERS, p=[0.5, 0.5])

# ---------- Geography and loyalty ----------
zip_codes = [f"{np.random.randint(10000, 99999)}" for _ in range(NUM_PLAYERS)]
tier_ranks = np.random.choice(["Bronze", "Silver", "Gold", "Platinum"],
                              size=NUM_PLAYERS, p=[0.5, 0.3, 0.15, 0.05])
preferred_games = np.random.choice(
    ["Slots", "Blackjack", "Poker", "Roulette", "Baccarat"], size=NUM_PLAYERS)

df = pd.DataFrame({
    "PlayerID": player_ids,
    "UniversalID": universal_ids,
    "FirstName": first_names,
    "LastName": last_names,
    "FullName": full_names,
    "NetADT": net_adt,
    "SignUpDate": signup_dates,
    "LastVisitDate": last_visit_dates,
    "DaysSinceLastVisit": days_since_last_visit,
    "TotalVisitsMonth": total_visits_month,
    "TotalVisitsYear": total_visits_year,
    "AvgWager": avg_wager,
    "LifetimeValue": lifetime_value,
    "AccountBalance": account_balance,
    "DepositsMonth": deposits_month,
    "WithdrawalsMonth": withdrawals_month,
    "OfferSlots": offer_slots,
    "OfferTables": offer_tables,
    "OfferFood": offer_food,
    "OfferHotel": offer_hotel,
    "OfferTotal": offer_total,
    "CoinInSlots": coin_in_slots,
    "CoinInTables": coin_in_tables,
    "CoinInFood": coin_in_food,
    "CoinInHotel": coin_in_hotel,
    "CoinInTotal": coin_in_total,
    "RiskScore": risk_score,
    "CreditTier": credit_tiers,
    "HouseholdIncome": household_income,
    "CanEmail": can_email,
    "CanCall": can_call,
    "ZipCode": zip_codes,
    "TierRank": tier_ranks,
    "PreferredGame": preferred_games,
})

# Intentionally dirtying a slice of the data so the cleanup agent has real work.
df.loc[df.sample(frac=0.05).index, "NetADT"] = np.nan
df.loc[df.sample(frac=0.04).index, "CanEmail"] = ""
df.loc[df.sample(frac=0.03).index, "ZipCode"] = np.nan
df.loc[df.sample(frac=0.06).index, "CanCall"] = df["CanCall"].str.lower()

df.to_csv("sample_players.csv", index=False)

# Also write a blank template: same headers, no rows, so anyone can fill in their
# own players and upload it back into the tool.
df.head(0).to_csv("player_template.csv", index=False)

print(f"Created sample_players.csv with {len(df)} players and {len(df.columns)} columns.")
print("Created player_template.csv (headers only, for users to fill in).")