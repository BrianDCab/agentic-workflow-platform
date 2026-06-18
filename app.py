import os
import re
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

# Load my Groq key from .env so it never lives in the code itself.
load_dotenv()

st.set_page_config(page_title="Segmentation Agent", page_icon="🎲", layout="centered")

CYAN = "#22D3EE"; VIOLET = "#A78BFA"; AMBER = "#F5A623"

# ---------- Styling ----------
st.markdown("""
<style>
    .stApp { background-color: #000000; overflow-x: hidden; }
    h1, h2, h3 { color: #FFFFFF; letter-spacing: 0.3px; font-weight: 800; }
    .critical { color: #FFD23F; text-shadow: 0 0 8px rgba(255,210,63,0.7); font-weight: 600; }
    .goal-label { color: #FF5C7A; font-weight: 800; font-size: 1.05rem;
                  text-shadow: 0 0 8px rgba(225,29,72,0.4); margin-bottom: 2px; }
    .term { color: #22D3EE; font-weight: 600; cursor: help;
            border-bottom: 1px dotted rgba(34,211,238,0.7); text-shadow: 0 0 6px rgba(34,211,238,0.4); }
    .rec-do { color: #22D3EE; font-weight: 700; text-shadow: 0 0 6px rgba(34,211,238,0.5); }
    .rec-why { color: #7FE7F5; font-weight: 700; }
    .insight { background: rgba(34,211,238,0.06); border-left: 3px solid rgba(34,211,238,0.6);
               padding: 8px 12px; border-radius: 4px; margin: 6px 0 2px 0; color:#CFEfff; }
    .head-cyan   { color:#22D3EE; font-size:1.6rem; font-weight:800; text-shadow:0 0 10px rgba(34,211,238,0.4); margin:4px 0; }
    .head-violet { color:#A78BFA; font-size:1.6rem; font-weight:800; text-shadow:0 0 10px rgba(167,139,250,0.4); margin:4px 0; }
    .head-amber  { color:#F5A623; font-size:1.6rem; font-weight:800; text-shadow:0 0 10px rgba(245,166,35,0.4); margin:4px 0; }
    .beta-badge { display:inline-block; color:#FFD23F; font-weight:800; font-size:0.7rem;
                  border:1px solid rgba(255,210,63,0.7); border-radius:6px; padding:1px 7px; margin-left:8px;
                  text-shadow:0 0 8px rgba(255,210,63,0.9); box-shadow:0 0 10px rgba(255,210,63,0.25);
                  vertical-align:middle; letter-spacing:1px; }
    .divider { border:none; border-top:1px solid rgba(34,211,238,0.18); margin:30px 0 10px 0; }
    .stButton button, .stDownloadButton button {
        border: 1px solid rgba(34,211,238,0.5); border-radius: 9999px;
        background: rgba(34,211,238,0.10); color: #A5F3FC; font-weight: 700;
        box-shadow: 0 0 18px rgba(34,211,238,0.2); transition: all 0.2s ease;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        background: #22D3EE; color: #000000; transform: translateY(-1px);
    }
    @media (max-width: 640px) { .stButton button, .stDownloadButton button { width: 100%; } }
    div[role="radiogroup"] { gap: 6px; }
    div[role="radiogroup"] > label {
        border: 1px solid rgba(34,211,238,0.3); border-radius: 9999px;
        padding: 6px 16px; background: rgba(34,211,238,0.05); transition: all 0.2s ease;
    }
    div[role="radiogroup"] > label:hover {
        border-color: rgba(34,211,238,0.7); box-shadow: 0 0 12px rgba(34,211,238,0.25);
    }
    [data-testid="stSidebar"] { background-color: #060A0E; border-right: 1px solid rgba(34,211,238,0.15); }
    #chat-scroll { max-height: 40vh; overflow-y: auto; padding-right: 4px; }
    #bc-glow-cyan-l, #bc-glow-cyan-r, #bc-glow-ruby {
        position: fixed; border-radius: 50%; filter: blur(90px); z-index: 0; pointer-events: none;
    }
    #bc-glow-cyan-l { width:300px;height:300px;background:rgba(34,211,238,0.12);bottom:-140px;left:-60px;animation:bcPulse 18s ease-in-out infinite; }
    #bc-glow-cyan-r { width:240px;height:240px;background:rgba(34,211,238,0.10);bottom:-130px;right:-50px;animation:bcPulse 22s ease-in-out infinite; }
    #bc-glow-ruby   { width:180px;height:180px;background:rgba(225,29,72,0.10);bottom:-120px;left:45%;animation:bcPulse 26s ease-in-out infinite; }
    @keyframes bcPulse { 0%,100%{opacity:0.45;transform:scale(1);} 50%{opacity:0.75;transform:scale(1.08);} }
    #bc-footer { text-align:center; color:rgba(230,241,255,0.5); font-size:0.82rem; margin-top:48px; padding:18px 0; border-top:1px solid rgba(34,211,238,0.12); }
    #bc-footer a { color:#22D3EE; text-decoration:none; font-weight:700; }
    #bc-footer a:hover { text-shadow:0 0 8px rgba(34,211,238,0.7); }
    #bc-footer .sep { color:rgba(230,241,255,0.3); margin:0 8px; }
    .stApp > header, .block-container { position: relative; z-index: 2; }
</style>
<div id="bc-glow-cyan-l"></div><div id="bc-glow-cyan-r"></div><div id="bc-glow-ruby"></div>
""", unsafe_allow_html=True)


def divider():
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

def insight(text):
    st.markdown(f"<div class='insight'>{text}</div>", unsafe_allow_html=True)


GLOSSARY = {
    "NetADT": "Net Average Daily Theoretical. The casino's expected profit from a player per day of play, in currency. The core value metric.",
    "Recency": "How long since the record was last active. The strongest signal for spotting someone drifting away.",
    "Pareto": "A ranking of categories by size with a running cumulative percentage. Shows whether a few categories drive most of the total.",
    "Debt to Equity": "A company's debt divided by its equity. Above about 1.5 signals heavy leverage and higher risk.",
    "Segment": "A group of records sharing value and behavior traits, so each group can be acted on differently.",
    "Median": "The middle value when all values are lined up. Used to fill gaps because it is not skewed by extremes.",
    "Quartile": "One of four equal slices of the data by rank. Top quartile means the highest 25% by value.",
}

def term(label, key=None):
    definition = GLOSSARY.get(key or label, "").replace("'", "&#39;")
    return f"<span class='term' title='{definition}'>{label}</span>"

def money(x, sym):
    try:
        return f"{sym}{x:,.2f}"
    except Exception:
        return str(x)


def clean_money_series(s):
    cleaned = (s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True).replace("", pd.NA))
    return pd.to_numeric(cleaned, errors="coerce")


def format_recommendations(raw):
    html = []
    for line in raw.splitlines():
        s = line.strip()
        if not s: continue
        if s.startswith("###"):
            html.append(f"<div style='color:#FFFFFF;font-weight:800;font-size:1.05rem;margin-top:14px'>{s.lstrip('# ').strip()}</div>")
        elif s.lower().startswith("**what the data shows:**"):
            html.append(f"<div style='margin-top:4px'><b>What the data shows:</b> {s.split(':**',1)[1].strip()}</div>")
        elif s.lower().startswith("**do this:**"):
            html.append(f"<div style='margin-top:4px'><span class='rec-do'>Do this:</span> {s.split(':**',1)[1].strip()}</div>")
        elif s.lower().startswith("**why it helps:**"):
            html.append(f"<div style='margin-top:4px'><span class='rec-why'>Why it helps:</span> {s.split(':**',1)[1].strip()}</div>")
        else:
            html.append(f"<div>{s}</div>")
    return "<div>" + "".join(html) + "</div>"


# ============================================================
# Plain-English analysis lines (computed from real numbers, not AI)
# ============================================================
def segments_insight(counts, value_table, sym):
    top_seg = value_table.index[0]
    top_val = value_table.iloc[0]["TotalValue"]; total_val = value_table["TotalValue"].sum()
    top_recs = int(value_table.iloc[0]["Records"]); total_recs = int(value_table["Records"].sum())
    val_share = top_val / total_val * 100 if total_val else 0
    rec_share = top_recs / total_recs * 100 if total_recs else 0
    return (f"Your largest segment by value is {top_seg}, holding {money(top_val,sym)} "
            f"({val_share:.0f}% of all value) from {rec_share:.0f}% of records. "
            "When a small share of records holds most of the value, focus your effort there first.")

def distribution_insight(series, sym):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return "No numeric values to analyze."
    top10 = s.sort_values(ascending=False).head(max(1, int(len(s)*0.1))).sum()
    share = top10 / s.sum() * 100 if s.sum() else 0
    return (f"The median value is {money(s.median(),sym)} and the top 10% of records hold "
            f"about {share:.0f}% of total value. A wide gap between median and top means value "
            "is concentrated in a few records.")

def pareto_insight(pareto_raw, sym):
    top_cat = pareto_raw.iloc[0]["Category"]; top_cum = pareto_raw.iloc[0]["Cumulative %"]
    return (f"{top_cat} alone accounts for {top_cum:.0f}% of total offer dollars. "
            "Categories that drive most of the spend are where small changes move the budget most.")


# ============================================================
# Shared engine
# ============================================================

def get_recommendations(role, data_profile, seg_summary, val_summary, user_goal, sym, cur_name):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""You are {role} reviewing one specific uploaded dataset. All money figures are in {cur_name} ({sym}).

DATA PROFILE (real numbers from this file):
{data_profile}

HEADCOUNT BY SEGMENT:
{seg_summary}

TOTAL VALUE BY SEGMENT (in {sym}):
{val_summary}

THE USER'S GOAL: "{user_goal}"

Write exactly 4 items. Put each label on its own separate line. Use this exact format, with a blank line between items:

### [short title of the action]
**What the data shows:** one sentence with a real number from above. Always put the {sym} symbol directly before every money amount, like {sym}1,234.
**Do this:** one concrete, specific action.
**Why it helps:** one sentence tying it to the user's goal.

Every money figure must show the {sym} symbol. Keep it under 320 words. Do not use dashes as punctuation."""
    response = client.chat.completions.create(model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content


@st.cache_data(show_spinner=False)
def get_record_recommendation(role, name_label, detail_text, user_goal, sym, cur_name):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""You are {role}. All money is in {cur_name} ({sym}). Here is one specific record:

{detail_text}

The user's goal: "{user_goal}"

In 2 to 3 short sentences, recommend the single best next action for {name_label}.
Be specific and reference their actual numbers. Put the {sym} symbol directly before every
money amount, including debt, equity, liabilities, and revenue, like {sym}210,437.66.
Do not use dashes as punctuation."""
    response = client.chat.completions.create(model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content


def chat_answer(role, context, history, sym, cur_name):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    system = (f"You are {role} helping a user understand one specific analyzed dataset. "
              f"All money is in {cur_name} ({sym}); always show the {sym} symbol on amounts. "
              f"Use only the context below to answer. The roster lists individual records you "
              f"can answer about by name or ID. Match names loosely (ignore case and small "
              f"differences). If asked about a record truly not in the roster or something "
              f"unrelated, say so plainly. Keep answers short and concrete. Do not use dashes "
              f"as punctuation.\n\nCONTEXT:\n{context}")
    messages = [{"role": "system", "content": system}] + history
    response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
    return response.choices[0].message.content


@st.cache_data(show_spinner=False)
def describe_columns_ai(columns, sample_row):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = ("For each column, say in 4 words or fewer what it appears to be. "
              "Reply as 'column = description' lines, nothing else.\n"
              f"Columns and one sample value:\n" +
              "\n".join(f"{c} = {sample_row.get(c, '')}" for c in columns))
    try:
        r = client.chat.completions.create(model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}])
        out = {}
        for line in r.choices[0].message.content.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
        return out
    except Exception:
        return {}


def build_summary_report(title, health_lines, value_table, recs):
    out = [title.upper(), "=" * len(title), "", "Data health:"]
    out += [f"  {l}" for l in health_lines]
    out += ["", "Value by segment:", value_table.to_string(), "", "Findings and recommendations:", recs]
    return "\n".join(out)


def value_band_chart(series, label, sym, color):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return None
    try:
        bands = pd.qcut(s, q=5, duplicates="drop")
    except Exception:
        bands = pd.cut(s, bins=5)
    counts = bands.value_counts().sort_index()
    band_df = pd.DataFrame({
        "Band": [f"{sym}{int(i.left):,} to {sym}{int(i.right):,}" for i in counts.index],
        "Records": counts.values})
    label_clean = re.sub(r"[^0-9A-Za-z ]", "", str(label))
    return alt.Chart(band_df).mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Band:N", sort=None, title=f"{label_clean} band", axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Records:Q", title="Number of records"), tooltip=["Band","Records"]).properties(height=280)


def segment_chart(counts, meanings, color):
    data = pd.DataFrame({"Segment": counts.index, "Records": counts.values,
                         "Meaning": [meanings.get(s, "") for s in counts.index]})
    return alt.Chart(data).mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Segment:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Records:Q", title="Number of records"), tooltip=["Segment","Records","Meaning"]).properties(height=320)


def pareto_chart(pareto_raw, color):
    return alt.Chart(pareto_raw).mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Category:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("RawTotal:Q", title="Total offer dollars"),
        tooltip=["Category", "RawTotal", "Cumulative %"]).properties(height=280)


def build_roster(df, id_col, value_col):
    keep = [c for c in [id_col, "FullName", "FirstName", "LastName", "Segment",
                        value_col, "DaysSinceLastVisit", "MonthsSinceContact", "TierRank"]
            if c and c in df.columns]
    return df[keep].to_csv(index=False)


# ============================================================
# Mode A: players
# ============================================================
PLAYER_MEANINGS = {
    "VIP High Value": "Top quartile value and active. Your most important players.",
    "At Risk High Value": "High value but no visit in 180+ days. Worth a win back effort.",
    "Active Developing": "Recent activity, value still building. Nurture these.",
    "Casual": "Lower value, occasional activity. Low touch.",
    "Churned Low Value": "Lower value and gone a long time. Low priority.",
    "Needs Review": "Missing value data. Set aside for a manual look.",
}

def clean_data(df, value_col, consent_cols, flag_col):
    df = df.copy(); report = []; issues = 0
    for col in consent_cols:
        if col in df.columns:
            normalized = df[col].astype(str).str.strip().str.lower()
            df[col] = normalized.map({"yes":"Yes","no":"No"}).fillna("Unknown")
            report.append(f"Standardized {col} to Yes / No / Unknown.")
    if value_col in df.columns:
        df[value_col] = clean_money_series(df[value_col])
        if df[value_col].isna().any():
            med = df[value_col].median(); count = int(df[value_col].isna().sum()); issues += count
            df[value_col] = df[value_col].fillna(med)
            report.append(f"Filled {count} missing {value_col} values with the median ({med:,.2f}).")
    if flag_col and flag_col in df.columns:
        blank = df[flag_col].astype(str).str.strip() == ""
        nacount = int(df[flag_col].isna().sum() + blank.sum())
        if nacount:
            issues += nacount
            df[flag_col] = df[flag_col].replace("", pd.NA).fillna("MISSING")
            report.append(f"Flagged {nacount} missing {flag_col} values as MISSING.")
    if not report: report.append("No cleaning issues found. The data was already tidy.")
    return df, report, issues / max(len(df), 1)

def player_health(df):
    lines = [f"{len(df)} records loaded across {len(df.columns)} columns."]
    lines.append(f"{df.isna().sum().sum()/max(df.size,1)*100:.1f}% of all cells are empty.")
    if "CanEmail" in df.columns:
        lines.append(f"{(df['CanEmail'].astype(str).str.lower()=='yes').mean()*100:.0f}% are marked emailable.")
    if "DaysSinceLastVisit" in df.columns:
        lines.append(f"{(pd.to_numeric(df['DaysSinceLastVisit'],errors='coerce')>180).mean()*100:.0f}% have not visited in over 180 days.")
    return lines

def segment_players(df):
    df = df.copy()
    if "DaysSinceLastVisit" not in df.columns and "LastVisitDate" in df.columns:
        df["LastVisitDate"] = pd.to_datetime(df["LastVisitDate"], errors="coerce")
        df["DaysSinceLastVisit"] = (pd.Timestamp.now().normalize() - df["LastVisitDate"]).dt.days
    cut = df["NetADT"].quantile(0.75)
    def assign(row):
        v = row.get("NetADT", 0); r = row.get("DaysSinceLastVisit", 0)
        if pd.isna(v): return "Needs Review"
        if r is not None and r > 180: return "At Risk High Value" if v >= cut else "Churned Low Value"
        if v >= cut: return "VIP High Value"
        if r is not None and r <= 30: return "Active Developing"
        return "Casual"
    df["Segment"] = df.apply(assign, axis=1)
    return df, cut

def explain_player(row, cut, sym):
    v = row.get("NetADT", 0); r = row.get("DaysSinceLastVisit", None); parts = []
    if pd.isna(v): return "This player has no NetADT value, so they were set aside for review."
    if v >= cut:
        parts.append(f"Their NetADT of {money(v,sym)} is in the top quartile (cutoff {money(cut,sym)}), so they count as high value.")
    else:
        parts.append(f"Their NetADT of {money(v,sym)} is below the high value cutoff of {money(cut,sym)}.")
    if r is not None and not pd.isna(r):
        if r > 180: parts.append(f"They have not visited in {int(r)} days, past the 180 day mark, so recency flags them as drifting away.")
        elif r <= 30: parts.append(f"They visited {int(r)} days ago, which is recent.")
        else: parts.append(f"They last visited {int(r)} days ago.")
    parts.append(f"Together that places them in the {row['Segment']} segment.")
    return " ".join(parts)

def player_profile(df, sym):
    lines = [f"Total records: {len(df)}"]
    for col in ["NetADT","LifetimeValue","AvgWager","OfferTotal","CoinInTotal"]:
        if col in df.columns:
            s = pd.to_numeric(df[col],errors="coerce")
            lines.append(f"{col}: avg {money(s.mean(),sym)}, median {money(s.median(),sym)}, max {money(s.max(),sym)}")
    if "DaysSinceLastVisit" in df.columns:
        s = pd.to_numeric(df["DaysSinceLastVisit"],errors="coerce")
        lines.append(f"DaysSinceLastVisit: avg {s.mean():.0f}, max {s.max():.0f}")
    for col in ["TierRank","PreferredGame"]:
        if col in df.columns:
            top = df[col].value_counts().head(3)
            lines.append(f"{col} top values: " + ", ".join(f"{i} ({v})" for i,v in top.items()))
    return "\n".join(lines)

def player_pareto(df, sym):
    cols = [c for c in ["OfferSlots","OfferTables","OfferFood","OfferHotel"] if c in df.columns]
    if not cols: return None
    totals = {}
    for c in cols:
        s = clean_money_series(df[c])
        if s.notna().any():
            totals[c] = float(s.sum())
    if not totals: return None
    ser = pd.Series(totals).sort_values(ascending=False)
    p = pd.DataFrame({"Category": ser.index, "RawTotal": ser.values.round(2)})
    p["Cumulative %"] = (p["RawTotal"].cumsum()/p["RawTotal"].sum()*100).round(1)
    return p


# ============================================================
# Mode B: companies
# ============================================================
COMPANY_MEANINGS = {
    "Strategic Account": "Top quartile revenue with healthy leverage. Protect and grow.",
    "High Value At Risk": "Large revenue but heavily leveraged. Watch closely.",
    "Leveraged Watchlist": "Smaller and heavily leveraged. Monitor credit risk.",
    "Dormant": "No contact in over 12 months. Re-engage.",
    "Standard": "Mid portfolio, stable. Routine service.",
    "Needs Review": "Missing revenue data. Set aside for a manual look.",
}

def company_health(df):
    lines = [f"{len(df)} accounts loaded across {len(df.columns)} columns."]
    lines.append(f"{df.isna().sum().sum()/max(df.size,1)*100:.1f}% of all cells are empty.")
    if "MonthsSinceContact" in df.columns:
        lines.append(f"{(pd.to_numeric(df['MonthsSinceContact'],errors='coerce')>12).mean()*100:.0f}% not contacted in over 12 months.")
    if "RiskTier" in df.columns:
        lines.append(f"{(df['RiskTier']=='High Risk').mean()*100:.0f}% sit in the High Risk tier.")
    return lines

def segment_companies(df):
    df = df.copy()
    if "AnnualRevenue" in df.columns:
        df["AnnualRevenue"] = clean_money_series(df["AnnualRevenue"])
    cut = df["AnnualRevenue"].quantile(0.75)
    if "Debt" in df.columns and "Equity" in df.columns:
        d = clean_money_series(df["Debt"]); e = clean_money_series(df["Equity"]).replace(0, pd.NA)
        df["DebtToEquity"] = (d/e).round(2)
    def assign(row):
        rev=row.get("AnnualRevenue",0); dte=row.get("DebtToEquity",0); cold=row.get("MonthsSinceContact",0)
        if pd.isna(rev): return "Needs Review"
        if dte is not None and not pd.isna(dte) and dte>1.5:
            return "High Value At Risk" if rev>=cut else "Leveraged Watchlist"
        if rev>=cut: return "Strategic Account"
        if cold is not None and cold>12: return "Dormant"
        return "Standard"
    df["Segment"]=df.apply(assign,axis=1)
    return df, cut

def explain_company(row, cut, sym):
    rev=row.get("AnnualRevenue",0); dte=row.get("DebtToEquity",None); cold=row.get("MonthsSinceContact",None); parts=[]
    if pd.isna(rev): return "This account has no revenue figure, so it was set aside for review."
    if rev>=cut:
        parts.append(f"Its revenue of {money(rev,sym)} is in the top quartile (cutoff {money(cut,sym)}), so it is a large account.")
    else:
        parts.append(f"Its revenue of {money(rev,sym)} is below the large account cutoff of {money(cut,sym)}.")
    if dte is not None and not pd.isna(dte):
        parts.append(f"Its debt to equity ratio of {dte} is {'above 1.5, which signals heavy leverage.' if dte>1.5 else 'within a healthy range.'}")
    if cold is not None and not pd.isna(cold) and cold>12:
        parts.append(f"It has not been contacted in {int(cold)} months.")
    parts.append(f"Together that places it in the {row['Segment']} segment.")
    return " ".join(parts)

def company_profile(df, sym):
    lines=[f"Total accounts: {len(df)}"]
    for col in ["AnnualRevenue","Debt","Liabilities","Equity"]:
        if col in df.columns:
            s=clean_money_series(df[col])
            lines.append(f"{col}: avg {money(s.mean(),sym)}, median {money(s.median(),sym)}, max {money(s.max(),sym)}")
    for col in ["DebtToEquity","CreditScore","ProductsHeld"]:
        if col in df.columns:
            s=pd.to_numeric(df[col],errors="coerce")
            lines.append(f"{col}: avg {s.mean():.1f}, median {s.median():.1f}, max {s.max():.1f}")
    for col in ["Industry","RiskTier"]:
        if col in df.columns:
            top=df[col].value_counts().head(3)
            lines.append(f"{col} top values: " + ", ".join(f"{i} ({v})" for i,v in top.items()))
    return "\n".join(lines)


# ============================================================
# Mode C: custom
# ============================================================
CUSTOM_MEANINGS = {
    "High Value Active": "Top tier by value and recently active. Your best records.",
    "High Value At Risk": "Top tier by value but gone quiet. Worth re-engaging first.",
    "Mid Tier": "Middle of the pack by value, still active.",
    "Mid Tier Lapsed": "Middle value and gone quiet. Watch these.",
    "Low Value": "Bottom tier by value, still active. Low touch.",
    "Low Value Lapsed": "Bottom value and gone quiet. Lowest priority.",
    "Top Tier": "Highest value records (top 25%).",
    "Middle Tier": "Mid range by value.",
    "Bottom Tier": "Lowest value records (bottom 25%).",
    "Needs Review": "Value could not be read. Set aside for a manual look.",
}

def guess_mapping(df):
    cols = list(df.columns); lower = {c: c.lower() for c in cols}
    def find(keywords, avoid=()):
        for c in cols:
            if any(a in lower[c] for a in avoid): continue
            if any(k in lower[c] for k in keywords): return c
        return None
    id_guess = find(["member", "account", "customer", "player", "ref", "id", "code"])
    value_guess = find(["worth", "value", "revenue", "netadt", "spend", "amount", "ltv",
                        "lifetime", "balance", "sales", "theoretical"],
                       avoid=["row", "index", "no.", "count", "id"])
    recency_guess = find(["dayssince", "daysaway", "days away", "monthssince", "recency",
                         "lastvisit", "last_touch", "lasttouch", "lastcontact"])
    if value_guess is None:
        numeric = df.apply(clean_money_series)
        numeric = numeric.drop(columns=[c for c in cols if any(a in lower[c] for a in ["row","index","no.","count","id"])], errors="ignore")
        if not numeric.dropna(axis=1, how="all").empty:
            value_guess = numeric.std().idxmax()
    return id_guess, value_guess, recency_guess

def segment_custom(df, value_col, recency_col, approach):
    df = df.copy()
    df[value_col] = clean_money_series(df[value_col])
    cut = df[value_col].quantile(0.75); low = df[value_col].quantile(0.25)
    rec = None
    if recency_col and recency_col in df.columns:
        rec = clean_money_series(df[recency_col])
        if rec.isna().all():
            dt = pd.to_datetime(df[recency_col], errors="coerce")
            rec = (pd.Timestamp.now().normalize() - dt).dt.days
        df["_recency"] = rec
    def assign(row):
        v = row.get(value_col, None)
        if pd.isna(v): return "Needs Review"
        if approach == "Value + Recency" and "_recency" in df.columns:
            r = row.get("_recency", None)
            stale = r is not None and not pd.isna(r) and r > rec.median()
            if v >= cut: return "High Value At Risk" if stale else "High Value Active"
            if v <= low: return "Low Value Lapsed" if stale else "Low Value"
            return "Mid Tier Lapsed" if stale else "Mid Tier"
        if v >= cut: return "Top Tier"
        if v <= low: return "Bottom Tier"
        return "Middle Tier"
    df["Segment"] = df.apply(assign, axis=1)
    return df, cut

def custom_health(df, value_col):
    lines = [f"{len(df)} records loaded across {len(df.columns)} columns."]
    lines.append(f"{df.isna().sum().sum()/max(df.size,1)*100:.1f}% of all cells are empty.")
    parsed = clean_money_series(df[value_col])
    lines.append(f"{parsed.notna().mean()*100:.0f}% of the value column parsed into clean numbers.")
    return lines

def custom_profile(df, value_col, sym):
    s = clean_money_series(df[value_col])
    return (f"Total records: {len(df)}\n"
            f"{value_col}: avg {money(s.mean(),sym)}, median {money(s.median(),sym)}, max {money(s.max(),sym)}")

def explain_custom(row, cut, value_col, sym):
    v = row.get(value_col, None)
    if pd.isna(v): return f"This record has no usable {value_col}, so it was set aside for review."
    band = "in the top tier" if v >= cut else "below the top tier"
    return f"Its {value_col} of {money(v,sym)} is {band} (top cutoff {money(cut,sym)}), placing it in the {row['Segment']} segment."


# ============================================================
# Sidebar: chat AND record lookup. Rendered separately from the main flow so
# interacting here does not re-trigger the custom-mode column-mapping block.
# ============================================================
def explain_for_mode(mode, row, cut, value_col, sym):
    if mode == "Players / Customers": return explain_player(row, cut, sym)
    if mode == "Companies / Accounts": return explain_company(row, cut, sym)
    return explain_custom(row, cut, value_col, sym)

def render_sidebar():
    with st.sidebar:
        st.markdown("### Ask about your data")
        if not st.session_state.get("ran"):
            st.caption("Run the agent first. Then the chat and record lookup appear here.")
            return

        sym = st.session_state.sym; cur_name = st.session_state.cur_name
        role = st.session_state.chat_role; segmented = st.session_state.segmented
        id_col = st.session_state.id_col; value_col = st.session_state.value_col
        cut = st.session_state.cut; mode_run = st.session_state.mode_run
        goal = st.session_state.get("goal", "")

        # ----- Chat -----
        chat_on = st.toggle("Open chat", value=False)
        if chat_on:
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            question = st.chat_input("Ask about segments, records, or what to do")
            if st.session_state.chat_history and st.button("Clear chat"):
                st.session_state.chat_history = []
                st.rerun()
            if question:
                st.session_state.chat_history.append({"role": "user", "content": question})
                answer = chat_answer(role, st.session_state.chat_context,
                                     st.session_state.chat_history, sym, cur_name)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.markdown("<div id='chat-scroll'>", unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.caption("Flip this on to ask in plain English about your results.")

        st.markdown("---")
        # ----- Record lookup (lives here so it never disturbs the main page) -----
        if mode_run == "Custom / Any File":
            st.markdown("### Look up a record <span class='beta-badge'>BETA</span>", unsafe_allow_html=True)
        else:
            st.markdown("### Look up a record")
        if not (id_col and id_col in segmented.columns):
            st.caption("This dataset has no ID column mapped, so individual lookup is off.")
            return
        has_names = "FullName" in segmented.columns
        if has_names:
            labels = [f"{r['FullName']} ({r[id_col]})" for _, r in segmented.iterrows()]
        else:
            labels = segmented[id_col].astype(str).tolist()
        label_to_id = dict(zip(labels, segmented[id_col].tolist()))
        picked = st.selectbox("Search by name or ID", labels)
        chosen = segmented[segmented[id_col] == label_to_id[picked]].iloc[0]
        name_label = (f"{chosen['FullName']} ({chosen[id_col]})" if has_names else str(chosen[id_col]))

        st.markdown(f"**{name_label}**")
        st.info(explain_for_mode(mode_run, chosen, cut, value_col, sym))
        with st.expander("Full details"):
            rows = [{"Field": c, "Value": str(chosen[c])} for c in segmented.columns]
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        detail_text = "\n".join(f"{c}: {chosen[c]}" for c in segmented.columns)
        if st.button("Get recommendation for this record"):
            with st.spinner("Thinking..."):
                rec = get_record_recommendation(role, name_label, detail_text,
                        goal or "improve outcomes", sym, cur_name)
            st.markdown("**Recommended action:**")
            st.markdown(rec)


# ================= MAIN UI =================

st.markdown("**Mode**")
mode = st.radio("Mode", ["Players / Customers", "Companies / Accounts", "Custom / Any File"],
                horizontal=True, label_visibility="collapsed")

st.caption("Players / Customers: individual people like casino players or retail customers.  •  "
           "Companies / Accounts: business accounts with revenue, debt, equity.  •  "
           "Custom / Any File: upload any spreadsheet and map your own columns.")

if mode == "Players / Customers":
    cfg = {"title":"Player Segmentation Agent","sample":"sample_players.csv","template":"player_template.csv",
           "id_col":"PlayerID","value_col":"NetADT","consent":["CanEmail","CanCall"],"flag":"ZipCode",
           "role":"a casino marketing analyst",
           "placeholder":"Example: Find high value players who are slipping away so I can win them back",
           "default_goal":"improve overall player value and retention"}
elif mode == "Companies / Accounts":
    cfg = {"title":"Account Segmentation Agent","sample":"sample_companies.csv","template":None,
           "id_col":"AccountID","value_col":"AnnualRevenue","consent":[],"flag":"Industry",
           "role":"a commercial banking portfolio analyst",
           "placeholder":"Example: Find large accounts that are over leveraged and need a check in",
           "default_goal":"grow the portfolio while managing risk"}
else:
    cfg = {"title":"Custom Segmentation Agent","sample":None,"template":None,
           "id_col":None,"value_col":None,"consent":[],"flag":None,
           "role":"a data analyst",
           "placeholder":"Example: Find my highest value records that have gone quiet",
           "default_goal":"find the most valuable records and what to do about them"}

if mode == "Custom / Any File":
    st.markdown(f"<h1 style='display:inline'>{cfg['title']}</h1>"
                "<span class='beta-badge'>BETA</span>", unsafe_allow_html=True)
else:
    st.title(cfg["title"])
st.caption("Upload data, or load the demo. Get clean segments, clear visuals, and tailored recommendations.")

st.subheader("1. Settings")
check_mode = st.radio("How carefully should the agent check with you?",
    ["Smart gates (pause only when something looks off)","Pause at every stage","Run straight through (no pauses)"])

currency = st.selectbox("Currency for money figures", ["USD $", "EUR €", "GBP £", "CAD $", "AUD $"], index=0)
cur_name = currency.split()[0]; sym = currency.split()[-1]

st.markdown("<div class='goal-label'>What are you looking to do?</div>", unsafe_allow_html=True)
st.caption("Type your goal in plain English here, like 'win back my high value players who went quiet'. "
           "This steers the recommendations. For free-form questions after running, use the chat in the left sidebar. "
           "This box is not a chat.")
user_goal = st.text_input("What are you looking to do?", placeholder=cfg["placeholder"], label_visibility="collapsed")

st.subheader("2. Data")
demo_clicked = False
if mode == "Custom / Any File":
    st.caption("Upload any CSV or Excel file. The tool detects your columns and asks you to confirm.")
else:
    st.caption("Step one: load the demo or upload a file. Step two: the Run button appears once data is loaded.")
    c1, c2, c3 = st.columns(3)
    demo_clicked = c1.button("Load Prerendered Demo Data")
    if cfg["sample"] and os.path.exists(cfg["sample"]):
        with open(cfg["sample"], "rb") as f:
            c2.download_button("Download Sample CSV", f, cfg["sample"], "text/csv")
    if cfg["template"] and os.path.exists(cfg["template"]):
        with open(cfg["template"], "rb") as f:
            c3.download_button("Download Blank Template", f, cfg["template"], "text/csv")

uploaded = st.file_uploader("Upload your own (CSV or Excel)", type=["csv","xlsx"])

if uploaded is not None:
    st.session_state.df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    st.session_state.ran = False
elif demo_clicked and cfg["sample"] and os.path.exists(cfg["sample"]):
    st.session_state.df = pd.read_csv(cfg["sample"])
    st.session_state.ran = False
    st.success(f"Demo data loaded from {cfg['sample']}.")

df = st.session_state.get("df")

custom_value = custom_id = custom_recency = custom_approach = None
if mode == "Custom / Any File" and df is not None:
    already_ran = st.session_state.get("ran") and st.session_state.get("mode_run") == mode
    with st.expander("Column mapping (click to change which columns are used)", expanded=not already_ran):
        st.caption("The tool guessed these and the AI gave a second opinion. Each field is explained below.")
        id_g, val_g, rec_g = guess_mapping(df)
        sample_row = df.head(1).to_dict("records")
        sample_row = sample_row[0] if sample_row else {}
        descs = describe_columns_ai(tuple(df.columns), sample_row)
        cols = list(df.columns); none_opt = "(none)"
        def sample_of(col):
            try: return str(df[col].dropna().iloc[0])[:30]
            except Exception: return ""

        st.markdown("**ID column** <span style='color:rgba(230,241,255,0.5);font-weight:400'>"
                    "the unique label for each record, like a customer number or name</span>", unsafe_allow_html=True)
        custom_id = st.selectbox("ID column", cols, index=cols.index(id_g) if id_g in cols else 0, label_visibility="collapsed")
        st.caption(f"Example value: {sample_of(custom_id)}" + (f"  •  AI thinks: {descs.get(custom_id,'')}" if descs.get(custom_id) else ""))

        st.markdown("**Value column** <span style='color:rgba(230,241,255,0.5);font-weight:400'>"
                    "the one number that says how valuable each record is</span>", unsafe_allow_html=True)
        custom_value = st.selectbox("Value column", cols, index=cols.index(val_g) if val_g in cols else 0, label_visibility="collapsed")
        st.caption(f"Example value: {sample_of(custom_value)}" + (f"  •  AI thinks: {descs.get(custom_value,'')}" if descs.get(custom_value) else ""))

        st.markdown("**Recency column (optional)** <span style='color:rgba(230,241,255,0.5);font-weight:400'>"
                    "how long since each record was last active, a date or a days/months since number</span>", unsafe_allow_html=True)
        rec_opts = [none_opt] + cols
        custom_recency = st.selectbox("Recency column", rec_opts, index=rec_opts.index(rec_g) if rec_g in rec_opts else 0, label_visibility="collapsed")
        if custom_recency != none_opt:
            st.caption(f"Example value: {sample_of(custom_recency)}" + (f"  •  AI thinks: {descs.get(custom_recency,'')}" if descs.get(custom_recency) else ""))
        else:
            custom_recency = None

        suggested = "Value + Recency" if custom_recency else "Value only"
        st.markdown(f"**Suggested approach: {suggested}.**")
        if suggested == "Value + Recency":
            st.caption("Value + Recency groups records by how valuable they are AND whether they have gone quiet.")
        else:
            st.caption("Value only ranks records into top, middle, and bottom tiers by value.")
        custom_approach = st.radio("Segmentation approach", ["Value + Recency", "Value only"],
                                   index=0 if suggested == "Value + Recency" else 1, horizontal=True)
    cfg["id_col"] = custom_id; cfg["value_col"] = custom_value

if df is None:
    st.info("No file loaded yet. Please upload a CSV or Excel file above (or load the demo in the named modes) to continue.")
else:
    with st.expander("Preview the data"):
        st.dataframe(df.head(10), width='stretch')

    st.subheader("3. Run")
    if st.button("Run the agent", type="primary"):
        if mode == "Custom / Any File" and (custom_value is None):
            st.warning("Please map a Value column in the Column mapping section above before running.")
        else:
            if mode == "Players / Customers":
                cleaned, clean_report, missing_rate = clean_data(df, cfg["value_col"], cfg["consent"], cfg["flag"])
                segmented, cut = segment_players(cleaned)
                profile = player_profile(segmented, sym); pareto = player_pareto(segmented, sym)
                meanings = PLAYER_MEANINGS; health = player_health(df)
            elif mode == "Companies / Accounts":
                cleaned, clean_report, missing_rate = clean_data(df, cfg["value_col"], cfg["consent"], cfg["flag"])
                segmented, cut = segment_companies(cleaned)
                profile = company_profile(segmented, sym); pareto = None
                meanings = COMPANY_MEANINGS; health = company_health(df)
            else:
                segmented, cut = segment_custom(df, custom_value, custom_recency, custom_approach)
                clean_report = [f"Cleaned the {custom_value} column into numbers, stripping symbols and commas."]
                missing_rate = clean_money_series(df[custom_value]).isna().mean()
                profile = custom_profile(segmented, custom_value, sym); pareto = None
                meanings = CUSTOM_MEANINGS; health = custom_health(df, custom_value)
            counts = segmented["Segment"].value_counts()
            value_table = segmented.groupby("Segment").agg(
                Records=(cfg["id_col"],"count"),
                TotalValue=(cfg["value_col"],"sum"),
                AvgValue=(cfg["value_col"],"mean"),
            ).round(2).sort_values("TotalValue", ascending=False)
            with st.spinner("Generating recommendations..."):
                goal = user_goal.strip() or cfg["default_goal"]
                recs = get_recommendations(cfg["role"], profile, counts.to_string(),
                                           value_table["TotalValue"].to_string(), goal, sym, cur_name)
            roster = build_roster(segmented, cfg["id_col"], cfg["value_col"])
            chat_context = (f"User goal: {goal}\n\nData profile:\n{profile}\n\n"
                            f"Segment headcounts:\n{counts.to_string()}\n\n"
                            f"Total value by segment:\n{value_table['TotalValue'].to_string()}\n\n"
                            f"Recommendations already given:\n{recs}\n\n"
                            f"RECORD ROSTER (one row per record):\n{roster}")
            st.session_state.update(dict(ran=True, segmented=segmented, cut=cut, meanings=meanings,
                pareto=pareto, health=health, counts=counts, value_table=value_table, recs=recs,
                missing_rate=missing_rate, clean_report=clean_report, sym=sym, cur_name=cur_name, mode_run=mode,
                chat_context=chat_context, chat_role=cfg["role"], chat_history=[], goal=goal,
                id_col=cfg["id_col"], value_col=cfg["value_col"]))

    if st.session_state.get("ran") and st.session_state.get("mode_run") == mode:
        meanings = st.session_state.meanings; pareto = st.session_state.pareto
        health = st.session_state.health; counts = st.session_state.counts
        value_table = st.session_state.value_table; recs = st.session_state.recs
        sym = st.session_state.sym; cur_name = st.session_state.cur_name
        segmented = st.session_state.segmented; value_col = st.session_state.value_col

        divider()
        st.subheader("Data health")
        st.caption("A quick read on the file before any analysis.")
        for line in health: st.write("- " + line)

        divider()
        st.subheader("Cleaning summary")
        for line in st.session_state.clean_report: st.write("- " + line)
        if st.session_state.missing_rate > 0.10:
            st.markdown(f"<span class='critical'>About {st.session_state.missing_rate*100:.0f}% of the value column "
                        "could not be read as numbers. Check you mapped the right column.</span>", unsafe_allow_html=True)

        divider()
        st.markdown("<div class='head-cyan'>Segments</div>", unsafe_allow_html=True)
        st.markdown("Cyan chart. Hover any bar to see what that " + term("Segment") + " means.", unsafe_allow_html=True)
        st.altair_chart(segment_chart(counts, meanings, CYAN), width='stretch')
        insight(segments_insight(counts, value_table, sym))

        divider()
        st.subheader("Value by segment")
        st.caption(f"Total and average value per group, in {cur_name} ({sym}). A small group can hold most of the value.")
        display_table = value_table.copy()
        display_table["TotalValue"] = display_table["TotalValue"].map(lambda x: money(x, sym))
        display_table["AvgValue"] = display_table["AvgValue"].map(lambda x: money(x, sym))
        st.dataframe(display_table, width='stretch')

        divider()
        st.markdown("<div class='head-violet'>Value distribution</div>", unsafe_allow_html=True)
        st.caption(f"Violet chart. Records grouped into equal-size value bands, in {cur_name} ({sym}). Rank based so skew does not bunch them.")
        band = value_band_chart(segmented[value_col], value_col, sym, VIOLET)
        if band is not None: st.altair_chart(band, width='stretch')
        insight(distribution_insight(segmented[value_col], sym))

        if pareto is not None:
            divider()
            st.markdown("<div class='head-amber'>Spend Concentration (Pareto)</div>", unsafe_allow_html=True)
            st.markdown("<span style='color:#F5A623'>Amber chart.</span> Total offer dollars by category. The "
                        + term("Pareto") + f" cumulative percent shows how concentrated the spend is. Figures in {cur_name} ({sym}).",
                        unsafe_allow_html=True)
            st.altair_chart(pareto_chart(pareto, AMBER), width='stretch')
            show = pareto.copy()
            show["Total Offer Dollars"] = show["RawTotal"].map(lambda x: money(x, sym))
            st.dataframe(show[["Category","Total Offer Dollars","Cumulative %"]], width='stretch', hide_index=True)
            insight(pareto_insight(pareto, sym))

        divider()
        st.subheader("Look up an individual record")
        st.caption("Open the left sidebar to search any record by name or ID, see why it landed in its segment, "
                   "and get a tailored AI action. It lives in the sidebar so it never disturbs this page.")

        divider()
        st.subheader("Recommended actions")
        st.caption("Grounded in your uploaded data and tailored to the goal you typed above.")
        st.markdown(format_recommendations(recs), unsafe_allow_html=True)

        divider()
        st.subheader("Glossary")
        with st.expander("Key terms explained"):
            for k, v in GLOSSARY.items():
                st.markdown(f"**{k}.** {v}")

        divider()
        st.subheader("Export")
        e1, e2 = st.columns(2)
        csv = segmented.to_csv(index=False).encode("utf-8")
        e1.download_button("Download segmented data (CSV)", csv, "segmented_output.csv", "text/csv")
        report_txt = build_summary_report(cfg["title"], health, value_table, recs).encode("utf-8")
        e2.download_button("Download summary report (TXT)", report_txt, "segmentation_summary.txt", "text/plain")

render_sidebar()

st.markdown(
    "<div id='bc-footer'>"
    "Built by <a href='https://briancabrera.io' target='_blank'>Brian Cabrera</a>"
    "<span class='sep'>|</span><a href='https://www.linkedin.com/in/briandacellcabrera/' target='_blank'>LinkedIn</a>"
    "<span class='sep'>|</span><a href='https://github.com/BrianDCab' target='_blank'>GitHub</a>"
    "<span class='sep'>|</span>Powered by AI</div>",
    unsafe_allow_html=True,
)