import os
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

# Load my Groq key from .env so it never lives in the code itself.
load_dotenv()

st.set_page_config(page_title="Segmentation Agent", page_icon="🎲", layout="centered")

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
    #chat-scroll { max-height: 55vh; overflow-y: auto; padding-right: 4px; }
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
# Shared engine
# ============================================================

def clean_data(df, value_col, consent_cols, flag_col):
    df = df.copy(); report = []; issues = 0
    for col in consent_cols:
        if col in df.columns:
            normalized = df[col].astype(str).str.strip().str.lower()
            df[col] = normalized.map({"yes":"Yes","no":"No"}).fillna("Unknown")
            report.append(f"Standardized {col} to Yes / No / Unknown.")
    if value_col in df.columns and df[value_col].isna().any():
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


def get_recommendations(role, data_profile, seg_summary, val_summary, user_goal, sym):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""You are {role} reviewing one specific uploaded dataset. All money figures are in {sym}.

DATA PROFILE (real numbers from this file):
{data_profile}

HEADCOUNT BY SEGMENT:
{seg_summary}

TOTAL VALUE BY SEGMENT (in {sym}):
{val_summary}

THE USER'S GOAL: "{user_goal}"

Write exactly 4 items. Put each label on its own separate line. Use this exact format, with a blank line between items:

### [short title of the action]
**What the data shows:** one sentence with a real number from above, including the {sym} symbol on money figures.
**Do this:** one concrete, specific action.
**Why it helps:** one sentence tying it to the user's goal.

Keep it under 320 words. Do not use dashes as punctuation."""
    response = client.chat.completions.create(model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content


def get_record_recommendation(role, name_label, detail_text, user_goal, sym):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""You are {role}. Money is in {sym}. Here is one specific record:

{detail_text}

The user's goal: "{user_goal}"

In 2 to 3 short sentences, recommend the single best next action for {name_label}.
Be specific and reference their actual numbers. Do not use dashes as punctuation."""
    response = client.chat.completions.create(model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content


def chat_answer(role, context, history, sym):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    system = (f"You are {role} helping a user understand one specific analyzed dataset. "
              f"Money is in {sym}. Use only the context below to answer. The roster lists "
              f"individual records you can answer about by name or ID. If asked about a record "
              f"not in the roster or something unrelated to this data, say so plainly. "
              f"Keep answers short and concrete. Do not use dashes as punctuation.\n\n"
              f"CONTEXT:\n{context}")
    messages = [{"role": "system", "content": system}] + history
    response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
    return response.choices[0].message.content


def build_summary_report(title, health_lines, value_table, recs):
    out = [title.upper(), "=" * len(title), "", "Data health:"]
    out += [f"  {l}" for l in health_lines]
    out += ["", "Value by segment:", value_table.to_string(), "", "Findings and recommendations:", recs]
    return "\n".join(out)


def value_band_chart(series, label, sym):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return None
    counts = pd.cut(s, bins=6).value_counts().sort_index()
    band_df = pd.DataFrame({
        "Band": [f"{sym}{int(i.left):,} to {sym}{int(i.right):,}" for i in counts.index],
        "Records": counts.values})
    return alt.Chart(band_df).mark_bar(color="#22D3EE", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Band:N", sort=None, title=f"{label} band"),
        y=alt.Y("Records:Q", title="Number of records"), tooltip=["Band","Records"]).properties(height=260)


def segment_chart(counts, meanings):
    data = pd.DataFrame({"Segment": counts.index, "Records": counts.values,
                         "Meaning": [meanings.get(s, "") for s in counts.index]})
    return alt.Chart(data).mark_bar(color="#22D3EE", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Segment:N", sort="-y", title=None),
        y=alt.Y("Records:Q", title="Number of records"), tooltip=["Segment","Records","Meaning"]).properties(height=300)


def build_roster(df, id_col, value_col):
    keep = [c for c in [id_col, "FullName", "FirstName", "LastName", "Segment",
                        value_col, "DaysSinceLastVisit", "MonthsSinceContact", "TierRank"]
            if c in df.columns]
    return df[keep].head(250).to_csv(index=False)


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
    totals = df[cols].sum().sort_values(ascending=False)
    p = pd.DataFrame({"Category": totals.index, "Total Offer Dollars": totals.values.round(2)})
    p["Cumulative %"] = (p["Total Offer Dollars"].cumsum()/p["Total Offer Dollars"].sum()*100).round(1)
    p["Total Offer Dollars"] = p["Total Offer Dollars"].map(lambda x: money(x, sym))
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
    df = df.copy(); cut = df["AnnualRevenue"].quantile(0.75)
    if "Debt" in df.columns and "Equity" in df.columns:
        df["DebtToEquity"] = (df["Debt"]/df["Equity"].replace(0,pd.NA)).round(2)
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
            s=pd.to_numeric(df[col],errors="coerce")
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
# Sidebar chat. Off by default so its input box does not cause the page to jump
# to the bottom when you run the agent. Turn it on with the toggle when wanted.
# ============================================================
def render_chat_sidebar():
    with st.sidebar:
        st.markdown("### Ask about your data")
        if not st.session_state.get("ran"):
            st.caption("Run the agent first. Then you can open the chat here.")
            return

        chat_on = st.toggle("Open AI Chat", value=False)
        if not chat_on:
            st.caption("Flip this on to ask about segments, what to do, or a specific record by name or ID.")
            return

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        question = st.chat_input("Ask about segments, records, or what to do")
        if st.session_state.chat_history and st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            answer = chat_answer(st.session_state.chat_role, st.session_state.chat_context,
                                 st.session_state.chat_history, st.session_state.sym)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        st.markdown("<div id='chat-scroll'>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        st.markdown("</div>", unsafe_allow_html=True)


# ================= UI =================

st.markdown("**Mode**")
mode = st.radio("Mode", ["Players / Customers", "Companies / Accounts"], horizontal=True, label_visibility="collapsed")

if mode == "Players / Customers":
    cfg = {"title":"Player Segmentation Agent","sample":"sample_players.csv","template":"player_template.csv",
           "id_col":"PlayerID","value_col":"NetADT","consent":["CanEmail","CanCall"],"flag":"ZipCode",
           "role":"a casino marketing analyst",
           "placeholder":"Example: Find high value players who are slipping away so I can win them back",
           "default_goal":"improve overall player value and retention"}
else:
    cfg = {"title":"Account Segmentation Agent","sample":"sample_companies.csv","template":None,
           "id_col":"AccountID","value_col":"AnnualRevenue","consent":[],"flag":"Industry",
           "role":"a commercial banking portfolio analyst",
           "placeholder":"Example: Find large accounts that are over leveraged and need a check in",
           "default_goal":"grow the portfolio while managing risk"}

st.title(cfg["title"])
st.caption("Upload data, or load the demo. Get clean segments, clear visuals, and tailored recommendations.")

st.subheader("1. Settings")
check_mode = st.radio("How carefully should the agent check with you?",
    ["Smart gates (pause only when something looks off)","Pause at every stage","Run straight through (no pauses)"])
with st.expander("Explain this in depth"):
    st.markdown("""
**Smart gates.** The agent pauses only when the data trips a real threshold, like a high share of missing values. You get interrupted only when your judgment actually adds something.

**Pause at every stage.** The agent stops after each step so you can review before it continues. Useful while you are still building trust in the tool.

**Run straight through.** No pauses. Best once you trust the output and just want speed.
""")

currency = st.selectbox("Currency for money figures", ["USD $", "EUR €", "GBP £", "CAD $", "AUD $"], index=0)
sym = currency.split()[-1]

st.markdown("<div class='goal-label'>What are you looking to do?</div>", unsafe_allow_html=True)
st.caption("This steers the recommendations below. Describe a goal, like winning back lapsed players. It is not a chat box.")
user_goal = st.text_input("What are you looking to do?", placeholder=cfg["placeholder"], label_visibility="collapsed")

st.subheader("2. Data")
st.caption("Step one: load the demo or upload a file. Step two: the Run button appears once data is loaded.")
c1, c2, c3 = st.columns(3)
demo_clicked = c1.button("Load Prerendered Demo Data")
if os.path.exists(cfg["sample"]):
    with open(cfg["sample"], "rb") as f:
        c2.download_button("Download Sample CSV", f, cfg["sample"], "text/csv")
if cfg["template"] and os.path.exists(cfg["template"]):
    with open(cfg["template"], "rb") as f:
        c3.download_button("Download Blank Template", f, cfg["template"], "text/csv")
uploaded = st.file_uploader("Or upload your own (CSV or Excel)", type=["csv","xlsx"])

if uploaded is not None:
    st.session_state.df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    st.session_state.ran = False
elif demo_clicked and os.path.exists(cfg["sample"]):
    st.session_state.df = pd.read_csv(cfg["sample"])
    st.session_state.ran = False
    st.success(f"Demo data loaded from {cfg['sample']}.")

df = st.session_state.get("df")

if df is None:
    st.info("Load the demo data or upload a file above to continue. The Run button will appear here.")
else:
    with st.expander("Preview the data"):
        st.dataframe(df.head(10), use_container_width=True)

    st.subheader("3. Run")
    if st.button("Run the agent", type="primary"):
        cleaned, clean_report, missing_rate = clean_data(df, cfg["value_col"], cfg["consent"], cfg["flag"])
        if mode == "Players / Customers":
            segmented, cut = segment_players(cleaned)
            profile = player_profile(segmented, sym); pareto = player_pareto(segmented, sym)
            meanings = PLAYER_MEANINGS
        else:
            segmented, cut = segment_companies(cleaned)
            profile = company_profile(segmented, sym); pareto = None
            meanings = COMPANY_MEANINGS
        health = player_health(df) if mode == "Players / Customers" else company_health(df)
        counts = segmented["Segment"].value_counts()
        value_table = segmented.groupby("Segment").agg(
            Records=(cfg["id_col"],"count"),
            TotalValue=(cfg["value_col"],"sum"),
            AvgValue=(cfg["value_col"],"mean"),
        ).round(2).sort_values("TotalValue", ascending=False)
        with st.spinner("Generating recommendations..."):
            goal = user_goal.strip() or cfg["default_goal"]
            recs = get_recommendations(cfg["role"], profile, counts.to_string(),
                                       value_table["TotalValue"].to_string(), goal, sym)
        roster = build_roster(segmented, cfg["id_col"], cfg["value_col"])
        chat_context = (f"User goal: {goal}\n\nData profile:\n{profile}\n\n"
                        f"Segment headcounts:\n{counts.to_string()}\n\n"
                        f"Total value by segment:\n{value_table['TotalValue'].to_string()}\n\n"
                        f"Recommendations already given:\n{recs}\n\n"
                        f"RECORD ROSTER (one row per record):\n{roster}")
        st.session_state.update(dict(ran=True, segmented=segmented, cut=cut, meanings=meanings,
            pareto=pareto, health=health, counts=counts, value_table=value_table, recs=recs,
            missing_rate=missing_rate, clean_report=clean_report, sym=sym, mode_run=mode,
            chat_context=chat_context, chat_role=cfg["role"], chat_history=[]))

    if st.session_state.get("ran") and st.session_state.get("mode_run") == mode:
        segmented = st.session_state.segmented; cut = st.session_state.cut
        meanings = st.session_state.meanings; pareto = st.session_state.pareto
        health = st.session_state.health; counts = st.session_state.counts
        value_table = st.session_state.value_table; recs = st.session_state.recs
        sym = st.session_state.sym

        st.subheader("Data health")
        st.caption("A quick read on the file before any analysis.")
        for line in health: st.write("- " + line)

        st.subheader("Cleaning summary")
        for line in st.session_state.clean_report: st.write("- " + line)
        if st.session_state.missing_rate > 0.10:
            st.markdown(f"<span class='critical'>About {st.session_state.missing_rate*100:.0f}% of rows had "
                        "missing key values, higher than usual. Review the cleaning above.</span>", unsafe_allow_html=True)

        st.subheader("Segments")
        st.markdown("Hover any bar to see what that " + term("Segment") + " means.", unsafe_allow_html=True)
        st.altair_chart(segment_chart(counts, meanings), use_container_width=True)

        st.subheader("Value by segment")
        st.caption(f"Total and average value per group, in {sym}. A small group can hold most of the value.")
        display_table = value_table.copy()
        display_table["TotalValue"] = display_table["TotalValue"].map(lambda x: money(x, sym))
        display_table["AvgValue"] = display_table["AvgValue"].map(lambda x: money(x, sym))
        st.dataframe(display_table, use_container_width=True)

        st.subheader("Value distribution")
        st.caption(f"How many records fall into each {cfg['value_col']} band, in {sym}.")
        band = value_band_chart(segmented[cfg["value_col"]], cfg["value_col"], sym)
        if band is not None: st.altair_chart(band, use_container_width=True)

        if pareto is not None:
            st.subheader("Spend Concentration (Pareto)")
            st.markdown("Total offer dollars by category. The " + term("Pareto")
                        + f" cumulative percent shows how concentrated the spend is. Figures in {sym}.", unsafe_allow_html=True)
            st.dataframe(pareto, use_container_width=True, hide_index=True)

        st.subheader("Why is this record here?")
        st.caption("Search by name or ID to see the full reasoning and a tailored action.")
        if cfg["id_col"] in segmented.columns:
            has_names = "FullName" in segmented.columns
            if has_names:
                labels = [f"{r['FullName']} ({r[cfg['id_col']]})" for _, r in segmented.iterrows()]
            else:
                labels = segmented[cfg["id_col"]].tolist()
            label_to_id = dict(zip(labels, segmented[cfg["id_col"]].tolist()))
            st.markdown("Record <span style='color:rgba(230,241,255,0.4);font-weight:400'>"
                        "- dropdown or type to search by name or ID</span>", unsafe_allow_html=True)
            picked = st.selectbox("Record", labels, label_visibility="collapsed")
            chosen = segmented[segmented[cfg["id_col"]] == label_to_id[picked]].iloc[0]
            name_label = (f"{chosen['FullName']} ({chosen[cfg['id_col']]})"
                          if has_names else str(chosen[cfg["id_col"]]))

            explain_fn = explain_player if mode == "Players / Customers" else explain_company
            st.markdown(f"### {name_label}")
            st.info(explain_fn(chosen, cut, sym))

            with st.expander("Full details for this record"):
                money_cols = {"NetADT","LifetimeValue","AvgWager","AccountBalance","OfferTotal",
                              "CoinInTotal","OfferSlots","OfferTables","OfferFood","OfferHotel",
                              "DepositsMonth","WithdrawalsMonth","AnnualRevenue","Debt","Liabilities","Equity"}
                rows = []
                for col in segmented.columns:
                    val = chosen[col]
                    if col in money_cols and pd.notna(val) and isinstance(val, (int, float)):
                        val = money(val, sym)
                    rows.append({"Field": col, "Value": val})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            detail_text = "\n".join(f"{c}: {chosen[c]}" for c in segmented.columns)
            rec_key = f"rec_{label_to_id[picked]}_{mode}"
            if st.session_state.get("active_rec_key") != rec_key:
                with st.spinner(f"Thinking about {name_label}..."):
                    st.session_state.active_rec = get_record_recommendation(
                        cfg["role"], name_label, detail_text,
                        user_goal.strip() or cfg["default_goal"], sym)
                    st.session_state.active_rec_key = rec_key
            st.markdown(f"**Recommended action for {name_label}:**")
            st.markdown(st.session_state.active_rec)

        st.subheader("Recommended actions")
        st.caption("Grounded in your uploaded data and tailored to the goal you typed above.")
        st.markdown(format_recommendations(recs), unsafe_allow_html=True)

        st.subheader("Glossary")
        with st.expander("Key terms explained"):
            for k, v in GLOSSARY.items():
                st.markdown(f"**{k}.** {v}")

        st.subheader("Export")
        e1, e2 = st.columns(2)
        csv = segmented.to_csv(index=False).encode("utf-8")
        e1.download_button("Download segmented data (CSV)", csv, "segmented_output.csv", "text/csv")
        report_txt = build_summary_report(cfg["title"], health, value_table, recs).encode("utf-8")
        e2.download_button("Download summary report (TXT)", report_txt, "segmentation_summary.txt", "text/plain")

render_chat_sidebar()

st.markdown(
    "<div id='bc-footer'>"
    "Built by <a href='https://briancabrera.io' target='_blank'>Brian Cabrera</a>"
    "<span class='sep'>|</span><a href='https://www.linkedin.com/in/briandacellcabrera/' target='_blank'>LinkedIn</a>"
    "<span class='sep'>|</span><a href='https://github.com/BrianDCab' target='_blank'>GitHub</a>"
    "<span class='sep'>|</span>Powered by AI</div>",
    unsafe_allow_html=True,
)