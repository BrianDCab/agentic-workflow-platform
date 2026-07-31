import os
import re
import json
import time
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai
import streamlit as st

# Load my API keys. Locally they come from .env; on Streamlit Cloud they come
# from the app's Secrets. Check Secrets first, then fall back to .env.
load_dotenv()

def get_key(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)

st.set_page_config(page_title="Segmentation Agent", page_icon="\U0001F3B2", layout="centered")

CYAN = "#22D3EE"; VIOLET = "#A78BFA"; AMBER = "#F5A623"

# ---------- Styling ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');
    .stApp {
        background: radial-gradient(1200px 700px at 15% -10%, rgba(34,211,238,0.07), transparent 60%),
                    radial-gradient(1000px 600px at 110% 15%, rgba(167,139,250,0.06), transparent 55%),
                    #050608;
        overflow-x: hidden;
    }
    html, body, .stApp, p, li, label, input, textarea, div[data-testid] {
        font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
    }
    h1, h2, h3 { color: #FFFFFF; letter-spacing: 0.3px; font-weight: 700;
                 font-family: 'Space Grotesk', 'Segoe UI', sans-serif; }
    .critical { color: #FFD23F; text-shadow: 0 0 8px rgba(255,210,63,0.7); font-weight: 600; }
    .goal-label { color: #FF5C7A; font-weight: 800; font-size: 1.05rem;
                  text-shadow: 0 0 8px rgba(225,29,72,0.4); margin-bottom: 2px; }
    .term { color: #22D3EE; font-weight: 600; cursor: help;
            border-bottom: 1px dotted rgba(34,211,238,0.7); text-shadow: 0 0 6px rgba(34,211,238,0.4); }
    .rec-do { color: #22D3EE; font-weight: 700; text-shadow: 0 0 6px rgba(34,211,238,0.5); }
    .rec-why { color: #7FE7F5; font-weight: 700; }
    .insight { background: rgba(34,211,238,0.06); border-left: 3px solid rgba(34,211,238,0.6);
               padding: 8px 12px; border-radius: 4px; margin: 6px 0 2px 0; color:#CFEfff; }
    .ai-banner { background: rgba(245,166,35,0.10); border:1px solid rgba(245,166,35,0.5);
                 color:#F5C77E; padding:6px 12px; border-radius:6px; font-size:0.85rem; margin:4px 0; }
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
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(34,211,238,0.09), rgba(34,211,238,0.03));
        border: 1px solid rgba(34,211,238,0.28); border-radius: 14px;
        padding: 14px 16px 10px 16px; box-shadow: inset 0 0 22px rgba(34,211,238,0.08);
    }
    [data-testid="stMetricLabel"] { color: rgba(230,241,255,0.6); text-transform: uppercase;
        letter-spacing: 1.6px; font-size: 0.72rem; }
    [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; color: #E8F1F5;
        text-shadow: 0 0 12px rgba(34,211,238,0.35); }
    div[role="radiogroup"] > label:has(input:checked) {
        background: rgba(34,211,238,0.18); border-color: #22D3EE; color: #FFFFFF;
        box-shadow: 0 0 14px rgba(34,211,238,0.35);
    }
    [data-testid="stExpander"] {
        border: 1px solid rgba(34,211,238,0.18); border-radius: 12px;
        background: rgba(255,255,255,0.015);
    }
    [data-testid="stFileUploader"] section {
        border: 1.5px dashed rgba(34,211,238,0.4); border-radius: 12px;
        background: rgba(34,211,238,0.03);
    }
    button[kind="primary"], [data-testid="stBaseButton-primary"] {
        background: linear-gradient(90deg, #22D3EE, #A78BFA) !important;
        color: #04070A !important; border: none !important; font-weight: 800 !important;
        box-shadow: 0 0 24px rgba(34,211,238,0.35) !important;
    }
    button[kind="primary"]:hover { filter: brightness(1.08); }
    .step-chip { display: flex; align-items: center; gap: 10px; margin: 26px 0 4px 0; }
    .step-num { display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; border-radius: 8px; font-family: 'Space Grotesk', sans-serif;
        font-weight: 700; color: #04070A; background: #22D3EE;
        box-shadow: 0 0 12px rgba(34,211,238,0.5); font-size: 0.9rem; }
    .step-title { color: #FFFFFF; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.25rem; }
    .eyebrow { letter-spacing: 3px; font-size: 0.72rem; color: rgba(34,211,238,0.85);
        font-weight: 700; margin-bottom: 2px; text-transform: uppercase; }
    @media (prefers-reduced-motion: reduce) {
        #bc-glow-cyan-l, #bc-glow-cyan-r, #bc-glow-ruby { animation: none; }
    }
    .stApp > header, .block-container { position: relative; z-index: 2; }
</style>
<div id="bc-glow-cyan-l"></div><div id="bc-glow-cyan-r"></div><div id="bc-glow-ruby"></div>
""", unsafe_allow_html=True)


def divider():
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

def insight(text):
    st.markdown(f"<div class='insight'>{text}</div>", unsafe_allow_html=True)


def money(x, sym):
    try:
        return f"{sym}{x:,.2f}"
    except Exception:
        return str(x)


def clean_money_series(s):
    cleaned = (s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True).replace("", pd.NA))
    return pd.to_numeric(cleaned, errors="coerce")


# Plain-English analysis lines (computed from real numbers, not AI).
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
# AI layer with failover: try Groq first, fall back to Gemini on any error
# (rate limit included), report which provider answered. Returns (text, label).
# ============================================================
def _groq_call(prompt, system=None):
    client = Groq(api_key=get_key("GROQ_API_KEY"))
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    r = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
    return r.choices[0].message.content

def _gemini_call(prompt, system=None):
    genai.configure(api_key=get_key("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    full = (system + "\n\n" + prompt) if system else prompt
    r = model.generate_content(full)
    return r.text

def ai_generate(prompt, system=None):
    try:
        return _groq_call(prompt, system), "Groq (llama-3.3-70b)"
    except Exception:
        try:
            return _gemini_call(prompt, system), "Gemini (backup)"
        except Exception:
            return ("The AI is unavailable right now (both the primary and backup "
                    "providers could not be reached). The data, charts, and segment "
                    "analytics above are all still accurate. Try the AI again shortly."), "none"


def get_recommendations(role, data_profile, seg_summary, val_summary, user_goal, sym, cur_name):
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
    return ai_generate(prompt)


@st.cache_data(show_spinner=False)
def get_record_recommendation(role, name_label, detail_text, user_goal, sym, cur_name):
    prompt = f"""You are {role}. All money is in {cur_name} ({sym}). Here is one specific record:

{detail_text}

The user's goal: "{user_goal}"

In 2 to 3 short sentences, recommend the single best next action for {name_label}.
Be specific and reference their actual numbers. Put the {sym} symbol directly before every
money amount, including debt, equity, liabilities, and revenue, like {sym}210,437.66.
Do not use dashes as punctuation."""
    return ai_generate(prompt)


def chat_answer(role, context, history, sym, cur_name):
    system = (f"You are {role} helping a user understand one specific analyzed dataset. "
              f"All money is in {cur_name} ({sym}); always show the {sym} symbol on amounts. "
              f"Use only the context below to answer. The roster lists individual records you "
              f"can answer about by name or ID. Match names loosely (ignore case and small "
              f"differences). If asked about a record truly not in the roster or something "
              f"unrelated, say so plainly. Keep answers short and concrete. Do not use dashes "
              f"as punctuation.\n\nCONTEXT:\n{context}")
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    return ai_generate(convo, system)


@st.cache_data(show_spinner=False)
def describe_columns_ai(columns, sample_row):
    prompt = ("For each column, say in 4 words or fewer what it appears to be. "
              "Reply as 'column = description' lines, nothing else.\n"
              f"Columns and one sample value:\n" +
              "\n".join(f"{c} = {sample_row.get(c, '')}" for c in columns))
    text, _ = ai_generate(prompt)
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


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
        y=alt.Y("Records:Q", title="Number of records"), tooltip=["Band", "Records"]).properties(height=280)


def segment_chart(counts, meanings, color):
    data = pd.DataFrame({"Segment": counts.index, "Records": counts.values,
                         "Meaning": [meanings.get(s, "") for s in counts.index]})
    return alt.Chart(data).mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Segment:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Records:Q", title="Number of records"), tooltip=["Segment", "Records", "Meaning"]).properties(height=320)


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
# Per-segment analytics, computed from real numbers (no AI, instant, accurate).
# ============================================================
def segment_overview_table(segmented, value_col, id_col, sym):
    g = segmented.groupby("Segment")
    rows = []
    total_val = clean_money_series(segmented[value_col]).sum()
    for seg, sub in g:
        vals = clean_money_series(sub[value_col])
        tv = vals.sum()
        rows.append({
            "Segment": seg,
            "Records": len(sub),
            "% of records": f"{len(sub)/len(segmented)*100:.0f}%",
            "Total value": money(tv, sym),
            "Avg value": money(vals.mean(), sym),
            "% of value": f"{(tv/total_val*100 if total_val else 0):.0f}%",
        })
    return pd.DataFrame(rows).sort_values("Records", ascending=False)

def segment_deep_dive(segmented, seg_name, value_col, id_col, sym, meanings):
    sub = segmented[segmented["Segment"] == seg_name].copy()
    vals = clean_money_series(sub[value_col])
    lines = []
    meaning = meanings.get(seg_name, "")
    if meaning:
        lines.append(f"**What this segment is:** {meaning}")
    lines.append(f"**Records:** {len(sub)}  ({len(sub)/len(segmented)*100:.0f}% of all records)")
    lines.append(f"**Total {value_col}:** {money(vals.sum(), sym)}")
    lines.append(f"**Average {value_col}:** {money(vals.mean(), sym)}")
    lines.append(f"**Highest {value_col}:** {money(vals.max(), sym)}")
    lines.append(f"**Lowest {value_col}:** {money(vals.min(), sym)}")
    top = sub.assign(_v=vals).sort_values("_v", ascending=False).head(10)
    top_rows = []
    for _, r in top.iterrows():
        label = (str(r["FullName"]) if "FullName" in sub.columns
                 else (str(r[id_col]) if id_col in sub.columns else "record"))
        top_rows.append({"Record": label, value_col: money(r["_v"], sym)})
    return lines, pd.DataFrame(top_rows)


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

def quarantine_rows(df, id_col):
    # Rows that cannot be identified cannot be acted on. Pull them out with a
    # reason instead of dropping them silently; the user can download and fix them.
    if not id_col or id_col not in df.columns:
        return df, None
    work = df.copy()
    ids = work[id_col].astype(str).str.strip()
    reasons = pd.Series("", index=work.index)
    missing = work[id_col].isna() | ids.isin(["", "nan", "None", "NaN"])
    reasons[missing] = f"Missing {id_col}"
    dupes = ids.duplicated(keep="first") & ~missing
    reasons[dupes] = f"Duplicate {id_col} (first occurrence kept)"
    bad = reasons != ""
    if not bad.any():
        return work, None
    rejects = work[bad].copy()
    rejects.insert(0, "QuarantineReason", reasons[bad])
    return work[~bad].copy(), rejects


def apply_trend(segmented, prev_df, id_col):
    # Join the prior period on the ID and compute period over period change in NetADT.
    # Declining means play dropped more than 15 percent; Rising means it grew that much.
    if id_col not in prev_df.columns or "NetADT" not in prev_df.columns:
        return segmented
    prev = prev_df[[id_col, "NetADT"]].copy()
    prev[id_col] = prev[id_col].astype(str).str.strip()
    prev["NetADT_Prev"] = clean_money_series(prev["NetADT"])
    prev = prev.drop(columns=["NetADT"]).drop_duplicates(subset=[id_col], keep="first")
    prev = prev.rename(columns={id_col: "_join_id"})
    out = segmented.copy()
    out["_join_id"] = out[id_col].astype(str).str.strip()
    out = out.merge(prev, on="_join_id", how="left")
    cur = pd.to_numeric(out["NetADT"], errors="coerce")
    out["TrendPct"] = ((cur - out["NetADT_Prev"]) / out["NetADT_Prev"].replace(0, pd.NA) * 100).round(1)
    def label(p):
        if pd.isna(p): return "No prior data"
        if p <= -15: return "Declining"
        if p >= 15: return "Rising"
        return "Stable"
    out["Trend"] = out["TrendPct"].map(label)
    return out.drop(columns=["_join_id"])


def clean_data(df, value_col, consent_cols, flag_col):
    df = df.copy(); report = []; issues = 0
    for col in consent_cols:
        if col in df.columns:
            normalized = df[col].astype(str).str.strip().str.lower()
            df[col] = normalized.map({"yes": "Yes", "no": "No"}).fillna("Unknown")
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
    for col in ["NetADT", "LifetimeValue", "AvgWager", "OfferTotal", "CoinInTotal"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            lines.append(f"{col}: avg {money(s.mean(),sym)}, median {money(s.median(),sym)}, max {money(s.max(),sym)}")
    if "DaysSinceLastVisit" in df.columns:
        s = pd.to_numeric(df["DaysSinceLastVisit"], errors="coerce")
        lines.append(f"DaysSinceLastVisit: avg {s.mean():.0f}, max {s.max():.0f}")
    for col in ["TierRank", "PreferredGame"]:
        if col in df.columns:
            top = df[col].value_counts().head(3)
            lines.append(f"{col} top values: " + ", ".join(f"{i} ({v})" for i, v in top.items()))
    return "\n".join(lines)

# Host worklist: suggested comp tiers scaled to player value. These are
# ILLUSTRATIVE defaults; a real property sets them from its own reinvestment
# policy. The cutoffs are adjustable in the UI so they can be tuned to match.
DEFAULT_COMP_TIERS = [
    (0,   "Free play offer"),
    (50,  "Complimentary meal"),
    (120, "Hotel night on us"),
    (250, "Suite plus event invite"),
]

def comp_tier_for(value, tiers):
    label = tiers[0][1]
    for cutoff, name in tiers:
        if value >= cutoff:
            label = name
    return label

def comp_band_short(value, tiers):
    for i, (cutoff, name) in enumerate(tiers):
        upper = tiers[i+1][0] if i+1 < len(tiers) else None
        if value >= cutoff and (upper is None or value < upper):
            if upper is None:
                return f"{name} band (above {cutoff:,.0f})"
            return f"{name} band ({cutoff:,.0f} to {upper:,.0f})"
    return tiers[0][1] + " band"

def comp_reason_full(value, tiers, sym):
    for i, (cutoff, name) in enumerate(tiers):
        upper = tiers[i+1][0] if i+1 < len(tiers) else None
        if value >= cutoff and (upper is None or value < upper):
            if upper is None:
                return (f"NetADT of {money(value,sym)} is above the top cutoff of "
                        f"{money(cutoff,sym)}, so the {name.lower()} tier fits, the highest offer "
                        "for the most valuable players.")
            return (f"NetADT of {money(value,sym)} falls between {money(cutoff,sym)} and "
                    f"{money(upper,sym)}, so a {name.lower()} matches their value without "
                    "over-investing on a lower-value player or under-rewarding a higher one.")
    return f"NetADT of {money(value,sym)} sits in the entry tier."

def host_reason(row, cut, sym):
    v = row.get("NetADT", 0); r = row.get("DaysSinceLastVisit", None)
    seg = row.get("Segment", "")
    days_known = r is not None and pd.notna(r)
    extra = ""
    tp = row.get("TrendPct")
    if tp is not None and pd.notna(tp) and tp <= -15:
        extra = f" Play is down {abs(tp):.0f}% versus the prior period."
    if seg == "At Risk High Value":
        when = f"not seen in {int(r)} days" if days_known else "gone quiet"
        return (f"High value ({money(v,sym)}/day) and {when}. "
                "Prime win-back call, reach out before they are gone." + extra)
    if seg == "VIP High Value":
        return (f"Top-tier value ({money(v,sym)}/day) and active. "
                "Strengthen the relationship and keep them loyal." + extra)
    return f"Value {money(v,sym)}/day. Worth a personal touch." + extra

def build_host_worklist(segmented, sym, tiers):
    hv = segmented[segmented["Segment"].isin(["At Risk High Value", "VIP High Value"])].copy()
    if hv.empty:
        return None
    hv["NetADT"] = pd.to_numeric(hv["NetADT"], errors="coerce")
    priority = {"At Risk High Value": 0, "VIP High Value": 1}
    hv["_prio"] = hv["Segment"].map(priority).fillna(2)
    has_trend = "TrendPct" in hv.columns
    if has_trend:
        hv = hv.sort_values(["_prio", "TrendPct", "NetADT"],
                            ascending=[True, True, False], na_position="last")
    else:
        hv = hv.sort_values(["_prio", "NetADT"], ascending=[True, False])
    rows = []
    for _, r in hv.iterrows():
        name = (f"{r['FullName']}" if "FullName" in hv.columns else str(r.get("PlayerID", "player")))
        row_out = {
            "Player": name,
            "Segment": r["Segment"],
            "NetADT": money(r.get("NetADT", 0), sym),
            "Days since visit": int(r["DaysSinceLastVisit"]) if pd.notna(r.get("DaysSinceLastVisit")) else "n/a",
            "Suggested comp": comp_tier_for(r.get("NetADT", 0), tiers),
            "Why this comp": comp_band_short(r.get("NetADT", 0), tiers),
            "Why reach out": host_reason(r, None, sym),
        }
        if has_trend:
            tp = r.get("TrendPct")
            row_out["Trend"] = f"{tp:+.0f}%" if pd.notna(tp) else "n/a"
        rows.append(row_out)
    return pd.DataFrame(rows)


def player_pareto(df, sym):
    cols = [c for c in ["OfferSlots", "OfferTables", "OfferFood", "OfferHotel"] if c in df.columns]
    if not cols: return None
    totals = {}
    for c in cols:
        s = clean_money_series(df[c])
        if s.notna().any():
            totals[c] = float(s.sum())
    if not totals: return None
    ser = pd.Series(totals).sort_values(ascending=False)
    p = pd.DataFrame({"Category": ser.index, "RawTotal": ser.values.round(2)})
    p["Cumulative %"] = (p["RawTotal"].cumsum() / p["RawTotal"].sum() * 100).round(1)
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
        df["DebtToEquity"] = (d / e).round(2)
    def assign(row):
        rev = row.get("AnnualRevenue", 0); dte = row.get("DebtToEquity", 0); cold = row.get("MonthsSinceContact", 0)
        if pd.isna(rev): return "Needs Review"
        if dte is not None and not pd.isna(dte) and dte > 1.5:
            return "High Value At Risk" if rev >= cut else "Leveraged Watchlist"
        if rev >= cut: return "Strategic Account"
        if cold is not None and cold > 12: return "Dormant"
        return "Standard"
    df["Segment"] = df.apply(assign, axis=1)
    return df, cut

def explain_company(row, cut, sym):
    rev = row.get("AnnualRevenue", 0); dte = row.get("DebtToEquity", None); cold = row.get("MonthsSinceContact", None); parts = []
    if pd.isna(rev): return "This account has no revenue figure, so it was set aside for review."
    if rev >= cut:
        parts.append(f"Its revenue of {money(rev,sym)} is in the top quartile (cutoff {money(cut,sym)}), so it is a large account.")
    else:
        parts.append(f"Its revenue of {money(rev,sym)} is below the large account cutoff of {money(cut,sym)}.")
    if dte is not None and not pd.isna(dte):
        parts.append(f"Its debt to equity ratio of {dte} is {'above 1.5, which signals heavy leverage.' if dte>1.5 else 'within a healthy range.'}")
    if cold is not None and not pd.isna(cold) and cold > 12:
        parts.append(f"It has not been contacted in {int(cold)} months.")
    parts.append(f"Together that places it in the {row['Segment']} segment.")
    return " ".join(parts)

def company_profile(df, sym):
    lines = [f"Total accounts: {len(df)}"]
    for col in ["AnnualRevenue", "Debt", "Liabilities", "Equity"]:
        if col in df.columns:
            s = clean_money_series(df[col])
            lines.append(f"{col}: avg {money(s.mean(),sym)}, median {money(s.median(),sym)}, max {money(s.max(),sym)}")
    for col in ["DebtToEquity", "CreditScore", "ProductsHeld"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            lines.append(f"{col}: avg {s.mean():.1f}, median {s.median():.1f}, max {s.max():.1f}")
    for col in ["Industry", "RiskTier"]:
        if col in df.columns:
            top = df[col].value_counts().head(3)
            lines.append(f"{col} top values: " + ", ".join(f"{i} ({v})" for i, v in top.items()))
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
        numeric = numeric.drop(columns=[c for c in cols if any(a in lower[c] for a in ["row", "index", "no.", "count", "id"])], errors="ignore")
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
# Autonomous agent (EXPERIMENTAL).
# ============================================================
def agent_tools_for(segmented_holder):
    def t_clean():
        df = segmented_holder["df"]
        cleaned, report, rate = clean_data(df, "NetADT", ["CanEmail", "CanCall"], "ZipCode")
        segmented_holder["df"] = cleaned
        return "Cleaned the data. " + " ".join(report)
    def t_segment():
        df = segmented_holder["df"]
        seg, cut = segment_players(df)
        segmented_holder["segmented"] = seg; segmented_holder["cut"] = cut
        counts = seg["Segment"].value_counts()
        return "Segmented players. Counts: " + counts.to_string().replace("\n", "; ")
    def t_value_table():
        seg = segmented_holder.get("segmented")
        if seg is None: return "No segments yet. Call segment first."
        vt = seg.groupby("Segment").agg(Records=("PlayerID", "count"),
             TotalValue=("NetADT", "sum")).round(2).sort_values("TotalValue", ascending=False)
        segmented_holder["value_table"] = vt
        return "Value by segment (total NetADT): " + vt["TotalValue"].to_string().replace("\n", "; ")
    def t_pareto():
        seg = segmented_holder.get("segmented")
        if seg is None: return "No segments yet. Call segment first."
        p = player_pareto(seg, segmented_holder["sym"])
        if p is None: return "No offer columns available for a Pareto."
        return "Offer concentration: " + p[["Category", "Cumulative %"]].to_string(index=False).replace("\n", "; ")
    return {
        "clean_data": ("Clean and standardize the raw data. Call this first.", t_clean),
        "segment_players": ("Group players into value/recency segments. Call after cleaning.", t_segment),
        "value_by_segment": ("Compute total and average value per segment. Call after segmenting.", t_value_table),
        "pareto": ("Find how concentrated offer spend is across categories.", t_pareto),
    }

def agent_decide(goal, tools, history, sym):
    tool_list = "\n".join(f"- {name}: {desc}" for name, (desc, _) in tools.items())
    sofar = "\n".join(history) if history else "Nothing done yet."
    prompt = (
        f"You are an autonomous data analysis agent. Money is in {sym}. "
        f"Your goal: \"{goal}\".\n\n"
        f"Tools you can call:\n{tool_list}\n\n"
        f"Steps taken so far and their results:\n{sofar}\n\n"
        "Decide the single next action. Reply with ONLY a JSON object, no other text, "
        "in this exact shape:\n"
        '{"thought": "one short sentence on why", "action": "tool_name or FINISH", '
        '"final": "if action is FINISH, your final answer here, else empty"}\n'
        "Use FINISH once you have enough to answer the goal. Do not repeat a tool that already succeeded."
    )
    text, provider = ai_generate(prompt)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, provider
    try:
        return json.loads(m.group(0)), provider
    except Exception:
        return None, provider

def render_agent_mode(df, sym, cur_name, goal):
    st.markdown("<div class='ai-banner'>Experimental autonomous mode. The AI chooses its own "
                "steps to reach your goal. It may be slower and occasionally take an odd path. "
                "This is the agentic core of the project.</div>", unsafe_allow_html=True)

    step_mode = st.radio("How should the agent run?",
        ["Hands-off (run start to finish)", "Step-by-step (approve each action)"], horizontal=True)

    MAX_STEPS = 6

    if st.button("Start the agent", type="primary"):
        st.session_state.agent_history = []
        st.session_state.agent_done = False
        st.session_state.agent_final = ""
        st.session_state.agent_holder = {"df": df.copy(), "sym": sym}
        st.session_state.agent_pending = None
        st.session_state.agent_running = True
        st.rerun()

    if not st.session_state.get("agent_running"):
        return

    history = st.session_state.get("agent_history", [])
    holder = st.session_state.get("agent_holder", {"df": df.copy(), "sym": sym})
    tools = agent_tools_for(holder)
    st.markdown(f"<div class='insight'>Working toward your goal: \u201c{goal}\u201d</div>",
                unsafe_allow_html=True)

    if (not st.session_state.get("agent_done")
            and st.session_state.get("agent_pending") is None
            and len(history) < MAX_STEPS):
        with st.spinner("Agent is deciding the next step..."):
            decision, provider = agent_decide(goal, tools, history, sym)
        if decision is None:
            st.session_state.agent_running = False
            st.error("The agent could not produce a valid decision (the AI may be rate-limited). "
                     "Press Start to try again.")
            if history:
                st.markdown("### Steps taken before the stop")
                for i, h in enumerate(history, 1):
                    st.markdown(f"{i}. {h}")
            return
        st.session_state.agent_pending = decision
        if provider == "Gemini (backup)":
            st.markdown("<div class='ai-banner'>Decision made by backup AI (Gemini).</div>",
                        unsafe_allow_html=True)

    pending = st.session_state.get("agent_pending")
    if pending is not None and not st.session_state.get("agent_done"):
        thought = pending.get("thought", "")
        action = pending.get("action", "")
        st.markdown(f"**Agent thinking:** {thought}")
        if action == "FINISH":
            st.session_state.agent_done = True
            st.session_state.agent_final = pending.get("final", "Done.")
            st.session_state.agent_pending = None
        elif action in tools:
            st.markdown(f"**Proposed action:** call `{action}`")
            go = True
            if step_mode.startswith("Step"):
                go = st.button(f"Approve and run `{action}`")
            if go:
                result = tools[action][1]()
                history.append(f"{action} -> {result}")
                st.session_state.agent_history = history
                st.session_state.agent_pending = None
                st.rerun()
        else:
            history.append(f"INVALID action '{action}' ignored.")
            st.session_state.agent_history = history
            st.session_state.agent_pending = None
            st.warning(f"The agent proposed an unknown action ('{action}'). Asking it to choose again.")
            st.rerun()

    if st.session_state.get("agent_done"):
        st.success("Agent finished.")
        st.markdown("### The agent's conclusion")
        st.markdown(st.session_state.get("agent_final", ""))
    elif len(history) >= MAX_STEPS:
        st.info("Reached the step limit. Here is what the agent gathered:")

    if history:
        st.markdown("### Steps the agent took")
        for i, h in enumerate(history, 1):
            st.markdown(f"{i}. {h}")


# ============================================================
# Sidebar: chat AND record lookup.
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
                answer, provider = chat_answer(role, st.session_state.chat_context,
                                     st.session_state.chat_history, sym, cur_name)
                if provider == "Gemini (backup)":
                    answer = "_(answered by backup AI)_\n\n" + answer
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.markdown("<div id='chat-scroll'>", unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.caption("Flip this on to ask in plain English about your results.")

        st.markdown("---")
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
                rec, provider = get_record_recommendation(role, name_label, detail_text,
                        goal or "improve outcomes", sym, cur_name)
            if provider == "Gemini (backup)":
                st.markdown("<div class='ai-banner'>Primary AI was busy, answered by backup (Gemini).</div>", unsafe_allow_html=True)
            elif provider == "none":
                st.markdown("<div class='ai-banner'>AI unavailable right now. The analytics above are still accurate.</div>", unsafe_allow_html=True)
            st.markdown("**Recommended action:**")
            st.markdown(rec)


# ================= MAIN UI =================

st.markdown("<div class='eyebrow'>Mode</div>", unsafe_allow_html=True)
mode = st.radio("Mode", ["Players / Customers", "Companies / Accounts", "Custom / Any File",
                         "Autonomous Agent", "How It Was Built"],
                horizontal=True, label_visibility="collapsed")

st.caption("Players / Customers: individual people like casino players or retail customers.  \u2022  "
           "Companies / Accounts: business accounts with revenue, debt, equity.  \u2022  "
           "Custom / Any File: upload any spreadsheet and map your own columns.  \u2022  "
           "Autonomous Agent: experimental, the AI plans its own analysis steps.  \u2022  "
           "How It Was Built: the architecture, stack, and engineering decisions behind this tool.")

if mode == "Players / Customers":
    cfg = {"title": "Player Segmentation Agent", "sample": "sample_players.csv", "template": "player_template.csv",
           "id_col": "PlayerID", "value_col": "NetADT", "consent": ["CanEmail", "CanCall"], "flag": "ZipCode",
           "role": "a casino marketing analyst",
           "placeholder": "Example: Find high value players who are slipping away so I can win them back",
           "default_goal": "improve overall player value and retention"}
elif mode == "Companies / Accounts":
    cfg = {"title": "Account Segmentation Agent", "sample": "sample_companies.csv", "template": "company_template.csv",
           "id_col": "AccountID", "value_col": "AnnualRevenue", "consent": [], "flag": "Industry",
           "role": "a commercial banking portfolio analyst",
           "placeholder": "Example: Find large accounts that are over leveraged and need a check in",
           "default_goal": "grow the portfolio while managing risk"}
elif mode == "Autonomous Agent":
    cfg = {"title": "Autonomous Agent", "sample": "sample_players.csv", "template": None,
           "id_col": "PlayerID", "value_col": "NetADT", "consent": ["CanEmail", "CanCall"], "flag": "ZipCode",
           "role": "an autonomous data analysis agent",
           "placeholder": "Example: Find which players are slipping away and worth winning back",
           "default_goal": "find the most valuable players at risk and what to do"}
else:
    cfg = {"title": "Custom Segmentation Agent", "sample": None, "template": None,
           "id_col": None, "value_col": None, "consent": [], "flag": None,
           "role": "a data analyst",
           "placeholder": "Example: Find my highest value records that have gone quiet",
           "default_goal": "find the most valuable records and what to do about them"}

if mode == "How It Was Built":
    st.title("How It Was Built")
    st.caption("The architecture, stack, and engineering decisions behind this tool. "
               "Written for anyone curious how it works under the hood.")

    st.markdown("<div class='head-cyan'>What this is</div>", unsafe_allow_html=True)
    st.markdown(
        "An end to end data application: it ingests raw spreadsheets, validates and cleans them, "
        "transforms records into value and recency segments, aggregates the results into "
        "analytics, and serves plain English recommendations through a large language model layer. "
        "It was designed around casino player development (NetADT, theoretical, host worklists) "
        "and then generalized so the same pipeline handles bank accounts, retail customers, or any "
        "records with value dataset.")

    divider()
    st.markdown("<div class='head-violet'>The stack</div>", unsafe_allow_html=True)
    st.markdown(
        "**Python** for everything. **pandas** for ingestion, cleaning, and transformation. "
        "**DuckDB** for in-app SQL queries over the segmented output. "
        "**Altair** for the charts. **Streamlit** for the interface and session state. "
        "**Groq (Llama 3.3 70B)** as the primary LLM with **Google Gemini 1.5 Flash** as automatic "
        "failover. Deployed on **Streamlit Community Cloud**, continuously redeployed from "
        "**GitHub** on every push, with API keys held in the platform's secrets manager, never in code.")

    divider()
    st.markdown("<div class='head-amber'>The data pipeline</div>", unsafe_allow_html=True)
    st.markdown(
        "**1. Ingest.** Accepts CSV or Excel. In custom mode, schema detection guesses the ID, "
        "value, and recency columns using keyword heuristics plus a statistical fallback, gets an "
        "AI second opinion on what each column appears to be, and asks the user to confirm. "
        "Nothing runs on an unconfirmed mapping.")
    st.markdown(
        "**2. Validate and clean.** Real exports are never clean, so the cleaning step assumes the "
        "worst: money parsed out of five different formats with a regex, missing values filled "
        "with the median (robust to skew), consent flags standardized, blanks flagged rather than "
        "silently dropped. Rows that fail identity validation (missing or duplicate IDs) are "
        "quarantined to a downloadable rejects file with a reason per row instead of being "
        "dropped silently. Every action is reported back to the user in a cleaning summary.")
    st.markdown(
        "**3. Transform.** Records are segmented by quantile cuts on the value column crossed "
        "with recency thresholds. Quantiles instead of fixed dollar cutoffs means the same logic "
        "works whether the file holds casino players or nine figure corporate accounts.")
    st.markdown(
        "**4. Aggregate.** Per segment rollups, value concentration (Pareto with cumulative "
        "percent), rank based distribution bands, and the host worklist with value scaled comp "
        "tiers. When a prior period file is provided, a period over period join computes each "
        "player's trend and flags who is declining. All computed locally in pandas, instant, "
        "and available even when the AI is not.")
    st.markdown(
        "**5. Serve.** The LLM layer turns the computed numbers into recommendations tied to the "
        "user's stated goal, answers questions in a chat grounded only in this dataset, and in "
        "agent mode plans its own sequence of pipeline steps.")

    divider()
    st.markdown("<div class='head-cyan'>Engineering decisions worth naming</div>", unsafe_allow_html=True)
    st.markdown(
        "**Provider failover.** Every AI call tries Groq first and falls back to Gemini on any "
        "failure, rate limits included. If both fail, the app degrades gracefully: the computed "
        "analytics still render and the user is told plainly that the AI is offline, instead of "
        "the page crashing.")
    st.markdown(
        "**Caching against rate limits.** The free LLM tiers meter tokens per day. Column "
        "descriptions and per record recommendations are cached so repeated interactions cost "
        "zero additional tokens. This constraint shaped the architecture: everything that can be "
        "computed without the AI, is.")
    st.markdown(
        "**A validated agent loop.** In autonomous mode the model returns a structured JSON "
        "decision naming its next tool. The loop extracts that JSON even when the model wraps it "
        "in prose, validates the chosen action against the real tool list, and rejects anything "
        "unknown instead of executing it. Autonomy with a guardrail.")
    st.markdown(
        "**Run observability.** Every pipeline run logs rows in and rows out per stage with "
        "timings, visible in the app, so a shrinking row count or a slow stage is diagnosable "
        "instead of invisible.")
    st.markdown(
        "**Secrets management.** Keys load from the deployment platform's secrets store in the "
        "cloud and from a gitignored .env locally. No credential has ever been committed.")

    divider()
    st.markdown("<div class='head-violet'>Honest constraints</div>", unsafe_allow_html=True)
    st.markdown(
        "The comp tiers in the host worklist are illustrative and adjustable, not a property's "
        "actual reinvestment math. The demo data is generated (seeded, reproducible) rather than "
        "real player data, no real PII touches this app. And the free tier LLM quota means the AI "
        "features can temporarily go quiet under heavy use while the computed analytics keep "
        "working. Each of those was a deliberate scoping decision, and knowing where the edges "
        "are is part of the build.")

    divider()
    st.markdown("<div class='head-amber'>Built by</div>", unsafe_allow_html=True)
    st.markdown(
        "Brian Cabrera, a software engineer and data professional with casino industry experience "
        "on the database side. Designed and built end to end, from the cleaning logic to the "
        "deployment. The source is public: "
        "[github.com/BrianDCab/agentic-workflow-platform](https://github.com/BrianDCab/agentic-workflow-platform)")

    render_sidebar()
    st.markdown(
        "<div id='bc-footer'>"
        "Built by <a href='https://briancabrera.io' target='_blank'>Brian Cabrera</a>"
        "<span class='sep'>|</span><a href='https://www.linkedin.com/in/briandacellcabrera/' target='_blank'>LinkedIn</a>"
        "<span class='sep'>|</span><a href='https://github.com/BrianDCab' target='_blank'>GitHub</a>"
        "<span class='sep'>|</span>Powered by AI</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if mode == "Autonomous Agent":
    st.markdown(f"<h1 style='display:inline'>{cfg['title']}</h1>"
                "<span class='beta-badge'>EXPERIMENTAL</span>", unsafe_allow_html=True)
    st.caption("Upload player data or load the demo, give the agent a goal, and watch it plan its own analysis.")

    currency = st.selectbox("Currency for money figures", ["USD $", "EUR \u20ac", "GBP \u00a3", "CAD $", "AUD $"], index=0)
    cur_name = currency.split()[0]; sym = currency.split()[-1]

    st.markdown("<div class='goal-label'>What should the agent figure out?</div>", unsafe_allow_html=True)
    agent_goal = st.text_input("Agent goal", placeholder=cfg["placeholder"], label_visibility="collapsed")

    a1, a2 = st.columns(2)
    a_demo = a1.button("Load Prerendered Demo Data")
    if cfg["sample"] and os.path.exists(cfg["sample"]):
        with open(cfg["sample"], "rb") as f:
            a2.download_button("Download Sample CSV", f, cfg["sample"], "text/csv")
    a_uploaded = st.file_uploader("Or upload player data (CSV or Excel)", type=["csv", "xlsx"], key="agent_upload")

    if a_uploaded is not None:
        st.session_state.agent_df = pd.read_csv(a_uploaded) if a_uploaded.name.endswith(".csv") else pd.read_excel(a_uploaded)
    elif a_demo and cfg["sample"] and os.path.exists(cfg["sample"]):
        st.session_state.agent_df = pd.read_csv(cfg["sample"])
        st.success(f"Demo data loaded from {cfg['sample']}.")

    agent_df = st.session_state.get("agent_df")
    if agent_df is None:
        st.info("Load the demo or upload player data to begin.")
    else:
        with st.expander("Preview the data"):
            st.dataframe(agent_df.head(10), width='stretch')
        divider()
        render_agent_mode(agent_df, sym, cur_name, agent_goal.strip() or cfg["default_goal"])

    render_sidebar()
    st.markdown(
        "<div id='bc-footer'>"
        "Built by <a href='https://briancabrera.io' target='_blank'>Brian Cabrera</a>"
        "<span class='sep'>|</span><a href='https://www.linkedin.com/in/briandacellcabrera/' target='_blank'>LinkedIn</a>"
        "<span class='sep'>|</span><a href='https://github.com/BrianDCab' target='_blank'>GitHub</a>"
        "<span class='sep'>|</span>Powered by AI</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if mode == "Custom / Any File":
    st.markdown(f"<h1 style='display:inline'>{cfg['title']}</h1>"
                "<span class='beta-badge'>BETA</span>", unsafe_allow_html=True)
else:
    st.title(cfg["title"])
st.caption("Upload data, or load the demo. Get clean segments, clear visuals, and tailored recommendations.")

st.markdown("<div class='step-chip'><span class='step-num'>1</span><span class='step-title'>Settings</span></div>", unsafe_allow_html=True)
check_mode = st.radio("How carefully should the agent check with you?",
    ["Smart gates (pause only when something looks off)", "Pause at every stage", "Run straight through (no pauses)"])

currency = st.selectbox("Currency for money figures", ["USD $", "EUR \u20ac", "GBP \u00a3", "CAD $", "AUD $"], index=0)
cur_name = currency.split()[0]; sym = currency.split()[-1]

st.markdown("<div class='goal-label'>What are you looking to do?</div>", unsafe_allow_html=True)
st.caption("Type your goal in plain English here, like 'win back my high value players who went quiet'. "
           "This steers the recommendations. For free-form questions after running, use the chat in the left sidebar. "
           "This box is not a chat.")
user_goal = st.text_input("What are you looking to do?", placeholder=cfg["placeholder"], label_visibility="collapsed")

st.markdown("<div class='step-chip'><span class='step-num'>2</span><span class='step-title'>Data</span></div>", unsafe_allow_html=True)
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

uploaded = st.file_uploader("Upload your own (CSV or Excel)", type=["csv", "xlsx"])

if uploaded is not None:
    st.session_state.df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    st.session_state.ran = False
elif demo_clicked and cfg["sample"] and os.path.exists(cfg["sample"]):
    st.session_state.df = pd.read_csv(cfg["sample"])
    st.session_state.ran = False
    st.success(f"Demo data loaded from {cfg['sample']}.")

if mode == "Players / Customers":
    st.caption("Optional: add last period's file to unlock trend analysis. It needs matching "
               "PlayerID and NetADT columns; the demo previous period pairs with the demo data.")
    p1, p2 = st.columns(2)
    prev_demo = p1.button("Load demo previous period")
    prev_up = p2.file_uploader("Previous period file (CSV or Excel)", type=["csv", "xlsx"],
                               key="prev_upload")
    if prev_up is not None:
        st.session_state.prev_df = pd.read_csv(prev_up) if prev_up.name.endswith(".csv") else pd.read_excel(prev_up)
        st.success("Previous period loaded. Run the agent to see trends.")
    elif prev_demo:
        if os.path.exists("sample_players_prev.csv"):
            st.session_state.prev_df = pd.read_csv("sample_players_prev.csv")
            st.success("Demo previous period loaded. Run the agent to see trends.")
        else:
            st.warning("sample_players_prev.csv not found. Run generate_prior_period.py once to create it.")
    if st.session_state.get("prev_df") is not None:
        st.caption(f"Previous period on deck: {len(st.session_state.prev_df)} rows. Trends will "
                   "compute on the next run.")
        if st.button("Clear previous period"):
            st.session_state.prev_df = None
            st.rerun()

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
        st.caption(f"Example value: {sample_of(custom_id)}" + (f"  \u2022  AI thinks: {descs.get(custom_id,'')}" if descs.get(custom_id) else ""))

        st.markdown("**Value column** <span style='color:rgba(230,241,255,0.5);font-weight:400'>"
                    "the one number that says how valuable each record is</span>", unsafe_allow_html=True)
        custom_value = st.selectbox("Value column", cols, index=cols.index(val_g) if val_g in cols else 0, label_visibility="collapsed")
        st.caption(f"Example value: {sample_of(custom_value)}" + (f"  \u2022  AI thinks: {descs.get(custom_value,'')}" if descs.get(custom_value) else ""))

        st.markdown("**Recency column (optional)** <span style='color:rgba(230,241,255,0.5);font-weight:400'>"
                    "how long since each record was last active, a date or a days/months since number</span>", unsafe_allow_html=True)
        rec_opts = [none_opt] + cols
        custom_recency = st.selectbox("Recency column", rec_opts, index=rec_opts.index(rec_g) if rec_g in rec_opts else 0, label_visibility="collapsed")
        if custom_recency != none_opt:
            st.caption(f"Example value: {sample_of(custom_recency)}" + (f"  \u2022  AI thinks: {descs.get(custom_recency,'')}" if descs.get(custom_recency) else ""))
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

    st.markdown("<div class='step-chip'><span class='step-num'>3</span><span class='step-title'>Run</span></div>", unsafe_allow_html=True)
    if st.button("Run the agent", type="primary"):
        if mode == "Custom / Any File" and (custom_value is None):
            st.warning("Please map a Value column in the Column mapping section above before running.")
        else:
            run_log = []
            rejects = None
            rows_in = len(df)
            if mode == "Players / Customers":
                t0 = time.perf_counter()
                kept, rejects = quarantine_rows(df, cfg["id_col"])
                run_log.append({"Stage": "Quarantine", "Rows in": rows_in, "Rows out": len(kept),
                                "Seconds": round(time.perf_counter() - t0, 3)})
                t0 = time.perf_counter()
                cleaned, clean_report, missing_rate = clean_data(kept, cfg["value_col"], cfg["consent"], cfg["flag"])
                run_log.append({"Stage": "Clean", "Rows in": len(kept), "Rows out": len(cleaned),
                                "Seconds": round(time.perf_counter() - t0, 3)})
                t0 = time.perf_counter()
                segmented, cut = segment_players(cleaned)
                prev_df = st.session_state.get("prev_df")
                if prev_df is not None:
                    segmented = apply_trend(segmented, prev_df, cfg["id_col"])
                run_log.append({"Stage": "Segment + Trend" if prev_df is not None else "Segment",
                                "Rows in": len(cleaned), "Rows out": len(segmented),
                                "Seconds": round(time.perf_counter() - t0, 3)})
                profile = player_profile(segmented, sym); pareto = player_pareto(segmented, sym)
                meanings = PLAYER_MEANINGS; health = player_health(df)
            elif mode == "Companies / Accounts":
                t0 = time.perf_counter()
                kept, rejects = quarantine_rows(df, cfg["id_col"])
                run_log.append({"Stage": "Quarantine", "Rows in": rows_in, "Rows out": len(kept),
                                "Seconds": round(time.perf_counter() - t0, 3)})
                t0 = time.perf_counter()
                cleaned, clean_report, missing_rate = clean_data(kept, cfg["value_col"], cfg["consent"], cfg["flag"])
                run_log.append({"Stage": "Clean", "Rows in": len(kept), "Rows out": len(cleaned),
                                "Seconds": round(time.perf_counter() - t0, 3)})
                t0 = time.perf_counter()
                segmented, cut = segment_companies(cleaned)
                run_log.append({"Stage": "Segment", "Rows in": len(cleaned), "Rows out": len(segmented),
                                "Seconds": round(time.perf_counter() - t0, 3)})
                profile = company_profile(segmented, sym); pareto = None
                meanings = COMPANY_MEANINGS; health = company_health(df)
            else:
                t0 = time.perf_counter()
                kept, rejects = quarantine_rows(df, cfg["id_col"])
                run_log.append({"Stage": "Quarantine", "Rows in": rows_in, "Rows out": len(kept),
                                "Seconds": round(time.perf_counter() - t0, 3)})
                t0 = time.perf_counter()
                segmented, cut = segment_custom(kept, custom_value, custom_recency, custom_approach)
                run_log.append({"Stage": "Clean + Segment", "Rows in": len(kept), "Rows out": len(segmented),
                                "Seconds": round(time.perf_counter() - t0, 3)})
                clean_report = [f"Cleaned the {custom_value} column into numbers, stripping symbols and commas."]
                missing_rate = clean_money_series(kept[custom_value]).isna().mean()
                profile = custom_profile(segmented, custom_value, sym); pareto = None
                meanings = CUSTOM_MEANINGS; health = custom_health(df, custom_value)
            t0 = time.perf_counter()
            counts = segmented["Segment"].value_counts()
            value_table = segmented.groupby("Segment").agg(
                Records=(cfg["id_col"], "count"),
                TotalValue=(cfg["value_col"], "sum"),
                AvgValue=(cfg["value_col"], "mean"),
            ).round(2).sort_values("TotalValue", ascending=False)
            run_log.append({"Stage": "Aggregate", "Rows in": len(segmented), "Rows out": len(segmented),
                            "Seconds": round(time.perf_counter() - t0, 3)})
            with st.spinner("Generating recommendations..."):
                goal = user_goal.strip() or cfg["default_goal"]
                t0 = time.perf_counter()
                recs, rec_provider = get_recommendations(cfg["role"], profile, counts.to_string(),
                                           value_table["TotalValue"].to_string(), goal, sym, cur_name)
                run_log.append({"Stage": f"AI recommendations ({rec_provider})", "Rows in": "-",
                                "Rows out": "-", "Seconds": round(time.perf_counter() - t0, 3)})
            roster = build_roster(segmented, cfg["id_col"], cfg["value_col"])
            chat_context = (f"User goal: {goal}\n\nData profile:\n{profile}\n\n"
                            f"Segment headcounts:\n{counts.to_string()}\n\n"
                            f"Total value by segment:\n{value_table['TotalValue'].to_string()}\n\n"
                            f"Recommendations already given:\n{recs}\n\n"
                            f"RECORD ROSTER (one row per record):\n{roster}")
            st.session_state.update(dict(ran=True, segmented=segmented, cut=cut, meanings=meanings,
                pareto=pareto, health=health, counts=counts, value_table=value_table, recs=recs,
                rec_provider=rec_provider, missing_rate=missing_rate, clean_report=clean_report,
                rejects=rejects, run_log=run_log,
                sym=sym, cur_name=cur_name, mode_run=mode,
                chat_context=chat_context, chat_role=cfg["role"], chat_history=[], goal=goal,
                id_col=cfg["id_col"], value_col=cfg["value_col"]))

    if st.session_state.get("ran") and st.session_state.get("mode_run") == mode:
        meanings = st.session_state.meanings; pareto = st.session_state.pareto
        health = st.session_state.health; counts = st.session_state.counts
        value_table = st.session_state.value_table; recs = st.session_state.recs
        sym = st.session_state.sym; cur_name = st.session_state.cur_name
        segmented = st.session_state.segmented; value_col = st.session_state.value_col
        id_col = st.session_state.id_col

        divider()
        m1, m2, m3, m4 = st.columns(4)
        _total_val = clean_money_series(segmented[value_col]).sum()
        _avg_val = clean_money_series(segmented[value_col]).mean()
        m1.metric("Records", f"{len(segmented):,}")
        m2.metric("Total value", money(_total_val, sym))
        m3.metric("Avg value", money(_avg_val, sym))
        m4.metric("Segments", f"{segmented['Segment'].nunique()}")

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
        st.subheader("Data quality")
        _rej = st.session_state.get("rejects")
        _rej_n = 0 if _rej is None else len(_rej)
        q1, q2, q3 = st.columns(3)
        q1.metric("Rows received", f"{len(segmented) + _rej_n:,}")
        q2.metric("Rows kept", f"{len(segmented):,}")
        q3.metric("Rows quarantined", f"{_rej_n:,}")
        if _rej_n:
            st.caption("Quarantined rows failed identity validation and were excluded from the "
                       "analysis, each with a reason. Download, fix, and re-upload.")
            st.dataframe(_rej.head(20).astype(str), width='stretch', hide_index=True)
            st.download_button("Download quarantined rows (CSV)",
                               _rej.to_csv(index=False).encode("utf-8"),
                               "quarantined_rows.csv", "text/csv")
        else:
            st.caption("No rows were quarantined. Every record passed identity validation.")
        with st.expander("Pipeline run log (rows in and out per stage, with timings)"):
            _log = st.session_state.get("run_log", [])
            if _log:
                st.dataframe(pd.DataFrame(_log).astype(str), width='stretch', hide_index=True)
            else:
                st.caption("No run log for this session yet.")

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
        st.subheader("Per-segment breakdown")
        st.caption("Overview of every segment side by side, then pick one to dig into.")
        overview = segment_overview_table(segmented, value_col, id_col, sym)
        st.dataframe(overview, width='stretch', hide_index=True)
        seg_choice = st.selectbox("Dig into a segment", overview["Segment"].tolist())
        lines, top_df = segment_deep_dive(segmented, seg_choice, value_col, id_col, sym, meanings)
        for ln in lines:
            st.markdown(ln)
        if not top_df.empty:
            st.caption("Top records in this segment by value:")
            st.dataframe(top_df, width='stretch', hide_index=True)

        if mode == "Players / Customers" and "Trend" in segmented.columns:
            divider()
            st.markdown("<div class='head-violet'>Trends vs prior period</div>", unsafe_allow_html=True)
            st.caption("Compared against the previous period file you loaded. Declining means play "
                       "dropped more than 15%; Rising means it grew that much.")
            _tc = segmented["Trend"].value_counts()
            r1, r2, r3 = st.columns(3)
            r1.metric("Rising", int(_tc.get("Rising", 0)))
            r2.metric("Stable", int(_tc.get("Stable", 0)))
            r3.metric("Declining", int(_tc.get("Declining", 0)))
            _dec = segmented[segmented["Trend"] == "Declining"].copy()
            if not _dec.empty:
                _dec["TrendPct"] = pd.to_numeric(_dec["TrendPct"], errors="coerce")
                _top_dec = _dec.sort_values("TrendPct").head(10)
                _tbl_cols = [c for c in ["FullName", "PlayerID", "Segment"] if c in _top_dec.columns]
                _tbl = _top_dec[_tbl_cols].copy()
                _tbl["Prior NetADT"] = _top_dec["NetADT_Prev"].map(lambda x: money(x, sym))
                _tbl["Current NetADT"] = pd.to_numeric(_top_dec["NetADT"], errors="coerce").map(lambda x: money(x, sym))
                _tbl["Change"] = _top_dec["TrendPct"].map(lambda x: f"{x:+.0f}%")
                st.caption("Biggest declines, the players to reach first:")
                st.dataframe(_tbl.astype(str), width='stretch', hide_index=True)
                insight(f"{len(_dec)} players are declining versus the prior period. Declining "
                        "high-value players are the most urgent win-back calls, they sort to the "
                        "top of the Host Worklist below with their trend shown.")
            else:
                insight("No players are declining versus the prior period.")

        if mode == "Players / Customers":
            divider()
            st.markdown("<div class='head-amber'>Host Worklist</div>", unsafe_allow_html=True)
            st.caption("Who a host should reach out to, ranked. At-risk high-value players come first. "
                       "Suggested comps are illustrative, tune the cutoffs to match your property's reinvestment policy.")
            tiers = list(DEFAULT_COMP_TIERS)
            with st.expander("Adjust comp tier cutoffs (NetADT thresholds)"):
                st.caption("Set the NetADT at which each comp tier kicks in. Defaults shown.")
                c_meal = st.number_input("Meal tier starts at", value=50, step=10)
                c_hotel = st.number_input("Hotel tier starts at", value=120, step=10)
                c_suite = st.number_input("Suite + event tier starts at", value=250, step=10)
                tiers = [(0, "Free play offer"), (c_meal, "Complimentary meal"),
                         (c_hotel, "Hotel night on us"), (c_suite, "Suite plus event invite")]
            worklist = build_host_worklist(segmented, sym, tiers)
            if worklist is not None:
                st.markdown(
                    f"<div class='insight'>How comps are suggested: "
                    f"<b>Free play</b> under {money(tiers[1][0],sym)}, "
                    f"<b>meal</b> {money(tiers[1][0],sym)} to {money(tiers[2][0],sym)}, "
                    f"<b>hotel night</b> {money(tiers[2][0],sym)} to {money(tiers[3][0],sym)}, "
                    f"<b>suite plus event</b> above {money(tiers[3][0],sym)}. "
                    "Higher daily value earns a larger offer.</div>",
                    unsafe_allow_html=True)
                st.dataframe(worklist, width='stretch', hide_index=True)
                wl_csv = worklist.to_csv(index=False).encode("utf-8")
                st.download_button("Download host worklist (CSV)", wl_csv, "host_worklist.csv", "text/csv")

                with st.expander("Why does a specific player get their comp tier?"):
                    pick_name = st.selectbox("Pick a player from the worklist", worklist["Player"].tolist())
                    if "FullName" in segmented.columns:
                        prow = segmented[segmented["FullName"] == pick_name]
                    elif "PlayerID" in segmented.columns:
                        prow = segmented[segmented["PlayerID"].astype(str) == pick_name]
                    else:
                        prow = segmented.iloc[0:0]
                    if not prow.empty:
                        pv = pd.to_numeric(prow.iloc[0].get("NetADT", 0), errors="coerce")
                        st.markdown(f"**{pick_name}**: {comp_reason_full(pv if pd.notna(pv) else 0, tiers, sym)}")
            else:
                st.info("No high-value players found to put on a host worklist for this dataset.")

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
            st.dataframe(show[["Category", "Total Offer Dollars", "Cumulative %"]], width='stretch', hide_index=True)
            insight(pareto_insight(pareto, sym))

        divider()
        st.subheader("Look up an individual record")
        st.caption("Open the left sidebar to search any record by name or ID, see why it landed in its segment, "
                   "and get a tailored AI action. It lives in the sidebar so it never disturbs this page.")

        divider()
        st.subheader("Recommended actions")
        _goal_shown = st.session_state.get("goal", "")
        if _goal_shown:
            st.markdown(f"<div class='insight'>Based on what you asked: \u201c{_goal_shown}\u201d</div>",
                        unsafe_allow_html=True)
        st.caption("Grounded in your uploaded data.")
        if st.session_state.get("rec_provider") == "Gemini (backup)":
            st.markdown("<div class='ai-banner'>Primary AI was busy, these were generated by the backup (Gemini).</div>", unsafe_allow_html=True)
        elif st.session_state.get("rec_provider") == "none":
            st.markdown("<div class='ai-banner'>AI was unavailable, so no recommendations were generated. The analytics above are still accurate.</div>", unsafe_allow_html=True)
        st.markdown(format_recommendations(recs), unsafe_allow_html=True)

        divider()
        st.subheader("Glossary")
        with st.expander("Key terms explained"):
            for k, v in GLOSSARY.items():
                st.markdown(f"**{k}.** {v}")

        divider()
        st.markdown("<div class='head-cyan'>Query with SQL</div>", unsafe_allow_html=True)
        st.caption("Your segmented output is loaded into an in-memory DuckDB table named records. "
                   "Write any SQL against it and download the result.")
        _default_sql = (f'SELECT Segment, COUNT(*) AS records, ROUND(SUM("{value_col}"), 2) AS total_value\n'
                        'FROM records\n'
                        'GROUP BY Segment\n'
                        'ORDER BY total_value DESC')
        sql_text = st.text_area("SQL query", value=_default_sql, height=140)
        if st.button("Run query"):
            try:
                import duckdb
                con = duckdb.connect()
                con.register("records", segmented)
                st.session_state.sql_result = con.execute(sql_text).df()
                st.session_state.sql_error = None
            except ImportError:
                st.session_state.sql_result = None
                st.session_state.sql_error = ("DuckDB is not installed. Add a duckdb line to "
                                              "requirements.txt for the cloud deploy, or run "
                                              "pip install duckdb locally, then restart the app.")
            except Exception as q_err:
                st.session_state.sql_result = None
                st.session_state.sql_error = f"Query failed: {q_err}"
        if st.session_state.get("sql_error"):
            st.error(st.session_state.sql_error)
        elif st.session_state.get("sql_result") is not None:
            st.dataframe(st.session_state.sql_result, width='stretch', hide_index=True)
            st.download_button("Download query result (CSV)",
                               st.session_state.sql_result.to_csv(index=False).encode("utf-8"),
                               "query_result.csv", "text/csv")

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