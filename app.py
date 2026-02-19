import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import numpy as np
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
CSV_PATH    = "data/bets.csv"
SHEET_NAME  = "Smart Money Bets"
CREDS_FILE  = "creds.json"
UNIT_SIZE   = 100
DFS_BOOKS   = ['PrizePicks', 'Betr', 'Dabble', 'Underdog', 'Sleeper', 'Draftkings6']
VALID_LEAGUES = ['NBA', 'NFL', 'NHL', 'NCAAB', 'NCAAF', 'Tennis', 'UFC']

# Tier system constants (mirrors tracker)
GOOD_ODDS_MIN, GOOD_ODDS_MAX = -150, 499
BAD_ODDS_MIN,  BAD_ODDS_MAX  = 500,  999
GOOD_LIQ_MIN,  GOOD_LIQ_MAX  = 2000, 10000
PRIME_HOURS      = {6, 13, 22}
CONSENSUS_THRESHOLD = 3

TIER_ORDER  = ['DIAMOND', 'GOLD', 'SILVER', 'STANDARD', 'WATCH']
TIER_COLORS = {
    'DIAMOND': '#00BFFF',
    'GOLD':    '#D4AF37',
    'SILVER':  '#C0C0C0',
    'STANDARD':'#2ECC71',
    'WATCH':   '#95A5A6',
}
TIER_EMOJI = {
    'DIAMOND': '💎', 'GOLD': '🥇', 'SILVER': '🥈',
    'STANDARD': '🔥', 'WATCH': '👁️',
}

st.set_page_config(
    page_title="Smart Money Tracker v4",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — dark terminal aesthetic
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.stMetric { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 16px; }
.stMetric label { color: #8b949e !important; font-size: 11px !important; letter-spacing: 2px; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace !important; }
.stMetric [data-testid="metric-container"] > div:nth-child(2) { font-family: 'IBM Plex Mono', monospace; font-size: 28px !important; font-weight: 700; }

.tier-badge {
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700;
    letter-spacing: 1px;
}
.preset-section { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
.insight-card { background: #161b22; border-left: 3px solid #58a6ff; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; font-size: 13px; }

div[data-testid="stTab"] button { font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 1px; }
.stDataFrame { border: 1px solid #21262d; border-radius: 8px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def _get_gspread_client():
    """
    Returns an authorised gspread client.
    - Deployed: reads credentials from st.secrets["gcp_service_account"]
    - Local dev: falls back to creds.json on disk
    
    To set up Streamlit Cloud secrets, go to your app's Settings → Secrets
    and paste the entire contents of creds.json under [gcp_service_account].
    """
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # 1. Try Streamlit Secrets (deployed environment)
    # Secrets are stored under [connections.gsheets] in Streamlit Cloud.
    # We exclude non-credential keys like 'type' and 'universe_domain'.
    CRED_KEYS = {
        "project_id", "private_key_id", "private_key", "client_email",
        "client_id", "auth_uri", "token_uri",
        "auth_provider_x509_cert_url", "client_x509_cert_url",
    }
    for secret_key in ("connections.gsheets", "gcp_service_account"):
        try:
            raw = dict(st.secrets[secret_key])
            creds_dict = {k: v for k, v in raw.items() if k in CRED_KEYS}
            creds_dict["type"] = "service_account"
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            return gspread.authorize(creds)
        except KeyError:
            continue
        except Exception:
            raise

    # 2. Fall back to local creds.json (local dev)
    if os.path.exists(CREDS_FILE):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
        return gspread.authorize(creds)

    raise FileNotFoundError(
        "No credentials found. Either:\n"
        "• Add [gcp_service_account] to your Streamlit Secrets (deployed), or\n"
        f"• Place creds.json in the project root (local dev)"
    )

def _parse_timestamps(df):
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def fetch_from_cloud():
    """
    Pull latest data from Google Sheets via gspread (same auth as Push).
    Saves local CSV cache and stores the DataFrame in session_state.
    Returns (df, error_string_or_None).
    """
    try:
        client  = _get_gspread_client()
        sheet   = client.open(SHEET_NAME).sheet1
        records = sheet.get_all_records()
        df = pd.DataFrame(records)
        df = _parse_timestamps(df)
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        st.session_state["df_raw"] = df
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

def load_data():
    """
    Load order:
      1. session_state  (populated by fetch_from_cloud or save_local_only)
      2. local CSV cache
      3. empty DataFrame — user must click Pull Cloud
    Never hits Google Sheets automatically; that only happens on explicit Pull.
    """
    if "df_raw" in st.session_state and not st.session_state["df_raw"].empty:
        return st.session_state["df_raw"]
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            df = _parse_timestamps(df)
            st.session_state["df_raw"] = df
            return df
        except Exception:
            pass
    return pd.DataFrame()

def save_local_only(df_to_save):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df_to_save.to_csv(CSV_PATH, index=False)
    st.session_state["df_raw"] = df_to_save
    process_data.clear()   # bust process cache so changes are visible immediately
    st.toast("💾 Saved locally!", icon="💾")

def sync_to_google_sheets(df):
    try:
        client = _get_gspread_client()
        sheet  = client.open(SHEET_NAME).sheet1
        sheet.clear()
        set_with_dataframe(sheet, df)
        st.toast("☁️ Synced to Google Sheets!", icon="✅")
    except Exception as e:
        st.error(f"Sync Failed: {e}")


# ─────────────────────────────────────────────────────────────
# CLASSIFICATION HELPERS  (fixed ordering — Player Prop before Total)
# ─────────────────────────────────────────────────────────────
def parse_odds_val(val):
    if pd.isna(val): return 0.0
    s = str(val).lower().replace('−', '-')
    if 'even' in s: return 100.0
    match = re.search(r'([-+]?\d+)', s)
    if match:
        try: return float(match.group(1))
        except: return 0.0
    return 0.0

def get_decimal_odds(american_odds):
    if pd.isna(american_odds) or american_odds == 0: return 0.0
    if american_odds > 0: return 1 + (american_odds / 100.0)
    return 1 + (100.0 / abs(american_odds))

def categorize_bet(market, selection):
    """Fixed: Player Prop checked before Total to avoid 'Total Points' misclassification."""
    m, s = str(market).lower(), str(selection).lower()
    if "moneyline" in m: return "Moneyline"
    if "spread" in m or "handicap" in m or "run line" in m or "puck line" in m: return "Point Spread"
    # ✅ Player Prop BEFORE Total
    if "player" in m or "milestone" in m or "props" in m: return "Player Prop"
    if any(x in m for x in ["shots", "sog", "assists", "rebounds", "threes", "touchdowns"]): return "Player Prop"
    # ✅ 'goals' and 'points' removed from keyword list to avoid Total Goals/Points misclassification
    if "total" in m or "over/under" in m: return "Total"
    if "to score" in s or re.search(r'\d+\+', s): return "Player Prop"
    if "over" in s or "under" in s: return "Total"
    return "Moneyline"

def get_bet_side(selection):
    s = str(selection).lower()
    if re.search(r'\bover\b', s): return "Over"
    if re.search(r'\bunder\b', s): return "Under"
    return "Other"

def extract_prop_category(market):
    """Fixed: combined categories checked before single categories."""
    m = str(market).lower().replace("player ", "").replace("alternate ", "").replace("alt ", "")
    if "milestone" in m: return "Milestone"
    # ✅ Combined categories FIRST
    if "points" in m and "rebounds" in m and "assists" in m: return "PRA"
    if "points" in m and "rebounds" in m: return "Pts+Reb"
    if "points" in m and "assists" in m: return "Pts+Ast"
    if "rebounds" in m and "assists" in m: return "Reb+Ast"
    if "blocks" in m and "steals" in m: return "Blk+Stl"
    # Singles
    if "points" in m: return "Points"
    if "rebounds" in m: return "Rebounds"
    if "assists" in m: return "Assists"
    if "threes" in m or "3-point" in m or "3pt" in m: return "Threes"
    if "blocks" in m: return "Blocks"
    if "steals" in m: return "Steals"
    if "turnovers" in m: return "Turnovers"
    if "shots" in m or "sog" in m: return "Shots on Goal"
    if "saves" in m: return "Saves"
    if "goals" in m: return "Goals"
    if "passing" in m: return "Passing"
    if "rushing" in m: return "Rushing"
    if "receiving" in m or "receptions" in m: return "Receiving"
    if "touchdown" in m or "score" in m: return "Touchdowns"
    if "double" in m: return "Double Double"
    if "triple" in m: return "Triple Double"
    return "Other"

def get_odds_bucket(val):
    if val < -750: return "< -750"
    if -750 <= val < -300: return "-750 to -300"
    if -300 <= val < -150: return "-300 to -150"
    if -150 <= val <= 150:  return "-150 to +150"
    if 150 < val <= 300:    return "+150 to +300"
    if 300 < val <= 750:    return "+300 to +750"
    return "> +750"

ODDS_BUCKET_ORDER = ["< -750", "-750 to -300", "-300 to -150", "-150 to +150", "+150 to +300", "+300 to +750", "> +750"]

def classify_tier(row):
    """Mirrors tracker classify_tier exactly."""
    s          = str(row.get('play_selection', '')).lower()
    is_under   = 'under' in s
    books      = [b.strip().strip('"') for b in str(row.get('sharp_book', '')).split(',') if b.strip()]
    consensus  = len(books)
    is_fanatics = 'fanatics' in str(row.get('play_book', '')).lower()
    bet_type   = categorize_bet(row.get('market', ''), row.get('play_selection', ''))
    is_prop_under = is_under and bet_type == 'Player Prop'

    try:    odds = float(str(row.get('play_odds', '0')).replace('+', ''))
    except: odds = 0
    good_odds = GOOD_ODDS_MIN <= odds <= GOOD_ODDS_MAX
    bad_odds  = BAD_ODDS_MIN  <= odds <= BAD_ODDS_MAX

    try:    liq = float(row.get('liquidity', 0))
    except: liq = 0
    good_liq = GOOD_LIQ_MIN <= liq <= GOOD_LIQ_MAX

    try:
        dt = pd.to_datetime(str(row.get('timestamp', ''))).to_pydatetime()
        prime_time = dt.hour in PRIME_HOURS
    except: prime_time = False

    if (bad_odds or is_fanatics) and consensus < CONSENSUS_THRESHOLD and not is_prop_under:
        return 'WATCH'
    if consensus >= CONSENSUS_THRESHOLD and good_odds and is_prop_under: return 'DIAMOND'
    if consensus >= CONSENSUS_THRESHOLD and good_odds and good_liq:      return 'DIAMOND'
    if consensus >= CONSENSUS_THRESHOLD and good_odds:                   return 'GOLD'
    if is_prop_under and good_liq and good_odds:                         return 'GOLD'
    if prime_time and good_liq and good_odds:                            return 'GOLD'
    if is_prop_under and good_odds:                                      return 'SILVER'
    if prime_time and good_odds:                                         return 'SILVER'
    return 'STANDARD'

def calculate_profit(odds_val, result):
    try: odds = float(odds_val)
    except: return 0.0
    if result == "Won":
        return UNIT_SIZE * (odds / 100.0) if odds > 0 else UNIT_SIZE * (100.0 / abs(odds))
    elif result == "Lost":
        return -float(UNIT_SIZE)
    return 0.0

def calculate_arb_percent(row):
    play  = parse_odds_val(row.get('play_odds', 0))
    sharp = parse_odds_val(row.get('sharp_odds', 0))
    if play == 0 or sharp == 0: return 0.0
    dp = get_decimal_odds(play); ds = get_decimal_odds(sharp)
    if dp == 0 or ds == 0: return 0.0
    total_imp = (1 / dp) + (1 / ds)
    return ((1 / total_imp) - 1) * 100 if total_imp else 0.0

def calculate_fade_profit(row):
    result = row.get('result', 'Pending')
    if result == 'Won': return -float(UNIT_SIZE)
    if result == 'Lost':
        odds = parse_odds_val(row.get('play_odds', 100))
        if odds == 0: return 0.0
        fade = odds * -1
        return UNIT_SIZE * (fade / 100.0) if fade > 0 else UNIT_SIZE * (100.0 / abs(fade))
    return 0.0


# ─────────────────────────────────────────────────────────────
# PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def process_data(raw_df):
    df = raw_df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Clean dirty rows
    if 'league' in df.columns:
        df = df[df['league'].isin(VALID_LEAGUES)]
    if 'play_book' in df.columns:
        df = df[~df['play_book'].isin(DFS_BOOKS)]
        df = df[df['play_book'].notna()]

    # Parse profit
    if 'profit' in df.columns:
        df['profit'] = pd.to_numeric(
            df['profit'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False),
            errors='coerce'
        ).fillna(0.0)

    # Derived columns
    df['odds_val']       = df['play_odds'].apply(parse_odds_val)
    df['odds_bucket']    = df['odds_val'].apply(get_odds_bucket)
    df['bet_type']       = df.apply(lambda x: categorize_bet(x.get('market',''), x.get('play_selection','')), axis=1)
    df['bet_side']       = df['play_selection'].apply(get_bet_side)
    df['prop_cat']       = df.apply(lambda x: extract_prop_category(x.get('market','')) if x['bet_type'] == 'Player Prop' else '', axis=1)
    df['tier']           = df.apply(classify_tier, axis=1)
    df['consensus']      = df['sharp_book'].astype(str).str.split(',').str.len().fillna(1).astype(int)
    df['primary_sharp']  = df['sharp_book'].astype(str).str.split(',').str[0].str.strip().str.strip('"')
    df['is_prop_under']  = (df['bet_type'] == 'Player Prop') & (df['bet_side'] == 'Under')
    df['arb_pct']        = df.apply(calculate_arb_percent, axis=1)

    # Combo label for leaderboard
    def combo(row):
        league = str(row.get('league', ''))
        bt = row['bet_type']
        side = row['bet_side']
        pc = row['prop_cat']
        if bt == 'Player Prop': return f"{side} {league} {pc}"
        if bt == 'Total':       return f"{side} {league} Game Total"
        return f"{league} {bt}"
    df['combo'] = df.apply(combo, axis=1)

    return df


# ─────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
    font=dict(family='IBM Plex Mono', color='#e6edf3', size=12),
    xaxis=dict(gridcolor='#21262d', zerolinecolor='#30363d'),
    yaxis=dict(gridcolor='#21262d', zerolinecolor='#30363d'),
    margin=dict(l=10, r=10, t=40, b=10),
)

def roi_color(v):
    if v >= 15: return '#00ff9f'
    if v >= 8:  return '#4ade80'
    if v >= 2:  return '#86efac'
    if v >= 0:  return '#bbf7d0'
    if v >= -5: return '#f87171'
    return '#ef4444'

def make_bar(df_plot, x, y, title="", h=300, color_by_sign=True, text_fmt=None):
    colors = [roi_color(v) if color_by_sign else '#58a6ff' for v in df_plot[y]]
    fig = go.Figure(go.Bar(
        x=df_plot[x], y=df_plot[y],
        marker_color=colors,
        text=[f"{v:.1f}%" if text_fmt == 'roi' else f"${v:,.0f}" for v in df_plot[y]],
        textposition='outside',
        textfont=dict(size=11),
    ))
    fig.add_hline(y=0, line_color='#30363d', line_width=2)
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=h)
    return fig

def make_hbar(df_plot, x, y, title="", h=300):
    colors = [roi_color(v) for v in df_plot[x]]
    fig = go.Figure(go.Bar(
        x=df_plot[x], y=df_plot[y], orientation='h',
        marker_color=colors,
        text=[f"{v:.1f}%" for v in df_plot[x]],
        textposition='outside',
        textfont=dict(size=11),
    ))
    fig.add_vline(x=0, line_color='#30363d', line_width=2)
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=h)
    return fig

def calc_roi_df(df, group_col, min_n=1):
    grp = df.groupby(group_col).agg(
        profit=('profit', 'sum'),
        n=('profit', 'count'),
        wins=('status', lambda x: (x == 'Won').sum()),
    ).reset_index()
    grp['roi'] = (grp['profit'] / (grp['n'] * UNIT_SIZE)) * 100
    grp['wr']  = (grp['wins'] / grp['n'].clip(lower=1)) * 100
    return grp[grp['n'] >= min_n].sort_values('roi', ascending=False)


# ─────────────────────────────────────────────────────────────
# MANUAL GRADER
# ─────────────────────────────────────────────────────────────
def render_manual_grader(df_full):
    st.header("📝 Bulk Manual Grader")
    if 'status' not in df_full.columns:
        st.error("Status column missing.")
        return

    open_mask = df_full['status'].str.lower().isin(['open', 'pending'])
    open_bets = df_full[open_mask].copy()

    if open_bets.empty:
        st.info("No open bets to grade!")
        return

    st.info(f"⚡ {len(open_bets)} pending bets — edit statuses below, then click Commit.")

    cols_to_show = ['timestamp', 'league', 'matchup', 'play_selection', 'market', 'play_odds', 'status']
    cols_to_show = [c for c in cols_to_show if c in open_bets.columns]

    edited_df = st.data_editor(
        open_bets[cols_to_show],
        column_config={
            "status": st.column_config.SelectboxColumn("Status", width="medium", options=["Open", "Won", "Lost", "Push"], required=True),
            "play_odds": st.column_config.NumberColumn("Odds", disabled=True),
            "matchup": st.column_config.TextColumn("Matchup", disabled=True),
            "play_selection": st.column_config.TextColumn("Selection", disabled=True),
        },
        hide_index=True, use_container_width=True,
        key="grader_editor", num_rows="fixed"
    )

    if st.button("💾 Commit Grades (Save Local)", type="primary"):
        changes = 0
        for index, row in edited_df.iterrows():
            if df_full.at[index, 'status'] != row['status'] and row['status'] in ['Won', 'Lost', 'Push']:
                changes += 1
                df_full.at[index, 'status'] = row['status']
                df_full.at[index, 'result'] = row['status']
                df_full.at[index, 'profit'] = round(calculate_profit(row['play_odds'], row['status']), 2)
        if changes > 0:
            save_local_only(df_full)
            st.success(f"✅ Graded {changes} bets. Click 'Push to Cloud' when done.")
            st.rerun()
        else:
            st.warning("No changes detected.")


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
st.title("💸 Smart Money Tracker v4")

# ── Sidebar: Data controls ──
st.sidebar.header("⚙️ Data Controls")
col_r, col_s = st.sidebar.columns(2)
with col_r:
    if st.button("🔄 Pull Cloud"):
        with st.spinner("Pulling from Google Sheets..."):
            df_pulled, err = fetch_from_cloud()
        if err:
            st.sidebar.error(f"Pull failed: {err}")
        else:
            st.sidebar.success(f"✅ Loaded {len(df_pulled):,} rows")
            st.rerun()
with col_s:
    if st.button("☁️ Push Cloud"):
        df_raw = load_data()
        with st.spinner("Syncing..."):
            sync_to_google_sheets(df_raw)

df_raw = load_data()
if df_raw.empty:
    st.info("No data found. Click 'Pull Cloud' to initialize.")
    st.stop()

df = process_data(df_raw)

# ─────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Filters")

# ── Quick presets ──
st.sidebar.subheader("Quick Presets")
preset = st.sidebar.radio("", [
    "All Bets",
    "NBA Props Only",
    "3+ Consensus Only",
    "Exclude Fanatics",
    "Best Edges (DIAMOND + GOLD)",
    "Prop Unders Only",
], label_visibility="collapsed")

# ── Date range ──
st.sidebar.markdown("**Date Range**")
min_date = df['timestamp'].min().date() if 'timestamp' in df.columns else datetime(2025,1,1).date()
max_date = df['timestamp'].max().date() if 'timestamp' in df.columns else datetime.today().date()
date_range = st.sidebar.date_input("", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# ── Metric toggle ──
st.sidebar.markdown("**Display Metric**")
metric_mode = st.sidebar.radio("", ["ROI (%)", "Total Profit ($)"], label_visibility="collapsed")
fade_mode   = st.sidebar.toggle("🔄 FADE MODE", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Manual Filters")

# ── Leagues ──
all_leagues = sorted(df['league'].dropna().unique())
sel_leagues = st.sidebar.multiselect("Leagues", all_leagues, default=all_leagues)

# ── Tiers ──
sel_tiers = st.sidebar.multiselect("Tier", TIER_ORDER, default=TIER_ORDER)

# ── Sharp books ──
all_sharps = sorted(df['primary_sharp'].dropna().unique())
sel_sharps = st.sidebar.multiselect("Sharp Book Signal", all_sharps, default=all_sharps)

# ── Play books ──
all_books = sorted(df['play_book'].dropna().unique()) if 'play_book' in df.columns else []
sel_books = st.sidebar.multiselect("Play Book", all_books, default=all_books)

# ── Bet types ──
sel_types = st.sidebar.multiselect("Bet Type", ['Moneyline','Player Prop','Point Spread','Total'], default=['Moneyline','Player Prop','Point Spread','Total'])

# ── Consensus ──
max_cons = int(df['consensus'].max()) if 'consensus' in df.columns else 6
cons_range = st.sidebar.slider("Consensus Books", 1, max_cons, (1, max_cons))

# ── Odds range ──
st.sidebar.markdown("**Odds Range**")
oc1, oc2 = st.sidebar.columns(2)
default_min_odds = int(df['odds_val'].min()) if not df.empty else -10000
default_max_odds = int(df['odds_val'].max()) if not df.empty else 10000
with oc1: min_odds = st.number_input("Min", value=default_min_odds, step=10)
with oc2: max_odds = st.number_input("Max", value=default_max_odds, step=10)

# ─────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────
df_f = df.copy()

# Apply preset FIRST, then allow manual overrides on top
if preset == "NBA Props Only":
    df_f = df_f[(df_f['league'] == 'NBA') & (df_f['bet_type'] == 'Player Prop')]
elif preset == "3+ Consensus Only":
    df_f = df_f[df_f['consensus'] >= 3]
elif preset == "Exclude Fanatics":
    df_f = df_f[df_f['play_book'] != 'Fanatics']
elif preset == "Best Edges (DIAMOND + GOLD)":
    df_f = df_f[df_f['tier'].isin(['DIAMOND', 'GOLD'])]
elif preset == "Prop Unders Only":
    df_f = df_f[df_f['is_prop_under']]

# Date
if len(date_range) == 2 and 'timestamp' in df_f.columns:
    df_f = df_f[(df_f['timestamp'].dt.date >= date_range[0]) & (df_f['timestamp'].dt.date <= date_range[1])]

# Fade mode — flip profit
if fade_mode:
    df_f['profit'] = df_f.apply(calculate_fade_profit, axis=1)
    st.sidebar.warning("⚠️ FADE MODE ACTIVE")

# Manual filters
if sel_leagues: df_f = df_f[df_f['league'].isin(sel_leagues)]
if sel_tiers:   df_f = df_f[df_f['tier'].isin(sel_tiers)]
if sel_sharps:  df_f = df_f[df_f['primary_sharp'].isin(sel_sharps)]
if sel_books and 'play_book' in df_f.columns:
    df_f = df_f[df_f['play_book'].isin(sel_books)]
if sel_types:   df_f = df_f[df_f['bet_type'].isin(sel_types)]
df_f = df_f[(df_f['consensus'] >= cons_range[0]) & (df_f['consensus'] <= cons_range[1])]
df_f = df_f[(df_f['odds_val'] >= min_odds) & (df_f['odds_val'] <= max_odds)]

# Closed bets only for performance analysis
closed = df_f[df_f['status'].isin(['Won', 'Lost', 'Push'])].copy()

# ─────────────────────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────────────────────
total_profit  = closed['profit'].sum() if not closed.empty else 0
total_wagered = len(closed) * UNIT_SIZE
roi_overall   = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
win_rate      = (closed['status'] == 'Won').sum() / max(len(closed[closed['status'] != 'Push']), 1) * 100
pending_n     = len(df_f[df_f['status'].isin(['Open', 'Pending'])])

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Bets",    f"{len(df_f):,}")
c2.metric("Settled",       f"{len(closed):,}")
c3.metric("Pending",       f"{pending_n:,}")
c4.metric("Total Profit",  f"${total_profit:,.0f}", delta=f"{roi_overall:.1f}% ROI")
c5.metric("Win Rate",      f"{win_rate:.1f}%")
c6.metric("Active Filter", preset if preset != "All Bets" else "All Bets")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_live, tab_tier, tab_analysis, tab_props, tab_odds, tab_rolling, tab_leaderboard, tab_sim, tab_grader = st.tabs([
    "📊 Live Log",
    "💎 Tier Performance",
    "📈 Deep Dive",
    "🏀 Prop Breakdown",
    "🎲 Odds Analysis",
    "📉 Rolling ROI",
    "🏆 Leaderboard",
    "💰 Simulator",
    "📝 Grader",
])

# ─── LIVE LOG ───
with tab_live:
    st.subheader("Bet History")
    display_cols = ['timestamp', 'tier', 'league', 'matchup', 'bet_type', 'prop_cat', 'play_selection',
                    'bet_side', 'play_odds', 'play_book', 'primary_sharp', 'consensus', 'status', 'profit']
    display_cols = [c for c in display_cols if c in df_f.columns]
    display_df = df_f[display_cols].copy().sort_values('timestamp', ascending=False)

    # Color-code tier in display
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "tier": st.column_config.TextColumn("Tier", width="small"),
            "profit": st.column_config.NumberColumn("Profit", format="$%.2f"),
            "consensus": st.column_config.NumberColumn("# Books", width="small"),
            "play_odds": st.column_config.NumberColumn("Odds", format="%d"),
        }
    )

# ─── TIER PERFORMANCE ───
with tab_tier:
    st.subheader("💎 Performance by Tier")
    if closed.empty:
        st.warning("No settled bets in current filter.")
    else:
        tier_stats = calc_roi_df(closed, 'tier', min_n=5)
        tier_stats['tier'] = pd.Categorical(tier_stats['tier'], categories=TIER_ORDER, ordered=True)
        tier_stats = tier_stats.sort_values('tier')

        # Tier summary cards
        cols_t = st.columns(len(TIER_ORDER))
        for i, tier in enumerate(TIER_ORDER):
            row = tier_stats[tier_stats['tier'] == tier]
            if row.empty:
                cols_t[i].metric(f"{TIER_EMOJI.get(tier,'')} {tier}", "N/A")
            else:
                r = row.iloc[0]
                cols_t[i].metric(
                    f"{TIER_EMOJI.get(tier,'')} {tier}",
                    f"{r['roi']:.1f}% ROI",
                    f"N={r['n']:,} · WR {r['wr']:.0f}%"
                )

        st.markdown("---")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            metric = 'roi' if metric_mode == "ROI (%)" else 'profit'
            fig = make_bar(tier_stats, 'tier', metric, "ROI by Tier", text_fmt='roi' if metric == 'roi' else None)
            st.plotly_chart(fig, use_container_width=True)

        with col_t2:
            # Tier profit over time (cumulative)
            ts = closed[closed['timestamp'].notna()].sort_values('timestamp')
            if not ts.empty:
                tier_lines = []
                for tier in TIER_ORDER:
                    sub = ts[ts['tier'] == tier].copy()
                    if len(sub) < 3: continue
                    sub['cum_profit'] = sub['profit'].cumsum()
                    sub['tier_label'] = tier
                    tier_lines.append(sub[['timestamp','cum_profit','tier_label']])
                if tier_lines:
                    tier_ts = pd.concat(tier_lines)
                    fig2 = px.line(tier_ts, x='timestamp', y='cum_profit', color='tier_label',
                                   title="Cumulative Profit by Tier",
                                   color_discrete_map=TIER_COLORS)
                    fig2.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig2, use_container_width=True)

        # Tier detail table
        display_tier = tier_stats.copy()
        display_tier['roi']    = display_tier['roi'].map('{:.1f}%'.format)
        display_tier['profit'] = display_tier['profit'].map('${:,.0f}'.format)
        display_tier['wr']     = display_tier['wr'].map('{:.1f}%'.format)
        st.dataframe(display_tier.rename(columns={'n':'Bets','roi':'ROI','profit':'Profit','wr':'Win Rate'}),
                     use_container_width=True, hide_index=True)

        # Tier validation insight box
        st.markdown("---")
        st.markdown("**📌 Tier Validation (settled bets)**")
        for _, row in tier_stats.iterrows():
            emoji = TIER_EMOJI.get(row['tier'], '')
            color = TIER_COLORS.get(row['tier'], '#58a6ff')
            sign  = '+' if row['roi'] >= 0 else ''
            st.markdown(
                f'<div class="insight-card" style="border-left-color:{color}">'
                f'{emoji} <b>{row["tier"]}</b> — {sign}{row["roi"]:.1f}% ROI · '
                f'{row["n"]:,} bets · {row["wr"]:.0f}% WR · ${row["profit"]:,.0f} profit'
                f'</div>',
                unsafe_allow_html=True
            )

# ─── DEEP DIVE ───
with tab_analysis:
    if closed.empty:
        st.warning("No graded bets in current filter.")
    else:
        # Heatmap
        if 'league' in closed.columns and 'market' in closed.columns:
            st.subheader("🔥 League × Market Heatmap")
            hm = closed.groupby(['league', 'bet_type'])['profit'].agg(
                lambda x: (x.sum() / (len(x) * UNIT_SIZE)) * 100
            ).reset_index()
            hm.columns = ['league', 'bet_type', 'roi']
            fig_heat = px.density_heatmap(hm, x='bet_type', y='league', z='roi',
                                           color_continuous_scale='RdYlGn',
                                           text_auto='.1f',
                                           range_color=[-30, 30])
            fig_heat.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_heat, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("By League")
            lg_stats = calc_roi_df(closed, 'league', min_n=10)
            metric = 'roi' if metric_mode == "ROI (%)" else 'profit'
            st.plotly_chart(make_bar(lg_stats, 'league', metric, text_fmt='roi' if metric=='roi' else None), use_container_width=True)
        with col_b:
            st.subheader("By Bet Type")
            bt_stats = calc_roi_df(closed, 'bet_type', min_n=5)
            st.plotly_chart(make_bar(bt_stats, 'bet_type', metric, text_fmt='roi' if metric=='roi' else None), use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("By Play Book")
            bk_stats = calc_roi_df(closed, 'play_book', min_n=10)
            st.plotly_chart(make_bar(bk_stats, 'play_book', metric, text_fmt='roi' if metric=='roi' else None), use_container_width=True)
        with col_d:
            st.subheader("By Sharp Book Signal")
            sb_stats = calc_roi_df(closed, 'primary_sharp', min_n=10)
            st.plotly_chart(make_bar(sb_stats, 'primary_sharp', metric, text_fmt='roi' if metric=='roi' else None), use_container_width=True)

        # Consensus
        st.subheader("By Consensus Count")
        cs_stats = calc_roi_df(closed, 'consensus', min_n=5).sort_values('consensus')
        cs_stats['consensus'] = cs_stats['consensus'].astype(str) + ' books'
        st.plotly_chart(make_bar(cs_stats, 'consensus', metric, text_fmt='roi' if metric=='roi' else None, h=250), use_container_width=True)

# ─── PROP BREAKDOWN ───
with tab_props:
    st.subheader("🏀 Player Prop Analysis")
    props_closed = closed[closed['bet_type'] == 'Player Prop'].copy()

    if props_closed.empty:
        st.warning("No settled player prop bets in current filter.")
    else:
        # Over vs Under
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            side_stats = calc_roi_df(props_closed, 'bet_side', min_n=5)
            fig_side = go.Figure()
            for _, row in side_stats.iterrows():
                fig_side.add_trace(go.Bar(
                    x=[row['bet_side']], y=[row['roi']],
                    marker_color=roi_color(row['roi']),
                    text=f"{row['roi']:.1f}%<br>N={row['n']:,}<br>WR {row['wr']:.0f}%",
                    textposition='inside',
                    name=row['bet_side']
                ))
            fig_side.add_hline(y=0, line_color='#30363d')
            fig_side.update_layout(**PLOTLY_LAYOUT, title="Prop Over vs Under", showlegend=False, height=300)
            st.plotly_chart(fig_side, use_container_width=True)

        with col_p2:
            # Prop cat summary
            pc_stats = calc_roi_df(props_closed, 'prop_cat', min_n=10).head(8)
            st.plotly_chart(make_hbar(pc_stats, 'roi', 'prop_cat', "Top Prop Categories by ROI", h=300), use_container_width=True)

        # Prop category × Side matrix
        st.subheader("Prop Category × Over/Under")
        prop_cross = props_closed.groupby(['prop_cat', 'bet_side']).agg(
            profit=('profit', 'sum'), n=('profit', 'count')
        ).reset_index()
        prop_cross['roi'] = (prop_cross['profit'] / (prop_cross['n'] * UNIT_SIZE)) * 100
        prop_cross = prop_cross[prop_cross['n'] >= 5]

        fig_cross = px.bar(prop_cross, x='prop_cat', y='roi', color='bet_side',
                           barmode='group', color_discrete_map={'Over': '#f87171', 'Under': '#4ade80', 'Other': '#8b949e'},
                           text_auto='.1f')
        fig_cross.add_hline(y=0, line_color='#30363d', line_width=2)
        fig_cross.update_layout(**PLOTLY_LAYOUT, height=350, xaxis_tickangle=-30)
        st.plotly_chart(fig_cross, use_container_width=True)

        # NBA prop under category detail
        st.subheader("NBA Prop Unders — Category Detail")
        nba_under = props_closed[(props_closed['league'] == 'NBA') & (props_closed['bet_side'] == 'Under')]
        if not nba_under.empty:
            nba_cat = calc_roi_df(nba_under, 'prop_cat', min_n=5).sort_values('roi', ascending=False)
            st.plotly_chart(make_hbar(nba_cat, 'roi', 'prop_cat', "NBA Under ROI by Category", h=max(300, len(nba_cat)*35)), use_container_width=True)

            # Detail table
            display_nba = nba_cat.copy()
            display_nba['roi']    = display_nba['roi'].map('{:.1f}%'.format)
            display_nba['profit'] = display_nba['profit'].map('${:,.0f}'.format)
            display_nba['wr']     = display_nba['wr'].map('{:.1f}%'.format)
            st.dataframe(display_nba.rename(columns={'prop_cat':'Category','n':'Bets','roi':'ROI','profit':'Profit','wr':'WR'}),
                         use_container_width=True, hide_index=True)

# ─── ODDS ANALYSIS ───
with tab_odds:
    st.subheader("🎲 Profitability by Odds Range")
    if closed.empty:
        st.warning("No graded bets to analyze.")
    else:
        odds_stats = closed.groupby('odds_bucket').agg(
            profit=('profit','sum'), n=('profit','count')
        ).reset_index()
        odds_stats['roi'] = (odds_stats['profit'] / (odds_stats['n'] * UNIT_SIZE)) * 100
        odds_stats['odds_bucket'] = pd.Categorical(odds_stats['odds_bucket'], categories=ODDS_BUCKET_ORDER, ordered=True)
        odds_stats = odds_stats.sort_values('odds_bucket')

        metric_col = 'roi' if metric_mode == "ROI (%)" else 'profit'
        st.plotly_chart(make_bar(odds_stats, 'odds_bucket', metric_col, "Performance by Odds Range",
                                 text_fmt='roi' if metric_col=='roi' else None, h=320), use_container_width=True)

        display_odds = odds_stats.copy()
        display_odds['roi']    = display_odds['roi'].map('{:.1f}%'.format)
        display_odds['profit'] = display_odds['profit'].map('${:,.2f}'.format)
        st.dataframe(display_odds.rename(columns={'odds_bucket':'Odds Range','n':'Bets','roi':'ROI','profit':'Profit'}),
                     use_container_width=True, hide_index=True)

# ─── ROLLING ROI ───
with tab_rolling:
    st.subheader("📉 Rolling ROI Trend")
    if closed.empty or 'timestamp' not in closed.columns:
        st.warning("No data for trend analysis.")
    else:
        roll_df = closed[closed['timestamp'].notna()].sort_values('timestamp').copy()

        col_rw, col_rb = st.columns(2)
        with col_rw:
            window = st.slider("Rolling window (bets)", 10, 200, 50, step=10)
        with col_rb:
            roll_by = st.radio("Group by", ["All Bets", "By Tier"], horizontal=True)

        # Rolling ROI = rolling mean profit / unit_size * 100
        if roll_by == "All Bets":
            roll_df['rolling_roi'] = roll_df['profit'].rolling(window, min_periods=max(5, window//4)).mean() / UNIT_SIZE * 100
            roll_df['cum_profit']  = roll_df['profit'].cumsum()

            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=roll_df['timestamp'], y=roll_df['rolling_roi'],
                name=f'{window}-bet Rolling ROI', line=dict(color='#58a6ff', width=2),
            ))
            fig_roll.add_hline(y=0, line_color='#30363d', line_dash='dash')
            fig_roll.update_layout(**PLOTLY_LAYOUT, title=f"{window}-Bet Rolling ROI", yaxis_title="ROI (%)", height=350)
            st.plotly_chart(fig_roll, use_container_width=True)

            fig_cum = go.Figure(go.Scatter(
                x=roll_df['timestamp'], y=roll_df['cum_profit'],
                fill='tozeroy',
                line=dict(color='#4ade80' if roll_df['cum_profit'].iloc[-1] >= 0 else '#f87171', width=2),
                fillcolor='rgba(74,222,128,0.1)',
            ))
            fig_cum.add_hline(y=0, line_color='#30363d')
            fig_cum.update_layout(**PLOTLY_LAYOUT, title="Cumulative Profit", yaxis_title="$ Profit", height=300)
            st.plotly_chart(fig_cum, use_container_width=True)

        else:
            fig_tier_roll = go.Figure()
            for tier in TIER_ORDER:
                sub = roll_df[roll_df['tier'] == tier].copy()
                if len(sub) < window // 2: continue
                sub['rolling_roi'] = sub['profit'].rolling(window, min_periods=max(5, window//4)).mean() / UNIT_SIZE * 100
                fig_tier_roll.add_trace(go.Scatter(
                    x=sub['timestamp'], y=sub['rolling_roi'],
                    name=f"{TIER_EMOJI.get(tier,'')} {tier}",
                    line=dict(color=TIER_COLORS.get(tier, '#58a6ff'), width=2),
                ))
            fig_tier_roll.add_hline(y=0, line_color='#30363d', line_dash='dash')
            fig_tier_roll.update_layout(**PLOTLY_LAYOUT, title=f"{window}-Bet Rolling ROI by Tier", height=400)
            st.plotly_chart(fig_tier_roll, use_container_width=True)

        # Monthly breakdown table
        st.subheader("Monthly Performance")
        roll_df['month'] = roll_df['timestamp'].dt.to_period('M').astype(str)
        monthly = roll_df.groupby('month').agg(profit=('profit','sum'), n=('profit','count')).reset_index()
        monthly['roi'] = (monthly['profit'] / (monthly['n'] * UNIT_SIZE)) * 100
        monthly['roi_fmt']    = monthly['roi'].map('{:+.1f}%'.format)
        monthly['profit_fmt'] = monthly['profit'].map('${:,.0f}'.format)
        st.dataframe(monthly[['month','n','roi_fmt','profit_fmt']].rename(
            columns={'month':'Month','n':'Bets','roi_fmt':'ROI','profit_fmt':'Profit'}),
            use_container_width=True, hide_index=True)

# ─── LEADERBOARD ───
with tab_leaderboard:
    st.subheader("🏆 Most Profitable Categories")
    if closed.empty:
        st.warning("No settled bets available.")
    else:
        col_lb1, col_lb2 = st.columns([1, 3])
        with col_lb1:
            min_bets = st.slider("Min Sample Size", 1, 100, 10)
            sort_by  = st.radio("Sort by", ["ROI", "Total Profit"], index=0)

        lb = closed.groupby('combo').agg(
            profit=('profit','sum'),
            n=('profit','count'),
            wins=('status', lambda x: (x == 'Won').sum()),
        ).reset_index()
        lb['roi'] = (lb['profit'] / (lb['n'] * UNIT_SIZE)) * 100
        lb['wr']  = lb['wins'] / lb['n'].clip(lower=1) * 100
        lb = lb[lb['n'] >= min_bets]
        lb = lb.sort_values('roi' if sort_by == 'ROI' else 'profit', ascending=False)

        col_lb2.metric("Categories shown", f"{len(lb)}")

        display_lb = lb.copy()
        display_lb['roi']    = display_lb['roi'].map('{:+.1f}%'.format)
        display_lb['profit'] = display_lb['profit'].map('${:,.0f}'.format)
        display_lb['wr']     = display_lb['wr'].map('{:.1f}%'.format)
        st.dataframe(
            display_lb[['combo','n','roi','profit','wr']].rename(
                columns={'combo':'Category','n':'Bets','roi':'ROI','profit':'Profit','wr':'Win Rate'}),
            use_container_width=True, height=600, hide_index=True
        )

# ─── SIMULATOR ───
with tab_sim:
    st.subheader("💰 Bankroll Simulator")
    if closed.empty:
        st.warning("No graded bets to simulate.")
    else:
        col_s1, col_s2, col_s3 = st.columns(3)
        start_bankroll = col_s1.number_input("Starting Bankroll ($)", value=10000, step=500)
        pct_stake      = col_s2.slider("% Stake Strategy", 0.5, 5.0, 2.0, step=0.5) / 100.0
        sim_tier_filter = col_s3.multiselect("Simulate Tiers", TIER_ORDER, default=TIER_ORDER)

        sim_df = closed[closed['tier'].isin(sim_tier_filter)].sort_values('timestamp').copy()

        if sim_df.empty:
            st.warning("No bets match simulation filter.")
        else:
            sim_df['flat_bankroll'] = start_bankroll + sim_df['profit'].cumsum()

            def get_multiplier(row):
                if row['result'] == 'Won':
                    odds = parse_odds_val(row['play_odds'])
                    return (odds / 100.0) if odds > 0 else (100.0 / abs(odds))
                elif row['result'] == 'Lost': return -1.0
                return 0.0

            sim_df['multiplier']   = sim_df.apply(get_multiplier, axis=1)
            sim_df['growth_factor'] = 1 + (pct_stake * sim_df['multiplier'])
            sim_df['pct_bankroll']  = start_bankroll * sim_df['growth_factor'].cumprod()

            fig_sim = px.line(sim_df, x='timestamp', y=['flat_bankroll', 'pct_bankroll'],
                              title=f"Flat vs {pct_stake*100:.1f}% Compounding — {', '.join(sim_tier_filter)} tiers",
                              labels={'value': 'Bankroll ($)', 'variable': 'Strategy'},
                              color_discrete_map={'flat_bankroll': '#58a6ff', 'pct_bankroll': '#4ade80'})
            fig_sim.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_sim, use_container_width=True)

            final_flat = sim_df['flat_bankroll'].iloc[-1]
            final_pct  = sim_df['pct_bankroll'].iloc[-1]
            cs1, cs2, cs3 = st.columns(3)
            cs1.metric("Final (Flat)",         f"${final_flat:,.0f}", delta=f"${final_flat-start_bankroll:,.0f}")
            cs2.metric(f"Final ({pct_stake*100:.0f}% Compound)", f"${final_pct:,.0f}", delta=f"${final_pct-start_bankroll:,.0f}")
            cs3.metric("Bets Simulated",        f"{len(sim_df):,}")

# ─── GRADER ───
with tab_grader:
    render_manual_grader(df_raw)

# ─── DEBUG ───
with st.expander("🛠️ Debug"):
    st.write("Processed data shape:", df.shape)
    st.write("Filtered shape:", df_f.shape)
    st.write("Closed bets:", len(closed))
    st.write("Tier distribution:", df_f['tier'].value_counts().to_dict())
    st.write("Primary sharp distribution:", df_f['primary_sharp'].value_counts().to_dict())