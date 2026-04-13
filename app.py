import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from datetime import datetime, timedelta

from shared_logic import (
    CSV_PATH, UNIT_SIZE,
    TIER_ORDER, TIER_COLORS, TIER_EMOJI, ODDS_BUCKET_ORDER,
    CONSENSUS_THRESHOLD,
    parse_odds_val, calculate_profit, calculate_arb_percent,
    clean_raw_df, add_derived_columns,
)
from db_utils import load_bets, get_date_range, count_bets

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Money Tracker", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;600&display=swap');
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117; color: #e6edf3;
}
h1,h2,h3 { font-family: 'IBM Plex Mono', monospace; }
.stMetric { background:#161b22; border:1px solid #21262d; border-radius:10px; padding:16px; }
.stMetric label { color:#8b949e !important; font-size:11px !important; letter-spacing:2px;
    text-transform:uppercase; font-family:'IBM Plex Mono',monospace !important; }
.insight-card { background:#161b22; border-left:3px solid #58a6ff; border-radius:6px;
    padding:10px 14px; margin-bottom:8px; font-size:13px; }
div[data-testid="stTab"] button { font-family:'IBM Plex Mono',monospace; font-size:12px; letter-spacing:1px; }
section[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #21262d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_from_db(days_back: int) -> pd.DataFrame:
    raw = load_bets(days_back=days_back if days_back > 0 else None)
    if raw.empty:
        return pd.DataFrame()
    df = clean_raw_df(raw)
    df = add_derived_columns(df)
    
    # --- TIMEZONE FIX ---
    # The database stores time in UTC. This converts the entire 
    # dataframe to Eastern Time so the date picker works perfectly.
    if 'timestamp' in df.columns:
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern')
        
    return df

def bust_cache():
    fetch_from_db.clear()
    fetch_parlays.clear()
    st.rerun()


@st.cache_data(ttl=300)
def fetch_parlays() -> pd.DataFrame:
    import psycopg2
    db_url = os.environ.get('DATABASE_URL', 'postgresql://tracker:Sh%40dam949@104.131.111.111:5432/smartmoney')
    conn = psycopg2.connect(db_url, sslmode='disable')
    df = pd.read_sql("""
        SELECT id, created_at, n_legs, book,
               leg1_sel, leg1_book, leg1_odds, leg1_market, leg1_league, leg1_tier, leg1_edge, leg1_twroi,
               leg2_sel, leg2_book, leg2_odds, leg2_market, leg2_league, leg2_tier, leg2_edge, leg2_twroi,
               leg3_sel, leg3_book, leg3_odds, leg3_market, leg3_league, leg3_tier, leg3_edge, leg3_twroi,
               parlay_odds, ev_pct, status, profit, graded_at
        FROM parlays
        ORDER BY created_at DESC
    """, conn)
    conn.close()
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True).dt.tz_convert('US/Eastern')
    df['n_legs'] = df['n_legs'].fillna(2).astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────
LAYOUT = dict(paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
              font=dict(family='IBM Plex Mono', color='#e6edf3', size=12),
              xaxis=dict(gridcolor='#21262d', zerolinecolor='#30363d'),
              yaxis=dict(gridcolor='#21262d', zerolinecolor='#30363d'),
              margin=dict(l=10, r=10, t=40, b=10))

def roi_color(v):
    if v >= 15: return '#00ff9f'
    if v >= 8:  return '#4ade80'
    if v >= 2:  return '#86efac'
    if v >= 0:  return '#bbf7d0'
    if v >= -5: return '#f87171'
    return '#ef4444'

def bar(df_plot, x, y, title="", h=280, text_fmt='roi'):
    colors = [roi_color(v) for v in df_plot[y]]
    texts  = [f"{v:+.1f}%" if text_fmt=='roi' else f"${v:,.0f}" for v in df_plot[y]]
    fig = go.Figure(go.Bar(x=df_plot[x], y=df_plot[y], marker_color=colors,
                           text=texts, textposition='outside', textfont=dict(size=11)))
    fig.add_hline(y=0, line_color='#30363d', line_width=2)
    fig.update_layout(**LAYOUT, title=title, height=h)
    return fig

def hbar(df_plot, x, y, title="", h=300):
    colors = [roi_color(v) for v in df_plot[x]]
    fig = go.Figure(go.Bar(x=df_plot[x], y=df_plot[y], orientation='h',
                           marker_color=colors,
                           text=[f"{v:+.1f}%" for v in df_plot[x]],
                           textposition='outside', textfont=dict(size=11)))
    fig.add_vline(x=0, line_color='#30363d', line_width=2)
    fig.update_layout(**LAYOUT, title=title, height=h)
    return fig

def calc_roi(df, group_col, min_n=5):
    g = df.groupby(group_col).agg(
        profit=('profit','sum'), n=('profit','count'),
        wins=('status', lambda x: (x=='Won').sum())
    ).reset_index()
    g['roi'] = (g['profit'] / (g['n'] * UNIT_SIZE)) * 100
    g['wr']  = g['wins'] / g['n'].clip(lower=1) * 100
    return g[g['n'] >= min_n].sort_values('roi', ascending=False)


# ─────────────────────────────────────────────────────────────
# EDGE SCORE CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────

MY_SCORE_BUCKETS = [
    (0,  20,  'AVOID', '#ef4444'),
    (20, 35,  'LEAN',  '#f59e0b'),
    (35, 55,  'FAIR',  '#4ade80'),
    (55, 70,  'GOOD',  '#00ff9f'),
    (70, 101, 'HIGH',  '#00ffcc'),
]
GEM_SCORE_BUCKETS = [
    (0,  52, 'SKIP',      '#ef4444'),
    (52, 54, 'NEAR',      '#f59e0b'),
    (54, 56, 'BET 1u',    '#4ade80'),
    (56, 70, 'BET 0.25u', '#00ff9f'),
]
SMASH_SCORE_BUCKETS = [
    (0,  52, 'SKIP (<52)',   '#ef4444'),
    (52, 55, 'LEAN (52-55)', '#f59e0b'),
    (55, 58, 'PLAY (55-58)', '#4ade80'),
    (58, 101, 'SMASH (58+)', '#00ff9f'),
]

def score_bucket_roi(df, score_col, buckets, min_n=2):
    settled = df[df['status'].isin(['Won','Lost'])].dropna(subset=[score_col])
    rows = []
    for lo, hi, lbl, color in buckets:
        sub = settled[(settled[score_col] >= lo) & (settled[score_col] < hi)]
        if len(sub) < min_n:
            continue
        roi = sub['profit'].sum() / (len(sub) * UNIT_SIZE) * 100
        wr  = (sub['status'] == 'Won').mean() * 100
        rows.append({'bucket': lbl, 'lo': lo, 'roi': roi, 'n': len(sub), 'wr': wr, 'color': color})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
st.title("💸 Smart Money Tracker")

# ── Sidebar ──
st.sidebar.header("⚙️ Data Controls")
st.sidebar.markdown("**Data Window**")
window_options = {"Last 14 days":14,"Last 30 days":30,"Last 60 days":60,"Last 90 days":90,"All time":0}
selected_window = st.sidebar.selectbox("", list(window_options.keys()), index=2, label_visibility="collapsed")
days_back = window_options[selected_window]

if st.sidebar.button("🔄 Refresh Data"):
    bust_cache()

try:
    total_in_db = count_bets()
    st.sidebar.caption(f"📦 {total_in_db:,} total bets in database")
except Exception:
    pass

with st.spinner(f"Loading {selected_window.lower()}..."):
    df = fetch_from_db(days_back)

if df.empty:
    st.info("No data found. Check your DATABASE_URL connection.")
    st.stop()

# ── CACHE-BUSTING TYPE SANITIZATION ──
df = df.loc[:, ~df.columns.duplicated()].copy()

if 'profit' in df.columns:
    df['profit'] = pd.to_numeric(df['profit'], errors='coerce').fillna(0.0).astype(float)
if 'liquidity' in df.columns:
    df['liquidity'] = pd.to_numeric(df['liquidity'], errors='coerce').fillna(0.0).astype(float)


# ── Filters ──
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Filters")

# ── TOP-LEVEL: Bet Status Scope ──────────────────────────────
st.sidebar.markdown("**Bet Status Scope**")
status_scope = st.sidebar.radio(
    "",
    ["Exclude Expired", "All Bets (incl. Expired)"],
    index=0,
    label_visibility="collapsed",
    key="status_scope"
)
st.sidebar.markdown("---")

st.sidebar.subheader("Quick Presets")
preset = st.sidebar.radio("", [
    "All Bets","NBA Props Only","3+ Consensus Only",
    "Exclude Fanatics","Best Edges (DIAMOND + GOLD)","Prop Unders Only",
    "Alerted Bets Only"
], label_visibility="collapsed")

st.sidebar.markdown("**Date Range**")
min_date   = df['timestamp'].min().date()
max_date   = df['timestamp'].max().date()
date_range = st.sidebar.date_input("", value=(min_date, max_date), min_value=min_date, max_value=max_date)

st.sidebar.markdown("**Display Metric**")
metric_mode = st.sidebar.radio("", ["ROI (%)", "Total Profit ($)"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.subheader("Manual Filters")
all_leagues = sorted(df['league'].dropna().unique())
sel_leagues = st.sidebar.multiselect("Leagues", all_leagues, default=all_leagues)
sel_tiers   = st.sidebar.multiselect("Tier", TIER_ORDER, default=TIER_ORDER)
all_sharps  = sorted(df['primary_sharp'].dropna().unique())
sel_sharps  = st.sidebar.multiselect("Sharp Book Signal", all_sharps, default=all_sharps)
all_books   = sorted(df['play_book'].dropna().unique())
sel_books   = st.sidebar.multiselect("Play Book", all_books, default=all_books)
sel_types   = st.sidebar.multiselect("Bet Type",
    ['Moneyline','Player Prop','Point Spread','Total'],
    default=['Moneyline','Player Prop','Point Spread','Total'])
max_cons   = int(df['consensus'].max())
cons_range = st.sidebar.slider("Consensus Books", 1, max_cons, (1, max_cons))
oc1, oc2   = st.sidebar.columns(2)
min_odds   = oc1.number_input("Min Odds", value=int(df['odds_val'].min()), step=10)
max_odds   = oc2.number_input("Max Odds", value=int(df['odds_val'].max()), step=10)


HAS_MY_SCORE_SIDEBAR = 'edge_score' in df.columns and df['edge_score'].notna().sum() > 0
if HAS_MY_SCORE_SIDEBAR:
    st.sidebar.markdown("**Edge Score Range**")
    ec1, ec2 = st.sidebar.columns(2)
    min_edge = ec1.number_input("Min Edge", value=0, min_value=0, max_value=100, step=1, key="min_edge")
    max_edge = ec2.number_input("Max Edge", value=100, min_value=0, max_value=100, step=1, key="max_edge")
else:
    min_edge, max_edge = 0, 100

HAS_GEM_SCORE_SIDEBAR = 'gem_score' in df.columns and df['gem_score'].notna().sum() > 0
if HAS_GEM_SCORE_SIDEBAR:
    st.sidebar.markdown("**Gem Score Range**")
    gc1, gc2 = st.sidebar.columns(2)
    min_gem = gc1.number_input("Min Gem", value=0.0, min_value=0.0, max_value=100.0, step=1.0, key="min_gem")
    max_gem = gc2.number_input("Max Gem", value=100.0, min_value=0.0, max_value=100.0, step=1.0, key="max_gem")
else:
    min_gem, max_gem = 0.0, 100.0

HAS_SMASH_SCORE_SIDEBAR = 'smash_score' in df.columns and df['smash_score'].notna().sum() > 0
if HAS_SMASH_SCORE_SIDEBAR:
    st.sidebar.markdown("**Smash Score Range**")
    sc1, sc2 = st.sidebar.columns(2)
    min_smash = sc1.number_input("Min Smash", value=0.0, min_value=0.0, max_value=100.0, step=1.0, key="min_smash")
    max_smash = sc2.number_input("Max Smash", value=100.0, min_value=0.0, max_value=100.0, step=1.0, key="max_smash")
else:
    min_smash, max_smash = 0.0, 100.0

# ── Apply filters ──
df_f = df.copy()

# ── TOP-LEVEL SCOPE FILTER (applied first, before all sub-filters) ──
if status_scope == "Exclude Expired":
    df_f = df_f[df_f['status'] != 'Expired']

# ── Preset filters ──
if preset == "NBA Props Only":         df_f = df_f[(df_f['league']=='NBA')&(df_f['bet_type']=='Player Prop')]
elif preset == "3+ Consensus Only":    df_f = df_f[df_f['consensus']>=3]
elif preset == "Exclude Fanatics":     df_f = df_f[df_f['play_book']!='Fanatics']
elif preset == "Best Edges (DIAMOND + GOLD)": df_f = df_f[df_f['tier'].isin(['DIAMOND','GOLD'])]
elif preset == "Prop Unders Only":     df_f = df_f[df_f['is_prop_under']]
elif preset == "Alerted Bets Only":
    if 'alerted' in df_f.columns:
        df_f = df_f[df_f['alerted'] == True]
    else:
        st.sidebar.warning("No 'alerted' column found in database.")

if len(date_range)==2:
    df_f = df_f[(df_f['timestamp'].dt.date>=date_range[0])&(df_f['timestamp'].dt.date<=date_range[1])]
df_f = df_f[df_f['league'].isin(sel_leagues)]
df_f = df_f[df_f['tier'].isin(sel_tiers)]
df_f = df_f[df_f['primary_sharp'].isin(sel_sharps)]
df_f = df_f[df_f['play_book'].isin(sel_books)]
df_f = df_f[df_f['bet_type'].isin(sel_types)]
df_f = df_f[(df_f['consensus']>=cons_range[0])&(df_f['consensus']<=cons_range[1])]
df_f = df_f[(df_f['odds_val']>=min_odds)&(df_f['odds_val']<=max_odds)]
if HAS_MY_SCORE_SIDEBAR and (min_edge > 0 or max_edge < 100):
    df_f = df_f[df_f['edge_score'].notna() & (df_f['edge_score'] >= min_edge) & (df_f['edge_score'] <= max_edge)]
if HAS_GEM_SCORE_SIDEBAR and (min_gem > 0.0 or max_gem < 100.0):
    df_f = df_f[df_f['gem_score'].notna() & (df_f['gem_score'] >= min_gem) & (df_f['gem_score'] <= max_gem)]
if HAS_SMASH_SCORE_SIDEBAR and (min_smash > 0.0 or max_smash < 100.0):
    df_f = df_f[df_f['smash_score'].notna() & (df_f['smash_score'] >= min_smash) & (df_f['smash_score'] <= max_smash)]

closed = df_f[df_f['status'].isin(['Won','Lost','Push'])].copy()

# ── Score column detection ──
HAS_MY_SCORE  = 'edge_score' in df_f.columns and df_f['edge_score'].notna().sum() > 0
HAS_GEM_SCORE = 'gem_score'  in df_f.columns and df_f['gem_score'].notna().sum() > 0
HAS_SMASH_SCORE = 'smash_score' in df_f.columns and df_f['smash_score'].notna().sum() > 0

# ── Top metrics ──
total_profit  = closed['profit'].sum() if not closed.empty else 0
total_wagered = len(closed) * UNIT_SIZE
roi_overall   = (total_profit/total_wagered*100) if total_wagered>0 else 0
win_rate      = (closed['status']=='Won').sum()/max(len(closed[closed['status']!='Push']),1)*100
pending_n     = len(df_f[df_f['status'].isin(['Open','Pending'])])
expired_n     = len(df_f[df_f['status'] == 'Expired'])

# likely_missed count — only meaningful when expired bets are shown
if 'likely_missed' in df_f.columns:
    likely_missed_n = int(df_f[(df_f['status'] == 'Expired') & (df_f['likely_missed'] == True)].shape[0])
else:
    likely_missed_n = 0

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("Total Bets",  f"{len(df_f):,}")
c2.metric("Settled",     f"{len(closed):,}")
c3.metric("Pending",     f"{pending_n:,}")
c4.metric("Profit",      f"${total_profit:,.0f}", delta=f"{roi_overall:.1f}% ROI")
c5.metric("Win Rate",    f"{win_rate:.1f}%")
c6.metric("Filter",      preset if preset!="All Bets" else selected_window)
c7.metric("Expired",     f"{expired_n:,}",
          delta=f"{likely_missed_n} likely missed" if likely_missed_n > 0 else None,
          delta_color="inverse" if likely_missed_n > 0 else "normal")
st.markdown("---")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
(tab_log, tab_tier, tab_analysis, tab_props, tab_odds,
 tab_rolling, tab_leaderboard, tab_sharps, tab_edge, tab_sim, tab_parlays) = st.tabs([
    "📊 Live Log","💎 Tier Performance","📈 Deep Dive",
    "🏀 Prop Breakdown","🎲 Odds Analysis","📉 Rolling ROI",
    "🏆 Leaderboard","🤝 Sharp Agreement","🎯 Edge Scores", "🧪 Simulator", "🎰 Parlays"
])


# ─── LIVE LOG ───────────────────────────────────────────────
with tab_log:
    st.subheader("Bet History")
    base_cols = ['timestamp','tier','league','matchup','bet_type','prop_cat',
                 'play_selection','bet_side','play_odds','play_book',
                 'primary_sharp','consensus','status','profit']
    # Show likely_missed in the log when expired bets are included
    extra_cols = []
    if status_scope == "All Bets (incl. Expired)" and 'likely_missed' in df_f.columns:
        extra_cols = ['likely_missed']
    score_cols = [c for c in ['edge_score','gem_score','smash_score'] if c in df_f.columns and df_f[c].notna().any()]
    show_cols  = [c for c in base_cols + extra_cols + score_cols if c in df_f.columns]

    col_config = {
        "profit":    st.column_config.NumberColumn("Profit",  format="$%.2f"),
        "play_odds": st.column_config.NumberColumn("Odds",    format="%d"),
        "consensus": st.column_config.NumberColumn("# Books", width="small"),
        "tier":      st.column_config.TextColumn("Tier",      width="small"),
    }
    if 'likely_missed' in show_cols:
        col_config["likely_missed"] = st.column_config.CheckboxColumn("Likely Missed?", width="small")
    if HAS_MY_SCORE:
        col_config["edge_score"] = st.column_config.ProgressColumn(
            "Edge Score", min_value=0, max_value=100, format="%.0f")
    if HAS_GEM_SCORE:
        col_config["gem_score"] = st.column_config.ProgressColumn(
            "Gem Score", min_value=0, max_value=70, format="%.1f")
    if HAS_SMASH_SCORE:
        col_config["smash_score"] = st.column_config.ProgressColumn(
            "Smash Score", min_value=0, max_value=100, format="%.1f")

    st.dataframe(df_f[show_cols].sort_values('timestamp', ascending=False),
                 use_container_width=True, column_config=col_config)


# ─── TIER PERFORMANCE ────────────────────────────────────────
with tab_tier:
    st.subheader("💎 Performance by Tier")
    if closed.empty:
        st.warning("No settled bets in current filter.")
    else:
        tier_stats = calc_roi(closed,'tier',min_n=2)
        tier_stats['tier'] = pd.Categorical(tier_stats['tier'],categories=TIER_ORDER,ordered=True)
        tier_stats = tier_stats.sort_values('tier')

        cols_t = st.columns(len(TIER_ORDER))
        for i, tier in enumerate(TIER_ORDER):
            row = tier_stats[tier_stats['tier']==tier]
            em  = TIER_EMOJI.get(tier,'')
            if row.empty: cols_t[i].metric(f"{em} {tier}","N/A")
            else:
                r = row.iloc[0]
                cols_t[i].metric(f"{em} {tier}",f"{r['roi']:.1f}% ROI",f"N={r['n']:,} · WR {r['wr']:.0f}%")

        st.markdown("---")
        col_t1,col_t2 = st.columns(2)
        with col_t1:
            metric = 'roi' if metric_mode=="ROI (%)" else 'profit'
            st.plotly_chart(bar(tier_stats,'tier',metric,"ROI by Tier",
                text_fmt='roi' if metric=='roi' else 'profit'),use_container_width=True)
        with col_t2:
            ts = closed[closed['timestamp'].notna()].sort_values('timestamp')
            lines = []
            for tier in TIER_ORDER:
                sub = ts[ts['tier']==tier].copy()
                if len(sub)<5: continue
                sub['cum']=sub['profit'].cumsum(); sub['lbl']=tier
                lines.append(sub[['timestamp','cum','lbl']])
            if lines:
                tdf = pd.concat(lines)
                fig2 = px.line(tdf,x='timestamp',y='cum',color='lbl',
                    title="Cumulative Profit by Tier",color_discrete_map=TIER_COLORS)
                fig2.update_layout(**LAYOUT)
                st.plotly_chart(fig2,use_container_width=True)

        disp = tier_stats.copy()
        disp['roi']    = disp['roi'].map('{:+.1f}%'.format)
        disp['profit'] = disp['profit'].map('${:,.0f}'.format)
        disp['wr']     = disp['wr'].map('{:.1f}%'.format)
        st.dataframe(disp.rename(columns={'n':'Bets','roi':'ROI','profit':'Profit','wr':'Win Rate'}),
                     use_container_width=True,hide_index=True)
        st.markdown("---")
        for _,r in tier_stats.iterrows():
            em=TIER_EMOJI.get(r['tier'],''); color=TIER_COLORS.get(r['tier'],'#58a6ff')
            st.markdown(f'<div class="insight-card" style="border-left-color:{color}">'
                f'{em} <b>{r["tier"]}</b> — {r["roi"]:+.1f}% ROI · {r["n"]:,} bets · '
                f'{r["wr"]:.0f}% WR · ${r["profit"]:,.0f}</div>',unsafe_allow_html=True)


# ─── DEEP DIVE ───────────────────────────────────────────────
with tab_analysis:
    if closed.empty:
        st.warning("No settled bets.")
    else:
        st.subheader("🔥 League × Bet Type Heatmap")
        hm = closed.groupby(['league','bet_type'])['profit'].agg(
            lambda x: (x.sum()/(len(x)*UNIT_SIZE))*100).reset_index()
        hm.columns=['league','bet_type','roi']
        fig_hm = px.density_heatmap(hm,x='bet_type',y='league',z='roi',
            color_continuous_scale='RdYlGn',text_auto='.1f',range_color=[-30,30])
        fig_hm.update_layout(**LAYOUT)
        st.plotly_chart(fig_hm,use_container_width=True)

        st.markdown("---")
        st.subheader("⚖️ Current TWROI Breakdown")
        st.caption("Visualizing the Time-Weighted ROI (50% 7-day, 30% 14-day, 20% 30-day) for your current filters.")
        
        now = pd.Timestamp.now(tz='US/Eastern')
        s30_df = closed[closed['timestamp'] >= now - pd.Timedelta(days=30)]
        s14_df = closed[closed['timestamp'] >= now - pd.Timedelta(days=14)]
        s7_df  = closed[closed['timestamp'] >= now - pd.Timedelta(days=7)]

        def _get_roi(sub_df):
            if sub_df.empty: return 0.0
            return (sub_df['profit'].sum() / (len(sub_df) * UNIT_SIZE)) * 100

        roi_30 = _get_roi(s30_df)
        roi_14 = _get_roi(s14_df)
        roi_7  = _get_roi(s7_df)
        calc_twroi = (roi_7 * 0.50) + (roi_14 * 0.30) + (roi_30 * 0.20)

        twroi_data = pd.DataFrame({
            'Window': ['30-Day (20% weight)', '14-Day (30% weight)', '7-Day (50% weight)', 'Final TWROI'],
            'ROI': [roi_30, roi_14, roi_7, calc_twroi]
        })

        fig_twroi = go.Figure(go.Bar(
            x=twroi_data['Window'], 
            y=twroi_data['ROI'],
            marker_color=[roi_color(r) for r in twroi_data['ROI']],
            text=[f"{r:+.1f}%" for r in twroi_data['ROI']],
            textposition='outside', 
            textfont=dict(size=13, weight='bold')
        ))
        fig_twroi.add_hline(y=0, line_color='#30363d', line_width=2)
        fig_twroi.add_shape(type="rect", x0=2.5, x1=3.5, y0=min(0, calc_twroi - 2), y1=max(0, calc_twroi + 2),
                            line=dict(color="#58a6ff", width=2, dash="dash"), fillcolor="rgba(0,0,0,0)")
        fig_twroi.update_layout(**LAYOUT, height=300, yaxis_title="ROI (%)")
        st.plotly_chart(fig_twroi, use_container_width=True)
        st.markdown("---")

        metric = 'roi' if metric_mode=="ROI (%)" else 'profit'
        tf = 'roi' if metric=='roi' else 'profit'
        col_a,col_b = st.columns(2)
        with col_a:
            st.subheader("By League")
            st.plotly_chart(bar(calc_roi(closed,'league',10),'league',metric,text_fmt=tf),use_container_width=True, key="chart_league")
        with col_b:
            st.subheader("By Bet Type")
            st.plotly_chart(bar(calc_roi(closed,'bet_type',5),'bet_type',metric,text_fmt=tf),use_container_width=True, key="chart_bet_type")
        
        col_c,col_d = st.columns(2)
        with col_c:
            st.subheader("By Play Book")
            st.plotly_chart(bar(calc_roi(closed,'play_book',10),'play_book',metric,text_fmt=tf),use_container_width=True, key="chart_playbook")
        with col_d:
            st.subheader("By Sharp Book Signal")
            st.plotly_chart(bar(calc_roi(closed,'primary_sharp',10),'primary_sharp',metric,text_fmt=tf),use_container_width=True, key="chart_sharp")
        
        st.subheader("By Consensus Count")
        cs_stats = calc_roi(closed,'consensus',5).sort_values('consensus')
        cs_stats['consensus'] = cs_stats['consensus'].astype(str)+' books'
        st.plotly_chart(bar(cs_stats,'consensus',metric,text_fmt=tf,h=240),use_container_width=True, key="chart_consensus")

        st.subheader("By Time of Day (Hour)")
        closed_hr = closed.copy()
        if 'timestamp' in closed_hr.columns:
            closed_hr['hour'] = closed_hr['timestamp'].dt.hour
            hr_stats = calc_roi(closed_hr, 'hour', 5).sort_values('hour')
            hr_stats['hour_lbl'] = hr_stats['hour'].apply(lambda h: f"{h%12 or 12} {'AM' if h < 12 else 'PM'}")
            fig_hr = bar(hr_stats, 'hour_lbl', metric, text_fmt=tf, h=280)
            fig_hr.update_xaxes(categoryorder='array', categoryarray=hr_stats['hour_lbl'])
            st.plotly_chart(fig_hr, use_container_width=True, key="chart_hour")

        st.markdown("---")
        st.subheader("💧 Liquidity Sweet Spot")
        st.caption("Find the exact market depth where your edge is strongest.")
        
        liq_df = closed[(closed['liquidity'] > 0) & (closed['liquidity'].notna())].copy()
        
        if not liq_df.empty:
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                fig_liq = px.scatter(
                    liq_df, x='liquidity', y='profit', color='tier',
                    color_discrete_map=TIER_COLORS, opacity=0.65,
                    hover_data={'league': True, 'bet_type': True, 'play_selection': True, 'odds_val': True},
                    title="Every Bet: Profit vs. Liquidity"
                )
                fig_liq.add_hline(y=0, line_color='#30363d', line_width=2)
                fig_liq.update_layout(**LAYOUT, height=350, xaxis_title="Market Liquidity ($)", yaxis_title="Profit ($)")
                st.plotly_chart(fig_liq, use_container_width=True)
            with col_l2:
                liq_df['liq_bucket'] = pd.cut(
                    liq_df['liquidity'], 
                    bins=[0, 1000, 2500, 5000, 10000, 100000], 
                    labels=['Under $1k', '$1k - $2.5k', '$2.5k - $5k', '$5k - $10k', '$10k+']
                )
                liq_stats = calc_roi(liq_df, 'liq_bucket', 5)
                liq_stats['liq_bucket'] = pd.Categorical(
                    liq_stats['liq_bucket'], 
                    categories=['Under $1k', '$1k - $2.5k', '$2.5k - $5k', '$5k - $10k', '$10k+'], 
                    ordered=True
                )
                liq_stats = liq_stats.sort_values('liq_bucket')
                metric_to_plot = 'roi' if metric_mode == "ROI (%)" else 'profit'
                text_format = 'roi' if metric_to_plot == 'roi' else 'profit'
                fig_liq_bar = bar(liq_stats, 'liq_bucket', metric_to_plot, title="ROI by Liquidity Bucket", text_fmt=text_format, h=350)
                st.plotly_chart(fig_liq_bar, use_container_width=True)
        else:
            st.info("No liquidity data available for settled bets.")


# ─── PROP BREAKDOWN ──────────────────────────────────────────
with tab_props:
    props_closed = closed[closed['bet_type']=='Player Prop'].copy()
    st.subheader("🏀 Player Prop Analysis")
    if props_closed.empty:
        st.warning("No settled prop bets in current filter.")
    else:
        col_p1,col_p2 = st.columns(2)
        with col_p1:
            side_s = calc_roi(props_closed,'bet_side',5)
            fig_side = go.Figure()
            for _,r in side_s.iterrows():
                fig_side.add_trace(go.Bar(x=[r['bet_side']],y=[r['roi']],
                    marker_color=roi_color(r['roi']),
                    text=f"{r['roi']:+.1f}%<br>N={r['n']:,}<br>WR {r['wr']:.0f}%",
                    textposition='inside',name=r['bet_side']))
            fig_side.add_hline(y=0,line_color='#30363d')
            fig_side.update_layout(**LAYOUT,title="Prop Over vs Under",showlegend=False,height=300)
            st.plotly_chart(fig_side,use_container_width=True)
        with col_p2:
            pc_s = calc_roi(props_closed,'prop_cat',10).head(8)
            st.plotly_chart(hbar(pc_s,'roi','prop_cat',"Top Prop Categories",h=300),use_container_width=True)

        st.subheader("Prop Category × Over/Under")
        cross = props_closed.groupby(['prop_cat','bet_side']).agg(
            profit=('profit','sum'),n=('profit','count')).reset_index()
        cross['roi'] = (cross['profit']/(cross['n']*UNIT_SIZE))*100
        cross = cross[cross['n']>=5]
        fig_x = px.bar(cross,x='prop_cat',y='roi',color='bet_side',barmode='group',
            color_discrete_map={'Over':'#f87171','Under':'#4ade80','Other':'#8b949e'},text_auto='.1f')
        fig_x.add_hline(y=0,line_color='#30363d',line_width=2)
        fig_x.update_layout(**LAYOUT,height=350,xaxis_tickangle=-30)
        st.plotly_chart(fig_x,use_container_width=True)

        st.subheader("NBA Prop Unders — Category Detail")
        nba_u = props_closed[(props_closed['league']=='NBA')&(props_closed['bet_side']=='Under')]
        if not nba_u.empty:
            nba_cat = calc_roi(nba_u,'prop_cat',5).sort_values('roi',ascending=False)
            st.plotly_chart(hbar(nba_cat,'roi','prop_cat',"NBA Under ROI by Category",
                h=max(300,len(nba_cat)*35)),use_container_width=True)
            disp = nba_cat.copy()
            disp['roi']=disp['roi'].map('{:+.1f}%'.format)
            disp['profit']=disp['profit'].map('${:,.0f}'.format)
            disp['wr']=disp['wr'].map('{:.1f}%'.format)
            st.dataframe(disp[['prop_cat','n','roi','profit','wr']].rename(
                columns={'prop_cat':'Category','n':'Bets','roi':'ROI','profit':'Profit','wr':'WR'}),
                use_container_width=True,hide_index=True)


# ─── ODDS ANALYSIS ───────────────────────────────────────────
with tab_odds:
    st.subheader("🎲 Profitability by Odds Range")
    if closed.empty:
        st.warning("No settled bets.")
    else:
        odds_s = closed.groupby('odds_bucket').agg(profit=('profit','sum'),n=('profit','count')).reset_index()
        odds_s['roi']=(odds_s['profit']/(odds_s['n']*UNIT_SIZE))*100
        odds_s['odds_bucket']=pd.Categorical(odds_s['odds_bucket'],categories=ODDS_BUCKET_ORDER,ordered=True)
        odds_s=odds_s.sort_values('odds_bucket')
        metric='roi' if metric_mode=="ROI (%)" else 'profit'
        st.plotly_chart(bar(odds_s,'odds_bucket',metric,
            text_fmt='roi' if metric=='roi' else 'profit',h=320),use_container_width=True)
        disp=odds_s.copy()
        disp['roi']=disp['roi'].map('{:+.1f}%'.format)
        disp['profit']=disp['profit'].map('${:,.2f}'.format)
        st.dataframe(disp.rename(columns={'odds_bucket':'Odds Range','n':'Bets','roi':'ROI','profit':'Profit'}),
                     use_container_width=True,hide_index=True)


# ─── ROLLING ROI ─────────────────────────────────────────────
with tab_rolling:
    st.subheader("📉 Rolling ROI Trend")
    if closed.empty or 'timestamp' not in closed.columns:
        st.warning("No data.")
    else:
        roll_df=closed[closed['timestamp'].notna()].sort_values('timestamp').copy()
        cw,cb=st.columns(2)
        window=cw.slider("Rolling window (bets)",10,200,50,step=10)
        roll_by=cb.radio("Group by",["All Bets","By Tier"],horizontal=True)

        if roll_by=="All Bets":
            roll_df['rolling_roi']=(roll_df['profit']
                .rolling(window,min_periods=max(5,window//4)).mean()/UNIT_SIZE*100)
            roll_df['cum_profit']=roll_df['profit'].cumsum()
            fig_r=go.Figure(go.Scatter(x=roll_df['timestamp'],y=roll_df['rolling_roi'],
                name=f'{window}-bet Rolling ROI',line=dict(color='#58a6ff',width=2)))
            fig_r.add_hline(y=0,line_color='#30363d',line_dash='dash')
            fig_r.update_layout(**LAYOUT,title=f"{window}-Bet Rolling ROI",yaxis_title="ROI (%)",height=350)
            st.plotly_chart(fig_r,use_container_width=True)
            last_val=roll_df['cum_profit'].iloc[-1]
            fig_c=go.Figure(go.Scatter(x=roll_df['timestamp'],y=roll_df['cum_profit'],
                fill='tozeroy',line=dict(color='#4ade80' if last_val>=0 else '#f87171',width=2),
                fillcolor='rgba(74,222,128,0.1)'))
            fig_c.add_hline(y=0,line_color='#30363d')
            fig_c.update_layout(**LAYOUT,title="Cumulative Profit",height=280)
            st.plotly_chart(fig_c,use_container_width=True)
        else:
            fig_tr=go.Figure()
            for tier in TIER_ORDER:
                sub=roll_df[roll_df['tier']==tier].copy()
                if len(sub)<window//2: continue
                sub['rr']=(sub['profit'].rolling(window,min_periods=max(5,window//4)).mean()/UNIT_SIZE*100)
                fig_tr.add_trace(go.Scatter(x=sub['timestamp'],y=sub['rr'],
                    name=f"{TIER_EMOJI.get(tier,'')} {tier}",
                    line=dict(color=TIER_COLORS.get(tier,'#58a6ff'),width=2)))
            fig_tr.add_hline(y=0,line_color='#30363d',line_dash='dash')
            fig_tr.update_layout(**LAYOUT,title=f"{window}-Bet Rolling ROI by Tier",height=400)
            st.plotly_chart(fig_tr,use_container_width=True)

        st.subheader("Monthly Performance")
        roll_df['month']=roll_df['timestamp'].dt.to_period('M').astype(str)
        mo=roll_df.groupby('month').agg(profit=('profit','sum'),n=('profit','count')).reset_index()
        mo['roi']=(mo['profit']/(mo['n']*UNIT_SIZE))*100
        mo['roi_fmt']=mo['roi'].map('{:+.1f}%'.format)
        mo['profit_fmt']=mo['profit'].map('${:,.0f}'.format)
        st.dataframe(mo[['month','n','roi_fmt','profit_fmt']].rename(
            columns={'month':'Month','n':'Bets','roi_fmt':'ROI','profit_fmt':'Profit'}),
            use_container_width=True,hide_index=True)


# ─── LEADERBOARD ─────────────────────────────────────────────
with tab_leaderboard:
    st.subheader("🏆 Most Profitable Categories")
    if closed.empty:
        st.warning("No settled bets.")
    else:
        lb1,lb2=st.columns([1,3])
        min_bets=lb1.slider("Min Sample",1,100,10)
        sort_by=lb1.radio("Sort by",["ROI","Profit"])
        lb=closed.groupby('combo').agg(profit=('profit','sum'),n=('profit','count'),
            wins=('status',lambda x:(x=='Won').sum())).reset_index()
        lb['roi']=(lb['profit']/(lb['n']*UNIT_SIZE))*100
        lb['wr']=lb['wins']/lb['n'].clip(lower=1)*100
        lb=lb[lb['n']>=min_bets].sort_values('roi' if sort_by=='ROI' else 'profit',ascending=False)
        lb2.metric("Categories shown",f"{len(lb)}")
        disp=lb.copy()
        disp['roi']=disp['roi'].map('{:+.1f}%'.format)
        disp['profit']=disp['profit'].map('${:,.0f}'.format)
        disp['wr']=disp['wr'].map('{:.1f}%'.format)
        st.dataframe(disp[['combo','n','roi','profit','wr']].rename(
            columns={'combo':'Category','n':'Bets','roi':'ROI','profit':'Profit','wr':'Win Rate'}),
            use_container_width=True,height=600,hide_index=True)


# ─── SHARP AGREEMENT ─────────────────────────────────────────
with tab_sharps:
    st.subheader("🤝 Sharp Book Agreement Matrix")
    st.caption("How often each pair of sharp books appear together in the same signal — "
               "and whether that co-occurrence actually outperforms single-book signals.")
    if closed.empty:
        st.warning("No settled bets.")
    else:
        ALL_BOOKS = ['Prophet','NoVigApp','Pinnacle','4cx','Polymarket','Kalshi']
        for b in ALL_BOOKS:
            closed[f'_has_{b}'] = closed['sharp_book'].apply(lambda x: b in str(x))

        co_count = pd.DataFrame(0,index=ALL_BOOKS,columns=ALL_BOOKS)
        co_roi   = pd.DataFrame(np.nan,index=ALL_BOOKS,columns=ALL_BOOKS)
        for i,b1 in enumerate(ALL_BOOKS):
            for j,b2 in enumerate(ALL_BOOKS):
                if i==j:
                    solo = closed[closed[f'_has_{b1}']&(closed['consensus']==1)]
                    co_count.loc[b1,b2]=len(solo)
                    if len(solo)>=5: co_roi.loc[b1,b2]=(solo['profit'].sum()/(len(solo)*UNIT_SIZE))*100
                else:
                    both = closed[closed[f'_has_{b1}']&closed[f'_has_{b2}']]
                    co_count.loc[b1,b2]=len(both)
                    if len(both)>=5: co_roi.loc[b1,b2]=(both['profit'].sum()/(len(both)*UNIT_SIZE))*100

        col_m1,col_m2=st.columns(2)
        hm_l={**LAYOUT,'height':400}
        hm_l['xaxis']=dict(side='bottom',gridcolor='#21262d')
        hm_l['yaxis']=dict(gridcolor='#21262d')
        with col_m1:
            st.markdown("**Co-occurrence Count**")
            fig_cnt=go.Figure(go.Heatmap(z=co_count.values,x=ALL_BOOKS,y=ALL_BOOKS,
                colorscale='Blues',text=co_count.values.astype(int),
                texttemplate='%{text}',textfont=dict(size=12),showscale=True))
            fig_cnt.update_layout(**hm_l)
            st.plotly_chart(fig_cnt,use_container_width=True)
        with col_m2:
            st.markdown("**ROI When Books Agree**")
            fig_roi=go.Figure(go.Heatmap(z=co_roi.values.astype(float),x=ALL_BOOKS,y=ALL_BOOKS,
                colorscale='RdYlGn',zmid=0,zmin=-20,zmax=20,
                text=co_roi.round(1).values,texttemplate='%{text:.1f}%',
                textfont=dict(size=12),showscale=True))
            fig_roi.update_layout(**hm_l)
            st.plotly_chart(fig_roi,use_container_width=True)

        st.subheader("Single-Book vs Multi-Book ROI per Sharp Source")
        rows=[]
        for b in ALL_BOOKS:
            has=closed[closed[f'_has_{b}']]; solo=has[has['consensus']==1]
            multi=has[has['consensus']>=2]; three=has[has['consensus']>=3]
            def _roi(g): return (g['profit'].sum()/(len(g)*UNIT_SIZE)*100) if len(g)>=5 else None
            rows.append({'Sharp Book':b,'All Bets N':len(has),
                'All ROI':f"{_roi(has):+.1f}%" if _roi(has) is not None else '—',
                'Solo N':len(solo),'Solo ROI':f"{_roi(solo):+.1f}%" if _roi(solo) is not None else '—',
                '2+ Books N':len(multi),'2+ ROI':f"{_roi(multi):+.1f}%" if _roi(multi) is not None else '—',
                '3+ Books N':len(three),'3+ ROI':f"{_roi(three):+.1f}%" if _roi(three) is not None else '—'})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

        st.subheader("Most Independent Book Pairs")
        pairs=[]
        for i,b1 in enumerate(ALL_BOOKS):
            for j,b2 in enumerate(ALL_BOOKS):
                if j<=i: continue
                n1=closed[f'_has_{b1}'].sum(); n2=closed[f'_has_{b2}'].sum()
                both=int(co_count.loc[b1,b2])
                if n1<20 or n2<20: continue
                overlap_pct=both/min(n1,n2)*100 if min(n1,n2)>0 else 0
                roi_val=co_roi.loc[b1,b2]
                pairs.append({'Books':f"{b1} + {b2}",'Co-signals':both,
                    'Overlap %':f"{overlap_pct:.1f}%",
                    'Co-signal ROI':f"{roi_val:+.1f}%" if not np.isnan(roi_val) else '—'})
        if pairs:
            st.dataframe(pd.DataFrame(pairs).sort_values('Co-signals'),use_container_width=True,hide_index=True)

        for b in ALL_BOOKS:
            if f'_has_{b}' in closed.columns: closed.drop(columns=[f'_has_{b}'],inplace=True)


# ─────────────────────────────────────────────────────────────
# 🎯 EDGE SCORES TAB
# ─────────────────────────────────────────────────────────────
with tab_edge:
    st.subheader("🎯 Edge Score Analysis")
    st.caption("Validating whether higher scores actually predict better outcomes on your real bets.")

    if not HAS_MY_SCORE and not HAS_GEM_SCORE and not HAS_SMASH_SCORE:
        st.info(
            "No edge scores in the database yet. "
            "Scores are recorded on new bets as they are ingested by the tracker. "
            "Run the tracker then refresh to see data here."
        )
        st.stop()

    settled_e = closed.copy()
    
    base_prof = float(settled_e['profit'].sum())
    base_wager = max(len(settled_e) * float(UNIT_SIZE), 1)
    baseline_roi = (base_prof / base_wager) * 100

    st.markdown("### 📊 Score Bucket → Actual ROI")
    st.caption("The core validation: do higher scores deliver higher ROI on your settled bets?")

    val_c1, val_c2, val_c3 = st.columns(3)

    with val_c1:
        if HAS_MY_SCORE:
            bkt = score_bucket_roi(settled_e, 'edge_score', MY_SCORE_BUCKETS, min_n=2)
            if not bkt.empty:
                fig_v = go.Figure(go.Bar(
                    x=bkt['bucket'], y=bkt['roi'],
                    marker_color=bkt['color'],
                    text=[f"{r:+.1f}%<br>N={n:,}" for r,n in zip(bkt['roi'],bkt['n'])],
                    textposition='outside', textfont=dict(size=11)
                ))
                fig_v.add_hline(y=0, line_color='#30363d', line_width=2)
                fig_v.add_hline(y=baseline_roi, line_color='#58a6ff', line_dash='dash',
                                line_width=1, annotation_text=f"baseline {baseline_roi:+.1f}%",
                                annotation_position="bottom right")
                fig_v.update_layout(**LAYOUT, title="My Edge Score → ROI", height=320, yaxis_title="ROI (%)")
                st.plotly_chart(fig_v, use_container_width=True)
                if len(bkt) >= 2:
                    mono = all(bkt['roi'].iloc[i] <= bkt['roi'].iloc[i+1] for i in range(len(bkt)-1))
                    st.caption(f"Monotonically increasing: {'✅ Yes' if mono else '⚠️ Not yet — needs more forward data'}")

    with val_c2:
        if HAS_GEM_SCORE:
            gbkt = score_bucket_roi(settled_e, 'gem_score', GEM_SCORE_BUCKETS, min_n=2)
            if not gbkt.empty:
                fig_gv = go.Figure(go.Bar(
                    x=gbkt['bucket'], y=gbkt['roi'],
                    marker_color=gbkt['color'],
                    text=[f"{r:+.1f}%<br>N={n:,}" for r,n in zip(gbkt['roi'],gbkt['n'])],
                    textposition='outside', textfont=dict(size=11)
                ))
                fig_gv.add_hline(y=0, line_color='#30363d', line_width=2)
                fig_gv.add_hline(y=baseline_roi, line_color='#f59e0b', line_dash='dash',
                                 line_width=1, annotation_text=f"baseline {baseline_roi:+.1f}%",
                                 annotation_position="bottom right")
                fig_gv.update_layout(**LAYOUT, title="Gem Edge Score → ROI", height=320, yaxis_title="ROI (%)")
                st.plotly_chart(fig_gv, use_container_width=True)

    with val_c3:
        if HAS_SMASH_SCORE:
            sbkt = score_bucket_roi(settled_e, 'smash_score', SMASH_SCORE_BUCKETS, min_n=2)
            if not sbkt.empty:
                fig_sv = go.Figure(go.Bar(
                    x=sbkt['bucket'], y=sbkt['roi'],
                    marker_color=sbkt['color'],
                    text=[f"{r:+.1f}%<br>N={n:,}" for r,n in zip(sbkt['roi'],sbkt['n'])],
                    textposition='outside', textfont=dict(size=11)
                ))
                fig_sv.add_hline(y=0, line_color='#30363d', line_width=2)
                fig_sv.add_hline(y=baseline_roi, line_color='#00ffcc', line_dash='dash',
                                 line_width=1, annotation_text=f"baseline {baseline_roi:+.1f}%",
                                 annotation_position="bottom right")
                fig_sv.update_layout(**LAYOUT, title="Smash Score → ROI", height=320, yaxis_title="ROI (%)")
                st.plotly_chart(fig_sv, use_container_width=True)

    st.markdown("---")

    st.markdown("### 📅 Score Quality Over Time")
    st.caption("Rising average score = tracker is finding better-quality bets.")

    time_df = df_f[df_f['timestamp'].notna()].copy()
    time_df['week'] = time_df['timestamp'].dt.to_period('W').astype(str)

    fig_time = go.Figure()
    if HAS_MY_SCORE:
        wk = time_df.dropna(subset=['edge_score']).groupby('week')['edge_score'].agg(['mean','count']).reset_index()
        wk.columns=['week','avg','n']
        wk = wk[wk['n']>=5]
        fig_time.add_trace(go.Scatter(x=wk['week'],y=wk['avg'],name='My Edge Score (avg)',
            line=dict(color='#4ade80',width=2),mode='lines+markers'))
    if HAS_GEM_SCORE:
        wkg = time_df.dropna(subset=['gem_score']).groupby('week')['gem_score'].agg(['mean','count']).reset_index()
        wkg.columns=['week','avg','n']
        wkg = wkg[wkg['n']>=5]
        wkg['avg_norm'] = wkg['avg'] / 70 * 100
        fig_time.add_trace(go.Scatter(x=wkg['week'],y=wkg['avg_norm'],name='Gem Score (norm. to 100)',
            line=dict(color='#f59e0b',width=2),mode='lines+markers'))
    if HAS_SMASH_SCORE:
        wks = time_df.dropna(subset=['smash_score']).groupby('week')['smash_score'].agg(['mean','count']).reset_index()
        wks.columns=['week','avg','n']
        wks = wks[wks['n']>=5]
        fig_time.add_trace(go.Scatter(x=wks['week'],y=wks['avg'],name='Smash Score (avg)',
            line=dict(color='#00ffcc',width=2),mode='lines+markers'))

    fig_time.update_layout(**LAYOUT,title="Weekly Average Score",height=300,
                            yaxis_title="Score (scaled)",xaxis_tickangle=-30)
    st.plotly_chart(fig_time, use_container_width=True)

    hist_c1, hist_c2, hist_c3 = st.columns(3)
    with hist_c1:
        if HAS_MY_SCORE:
            fig_h = go.Figure(go.Histogram(x=df_f['edge_score'].dropna(),nbinsx=20,
                marker_color='#4ade80',opacity=0.8))
            for lo,_,lbl,color in MY_SCORE_BUCKETS[1:]:
                fig_h.add_vline(x=lo,line_color=color,line_dash='dash',line_width=1,
                    annotation_text=lbl,annotation_position="top right",annotation_font_size=9)
            fig_h.update_layout(**LAYOUT,title="My Edge Score Distribution",height=260,
                                 xaxis_title="Score",yaxis_title="# Bets")
            st.plotly_chart(fig_h, use_container_width=True)
    with hist_c2:
        if HAS_GEM_SCORE:
            fig_gh = go.Figure(go.Histogram(x=df_f['gem_score'].dropna(),nbinsx=20,
                marker_color='#f59e0b',opacity=0.8))
            for lo,_,lbl,color in GEM_SCORE_BUCKETS[1:]:
                fig_gh.add_vline(x=lo,line_color=color,line_dash='dash',line_width=1,
                    annotation_text=lbl,annotation_position="top right",annotation_font_size=9)
            fig_gh.update_layout(**LAYOUT,title="Gem Score Distribution",height=260,
                                  xaxis_title="Score",yaxis_title="# Bets")
            st.plotly_chart(fig_gh, use_container_width=True)
    with hist_c3:
        if HAS_SMASH_SCORE:
            fig_sh = go.Figure(go.Histogram(x=df_f['smash_score'].dropna(),nbinsx=20,
                marker_color='#00ffcc',opacity=0.8))
            for lo,_,lbl,color in SMASH_SCORE_BUCKETS[1:]:
                fig_sh.add_vline(x=lo,line_color=color,line_dash='dash',line_width=1,
                    annotation_text=lbl,annotation_position="top right",annotation_font_size=9)
            fig_sh.update_layout(**LAYOUT,title="Smash Score Distribution",height=260,
                                  xaxis_title="Score",yaxis_title="# Bets")
            st.plotly_chart(fig_sh, use_container_width=True)

    st.markdown("---")

    st.markdown("### 🏆 Best Scoring Book × Market Combinations")
    st.caption("Which combos score highest AND deliver real ROI?")

    if HAS_MY_SCORE and not settled_e.empty:
        if 'bet_side' not in settled_e.columns:
            settled_e['bet_side'] = 'Other'
        settled_e['bms_key'] = (settled_e['play_book'].fillna('')+' · '+
                                 settled_e['market'].fillna('')+' · '+
                                 settled_e['bet_side'].fillna(''))

        bms_min = st.slider("Min bets per combo", 5, 50, 10, key='bms_min')
        
        def safe_roi(x):
            val = x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x
            prof_val = pd.to_numeric(val, errors='coerce').fillna(0.0).sum()
            return (float(prof_val) / (len(val) * float(UNIT_SIZE))) * 100

        combo_s = settled_e.groupby('bms_key').agg(
            avg_score=('edge_score','mean'),
            roi=('profit', safe_roi),
            n=('profit','count'),
            wr=('status',lambda x: (x=='Won').sum()/max(len(x),1)*100),
        ).reset_index()
        combo_s = combo_s[combo_s['n']>=bms_min].sort_values('avg_score',ascending=False).head(20)

        fig_bms = go.Figure(go.Bar(
            x=combo_s['avg_score'], y=combo_s['bms_key'], orientation='h',
            marker_color=[roi_color(r) for r in combo_s['roi']],
            text=[f"Score:{s:.0f}  ROI:{r:+.1f}%  N={n}"
                  for s,r,n in zip(combo_s['avg_score'],combo_s['roi'],combo_s['n'])],
            textposition='outside', textfont=dict(size=10),
        ))
        fig_bms.update_layout(**LAYOUT,
            title="Top 20 Combos by Avg Edge Score  (bar color = actual ROI)",
            height=max(380, len(combo_s)*28), xaxis_title="Avg Edge Score"
        )
        fig_bms.update_xaxes(range=[0,100])
        st.plotly_chart(fig_bms, use_container_width=True)

        disp = combo_s.copy()
        disp['avg_score'] = disp['avg_score'].map('{:.0f}'.format)
        disp['roi']       = disp['roi'].map('{:+.1f}%'.format)
        disp['wr']        = disp['wr'].map('{:.0f}%'.format)
        st.dataframe(disp.rename(columns={'bms_key':'Combo','avg_score':'Avg Score',
                                           'n':'Bets','roi':'ROI','wr':'WR'}),
                     use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("### ⚖️ Model Comparison — Head to Head")
    st.caption("How do the models stack up when filtering your real bets?")

    if HAS_MY_SCORE and HAS_GEM_SCORE:
        both = settled_e.dropna(subset=['edge_score','gem_score']).copy()
        both = both.loc[:, ~both.columns.duplicated()]
        both['profit'] = pd.to_numeric(both['profit'], errors='coerce').fillna(0.0).astype(float)
        
        if len(both) >= 20:
            sample = both.sample(min(2000,len(both)),random_state=42)
            fig_sc = px.scatter(sample, x='edge_score', y='gem_score', color='status',
                color_discrete_map={'Won':'#4ade80','Lost':'#f87171'},
                opacity=0.45, title="My Score vs Gem Score — each dot = one settled bet",
                labels={'edge_score':'My Edge Score (0-100)','gem_score':'Gem Score (0-70)'})
            fig_sc.add_vline(x=45, line_dash='dash', line_color='#4ade80', line_width=1,
                             annotation_text="My FAIR", annotation_font_size=10)
            fig_sc.add_hline(y=54, line_dash='dash', line_color='#f59e0b', line_width=1,
                             annotation_text="Gem BET", annotation_font_size=10)
            fig_sc.update_layout(**LAYOUT, height=400)
            st.plotly_chart(fig_sc, use_container_width=True)

            st.markdown("**Cumulative Profit: Which Filter Wins?**")
            ts = both.sort_values('timestamp').copy()
            
            prof_data_ts = ts['profit'].iloc[:, 0] if isinstance(ts['profit'], pd.DataFrame) else ts['profit']
            ts['safe_profit'] = pd.to_numeric(prof_data_ts, errors='coerce').fillna(0.0).astype(float)
            
            MY_T, GEM_T = 45, 54
            ts['pnl_both'] = ts.apply(lambda r: float(r['safe_profit']) if (r['edge_score']>=MY_T and r['gem_score']>=GEM_T) else 0.0, axis=1).cumsum()
            ts['pnl_my']   = ts.apply(lambda r: float(r['safe_profit']) if r['edge_score']>=MY_T else 0.0, axis=1).cumsum()
            ts['pnl_gem']  = ts.apply(lambda r: float(r['safe_profit']) if r['gem_score']>=GEM_T else 0.0, axis=1).cumsum()
            
            if HAS_SMASH_SCORE:
                ts['pnl_smash'] = ts.apply(lambda r: float(r['safe_profit']) if pd.notna(r.get('smash_score')) and r.get('smash_score', 0)>=55 else 0.0, axis=1).cumsum()
            
            ts['pnl_all']  = ts['safe_profit'].cumsum()

            fig_cum = go.Figure()
            if HAS_SMASH_SCORE:
                fig_cum.add_trace(go.Scatter(x=ts['timestamp'],y=ts['pnl_smash'],name='🤖 Smash score ≥55',
                    line=dict(color='#00ffcc',width=2)))
            fig_cum.add_trace(go.Scatter(x=ts['timestamp'],y=ts['pnl_both'],name='✅ Both agree (My+Gem)',
                line=dict(color='#00ff9f',width=2)))
            fig_cum.add_trace(go.Scatter(x=ts['timestamp'],y=ts['pnl_my'],name='📊 My score ≥45',
                line=dict(color='#4ade80',width=1,dash='dot')))
            fig_cum.add_trace(go.Scatter(x=ts['timestamp'],y=ts['pnl_gem'],name='💎 Gem score ≥54',
                line=dict(color='#f59e0b',width=1,dash='dot')))
            fig_cum.add_trace(go.Scatter(x=ts['timestamp'],y=ts['pnl_all'],name='All bets',
                line=dict(color='#8b949e',width=1)))
            fig_cum.add_hline(y=0,line_color='#30363d')
            fig_cum.update_layout(**LAYOUT,title="Cumulative Profit by Filter Strategy",
                                   height=350,yaxis_title="Profit ($)")
            st.plotly_chart(fig_cum, use_container_width=True)

    elif HAS_MY_SCORE:
        st.info("Gem score not in DB yet. Train and run: `python3 gem_edge_score.py train data/bets.csv`")
    elif HAS_GEM_SCORE:
        st.info("My edge score not in DB yet.")

    st.markdown("---")
    st.markdown("### 📋 Quick Summary")
    
    cols = st.columns(6)
    idx = 0
    if HAS_MY_SCORE:
        n_scored   = df_f['edge_score'].notna().sum()
        n_high     = (df_f['edge_score']>=65).sum()
        cols[idx].metric("Bets with Score",  f"{n_scored:,}")
        cols[idx+1].metric("High Conf (65+)",  f"{n_high:,}", delta=f"{n_high/max(n_scored,1)*100:.0f}% of scored")
        idx += 2
    if HAS_GEM_SCORE:
        n_gem_bet  = (df_f['gem_score']>=54).sum()
        n_gem_t1   = ((df_f['gem_score']>=54)&(df_f['gem_score']<56)).sum()
        cols[idx].metric("Gem BET signals",  f"{n_gem_bet:,}")
        cols[idx+1].metric("Gem Tier 1 (54-56)",f"{n_gem_t1:,}", delta="1-unit signals")
        idx += 2
    if HAS_SMASH_SCORE:
        n_smash_play   = (df_f['smash_score']>=55).sum()
        n_smash_strong = (df_f['smash_score']>=58).sum()
        cols[idx].metric("Smash PLAY (55+)",  f"{n_smash_play:,}")
        cols[idx+1].metric("Smash STRONG (58+)",f"{n_smash_strong:,}", delta="Top tier signals")


# ─── TWROI SIMULATOR ─────────────────────────────────────────
with tab_sim:
    st.subheader("🧪 Historical TWROI Simulator")
    st.caption("Runs a point-in-time simulation to see how your profit and ROI change if you strictly require positive TWROI and positive Book TWROI.")
    
    st.info("💡 **Required:** Set the **Data Window** in the left sidebar to **'All time'** before running.")

    sim_c1, sim_c2 = st.columns(2)
    group_by_opt = sim_c1.radio("Group results by:", ["Tier", "Edge Score Bucket", "Smash Score Bucket"], horizontal=True)

    default_start = datetime.now().replace(day=1).date()
    start_date = sim_c2.date_input("Simulation Start Date", value=default_start)

    alerted_only = st.checkbox("Alerted bets only (simulate betting everything that fired with positive book & market TWROI)", value=False)

    if st.button("▶️ Run Historical Simulation on Current Filters", type="primary"):
        with st.spinner("Running point-in-time calculations (this takes a few seconds)..."):

            hist_df = df[df['status'].isin(['Won', 'Lost', 'Push'])].copy()
            hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], utc=True)

            target_bets = df_f[df_f['status'].isin(['Won', 'Lost'])].copy()
            target_bets['timestamp'] = pd.to_datetime(target_bets['timestamp'], utc=True)
            target_bets = target_bets.sort_values('timestamp')

            if alerted_only and 'alerted' in target_bets.columns:
                target_bets = target_bets[target_bets['alerted'] == True]
            
            if group_by_opt == "Edge Score Bucket":
                target_bets['sim_group'] = pd.cut(target_bets['edge_score'], 
                                             bins=[-1, 35, 54, 69, 100], 
                                             labels=['<35 (Avoid/Lean)', '35-54 (Fair)', '55-69 (Good)', '70+ (High)'])
            elif group_by_opt == "Smash Score Bucket":
                target_bets['sim_group'] = pd.cut(target_bets['smash_score'], 
                                             bins=[-1, 52, 55, 58, 100], 
                                             labels=['<52 (Skip)', '52-55 (Lean)', '55-58 (Play)', '58+ (Smash)'])
            else:
                target_bets['sim_group'] = target_bets['tier'].fillna("Unknown")

            target_dt = pd.to_datetime(start_date, utc=True)
            target_bets = target_bets[target_bets['timestamp'] >= target_dt]

            if target_bets.empty:
                st.warning("No settled bets found in your current filter after this date.")
            else:
                progress_bar = st.progress(0)
                total_bets = len(target_bets)
                results = []
                
                for i, (idx, bet) in enumerate(target_bets.iterrows()):
                    if i % 50 == 0:
                        progress_bar.progress(min(i / total_bets, 1.0))
                        
                    past_bets = hist_df[hist_df['timestamp'] < bet['timestamp']]
                    subset = past_bets[
                        (past_bets['league'] == bet['league']) &
                        (past_bets['bet_type'] == bet['bet_type']) &
                        (past_bets['bet_side'] == bet['bet_side'])
                    ]
                    
                    t = bet['timestamp']
                    s30_df = subset[subset['timestamp'] >= t - pd.Timedelta(days=30)]
                    s14_df = subset[subset['timestamp'] >= t - pd.Timedelta(days=14)]
                    s7_df  = subset[subset['timestamp'] >= t - pd.Timedelta(days=7)]
                    
                    def calc_roi_sim(sub):
                        if len(sub) == 0: return 0.0
                        return (sub['profit'].sum() / (len(sub) * UNIT_SIZE)) * 100
                        
                    twroi = (calc_roi_sim(s7_df)*0.5) + (calc_roi_sim(s14_df)*0.3) + (calc_roi_sim(s30_df)*0.2)
                    
                    bk30_df = s30_df[s30_df['play_book'].astype(str).str.contains(str(bet['play_book']), case=False, na=False)]
                    bk14_df = s14_df[s14_df['play_book'].astype(str).str.contains(str(bet['play_book']), case=False, na=False)]
                    bk7_df  = s7_df[s7_df['play_book'].astype(str).str.contains(str(bet['play_book']), case=False, na=False)]
                    
                    bk_twroi = (calc_roi_sim(bk7_df)*0.5) + (calc_roi_sim(bk14_df)*0.3) + (calc_roi_sim(bk30_df)*0.2)
                    passed_filter = (twroi > 0) and (bk_twroi > 0) and (len(s30_df) >= 10)
                    
                    results.append({
                        'group': bet['sim_group'],
                        'profit': bet['profit'],
                        'passed': passed_filter
                    })
                
                progress_bar.empty()
                res_df = pd.DataFrame(results)
                
                summary = []
                for grp in res_df['group'].dropna().unique():
                    grp_bets = res_df[res_df['group'] == grp]
                    if grp_bets.empty: continue
                    base_n = len(grp_bets)
                    base_prof = grp_bets['profit'].sum()
                    base_roi = (base_prof / (base_n * UNIT_SIZE)) * 100 if base_n > 0 else 0
                    filt_bets = grp_bets[grp_bets['passed']]
                    filt_n = len(filt_bets)
                    filt_prof = filt_bets['profit'].sum()
                    filt_roi = (filt_prof / (filt_n * UNIT_SIZE)) * 100 if filt_n > 0 else 0
                    summary.append({
                        'Group': grp,
                        'Base Bets': base_n, 'Base ROI (%)': base_roi, 'Base Profit': base_prof,
                        'Filt Bets': filt_n, 'Filt ROI (%)': filt_roi, 'Filt Profit': filt_prof,
                        'ROI Delta': filt_roi - base_roi
                    })
                
                sum_df = pd.DataFrame(summary).sort_values('Base Bets', ascending=False)
                
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Bar(
                    x=sum_df['Group'], y=sum_df['Base ROI (%)'],
                    name='Baseline ROI', marker_color='#30363d',
                    text=[f"{r:+.1f}%" for r in sum_df['Base ROI (%)']], textposition='auto'
                ))
                fig_sim.add_trace(go.Bar(
                    x=sum_df['Group'], y=sum_df['Filt ROI (%)'],
                    name='Filtered ROI (TWROI > 0)', marker_color='#4ade80',
                    text=[f"{r:+.1f}%" for r in sum_df['Filt ROI (%)']], textposition='auto'
                ))
                fig_sim.update_layout(**LAYOUT, barmode='group',
                    title=f"Baseline vs Filtered ROI by {group_by_opt}", height=400)
                st.plotly_chart(fig_sim, use_container_width=True)
                
                disp_df = sum_df.copy()
                disp_df['Base ROI (%)'] = disp_df['Base ROI (%)'].map('{:+.1f}%'.format)
                disp_df['Filt ROI (%)'] = disp_df['Filt ROI (%)'].map('{:+.1f}%'.format)
                disp_df['ROI Delta']    = disp_df['ROI Delta'].map('{:+.1f}%'.format)
                disp_df['Base Profit']  = disp_df['Base Profit'].map('${:,.0f}'.format)
                disp_df['Filt Profit']  = disp_df['Filt Profit'].map('${:,.0f}'.format)
                st.dataframe(disp_df, use_container_width=True, hide_index=True)


# ─── PARLAYS ─────────────────────────────────────────────────
with tab_parlays:
    st.subheader("🎰 Parlay Tracker")
    try:
        pdf = fetch_parlays()
    except Exception as e:
        st.error(f"Could not load parlays: {e}")
        pdf = pd.DataFrame()

    if pdf.empty:
        st.info("No parlays recorded yet. They'll appear here once the tracker fires its first parlay alert.")
    else:
        settled = pdf[pdf['status'].isin(['Won', 'Lost', 'Push'])]
        open_p  = pdf[pdf['status'] == 'Open']

        # ── Summary metrics ──────────────────────────────────────
        if not settled.empty:
            total_n   = len(settled)
            total_w   = (settled['status'] == 'Won').sum()
            total_l   = (settled['status'] == 'Lost').sum()
            total_push= (settled['status'] == 'Push').sum()
            total_pnl = settled['profit'].sum()
            total_roi = total_pnl / (total_n * 100) * 100

            mc = st.columns(5)
            mc[0].metric("Total Parlays", total_n)
            mc[1].metric("Record", f"{total_w}W / {total_l}L / {total_push}P")
            mc[2].metric("ROI", f"{total_roi:+.1f}%")
            mc[3].metric("P&L", f"${total_pnl:+,.0f}", delta=f"{total_pnl/100:+.2f}u")
            mc[4].metric("Open", len(open_p))
            st.markdown("---")

            # ── By leg count ────────────────────────────────────
            st.markdown("**Performance by Parlay Size**")
            size_rows = []
            for n in sorted(settled['n_legs'].unique()):
                sub = settled[settled['n_legs'] == n]
                n_w = (sub['status'] == 'Won').sum()
                n_l = (sub['status'] == 'Lost').sum()
                n_p = (sub['status'] == 'Push').sum()
                pnl = sub['profit'].sum()
                roi = pnl / (len(sub) * 100) * 100
                size_rows.append({
                    'Size': f"{n}-Leg",
                    'Count': len(sub),
                    'W': n_w, 'L': n_l, 'P': n_p,
                    'Win%': f"{n_w/len(sub)*100:.0f}%",
                    'ROI': f"{roi:+.1f}%",
                    'P&L': f"${pnl:+,.0f}",
                })
            st.dataframe(pd.DataFrame(size_rows), use_container_width=True, hide_index=True)

            # ── By book ─────────────────────────────────────────
            st.markdown("**Performance by Book**")
            book_rows = []
            for bk in settled['book'].dropna().unique():
                sub = settled[settled['book'] == bk]
                if len(sub) < 2: continue
                n_w = (sub['status'] == 'Won').sum()
                n_l = (sub['status'] == 'Lost').sum()
                pnl = sub['profit'].sum()
                roi = pnl / (len(sub) * 100) * 100
                book_rows.append({'Book': bk, 'Count': len(sub), 'W': n_w, 'L': n_l,
                                  'ROI': f"{roi:+.1f}%", 'P&L': f"${pnl:+,.0f}"})
            if book_rows:
                st.dataframe(pd.DataFrame(book_rows).sort_values('Count', ascending=False),
                             use_container_width=True, hide_index=True)

            # ── Cumulative P&L chart ─────────────────────────────
            st.markdown("**Cumulative P&L (settled parlays)**")
            chart_df = settled.sort_values('created_at').copy()
            chart_df['cumulative_pnl'] = chart_df['profit'].cumsum() / 100
            fig_cum = px.line(chart_df, x='created_at', y='cumulative_pnl',
                              labels={'created_at': '', 'cumulative_pnl': 'Units'},
                              color_discrete_sequence=['#9B59B6'])
            fig_cum.update_layout(**LAYOUT, height=300)
            fig_cum.add_hline(y=0, line_dash='dash', line_color='#30363d')
            st.plotly_chart(fig_cum, use_container_width=True)

        st.markdown("---")

        # ── Full parlay log ──────────────────────────────────────
        st.markdown("**Full Parlay Log**")
        p_filter = st.radio("Filter", ["All", "Open", "Won", "Lost", "Push"],
                            horizontal=True, key="parlay_filter")
        show_df = pdf if p_filter == "All" else pdf[pdf['status'] == p_filter]

        def fmt_parlay_row(row):
            def ostrs(o):
                if o is None or (isinstance(o, float) and pd.isna(o)): return "—"
                o = int(o); return f"+{o}" if o >= 0 else str(o)
            legs = [f"{row['leg1_sel']} ({ostrs(row['leg1_odds'])})",
                    f"{row['leg2_sel']} ({ostrs(row['leg2_odds'])})"]
            if row['n_legs'] == 3 and pd.notna(row.get('leg3_sel')):
                legs.append(f"{row['leg3_sel']} ({ostrs(row['leg3_odds'])})")
            return " + ".join(legs)

        log_rows = []
        for _, row in show_df.iterrows():
            status_emoji = {"Won": "✅", "Lost": "❌", "Push": "⏸️", "Open": "⏳"}.get(row['status'], "")
            ev = f"{row['ev_pct']:+.1f}%" if pd.notna(row.get('ev_pct')) else "—"
            pnl = f"${row['profit']:+,.0f}" if row['status'] != 'Open' else "—"
            log_rows.append({
                '': status_emoji,
                'Date': row['created_at'].strftime('%m/%d %H:%M') if pd.notna(row['created_at']) else '—',
                'Legs': fmt_parlay_row(row),
                'Size': f"{row['n_legs']}-Leg",
                'Book': row['book'],
                'Odds': (f"+{int(row['parlay_odds'])}" if row['parlay_odds'] >= 0 else str(int(row['parlay_odds']))) if pd.notna(row.get('parlay_odds')) else '—',
                'Est EV': ev,
                'Status': row['status'],
                'P&L': pnl,
            })

        if log_rows:
            st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No parlays match this filter.")


# ─── DEBUG ────────────────────────────────────────────────────
with st.expander("🛠️ Debug"):
    st.write("Data window:", selected_window)
    st.write("Status scope:", status_scope)
    st.write("Processed rows:", len(df))
    st.write("Filtered rows:", len(df_f))
    st.write("Settled rows:", len(closed))
    st.write("Expired rows:", expired_n)
    st.write("Likely missed:", likely_missed_n)
    st.write("Has edge_score:", HAS_MY_SCORE)
    st.write("Has gem_score:", HAS_GEM_SCORE)
    st.write("Has smash_score:", HAS_SMASH_SCORE)
    st.write("Tier distribution:", df_f['tier'].value_counts().to_dict())
    st.write("Sharp book distribution:", df_f['primary_sharp'].value_counts().to_dict())
    try:
        st.write("Total in DB:", count_bets())
    except Exception as e:
        st.write("DB count error:", str(e))
