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
    return df


def bust_cache():
    fetch_from_db.clear()
    st.rerun()


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

def score_bucket_roi(df, score_col, buckets, min_n=10):
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

# ── Filters ──
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Filters")
st.sidebar.subheader("Quick Presets")
preset = st.sidebar.radio("", [
    "All Bets","NBA Props Only","3+ Consensus Only",
    "Exclude Fanatics","Best Edges (DIAMOND + GOLD)","Prop Unders Only",
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

# ── Apply filters ──
df_f = df.copy()
if preset == "NBA Props Only":         df_f = df_f[(df_f['league']=='NBA')&(df_f['bet_type']=='Player Prop')]
elif preset == "3+ Consensus Only":    df_f = df_f[df_f['consensus']>=3]
elif preset == "Exclude Fanatics":     df_f = df_f[df_f['play_book']!='Fanatics']
elif preset == "Best Edges (DIAMOND + GOLD)": df_f = df_f[df_f['tier'].isin(['DIAMOND','GOLD'])]
elif preset == "Prop Unders Only":     df_f = df_f[df_f['is_prop_under']]
if len(date_range)==2:
    df_f = df_f[(df_f['timestamp'].dt.date>=date_range[0])&(df_f['timestamp'].dt.date<=date_range[1])]
df_f = df_f[df_f['league'].isin(sel_leagues)]
df_f = df_f[df_f['tier'].isin(sel_tiers)]
df_f = df_f[df_f['primary_sharp'].isin(sel_sharps)]
df_f = df_f[df_f['play_book'].isin(sel_books)]
df_f = df_f[df_f['bet_type'].isin(sel_types)]
df_f = df_f[(df_f['consensus']>=cons_range[0])&(df_f['consensus']<=cons_range[1])]
df_f = df_f[(df_f['odds_val']>=min_odds)&(df_f['odds_val']<=max_odds)]

closed = df_f[df_f['status'].isin(['Won','Lost','Push'])].copy()

# ── Score column detection ──
HAS_MY_SCORE  = 'edge_score' in df_f.columns and df_f['edge_score'].notna().sum() > 0
HAS_GEM_SCORE = 'gem_score'  in df_f.columns and df_f['gem_score'].notna().sum() > 0

# ── Top metrics ──
total_profit  = closed['profit'].sum() if not closed.empty else 0
total_wagered = len(closed) * UNIT_SIZE
roi_overall   = (total_profit/total_wagered*100) if total_wagered>0 else 0
win_rate      = (closed['status']=='Won').sum()/max(len(closed[closed['status']!='Push']),1)*100
pending_n     = len(df_f[df_f['status'].isin(['Open','Pending'])])

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Total Bets",  f"{len(df_f):,}")
c2.metric("Settled",     f"{len(closed):,}")
c3.metric("Pending",     f"{pending_n:,}")
c4.metric("Profit",      f"${total_profit:,.0f}", delta=f"{roi_overall:.1f}% ROI")
c5.metric("Win Rate",    f"{win_rate:.1f}%")
c6.metric("Filter",      preset if preset!="All Bets" else selected_window)
st.markdown("---")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
(tab_log, tab_tier, tab_analysis, tab_props, tab_odds,
 tab_rolling, tab_leaderboard, tab_sharps, tab_edge) = st.tabs([
    "📊 Live Log","💎 Tier Performance","📈 Deep Dive",
    "🏀 Prop Breakdown","🎲 Odds Analysis","📉 Rolling ROI",
    "🏆 Leaderboard","🤝 Sharp Agreement","🎯 Edge Scores",
])


# ─── LIVE LOG ───────────────────────────────────────────────
with tab_log:
    st.subheader("Bet History")
    base_cols = ['timestamp','tier','league','matchup','bet_type','prop_cat',
                 'play_selection','bet_side','play_odds','play_book',
                 'primary_sharp','consensus','status','profit']
    score_cols = [c for c in ['edge_score','gem_score'] if c in df_f.columns and df_f[c].notna().any()]
    show_cols  = [c for c in base_cols + score_cols if c in df_f.columns]

    col_config = {
        "profit":    st.column_config.NumberColumn("Profit",  format="$%.2f"),
        "play_odds": st.column_config.NumberColumn("Odds",    format="%d"),
        "consensus": st.column_config.NumberColumn("# Books", width="small"),
        "tier":      st.column_config.TextColumn("Tier",      width="small"),
    }
    if HAS_MY_SCORE:
        col_config["edge_score"] = st.column_config.ProgressColumn(
            "Edge Score", min_value=0, max_value=100, format="%.0f")
    if HAS_GEM_SCORE:
        col_config["gem_score"] = st.column_config.ProgressColumn(
            "Gem Score", min_value=0, max_value=70, format="%.1f")

    st.dataframe(df_f[show_cols].sort_values('timestamp', ascending=False),
                 use_container_width=True, column_config=col_config)


# ─── TIER PERFORMANCE ────────────────────────────────────────
with tab_tier:
    st.subheader("💎 Performance by Tier")
    if closed.empty:
        st.warning("No settled bets in current filter.")
    else:
        tier_stats = calc_roi(closed,'tier',min_n=5)
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

        metric = 'roi' if metric_mode=="ROI (%)" else 'profit'
        tf = 'roi' if metric=='roi' else 'profit'
        col_a,col_b = st.columns(2)
        with col_a:
            st.subheader("By League")
            st.plotly_chart(bar(calc_roi(closed,'league',10),'league',metric,text_fmt=tf),use_container_width=True)
        with col_b:
            st.subheader("By Bet Type")
            st.plotly_chart(bar(calc_roi(closed,'bet_type',5),'bet_type',metric,text_fmt=tf),use_container_width=True)
        col_c,col_d = st.columns(2)
        with col_c:
            st.subheader("By Play Book")
            st.plotly_chart(bar(calc_roi(closed,'play_book',10),'play_book',metric,text_fmt=tf),use_container_width=True)
        with col_d:
            st.subheader("By Sharp Book Signal")
            st.plotly_chart(bar(calc_roi(closed,'primary_sharp',10),'primary_sharp',metric,text_fmt=tf),use_container_width=True)
        st.subheader("By Consensus Count")
        cs_stats = calc_roi(closed,'consensus',5).sort_values('consensus')
        cs_stats['consensus'] = cs_stats['consensus'].astype(str)+' books'
        st.plotly_chart(bar(cs_stats,'consensus',metric,text_fmt=tf,h=240),use_container_width=True)


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

    if not HAS_MY_SCORE and not HAS_GEM_SCORE:
        st.info(
            "No edge scores in the database yet. "
            "Scores are recorded on new bets as they are ingested by the tracker. "
            "Run the tracker then refresh to see data here."
        )
        st.stop()

    settled_e = closed.copy()
    baseline_roi = (settled_e['profit'].sum() / max(len(settled_e)*UNIT_SIZE, 1)) * 100

    # ── 1. SCORE vs ACTUAL ROI ───────────────────────────────
    st.markdown("### 📊 Score Bucket → Actual ROI")
    st.caption("The core validation: do higher scores deliver higher ROI on your settled bets?")

    val_c1, val_c2 = st.columns(2)

    with val_c1:
        if HAS_MY_SCORE:
            bkt = score_bucket_roi(settled_e, 'edge_score', MY_SCORE_BUCKETS, min_n=10)
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
            gbkt = score_bucket_roi(settled_e, 'gem_score', GEM_SCORE_BUCKETS, min_n=10)
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

    st.markdown("---")

    # ── 2. SCORE DISTRIBUTION OVER TIME ─────────────────────
    st.markdown("### 📅 Score Quality Over Time")
    st.caption("Rising average score = tracker is finding better-quality bets. Flat or falling = signal degrading.")

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
        wkg['avg_norm'] = wkg['avg'] / 70 * 100   # normalise 0-70 → 0-100 for same axis
        fig_time.add_trace(go.Scatter(x=wkg['week'],y=wkg['avg_norm'],name='Gem Score (norm. to 100)',
            line=dict(color='#f59e0b',width=2),mode='lines+markers'))
    fig_time.update_layout(**LAYOUT,title="Weekly Average Score",height=300,
                            yaxis_title="Score (both on 0-100 scale)",xaxis_tickangle=-30)
    st.plotly_chart(fig_time, use_container_width=True)

    # Histograms
    hist_c1, hist_c2 = st.columns(2)
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

    st.markdown("---")

    # ── 3. BEST SCORING BOOK × MARKET COMBOS ─────────────────
    st.markdown("### 🏆 Best Scoring Book × Market Combinations")
    st.caption("Which combos score highest AND deliver real ROI? Color = actual ROI, bar length = avg score.")

    if HAS_MY_SCORE and not settled_e.empty:
        # Build the bet_side column if it doesn't exist
        if 'bet_side' not in settled_e.columns:
            settled_e['bet_side'] = 'Other'
        settled_e['bms_key'] = (settled_e['play_book'].fillna('')+' · '+
                                 settled_e['market'].fillna('')+' · '+
                                 settled_e['bet_side'].fillna(''))

        bms_min = st.slider("Min bets per combo", 5, 50, 10, key='bms_min')
        combo_s = settled_e.groupby('bms_key').agg(
            avg_score=('edge_score','mean'),
            roi=('profit',lambda x: x.sum()/(len(x)*UNIT_SIZE)*100),
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
            height=max(380, len(combo_s)*28), xaxis_title="Avg Edge Score",
            xaxis=dict(range=[0,100],gridcolor='#21262d'))
        st.plotly_chart(fig_bms, use_container_width=True)

        disp = combo_s.copy()
        disp['avg_score'] = disp['avg_score'].map('{:.0f}'.format)
        disp['roi']       = disp['roi'].map('{:+.1f}%'.format)
        disp['wr']        = disp['wr'].map('{:.0f}%'.format)
        st.dataframe(disp.rename(columns={'bms_key':'Combo','avg_score':'Avg Score',
                                           'n':'Bets','roi':'ROI','wr':'WR'}),
                     use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── 4. MY SCORE vs GEM SCORE ─────────────────────────────
    st.markdown("### ⚖️ My Score vs Gem Score — Head to Head")
    st.caption(
        "**Correlation: ~0.15** — these two scores measure almost entirely different things. "
        "My score uses historical book × market lookup tables. "
        "Gem uses a logistic regression across all features. "
        "When both agree it's the highest-conviction signal."
    )

    if HAS_MY_SCORE and HAS_GEM_SCORE:
        both = settled_e.dropna(subset=['edge_score','gem_score']).copy()

        if len(both) >= 20:
            # Scatter
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

            # Quadrant analysis
            st.markdown("**Quadrant ROI — When Both Models Agree**")
            MY_T, GEM_T = 45, 54
            both_good = both[(both['edge_score']>=MY_T)&(both['gem_score']>=GEM_T)]
            my_only   = both[(both['edge_score']>=MY_T)&(both['gem_score']< GEM_T)]
            gem_only  = both[(both['edge_score']< MY_T)&(both['gem_score']>=GEM_T)]
            both_skip = both[(both['edge_score']< MY_T)&(both['gem_score']< GEM_T)]

            q1,q2,q3,q4 = st.columns(4)
            def quad_metric(sub, label, col):
                if len(sub)<5: col.metric(label,"N/A",f"N={len(sub)}"); return
                r = sub['profit'].sum()/(len(sub)*UNIT_SIZE)*100
                col.metric(label,f"{r:+.1f}% ROI",f"N={len(sub):,} bets")

            quad_metric("✅ Both Like It",   both_good, q1)
            quad_metric("📊 My Score Only",  my_only,   q2)
            quad_metric("💎 Gem Only",       gem_only,  q3)
            quad_metric("❌ Both Skip",      both_skip, q4)
            st.caption("Use full unit size when both models agree — that's your highest-conviction filter.")

            # Cumulative profit by filter strategy
            st.markdown("**Cumulative Profit: Which Filter Wins?**")
            ts = both.sort_values('timestamp').copy()
            ts['pnl_both'] = ts.apply(lambda r: r['profit'] if (r['edge_score']>=MY_T and r['gem_score']>=GEM_T) else 0, axis=1).cumsum()
            ts['pnl_my']   = ts.apply(lambda r: r['profit'] if r['edge_score']>=MY_T else 0, axis=1).cumsum()
            ts['pnl_gem']  = ts.apply(lambda r: r['profit'] if r['gem_score']>=GEM_T else 0, axis=1).cumsum()
            ts['pnl_all']  = ts['profit'].cumsum()

            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(x=ts['timestamp'],y=ts['pnl_both'],name='✅ Both agree',
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

    # ── Summary metrics ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Quick Summary")
    sm1,sm2,sm3,sm4 = st.columns(4)
    if HAS_MY_SCORE:
        n_scored   = df_f['edge_score'].notna().sum()
        n_high     = (df_f['edge_score']>=65).sum()
        sm1.metric("Bets with Score",  f"{n_scored:,}")
        sm2.metric("High Conf (65+)",  f"{n_high:,}",
                    delta=f"{n_high/max(n_scored,1)*100:.0f}% of scored")
    if HAS_GEM_SCORE:
        n_gem_bet  = (df_f['gem_score']>=54).sum()
        n_gem_t1   = ((df_f['gem_score']>=54)&(df_f['gem_score']<56)).sum()
        sm3.metric("Gem BET signals",  f"{n_gem_bet:,}")
        sm4.metric("Gem Tier 1 (54-56)",f"{n_gem_t1:,}", delta="1-unit signals")


# ─── DEBUG ────────────────────────────────────────────────────
with st.expander("🛠️ Debug"):
    st.write("Data window:", selected_window)
    st.write("Processed rows:", len(df))
    st.write("Filtered rows:", len(df_f))
    st.write("Settled rows:", len(closed))
    st.write("Has edge_score:", HAS_MY_SCORE)
    st.write("Has gem_score:", HAS_GEM_SCORE)
    st.write("Tier distribution:", df_f['tier'].value_counts().to_dict())
    st.write("Sharp book distribution:", df_f['primary_sharp'].value_counts().to_dict())
    try:
        st.write("Total in DB:", count_bets())
    except Exception as e:
        st.write("DB count error:", str(e))
