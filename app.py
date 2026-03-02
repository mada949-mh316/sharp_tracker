import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe

from shared_logic import (
    # Constants
    CSV_PATH, SHEET_NAME, CREDS_FILE, DFS_BOOKS, UNIT_SIZE,
    TIER_ORDER, TIER_COLORS, TIER_EMOJI, ODDS_BUCKET_ORDER,
    CONSENSUS_THRESHOLD,
    # Functions
    parse_odds_val, calculate_profit, calculate_arb_percent,
    clean_raw_df, add_derived_columns,
)

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
# GOOGLE SHEETS AUTH
# ─────────────────────────────────────────────────────────────
def _get_gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    CRED_KEYS = {"project_id","private_key_id","private_key","client_email",
                 "client_id","auth_uri","token_uri",
                 "auth_provider_x509_cert_url","client_x509_cert_url"}

    def _auth(raw):
        d = {k: v for k, v in raw.items() if k in CRED_KEYS}
        d["type"] = "service_account"
        return gspread.authorize(ServiceAccountCredentials.from_json_keyfile_dict(d, scope))

    for key in ("connections.gsheets", "gcp_service_account"):
        try:
            parts = key.split(".")
            node = st.secrets
            for p in parts:
                node = node[p]
            return _auth(dict(node))
        except KeyError:
            continue
        except Exception:
            raise

    # Try nested: st.secrets["connections"]["gsheets"]
    try:
        return _auth(dict(st.secrets["connections"]["gsheets"]))
    except (KeyError, AttributeError):
        pass

    if "private_key" in st.secrets:
        return _auth(dict(st.secrets))

    if os.path.exists(CREDS_FILE):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
        return gspread.authorize(creds)

    try:    top_keys = list(st.secrets.keys())
    except: top_keys = ["(unreadable)"]
    raise RuntimeError(
        f"No credentials found. Secret keys visible: {top_keys}\n"
        "Add [connections.gsheets] to Streamlit Secrets or place creds.json locally."
    )


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def _parse_timestamps(df):
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def fetch_from_cloud():
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

def save_and_push(df):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    st.session_state["df_raw"] = df
    process_data.clear()

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
# PROCESSING (cached — busted on save)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def process_data(raw_df_hash):
    """Accepts a hash key so cache busts when data changes."""
    df_raw = st.session_state.get("df_raw", pd.DataFrame())
    if df_raw.empty:
        return pd.DataFrame()
    df = clean_raw_df(df_raw)
    df = add_derived_columns(df)
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
# MAIN APP
# ─────────────────────────────────────────────────────────────
st.title("💸 Smart Money Tracker")

# ── Sidebar: data controls ──
st.sidebar.header("⚙️ Data Controls")
cr, cs = st.sidebar.columns(2)
with cr:
    if st.button("🔄 Pull Cloud"):
        with st.spinner("Pulling..."):
            df_pulled, err = fetch_from_cloud()
        if err:
            st.sidebar.error(f"Pull failed: {err}")
        else:
            st.sidebar.success(f"✅ {len(df_pulled):,} rows loaded")
            st.rerun()
with cs:
    if st.button("☁️ Push Cloud"):
        with st.spinner("Syncing..."):
            sync_to_google_sheets(load_data())

# Load & process
df_raw = load_data()
if df_raw.empty:
    st.info("No data found. Click '🔄 Pull Cloud' to initialize.")
    st.stop()

# Use a hash of the dataframe shape+checksum as cache key
raw_hash = str(len(df_raw)) + str(df_raw.iloc[-1].to_string() if len(df_raw) else "")
df = process_data(raw_hash)
if df.empty:
    st.warning("Data loaded but processing failed. Try Pull Cloud again.")
    st.stop()

# ── Sidebar: filters ──
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Filters")

st.sidebar.subheader("Quick Presets")
preset = st.sidebar.radio("", [
    "All Bets", "NBA Props Only", "3+ Consensus Only",
    "Exclude Fanatics", "Best Edges (DIAMOND + GOLD)", "Prop Unders Only",
], label_visibility="collapsed")

st.sidebar.markdown("**Date Range**")
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()
date_range = st.sidebar.date_input("", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)

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
max_cons    = int(df['consensus'].max())
cons_range  = st.sidebar.slider("Consensus Books", 1, max_cons, (1, max_cons))
oc1, oc2    = st.sidebar.columns(2)
min_odds    = oc1.number_input("Min Odds", value=int(df['odds_val'].min()), step=10)
max_odds    = oc2.number_input("Max Odds", value=int(df['odds_val'].max()), step=10)

HAS_MY_SCORE_SIDEBAR = 'edge_score' in df.columns and df['edge_score'].notna().sum() > 0
if HAS_MY_SCORE_SIDEBAR:
    st.sidebar.markdown("**Edge Score Range**")
    ec1, ec2 = st.sidebar.columns(2)
    min_edge = ec1.number_input("Min", value=0, min_value=0, max_value=100, step=1, key="min_edge")
    max_edge = ec2.number_input("Max", value=100, min_value=0, max_value=100, step=1, key="max_edge")
else:
    min_edge, max_edge = 0, 100

# ── Apply filters ──
df_f = df.copy()

if preset == "NBA Props Only":
    df_f = df_f[(df_f['league']=='NBA') & (df_f['bet_type']=='Player Prop')]
elif preset == "3+ Consensus Only":
    df_f = df_f[df_f['consensus'] >= 3]
elif preset == "Exclude Fanatics":
    df_f = df_f[df_f['play_book'] != 'Fanatics']
elif preset == "Best Edges (DIAMOND + GOLD)":
    df_f = df_f[df_f['tier'].isin(['DIAMOND','GOLD'])]
elif preset == "Prop Unders Only":
    df_f = df_f[df_f['is_prop_under']]

if len(date_range) == 2:
    df_f = df_f[(df_f['timestamp'].dt.date >= date_range[0]) &
                (df_f['timestamp'].dt.date <= date_range[1])]

df_f = df_f[df_f['league'].isin(sel_leagues)]
df_f = df_f[df_f['tier'].isin(sel_tiers)]
df_f = df_f[df_f['primary_sharp'].isin(sel_sharps)]
df_f = df_f[df_f['play_book'].isin(sel_books)]
df_f = df_f[df_f['bet_type'].isin(sel_types)]
df_f = df_f[(df_f['consensus'] >= cons_range[0]) & (df_f['consensus'] <= cons_range[1])]
df_f = df_f[(df_f['odds_val'] >= min_odds) & (df_f['odds_val'] <= max_odds)]
if HAS_MY_SCORE_SIDEBAR and (min_edge > 0 or max_edge < 100):
    df_f = df_f[df_f['edge_score'].notna() & (df_f['edge_score'] >= min_edge) & (df_f['edge_score'] <= max_edge)]

closed = df_f[df_f['status'].isin(['Won','Lost','Push'])].copy()

# ── Top metrics ──
total_profit  = closed['profit'].sum() if not closed.empty else 0
total_wagered = len(closed) * UNIT_SIZE
roi_overall   = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
win_rate      = (closed['status']=='Won').sum() / max(len(closed[closed['status']!='Push']),1) * 100
pending_n     = len(df_f[df_f['status'].isin(['Open','Pending'])])

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Total Bets",   f"{len(df_f):,}")
c2.metric("Settled",      f"{len(closed):,}")
c3.metric("Pending",      f"{pending_n:,}")
c4.metric("Profit",       f"${total_profit:,.0f}", delta=f"{roi_overall:.1f}% ROI")
c5.metric("Win Rate",     f"{win_rate:.1f}%")
c6.metric("Filter",       preset if preset != "All Bets" else "All")
st.markdown("---")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_log, tab_tier, tab_analysis, tab_props, tab_odds, tab_rolling, tab_leaderboard, tab_sharps = st.tabs([
    "📊 Live Log", "💎 Tier Performance", "📈 Deep Dive",
    "🏀 Prop Breakdown", "🎲 Odds Analysis", "📉 Rolling ROI",
    "🏆 Leaderboard", "🤝 Sharp Agreement",
])


# ─── LIVE LOG ───
with tab_log:
    st.subheader("Bet History")
    show_cols = ['timestamp','tier','league','matchup','bet_type','prop_cat',
                 'play_selection','bet_side','play_odds','play_book',
                 'primary_sharp','consensus','status','profit']
    show_cols = [c for c in show_cols if c in df_f.columns]
    st.dataframe(
        df_f[show_cols].sort_values('timestamp', ascending=False),
        use_container_width=True,
        column_config={
            "profit":    st.column_config.NumberColumn("Profit",  format="$%.2f"),
            "play_odds": st.column_config.NumberColumn("Odds",    format="%d"),
            "consensus": st.column_config.NumberColumn("# Books", width="small"),
            "tier":      st.column_config.TextColumn("Tier",      width="small"),
        }
    )


# ─── TIER PERFORMANCE ───
with tab_tier:
    st.subheader("💎 Performance by Tier")
    if closed.empty:
        st.warning("No settled bets in current filter.")
    else:
        tier_stats = calc_roi(closed, 'tier', min_n=5)
        tier_stats['tier'] = pd.Categorical(tier_stats['tier'], categories=TIER_ORDER, ordered=True)
        tier_stats = tier_stats.sort_values('tier')

        cols_t = st.columns(len(TIER_ORDER))
        for i, tier in enumerate(TIER_ORDER):
            row = tier_stats[tier_stats['tier'] == tier]
            em  = TIER_EMOJI.get(tier, '')
            if row.empty:
                cols_t[i].metric(f"{em} {tier}", "N/A")
            else:
                r = row.iloc[0]
                cols_t[i].metric(f"{em} {tier}",
                    f"{r['roi']:.1f}% ROI", f"N={r['n']:,} · WR {r['wr']:.0f}%")

        st.markdown("---")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            metric = 'roi' if metric_mode == "ROI (%)" else 'profit'
            st.plotly_chart(bar(tier_stats, 'tier', metric, "ROI by Tier",
                                text_fmt='roi' if metric=='roi' else 'profit'),
                            use_container_width=True)
        with col_t2:
            ts = closed[closed['timestamp'].notna()].sort_values('timestamp')
            lines = []
            for tier in TIER_ORDER:
                sub = ts[ts['tier']==tier].copy()
                if len(sub) < 5: continue
                sub['cum'] = sub['profit'].cumsum()
                sub['lbl'] = tier
                lines.append(sub[['timestamp','cum','lbl']])
            if lines:
                tdf = pd.concat(lines)
                fig2 = px.line(tdf, x='timestamp', y='cum', color='lbl',
                               title="Cumulative Profit by Tier",
                               color_discrete_map=TIER_COLORS)
                fig2.update_layout(**LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)

        # Detail table
        disp = tier_stats.copy()
        disp['roi']    = disp['roi'].map('{:+.1f}%'.format)
        disp['profit'] = disp['profit'].map('${:,.0f}'.format)
        disp['wr']     = disp['wr'].map('{:.1f}%'.format)
        st.dataframe(disp.rename(columns={'n':'Bets','roi':'ROI','profit':'Profit','wr':'Win Rate'}),
                     use_container_width=True, hide_index=True)

        # Validation insight cards
        st.markdown("---")
        for _, r in tier_stats.iterrows():
            em    = TIER_EMOJI.get(r['tier'], '')
            color = TIER_COLORS.get(r['tier'], '#58a6ff')
            st.markdown(
                f'<div class="insight-card" style="border-left-color:{color}">'
                f'{em} <b>{r["tier"]}</b> — {r["roi"]:+.1f}% ROI · '
                f'{r["n"]:,} bets · {r["wr"]:.0f}% WR · ${r["profit"]:,.0f}'
                f'</div>', unsafe_allow_html=True)

        # ── STANDARD_PLUS drill-down ──
        st.markdown("---")
        st.subheader("⭐ STANDARD_PLUS — Slice Analysis")
        st.caption(
            "STANDARD overall runs −1.3% ROI. These sub-slices run +14–15% historically. "
            "The tracker now flags them as STANDARD_PLUS automatically."
        )

        all_std = closed[closed['tier'].isin(['STANDARD', 'STANDARD_PLUS'])].copy()
        std_plus = closed[closed['tier'] == 'STANDARD_PLUS'].copy()
        std_base = closed[closed['tier'] == 'STANDARD'].copy()

        if all_std.empty:
            st.info("No settled STANDARD/STANDARD_PLUS bets in current filter.")
        else:
            # Summary comparison row
            def _roi_summary(sub, label):
                if sub.empty: return label, 0, 0, 0, 0.0
                n   = len(sub)
                roi = sub['profit'].sum() / (n * UNIT_SIZE) * 100
                wr  = (sub['status'] == 'Won').sum() / n * 100
                pnl = sub['profit'].sum()
                return label, roi, n, wr, pnl

            rows_summary = [
                _roi_summary(all_std,   "STANDARD (all)"),
                _roi_summary(std_plus,  "STANDARD_PLUS"),
                _roi_summary(std_base,  "STANDARD (excl. STD+)"),
            ]
            mc = st.columns(3)
            for col, (lbl, roi, n, wr, pnl) in zip(mc, rows_summary):
                delta_color = "normal" if roi >= 0 else "inverse"
                col.metric(lbl, f"{roi:+.1f}% ROI",
                           f"N={n:,} · WR {wr:.0f}% · ${pnl:+,.0f}",
                           delta_color=delta_color)

            st.markdown("---")

            # Two-column breakdown
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**STANDARD_PLUS vs STANDARD — by League**")
                lg_rows = []
                for lg in sorted(all_std['league'].dropna().unique()):
                    sp = std_plus[std_plus['league'] == lg]
                    sb = std_base[std_base['league'] == lg]
                    for sub, lbl in [(sp, 'STD+'), (sb, 'STANDARD')]:
                        if len(sub) < 10: continue
                        roi = sub['profit'].sum() / (len(sub) * UNIT_SIZE) * 100
                        lg_rows.append({'League': lg, 'Tier': lbl,
                                        'ROI': round(roi, 1), 'N': len(sub)})
                if lg_rows:
                    lg_df = pd.DataFrame(lg_rows)
                    fig_lg = px.bar(
                        lg_df, x='League', y='ROI', color='Tier', barmode='group',
                        color_discrete_map={'STD+': '#27AE60', 'STANDARD': '#2ECC71'},
                        title="ROI by League: STD+ vs STANDARD",
                        text=lg_df['ROI'].map('{:+.1f}%'.format),
                    )
                    fig_lg.add_hline(y=0, line_color='#30363d', line_width=1)
                    fig_lg.update_layout(**LAYOUT, height=320)
                    st.plotly_chart(fig_lg, use_container_width=True)

            with col_right:
                st.markdown("**STANDARD_PLUS vs STANDARD — by Bet Type**")
                bt_rows = []
                for bt in ['Moneyline', 'Point Spread', 'Total', 'Player Prop']:
                    sp = std_plus[std_plus['bet_type'] == bt]
                    sb = std_base[std_base['bet_type'] == bt]
                    for sub, lbl in [(sp, 'STD+'), (sb, 'STANDARD')]:
                        if len(sub) < 10: continue
                        roi = sub['profit'].sum() / (len(sub) * UNIT_SIZE) * 100
                        bt_rows.append({'Bet Type': bt, 'Tier': lbl,
                                        'ROI': round(roi, 1), 'N': len(sub)})
                if bt_rows:
                    bt_df = pd.DataFrame(bt_rows)
                    fig_bt = px.bar(
                        bt_df, x='Bet Type', y='ROI', color='Tier', barmode='group',
                        color_discrete_map={'STD+': '#27AE60', 'STANDARD': '#2ECC71'},
                        title="ROI by Bet Type: STD+ vs STANDARD",
                        text=bt_df['ROI'].map('{:+.1f}%'.format),
                    )
                    fig_bt.add_hline(y=0, line_color='#30363d', line_width=1)
                    fig_bt.update_layout(**LAYOUT, height=320)
                    st.plotly_chart(fig_bt, use_container_width=True)

            # Cumulative profit line: STD+ vs STANDARD vs combined
            st.markdown("**Cumulative Profit: STANDARD_PLUS vs STANDARD**")
            cum_lines = []
            for sub, lbl, color in [
                (std_plus,  'STANDARD_PLUS', '#27AE60'),
                (std_base,  'STANDARD',      '#2ECC71'),
                (all_std,   'STANDARD (all)','#58a6ff'),
            ]:
                if len(sub) < 5: continue
                s = sub[sub['timestamp'].notna()].sort_values('timestamp').copy()
                s['cum'] = s['profit'].cumsum()
                s['lbl'] = lbl
                cum_lines.append(s[['timestamp', 'cum', 'lbl']])
            if cum_lines:
                cdf = pd.concat(cum_lines)
                fig_cum = px.line(
                    cdf, x='timestamp', y='cum', color='lbl',
                    title="Cumulative P&L — STD+ vs STANDARD",
                    color_discrete_map={
                        'STANDARD_PLUS': '#27AE60',
                        'STANDARD':      '#2ECC71',
                        'STANDARD (all)':'#58a6ff',
                    },
                )
                fig_cum.add_hline(y=0, line_color='#30363d', line_width=1)
                fig_cum.update_layout(**LAYOUT, height=300)
                st.plotly_chart(fig_cum, use_container_width=True)

            # Sharp book breakdown for STANDARD_PLUS
            st.markdown("**STANDARD_PLUS — ROI by Sharp Book**")
            sp_sharp = calc_roi(std_plus, 'primary_sharp', min_n=10) if not std_plus.empty else pd.DataFrame()
            if not sp_sharp.empty:
                st.plotly_chart(
                    hbar(sp_sharp, 'roi', 'primary_sharp', "STD+ ROI by Sharp Book", h=250),
                    use_container_width=True)
            else:
                st.caption("Not enough data yet.")

            # Detail table: cross-cut STD+ bets by league × bet_type
            st.markdown("**STANDARD_PLUS — League × Bet Type breakdown**")
            if not std_plus.empty:
                cross = std_plus.groupby(['league', 'bet_type']).agg(
                    N=('profit', 'count'),
                    ROI=('profit', lambda x: round(x.sum() / (len(x) * UNIT_SIZE) * 100, 1)),
                    WR=('status', lambda x: round((x == 'Won').sum() / len(x) * 100, 1)),
                    PnL=('profit', lambda x: round(x.sum(), 0)),
                ).reset_index()
                cross = cross[cross['N'] >= 5].sort_values('ROI', ascending=False)
                cross['ROI'] = cross['ROI'].map('{:+.1f}%'.format)
                cross['WR']  = cross['WR'].map('{:.1f}%'.format)
                cross['PnL'] = cross['PnL'].map('${:,.0f}'.format)
                st.dataframe(cross.rename(columns={'league':'League','bet_type':'Bet Type'}),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("No STANDARD_PLUS bets settled yet — classifications will appear as bets close.")


# ─── DEEP DIVE ───
with tab_analysis:
    if closed.empty:
        st.warning("No settled bets.")
    else:
        st.subheader("🔥 League × Bet Type Heatmap")
        hm = closed.groupby(['league','bet_type'])['profit'].agg(
            lambda x: (x.sum()/(len(x)*UNIT_SIZE))*100
        ).reset_index()
        hm.columns = ['league','bet_type','roi']
        fig_hm = px.density_heatmap(hm, x='bet_type', y='league', z='roi',
                                     color_continuous_scale='RdYlGn',
                                     text_auto='.1f', range_color=[-30,30])
        fig_hm.update_layout(**LAYOUT)
        st.plotly_chart(fig_hm, use_container_width=True)

        metric = 'roi' if metric_mode == "ROI (%)" else 'profit'
        tf = 'roi' if metric=='roi' else 'profit'

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("By League")
            st.plotly_chart(bar(calc_roi(closed,'league',10), 'league', metric, text_fmt=tf),
                            use_container_width=True)
        with col_b:
            st.subheader("By Bet Type")
            st.plotly_chart(bar(calc_roi(closed,'bet_type',5), 'bet_type', metric, text_fmt=tf),
                            use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("By Play Book")
            st.plotly_chart(bar(calc_roi(closed,'play_book',10), 'play_book', metric, text_fmt=tf),
                            use_container_width=True)
        with col_d:
            st.subheader("By Sharp Book Signal")
            st.plotly_chart(bar(calc_roi(closed,'primary_sharp',10), 'primary_sharp', metric, text_fmt=tf),
                            use_container_width=True)

        st.subheader("By Consensus Count")
        cs_stats = calc_roi(closed,'consensus',5).sort_values('consensus')
        cs_stats['consensus'] = cs_stats['consensus'].astype(str) + ' books'
        st.plotly_chart(bar(cs_stats, 'consensus', metric, text_fmt=tf, h=240),
                        use_container_width=True)


# ─── PROP BREAKDOWN ───
with tab_props:
    props_closed = closed[closed['bet_type']=='Player Prop'].copy()
    st.subheader("🏀 Player Prop Analysis")

    if props_closed.empty:
        st.warning("No settled prop bets in current filter.")
    else:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            side_s = calc_roi(props_closed, 'bet_side', 5)
            fig_side = go.Figure()
            for _, r in side_s.iterrows():
                fig_side.add_trace(go.Bar(
                    x=[r['bet_side']], y=[r['roi']],
                    marker_color=roi_color(r['roi']),
                    text=f"{r['roi']:+.1f}%<br>N={r['n']:,}<br>WR {r['wr']:.0f}%",
                    textposition='inside', name=r['bet_side']
                ))
            fig_side.add_hline(y=0, line_color='#30363d')
            fig_side.update_layout(**LAYOUT, title="Prop Over vs Under", showlegend=False, height=300)
            st.plotly_chart(fig_side, use_container_width=True)
        with col_p2:
            pc_s = calc_roi(props_closed, 'prop_cat', 10).head(8)
            st.plotly_chart(hbar(pc_s, 'roi', 'prop_cat', "Top Prop Categories", h=300),
                            use_container_width=True)

        # Over vs Under by category
        st.subheader("Prop Category × Over/Under")
        cross = props_closed.groupby(['prop_cat','bet_side']).agg(
            profit=('profit','sum'), n=('profit','count')
        ).reset_index()
        cross['roi'] = (cross['profit']/(cross['n']*UNIT_SIZE))*100
        cross = cross[cross['n'] >= 5]
        fig_x = px.bar(cross, x='prop_cat', y='roi', color='bet_side', barmode='group',
                       color_discrete_map={'Over':'#f87171','Under':'#4ade80','Other':'#8b949e'},
                       text_auto='.1f')
        fig_x.add_hline(y=0, line_color='#30363d', line_width=2)
        fig_x.update_layout(**LAYOUT, height=350, xaxis_tickangle=-30)
        st.plotly_chart(fig_x, use_container_width=True)

        # NBA Prop Unders detail
        st.subheader("NBA Prop Unders — Category Detail")
        nba_u = props_closed[(props_closed['league']=='NBA') & (props_closed['bet_side']=='Under')]
        if not nba_u.empty:
            nba_cat = calc_roi(nba_u, 'prop_cat', 5).sort_values('roi', ascending=False)
            st.plotly_chart(hbar(nba_cat, 'roi', 'prop_cat', "NBA Under ROI by Category",
                                 h=max(300, len(nba_cat)*35)), use_container_width=True)
            disp = nba_cat.copy()
            disp['roi']    = disp['roi'].map('{:+.1f}%'.format)
            disp['profit'] = disp['profit'].map('${:,.0f}'.format)
            disp['wr']     = disp['wr'].map('{:.1f}%'.format)
            st.dataframe(disp[['prop_cat','n','roi','profit','wr']].rename(
                columns={'prop_cat':'Category','n':'Bets','roi':'ROI','profit':'Profit','wr':'WR'}),
                use_container_width=True, hide_index=True)


# ─── ODDS ANALYSIS ───
with tab_odds:
    st.subheader("🎲 Profitability by Odds Range")
    if closed.empty:
        st.warning("No settled bets.")
    else:
        odds_s = closed.groupby('odds_bucket').agg(
            profit=('profit','sum'), n=('profit','count')
        ).reset_index()
        odds_s['roi'] = (odds_s['profit']/(odds_s['n']*UNIT_SIZE))*100
        odds_s['odds_bucket'] = pd.Categorical(odds_s['odds_bucket'],
                                                categories=ODDS_BUCKET_ORDER, ordered=True)
        odds_s = odds_s.sort_values('odds_bucket')
        metric = 'roi' if metric_mode == "ROI (%)" else 'profit'
        st.plotly_chart(bar(odds_s, 'odds_bucket', metric,
                            text_fmt='roi' if metric=='roi' else 'profit', h=320),
                        use_container_width=True)
        disp = odds_s.copy()
        disp['roi']    = disp['roi'].map('{:+.1f}%'.format)
        disp['profit'] = disp['profit'].map('${:,.2f}'.format)
        st.dataframe(disp.rename(columns={'odds_bucket':'Odds Range','n':'Bets',
                                           'roi':'ROI','profit':'Profit'}),
                     use_container_width=True, hide_index=True)


# ─── ROLLING ROI ───
with tab_rolling:
    st.subheader("📉 Rolling ROI Trend")
    if closed.empty or 'timestamp' not in closed.columns:
        st.warning("No data.")
    else:
        roll_df = closed[closed['timestamp'].notna()].sort_values('timestamp').copy()
        cw, cb = st.columns(2)
        window  = cw.slider("Rolling window (bets)", 10, 200, 50, step=10)
        roll_by = cb.radio("Group by", ["All Bets", "By Tier"], horizontal=True)

        if roll_by == "All Bets":
            roll_df['rolling_roi'] = (roll_df['profit']
                .rolling(window, min_periods=max(5, window//4)).mean()
                / UNIT_SIZE * 100)
            roll_df['cum_profit']  = roll_df['profit'].cumsum()

            fig_r = go.Figure(go.Scatter(
                x=roll_df['timestamp'], y=roll_df['rolling_roi'],
                name=f'{window}-bet Rolling ROI', line=dict(color='#58a6ff', width=2)))
            fig_r.add_hline(y=0, line_color='#30363d', line_dash='dash')
            fig_r.update_layout(**LAYOUT, title=f"{window}-Bet Rolling ROI",
                                yaxis_title="ROI (%)", height=350)
            st.plotly_chart(fig_r, use_container_width=True)

            last_val = roll_df['cum_profit'].iloc[-1]
            fig_c = go.Figure(go.Scatter(
                x=roll_df['timestamp'], y=roll_df['cum_profit'],
                fill='tozeroy',
                line=dict(color='#4ade80' if last_val >= 0 else '#f87171', width=2),
                fillcolor='rgba(74,222,128,0.1)'))
            fig_c.add_hline(y=0, line_color='#30363d')
            fig_c.update_layout(**LAYOUT, title="Cumulative Profit", height=280)
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            fig_tr = go.Figure()
            for tier in TIER_ORDER:
                sub = roll_df[roll_df['tier']==tier].copy()
                if len(sub) < window//2: continue
                sub['rr'] = (sub['profit']
                    .rolling(window, min_periods=max(5,window//4)).mean()
                    / UNIT_SIZE * 100)
                fig_tr.add_trace(go.Scatter(
                    x=sub['timestamp'], y=sub['rr'],
                    name=f"{TIER_EMOJI.get(tier,'')} {tier}",
                    line=dict(color=TIER_COLORS.get(tier,'#58a6ff'), width=2)))
            fig_tr.add_hline(y=0, line_color='#30363d', line_dash='dash')
            fig_tr.update_layout(**LAYOUT, title=f"{window}-Bet Rolling ROI by Tier", height=400)
            st.plotly_chart(fig_tr, use_container_width=True)

        # Monthly table
        st.subheader("Monthly Performance")
        roll_df['month'] = roll_df['timestamp'].dt.to_period('M').astype(str)
        mo = roll_df.groupby('month').agg(profit=('profit','sum'), n=('profit','count')).reset_index()
        mo['roi'] = (mo['profit']/(mo['n']*UNIT_SIZE))*100
        mo['roi_fmt']    = mo['roi'].map('{:+.1f}%'.format)
        mo['profit_fmt'] = mo['profit'].map('${:,.0f}'.format)
        st.dataframe(mo[['month','n','roi_fmt','profit_fmt']].rename(
            columns={'month':'Month','n':'Bets','roi_fmt':'ROI','profit_fmt':'Profit'}),
            use_container_width=True, hide_index=True)


# ─── LEADERBOARD ───
with tab_leaderboard:
    st.subheader("🏆 Most Profitable Categories")
    if closed.empty:
        st.warning("No settled bets.")
    else:
        lb1, lb2 = st.columns([1, 3])
        min_bets = lb1.slider("Min Sample", 1, 100, 10)
        sort_by  = lb1.radio("Sort by", ["ROI", "Profit"])

        lb = closed.groupby('combo').agg(
            profit=('profit','sum'), n=('profit','count'),
            wins=('status', lambda x: (x=='Won').sum())
        ).reset_index()
        lb['roi'] = (lb['profit']/(lb['n']*UNIT_SIZE))*100
        lb['wr']  = lb['wins']/lb['n'].clip(lower=1)*100
        lb = lb[lb['n'] >= min_bets].sort_values(
            'roi' if sort_by=='ROI' else 'profit', ascending=False)

        lb2.metric("Categories shown", f"{len(lb)}")
        disp = lb.copy()
        disp['roi']    = disp['roi'].map('{:+.1f}%'.format)
        disp['profit'] = disp['profit'].map('${:,.0f}'.format)
        disp['wr']     = disp['wr'].map('{:.1f}%'.format)
        st.dataframe(disp[['combo','n','roi','profit','wr']].rename(
            columns={'combo':'Category','n':'Bets','roi':'ROI','profit':'Profit','wr':'Win Rate'}),
            use_container_width=True, height=600, hide_index=True)


# ─── SHARP BOOK AGREEMENT MATRIX ───
with tab_sharps:
    st.subheader("🤝 Sharp Book Agreement Matrix")
    st.caption("How often each pair of sharp books appear together in the same signal — "
               "and whether that co-occurrence actually outperforms single-book signals.")

    if closed.empty:
        st.warning("No settled bets.")
    else:
        ALL_BOOKS = ['Prophet','NoVigApp','Pinnacle','4cx','Polymarket','Kalshi']

        # ── Build presence matrix ──
        def has_book(sharp_str, book):
            return book in str(sharp_str)

        for b in ALL_BOOKS:
            closed[f'_has_{b}'] = closed['sharp_book'].apply(lambda x: has_book(x, b))

        # Co-occurrence count heatmap
        co_count = pd.DataFrame(0, index=ALL_BOOKS, columns=ALL_BOOKS)
        co_roi   = pd.DataFrame(np.nan, index=ALL_BOOKS, columns=ALL_BOOKS)

        for i, b1 in enumerate(ALL_BOOKS):
            for j, b2 in enumerate(ALL_BOOKS):
                if i == j:
                    # Diagonal = bets where this is the ONLY book (solo signal)
                    solo = closed[closed[f'_has_{b1}'] & (closed['consensus'] == 1)]
                    co_count.loc[b1, b2] = len(solo)
                    if len(solo) >= 5:
                        co_roi.loc[b1, b2] = (solo['profit'].sum() / (len(solo)*UNIT_SIZE)) * 100
                else:
                    both = closed[closed[f'_has_{b1}'] & closed[f'_has_{b2}']]
                    co_count.loc[b1, b2] = len(both)
                    if len(both) >= 5:
                        co_roi.loc[b1, b2] = (both['profit'].sum() / (len(both)*UNIT_SIZE)) * 100

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("**Co-occurrence Count**")
            st.caption("Diagonal = solo signals (only that book). Off-diagonal = bets both books flagged.")
            fig_cnt = go.Figure(go.Heatmap(
                z=co_count.values,
                x=ALL_BOOKS, y=ALL_BOOKS,
                colorscale='Blues',
                text=co_count.values.astype(int),
                texttemplate='%{text}',
                textfont=dict(size=12),
                showscale=True,
            ))
            hm_layout = {**LAYOUT, 'height': 400}
            hm_layout['xaxis'] = dict(side='bottom', gridcolor='#21262d')
            hm_layout['yaxis'] = dict(gridcolor='#21262d')
            fig_cnt.update_layout(**hm_layout)
            st.plotly_chart(fig_cnt, use_container_width=True)

        with col_m2:
            st.markdown("**ROI When Books Agree**")
            st.caption("Diagonal = solo signal ROI. Off-diagonal = ROI when both books agree. Grey = < 5 bets.")
            roi_display = co_roi.round(1)
            roi_vals    = co_roi.values.astype(float)
            fig_roi = go.Figure(go.Heatmap(
                z=roi_vals,
                x=ALL_BOOKS, y=ALL_BOOKS,
                colorscale='RdYlGn',
                zmid=0, zmin=-20, zmax=20,
                text=roi_display.values,
                texttemplate='%{text:.1f}%',
                textfont=dict(size=12),
                showscale=True,
            ))
            hm_layout = {**LAYOUT, 'height': 400}
            hm_layout['xaxis'] = dict(side='bottom', gridcolor='#21262d')
            hm_layout['yaxis'] = dict(gridcolor='#21262d')
            fig_roi.update_layout(**hm_layout)
            st.plotly_chart(fig_roi, use_container_width=True)

        # ── Per-book solo vs consensus breakdown table ──
        st.subheader("Single-Book vs Multi-Book ROI per Sharp Source")

        rows = []
        for b in ALL_BOOKS:
            has = closed[closed[f'_has_{b}']]
            solo  = has[has['consensus'] == 1]
            multi = has[has['consensus'] >= 2]
            three = has[has['consensus'] >= 3]

            def _roi(g): return (g['profit'].sum()/(len(g)*UNIT_SIZE)*100) if len(g) >= 5 else None

            rows.append({
                'Sharp Book':   b,
                'All Bets N':   len(has),
                'All ROI':      f"{_roi(has):+.1f}%" if _roi(has) is not None else '—',
                'Solo N':       len(solo),
                'Solo ROI':     f"{_roi(solo):+.1f}%" if _roi(solo) is not None else '—',
                '2+ Books N':   len(multi),
                '2+ ROI':       f"{_roi(multi):+.1f}%" if _roi(multi) is not None else '—',
                '3+ Books N':   len(three),
                '3+ ROI':       f"{_roi(three):+.1f}%" if _roi(three) is not None else '—',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Insight: which book pairs are most independent? ──
        st.subheader("Most Independent Book Pairs")
        st.caption("Pairs that rarely agree — their co-signals may carry more weight as genuine independent confirmation.")

        pairs = []
        for i, b1 in enumerate(ALL_BOOKS):
            for j, b2 in enumerate(ALL_BOOKS):
                if j <= i: continue
                n1   = closed[f'_has_{b1}'].sum()
                n2   = closed[f'_has_{b2}'].sum()
                both = int(co_count.loc[b1, b2])
                if n1 < 20 or n2 < 20: continue
                overlap_pct = both / min(n1, n2) * 100 if min(n1,n2) > 0 else 0
                roi_val = co_roi.loc[b1, b2]
                pairs.append({
                    'Books': f"{b1} + {b2}",
                    'Co-signals': both,
                    'Overlap %': f"{overlap_pct:.1f}%",
                    'Co-signal ROI': f"{roi_val:+.1f}%" if not np.isnan(roi_val) else '—',
                })
        if pairs:
            pairs_df = pd.DataFrame(pairs).sort_values('Co-signals')
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)

    # Clean up temp columns
    for b in ALL_BOOKS:
        col = f'_has_{b}'
        if col in closed.columns:
            closed.drop(columns=[col], inplace=True)


# ─── DEBUG ───
with st.expander("🛠️ Debug"):
    st.write("Raw rows:", len(df_raw))
    st.write("Processed rows:", len(df))
    st.write("Filtered rows:", len(df_f))
    st.write("Settled rows:", len(closed))
    st.write("Tier distribution:", df_f['tier'].value_counts().to_dict())
    st.write("Sharp book distribution:", df_f['primary_sharp'].value_counts().to_dict())
