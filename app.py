import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import re
from datetime import datetime, date
from streamlit_gsheets import GSheetsConnection

# --- CONFIGURATION ---
CSV_PATH = "data/bets.csv"
UNIT_SIZE = 100
DFS_BOOKS = ['PrizePicks', 'Betr', 'Dabble', 'Underdog', 'Sleeper', 'Draftkings6']
st.set_page_config(page_title="Smart Money Tracker", layout="wide")

# --- DATA LOADING ---
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        # ttl=10 ensures cache clears quickly
        df = conn.read(ttl=10)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error reading Google Sheet: {e}")
        return pd.DataFrame()

# --- HELPER: ROBUST ODDS PARSING ---
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
    else: return 1 + (100.0 / abs(american_odds))

# --- HELPER: ARBITRAGE CALCULATION ---
def calculate_arb_percent(row):
    play = parse_odds_val(row.get('play_odds', 0))
    sharp = parse_odds_val(row.get('sharp_odds', 0))
    if play == 0 or sharp == 0: return 0.0
    
    dec_play = get_decimal_odds(play)
    dec_sharp = get_decimal_odds(sharp)
    if dec_play == 0 or dec_sharp == 0: return 0.0

    imp_play = 1 / dec_play
    imp_sharp = 1 / dec_sharp
    total_imp = imp_play + imp_sharp
    if total_imp == 0: return 0.0
    
    return ((1 / total_imp) - 1) * 100

# --- HELPER: FADE PROFIT ---
def calculate_fade_profit(row):
    original_result = row.get('result', 'Pending')
    if original_result not in ['Won', 'Lost']: return 0.0
    fade_result = 'Lost' if original_result == 'Won' else 'Won'
    
    if fade_result == 'Lost': return -UNIT_SIZE
    
    original_odds = parse_odds_val(row.get('play_odds', 100))
    if original_odds == 0: return 0.0
    
    fade_odds = original_odds * -1
    if fade_odds > 0: return UNIT_SIZE * (fade_odds / 100.0)
    else: return UNIT_SIZE * (100.0 / abs(fade_odds))

# --- HELPER: PLOTTING ---
def plot_metric_bar(data, x_col, y_col, title, y_label, text_fmt):
    if data.empty: return px.bar(title="No Data")
    data['Outcome'] = data[y_col].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    data = data.sort_values(y_col, ascending=False)
    fig = px.bar(
        data, x=x_col, y=y_col, color='Outcome',
        color_discrete_map={'Positive': '#2ECC71', 'Negative': '#E74C3C'},
        text_auto=text_fmt, title=title
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=y_label)
    fig.add_hline(y=0, line_width=2, line_color="white", opacity=0.5)
    return fig

# --- HELPER: CLASSIFICATION ---
def categorize_bet(row):
    market = str(row.get('market', '')).lower()
    selection = str(row.get('play_selection', '')).lower()
    
    # Prop Keywords
    if "player" in market: return "Player Prop"
    if any(x in market for x in ["shots", "sog", "receptions", "saves", "goals", "assists", "rebounds", "points", "hits"]):
        return "Player Prop"
        
    if "moneyline" in market: return "Moneyline"
    if "spread" in market or "run line" in market or "puck line" in market or "handicap" in market: return "Spread"
    if "total" in market: return "Total"
    if "over" in selection or "under" in selection: return "Total"
    return "Moneyline"

def get_bet_side(selection):
    s = str(selection).lower()
    # REGEX: Matches "Under" only as a whole word (Fixes "Underdog")
    if re.search(r'\bover\b', s): return "Over"
    if re.search(r'\bunder\b', s): return "Under"
    return "Other"

def extract_prop_category_dashboard(market):
    m = str(market).lower().replace("player ", "").replace("alternate ", "").replace("game ", "")
    
    # 1. BASKETBALL / GENERIC
    if "points" in m and "rebounds" in m and "assists" in m: return "PRA"
    if "points" in m and "rebounds" in m: return "Pts + Reb"
    if "points" in m and "assists" in m: return "Pts + Ast"
    if "rebounds" in m and "assists" in m: return "Reb + Ast"
    
    if "points" in m: return "Points"
    if "rebounds" in m: return "Rebounds"
    if "assists" in m: return "Assists"
    if "threes" in m or "3-point" in m or "3pt" in m: return "Threes"
    if "blocks" in m or "blocked" in m: return "Blocks"
    if "steals" in m: return "Steals"
    if "turnovers" in m: return "Turnovers"
    
    # 2. NHL / SOCCER EXPANDED
    if "shots" in m or "sog" in m: return "Shots on Goal"
    if "saves" in m: return "Saves"
    if "goals" in m or "score" in m: return "Goals"
    if "hits" in m: return "Hits"
    if "faceoff" in m: return "Faceoffs"
    if "shutout" in m: return "Shutout"
    if "time on ice" in m or "toi" in m: return "Time On Ice"
    
    # 3. NFL
    if "receptions" in m: return "Receptions"
    if "passing" in m: return "Passing"
    if "rushing" in m: return "Rushing"
    if "receiving" in m: return "Receiving"
    if "touchdown" in m: return "Touchdowns"
    
    # 4. LINES
    if "total" in m: return "Total"
    if "spread" in m or "handicap" in m or "run line" in m or "puck line" in m: return "Spread"
    if "moneyline" in m: return "Moneyline"
    
    # 5. FALLBACK: Title Case (Eliminates "Other")
    return m.title()

# --- MAIN UI ---
st.title("💸 Smart Money Tracker")
df = load_data()

if df.empty:
    st.info("No bets tracked yet.")
else:
    cols = df.columns.tolist()
    sel_col = 'play_selection' if 'play_selection' in cols else 'selection'
    book_col = 'play_book' if 'play_book' in cols else 'sportsbook'
    sharp_col = 'sharp_book' if 'sharp_book' in cols else 'sharp_source'
    
    if book_col:
        df = df[~df[book_col].isin(DFS_BOOKS)]
        
    # --- APPLY CLASSIFICATIONS ---
    df['Bet Type'] = df.apply(categorize_bet, axis=1)
    df['Bet Side'] = df[sel_col].apply(get_bet_side)
    df['Prop Type'] = df['market'].apply(extract_prop_category_dashboard)
    
    def create_combo_category(row):
        league = str(row.get('league', 'Unknown'))
        prop = row['Prop Type']
        side = row['Bet Side']
        bet_type = row['Bet Type']

        # SPREAD/ML: Never show side
        if bet_type in ['Spread', 'Moneyline'] or prop in ['Spread', 'Moneyline']:
            return f"{league} {prop}"

        # TOTALS
        if bet_type == 'Total' or prop == 'Total':
            return f"{side} {league} Game Total"
            
        # PROPS
        if bet_type == 'Player Prop':
            return f"{side} {league} Player {prop}"

        # FALLBACK
        if side == "Other":
            return f"{league} {prop}"
            
        return f"{side} {league} {prop}"

    df['Combo Category'] = df.apply(create_combo_category, axis=1)

    # 2. CALCULATE METRICS
    if 'sharp_odds' in df.columns and 'play_odds' in df.columns:
        df['Arb %'] = df.apply(calculate_arb_percent, axis=1)
        
        def get_arb_bucket(val):
            if val == 0: return "None"
            if val < 0: return "Negative (No Arb)"
            if val < 1: return "0% - 1%"
            if val < 3: return "1% - 3%"
            if val < 5: return "3% - 5%"
            return "5%+"
        df['Arb Bucket'] = df['Arb %'].apply(get_arb_bucket)

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    metric_mode = st.sidebar.radio("Show Results As:", ["Total Profit ($)", "ROI (%)"], index=0)
    
    if metric_mode == "Total Profit ($)":
        agg_func = 'sum'
        y_label = "Profit ($)"
        text_fmt = '$.0f'
        metric_title = "Profit"
    else:
        agg_func = 'mean'
        y_label = "ROI (%)"
        text_fmt = '.1f%'
        metric_title = "ROI"

    st.sidebar.markdown("---")
    
    # Date Filter
    date_range = []
    if 'timestamp' in df.columns and not df.empty:
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        date_range = st.sidebar.date_input("Select Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    st.sidebar.markdown("---")
    
    # Fade Mode
    fade_mode = st.sidebar.toggle("🔄 FADE MODE", value=False)
    if fade_mode:
        st.sidebar.warning("⚠️ VIEWING OPPOSITE RESULTS")
        df['profit'] = df.apply(calculate_fade_profit, axis=1)
    
    st.sidebar.markdown("---")
    
    # Odds Filter
    col_min, col_max = st.sidebar.columns(2)
    min_odds_input = col_min.text_input("Min Odds", value="", placeholder="-150")
    max_odds_input = col_max.text_input("Max Odds", value="", placeholder="+150")

    st.sidebar.markdown("---")

    # Liquidity Filter
    col_liq_min, col_liq_max = st.sidebar.columns(2)
    min_liq_input = col_liq_min.text_input("Min Liq ($)", value="", placeholder="1000")
    max_liq_input = col_liq_max.text_input("Max Liq ($)", value="", placeholder="50000")

    # --- APPLY FILTERS ---
    df_filtered = df.copy()
    if 'timestamp' in df_filtered.columns and len(date_range) == 2:
        df_filtered = df_filtered[(df_filtered['timestamp'].dt.date >= date_range[0]) & (df_filtered['timestamp'].dt.date <= date_range[1])]
    
    all_leagues = df['league'].unique() if 'league' in df.columns else []
    selected_leagues = st.sidebar.multiselect("Filter by League", options=all_leagues, default=all_leagues)
    
    all_types = ['Moneyline', 'Spread', 'Total', 'Player Prop']
    selected_types = st.sidebar.multiselect("Filter by Type", options=all_types, default=all_types)

    all_sides = ['Over', 'Under', 'Other']
    selected_sides = st.sidebar.multiselect("Filter by Side", options=all_sides, default=all_sides)

    if 'league' in df.columns and selected_leagues:
        df_filtered = df_filtered[df_filtered['league'].isin(selected_leagues)]
    if selected_types:
        df_filtered = df_filtered[df_filtered['Bet Type'].isin(selected_types)]
    if selected_sides:
        df_filtered = df_filtered[df_filtered['Bet Side'].isin(selected_sides)]

    if min_odds_input or max_odds_input:
        df_filtered['decimal_odds'] = df_filtered['play_odds'].apply(lambda x: get_decimal_odds(parse_odds_val(x)))
        min_dec = get_decimal_odds(parse_odds_val(min_odds_input)) if min_odds_input else 0
        max_dec = get_decimal_odds(parse_odds_val(max_odds_input)) if max_odds_input else 999
        if min_dec > 0: df_filtered = df_filtered[df_filtered['decimal_odds'] >= min_dec]
        if max_dec < 999: df_filtered = df_filtered[df_filtered['decimal_odds'] <= max_dec]

    if min_liq_input or max_liq_input:
        if 'liquidity' in df_filtered.columns:
            df_filtered['liq_clean'] = pd.to_numeric(df_filtered['liquidity'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce').fillna(0)
            if min_liq_input:
                try: df_filtered = df_filtered[df_filtered['liq_clean'] >= float(min_liq_input)]
                except: pass
            if max_liq_input:
                try: df_filtered = df_filtered[df_filtered['liq_clean'] <= float(max_liq_input)]
                except: pass

    # --- METRICS ---
    closed_bets = df_filtered[df_filtered['status'] != "Open"].copy()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bets", len(df_filtered))
    col2.metric("Pending", len(df_filtered[df_filtered['status'] == "Open"]))
    
    if not closed_bets.empty and 'profit' in cols:
        total_profit = closed_bets['profit'].sum()
        total_wagered = len(closed_bets) * UNIT_SIZE
        roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0.0
        col3.metric("Total Profit", f"${total_profit:.2f}")
        col4.metric("ROI", f"{roi:.1f}%")
    else:
        col3.metric("Total Profit", "$0.00")
        col4.metric("ROI", "0.0%")

    st.markdown("---")

    # --- TABS ---
    tab_view, tab_analysis, tab_leaderboard = st.tabs(["📊 Live Log", "📈 Deep Dive", "🏆 Leaderboard"])

    with tab_view:
        st.subheader("Bet History")
        target_cols = ['timestamp', 'league', 'matchup', 'market', 'Combo Category', 'Bet Type', 'Bet Side', sel_col, 'play_odds', 'Arb %', book_col, sharp_col, 'liquidity', 'status', 'profit']
        final_cols = [c for c in target_cols if c in df_filtered.columns]
        display_df = df_filtered[final_cols].copy()
        if 'timestamp' in display_df.columns:
            display_df = display_df.sort_values(by='timestamp', ascending=False)
        st.dataframe(display_df, use_container_width=True)

    with tab_analysis:
        if closed_bets.empty:
            st.warning("No graded bets available.")
        else:
            if 'league' in closed_bets.columns and 'market' in closed_bets.columns:
                st.subheader(f"🔥 {metric_title} Heatmap")
                heatmap_data = closed_bets.groupby(['league', 'market'])['profit'].agg(agg_func).reset_index()
                fig_heat = px.density_heatmap(
                    heatmap_data, x="market", y="league", z="profit", text_auto=text_fmt,
                    color_continuous_scale="RdYlGn", range_color=[-500, 500]
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Bet Type")
                type_stats = closed_bets.groupby('Bet Type')['profit'].agg(agg_func).reset_index()
                st.plotly_chart(plot_metric_bar(type_stats, 'Bet Type', 'profit', "", y_label, text_fmt), use_container_width=True)
            with col_b:
                st.subheader("Over vs Under")
                side_stats = closed_bets.groupby('Bet Side')['profit'].agg(agg_func).reset_index()
                st.plotly_chart(plot_metric_bar(side_stats, 'Bet Side', 'profit', "", y_label, text_fmt), use_container_width=True)

            if sharp_col in closed_bets.columns:
                st.markdown("---")
                st.subheader("Sharp Source Analysis")
                sharp_data = closed_bets.copy().dropna(subset=[sharp_col])
                sharp_exploded = sharp_data.assign(sharp_split=sharp_data[sharp_col].astype(str).str.split(', ')).explode('sharp_split')
                sharp_stats = sharp_exploded.groupby('sharp_split')['profit'].agg(agg_func).reset_index()
                st.plotly_chart(plot_metric_bar(sharp_stats, 'sharp_split', 'profit', "", y_label, text_fmt), use_container_width=True)

    with tab_leaderboard:
        if closed_bets.empty:
            st.warning("No graded bets available.")
        else:
            st.subheader("🏆 Most Profitable Categories")
            min_bets = st.slider("Minimum Bet Sample Size", 1, 50, 5)
            
            leaderboard = closed_bets.groupby('Combo Category').agg(
                Total_Profit=('profit', 'sum'),
                Bet_Count=('profit', 'count')
            ).reset_index()
            
            leaderboard['ROI'] = (leaderboard['Total_Profit'] / (leaderboard['Bet_Count'] * UNIT_SIZE)) * 100
            leaderboard = leaderboard[leaderboard['Bet_Count'] >= min_bets]
            leaderboard = leaderboard.sort_values(by='ROI', ascending=False)
            
            display_lb = leaderboard.copy()
            display_lb['ROI'] = display_lb['ROI'].map('{:.1f}%'.format)
            display_lb['Total_Profit'] = display_lb['Total_Profit'].map('${:,.2f}'.format)
            st.dataframe(display_lb, use_container_width=True, height=600)

    # --- DEBUG SECTION ---
    with st.expander("🛠️ Debug: Uncategorized Markets"):
        st.write("Below are markets that might be missing keywords:")
        debug_df = df[['market', 'Prop Type', 'Combo Category']].drop_duplicates().sort_values('market')
        st.dataframe(debug_df, use_container_width=True)