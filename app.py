import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe, get_as_dataframe

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "bets.csv")
CREDS_FILE = os.path.join(BASE_DIR, "creds.json")

SHEET_NAME = "Smart Money Bets"
UNIT_SIZE = 100
DFS_BOOKS = ['PrizePicks', 'Betr', 'Dabble', 'Underdog', 'Sleeper', 'Draftkings6']

st.set_page_config(page_title="Smart Money Tracker v4.2", layout="wide")

# --- AUTHENTICATION HELPER ---
def get_cloud_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
            creds_dict = dict(st.secrets["connections"]["gsheets"])
            if "private_key" in creds_dict:
                raw_key = creds_dict["private_key"].strip('"').strip("'").replace("\\n", "\n")
                if "-----BEGIN PRIVATE KEY-----" not in raw_key: return None
                creds_dict["private_key"] = raw_key
            creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
            return gspread.authorize(creds)
        elif os.path.exists(CREDS_FILE):
            creds = Credentials.from_service_account_file(CREDS_FILE, scopes=scope)
            return gspread.authorize(creds)
        else: return None
    except Exception as e:
        st.error(f"Authentication Error: {e}")
        return None

# --- DATA LOADING (WITH TYPE CONVERSION) ---
@st.cache_data(ttl=3600)
def load_data(force_cloud=False):
    df = pd.DataFrame()
    
    # 1. Try Local CSV first
    if not force_cloud and os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
        except: pass 

    # 2. Cloud Fallback
    if df.empty:
        try:
            client = get_cloud_client()
            if client:
                sheet = client.open(SHEET_NAME).sheet1
                # Load as string to preserve data, then fix types later
                df = get_as_dataframe(sheet, evaluate_formulas=True, dtype=str)
                df = df.dropna(how='all')
                os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
                df.to_csv(CSV_PATH, index=False)
        except Exception as e:
            st.error(f"Error reading cloud data: {e}")
    
    # 🚨 CRITICAL FIX: ENSURE NUMERIC TYPES 🚨
    if not df.empty:
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Convert profit to float (force non-numeric to NaN, then fill with 0.0)
        if 'profit' in df.columns:
            df['profit'] = pd.to_numeric(df['profit'], errors='coerce').fillna(0.0)
            
        # Convert odds to float/int just in case
        if 'play_odds' in df.columns:
            df['play_odds'] = pd.to_numeric(df['play_odds'], errors='coerce').fillna(0)

    return df

# --- HELPER: SAVE LOCAL ---
def save_local_only(df_to_save):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df_to_save.to_csv(CSV_PATH, index=False)
    st.cache_data.clear() 
    st.toast("💾 Saved locally!", icon="💾")

# --- HELPER: CLOUD SYNC ---
def sync_to_google_sheets(df):
    try:
        client = get_cloud_client()
        if not client:
            st.error("❌ Auth failed. Check secrets/creds.")
            return
        sheet = client.open(SHEET_NAME).sheet1
        sheet.clear()
        set_with_dataframe(sheet, df)
        st.toast("☁️ Synced to Google Sheets!", icon="✅")
    except Exception as e:
        st.error(f"Sync Failed: {e}")

# --- HELPER: CALCULATIONS ---
def calculate_manual_profit(odds, result):
    try: odds = float(odds)
    except: return 0.0
    if result == "Won":
        return UNIT_SIZE * (odds / 100.0) if odds > 0 else UNIT_SIZE * (100.0 / abs(odds))
    elif result == "Lost":
        return -float(UNIT_SIZE)
    return 0.0

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

def calculate_arb_percent(row):
    play = parse_odds_val(row.get('play_odds', 0))
    sharp = parse_odds_val(row.get('sharp_odds', 0))
    if play == 0 or sharp == 0: return 0.0
    dec_play = get_decimal_odds(play)
    dec_sharp = get_decimal_odds(sharp)
    if dec_play == 0 or dec_sharp == 0: return 0.0
    total_imp = (1 / dec_play) + (1 / dec_sharp)
    if total_imp == 0: return 0.0
    return ((1 / total_imp) - 1) * 100

def calculate_fade_profit(row):
    original_result = row.get('result', 'Pending')
    if original_result not in ['Won', 'Lost']: return 0.0
    if original_result == 'Won':
        return -float(UNIT_SIZE)
    else: 
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
    if "player" in market: return "Player Prop"
    if "total" in market or "over/under" in market: return "Total"
    if market == "points": return "Total"
    if any(x in market for x in ["shots", "sog", "receptions", "saves", "goals", "assists", "rebounds", "hits"]):
        return "Player Prop"
    if "moneyline" in market: return "Moneyline"
    if "spread" in market or "run line" in market or "puck line" in market or "handicap" in market: return "Spread"
    if "over" in selection or "under" in selection: return "Total"
    return "Moneyline"

def get_bet_side(selection):
    s = str(selection).lower()
    if re.search(r'\bover\b', s): return "Over"
    if re.search(r'\bunder\b', s): return "Under"
    return "Other"

def extract_prop_category_dashboard(row):
    market = str(row.get('market', ''))
    league = str(row.get('league', ''))
    m = market.lower().replace("player ", "").replace("alternate ", "").replace("game ", "")
    if "total" in m: return "Total"
    if "points" in m:
        if "rebounds" in m or "assists" in m: pass 
        else:
            if "player" in market.lower(): return "Points"
            if league == "NHL": return "Points"
            return "Total"
    if "points" in m and "rebounds" in m and "assists" in m: return "PRA"
    if "points" in m and "rebounds" in m: return "Pts + Reb"
    if "points" in m and "assists" in m: return "Pts + Ast"
    if "rebounds" in m and "assists" in m: return "Reb + Ast"
    if "points" in m: return "Points"
    if "rebounds" in m: return "Rebounds"
    if "assists" in m: return "Assists"
    if "threes" in m or "3-point" in m or "3pt" in m: return "Threes"
    if "blocks" in m: return "Blocks"
    if "steals" in m: return "Steals"
    if "turnovers" in m: return "Turnovers"
    if "shots" in m or "sog" in m: return "Shots on Goal"
    if "saves" in m: return "Saves"
    if "goals" in m or "score" in m: return "Goals"
    if "hits" in m: return "Hits"
    if "faceoff" in m: return "Faceoffs"
    if "receptions" in m: return "Receptions"
    if "passing" in m: return "Passing"
    if "rushing" in m: return "Rushing"
    if "receiving" in m: return "Receiving"
    if "touchdown" in m: return "Touchdowns"
    if "spread" in m or "handicap" in m or "run line" in m or "puck line" in m: return "Spread"
    if "moneyline" in m: return "Moneyline"
    return m.title()

# --- ODDS BUCKETING ---
def get_odds_bucket(val):
    if val < -750: return "Less than -750"
    if -750 <= val < -300: return "-750 to -300"
    if -300 <= val < -150: return "-300 to -150"
    if -150 <= val <= 150: return "-150 to +150"
    if 150 < val <= 300: return "+150 to +300"
    if 300 < val <= 750: return "+300 to +750"
    if val > 750: return "+750 and Higher"
    return "Unknown"

# --- OPTIMIZED BULK GRADER ---
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

    st.info(f"⚡ Fast Mode: {len(open_bets)} pending bets. Edit in the table and click 'Commit' once.")

    cols_to_show = ['timestamp', 'league', 'matchup', 'play_selection', 'market', 'play_odds', 'status']
    cols_to_show = [c for c in cols_to_show if c in open_bets.columns]

    edited_df = st.data_editor(
        open_bets[cols_to_show],
        column_config={
            "status": st.column_config.SelectboxColumn(
                "Status", width="medium", options=["Open", "Won", "Lost", "Push"], required=True
            ),
            "play_odds": st.column_config.NumberColumn("Odds", disabled=True),
            "matchup": st.column_config.TextColumn("Matchup", disabled=True),
            "play_selection": st.column_config.TextColumn("Selection", disabled=True),
        },
        hide_index=True,
        use_container_width=True,
        key="grader_editor",
        num_rows="fixed"
    )

    if st.button("💾 Commit Grades (Local & Cloud)", type="primary"):
        changes_count = 0
        for index, row in edited_df.iterrows():
            original_status = df_full.at[index, 'status']
            new_status = row['status']
            
            if original_status != new_status and new_status in ['Won', 'Lost', 'Push']:
                changes_count += 1
                df_full.at[index, 'status'] = new_status
                df_full.at[index, 'result'] = new_status
                
                if new_status == 'Won':
                    profit = calculate_manual_profit(row['play_odds'], "Won")
                    df_full.at[index, 'profit'] = round(profit, 2)
                elif new_status == 'Lost':
                    profit = calculate_manual_profit(row['play_odds'], "Lost")
                    df_full.at[index, 'profit'] = round(profit, 2)
                elif new_status == 'Push':
                    df_full.at[index, 'profit'] = 0.0

        if changes_count > 0:
            if os.path.exists(CSV_PATH) or os.access(os.path.dirname(CSV_PATH), os.W_OK):
                save_local_only(df_full)
            else:
                st.warning("⚠️ Cloud Mode: Syncing to Google Sheets (Local save skipped).")

            with st.spinner("Syncing changes to Google Sheets..."):
                sync_to_google_sheets(df_full)
                
            st.rerun()
        else:
            st.warning("No changes detected.")

# --- MAIN UI ---
st.title("💸 Smart Money Tracker v4.2")

# SIDEBAR ACTIONS
st.sidebar.header("Data Controls")
col_refresh, col_sync = st.sidebar.columns(2)
with col_refresh:
    if st.button("🔄 Pull from Cloud"):
        st.cache_data.clear()
        load_data(force_cloud=True)
        st.rerun()
with col_sync:
    if st.button("☁️ Push to Cloud"):
        df = load_data() 
        with st.spinner("Syncing to Google Sheets..."):
            sync_to_google_sheets(df)

df = load_data()

if df.empty:
    st.info("No data found. Click 'Pull from Cloud' to initialize.")
else:
    # --- PROCESSING ---
    cols = df.columns.tolist()
    sel_col = 'play_selection' if 'play_selection' in cols else 'selection'
    book_col = 'play_book' if 'play_book' in cols else 'sportsbook'
    sharp_col = 'sharp_book' if 'sharp_book' in cols else 'sharp_source'
    
    if book_col:
        df = df[~df[book_col].isin(DFS_BOOKS)]
        
    df['Bet Type'] = df.apply(categorize_bet, axis=1)
    df['Bet Side'] = df[sel_col].apply(get_bet_side)
    df['Prop Type'] = df.apply(extract_prop_category_dashboard, axis=1)
    
    df['Odds Value'] = df['play_odds'].apply(parse_odds_val)
    df['Odds Bucket'] = df['Odds Value'].apply(get_odds_bucket)
    
    def create_combo_category(row):
        league = str(row.get('league', 'Unknown'))
        prop = row['Prop Type']
        side = row['Bet Side']
        bet_type = row['Bet Type']
        if bet_type in ['Spread', 'Moneyline'] or prop in ['Spread', 'Moneyline']: return f"{league} {prop}"
        if bet_type == 'Total' or prop == 'Total': return f"{side} {league} Game Total"
        if bet_type == 'Player Prop': return f"{side} {league} Player {prop}"
        return f"{side} {league} {prop}"

    df['Combo Category'] = df.apply(create_combo_category, axis=1)

    if 'sharp_odds' in df.columns and 'play_odds' in df.columns:
        df['Arb %'] = df.apply(calculate_arb_percent, axis=1)

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    metric_mode = st.sidebar.radio("Show Results As:", ["Total Profit ($)", "ROI (%)"], index=0)
    
    if metric_mode == "Total Profit ($)":
        agg_func = 'sum'; y_label = "Profit ($)"; text_fmt = '$.0f'; metric_title = "Profit"
    else:
        agg_func = 'mean'; y_label = "ROI (%)"; text_fmt = '.1f%'; metric_title = "ROI"

    st.sidebar.markdown("---")
    date_range = []
    if 'timestamp' in df.columns and not df.empty:
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        date_range = st.sidebar.date_input("Select Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    st.sidebar.markdown("---")
    
    # ODDS FILTER
    st.sidebar.subheader("Filter by Odds")
    col_o1, col_o2 = st.sidebar.columns(2)
    default_min = int(df['Odds Value'].min()) if not df.empty else -10000
    default_max = int(df['Odds Value'].max()) if not df.empty else 10000
    with col_o1: min_odds_input = st.number_input("Min Odds", value=default_min, step=10)
    with col_o2: max_odds_input = st.number_input("Max Odds", value=default_max, step=10)

    fade_mode = st.sidebar.toggle("🔄 FADE MODE", value=False)
    if fade_mode:
        st.sidebar.warning("⚠️ VIEWING OPPOSITE RESULTS")
        df['profit'] = df.apply(calculate_fade_profit, axis=1)
    
    st.sidebar.markdown("---")
    
    # --- FILTERING ---
    df_filtered = df.copy()
    if 'timestamp' in df_filtered.columns and len(date_range) == 2:
        df_filtered = df_filtered[(df_filtered['timestamp'].dt.date >= date_range[0]) & (df_filtered['timestamp'].dt.date <= date_range[1])]
    
    df_filtered = df_filtered[(df_filtered['Odds Value'] >= min_odds_input) & (df_filtered['Odds Value'] <= max_odds_input)]
    
    all_leagues = sorted(df['league'].unique()) if 'league' in df.columns else []
    selected_leagues = st.sidebar.multiselect("Filter by League", options=all_leagues, default=all_leagues)
    
    all_books = sorted(df[book_col].unique()) if book_col in df.columns else []
    selected_books = st.sidebar.multiselect("Filter by Sportsbook", options=all_books, default=all_books)

    # --- NEW: SHARP BOOK FILTER ---
    all_sharps = []
    if sharp_col in df.columns:
        all_sharps = sorted(df[sharp_col].dropna().astype(str).unique())
    selected_sharps = st.sidebar.multiselect("Filter by Sharp Source", options=all_sharps, default=all_sharps)
    # ------------------------------

    all_types = ['Moneyline', 'Spread', 'Total', 'Player Prop']
    selected_types = st.sidebar.multiselect("Filter by Bet Category", options=all_types, default=all_types)
    
    all_props = sorted(df['Prop Type'].unique())
    selected_props = st.sidebar.multiselect("Filter by Market/Prop", options=all_props, default=all_props)
    
    all_sides = ['Over', 'Under', 'Other']
    selected_sides = st.sidebar.multiselect("Filter by Side", options=all_sides, default=all_sides)

    if selected_leagues: df_filtered = df_filtered[df_filtered['league'].isin(selected_leagues)]
    if selected_books: df_filtered = df_filtered[df_filtered[book_col].isin(selected_books)]
    
    # --- APPLY SHARP FILTER ---
    if selected_sharps and sharp_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[sharp_col].astype(str).isin(selected_sharps)]
    # --------------------------

    if selected_types: df_filtered = df_filtered[df_filtered['Bet Type'].isin(selected_types)]
    if selected_props: df_filtered = df_filtered[df_filtered['Prop Type'].isin(selected_props)]
    if selected_sides: df_filtered = df_filtered[df_filtered['Bet Side'].isin(selected_sides)]

    # --- METRICS UI ---
    status_col = df_filtered['status'].str.lower()
    closed_bets = df_filtered[~status_col.isin(['open', 'pending'])].copy()
    pending_count = len(df_filtered[status_col.isin(['open', 'pending'])])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bets", len(df_filtered))
    col2.metric("Pending", pending_count)
    
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
    tab_view, tab_analysis, tab_odds, tab_leaderboard, tab_sim, tab_grader = st.tabs(["📊 Live Log", "📈 Deep Dive", "🎲 Odds Analysis", "🏆 Leaderboard", "💰 Simulator", "📝 Manual Grader"])

    with tab_view:
        st.subheader("Bet History")
        target_cols = ['timestamp', 'league', 'matchup', 'Prop Type', 'play_selection', 'market', 'Bet Side', 'play_odds', 'play_book', 'sharp_book', 'status', 'profit']
        final_cols = [c for c in target_cols if c in df_filtered.columns]
        display_df = df_filtered[final_cols].copy()
        if 'timestamp' in display_df.columns:
            display_df = display_df.sort_values(by='timestamp', ascending=False)
        st.dataframe(display_df, use_container_width=True)

    with tab_analysis:
        if closed_bets.empty:
            st.warning("No graded bets available.")
        else:
            col_chart_1, col_chart_2 = st.columns(2)
            with col_chart_1:
                st.subheader("🦅 Sharp Source Performance")
                if 'sharp_book' in closed_bets.columns:
                    sharp_stats = closed_bets.groupby('sharp_book')['profit'].agg(agg_func).reset_index()
                    sharp_stats = sharp_stats.sort_values('profit', ascending=False).head(10)
                    st.plotly_chart(plot_metric_bar(sharp_stats, 'sharp_book', 'profit', "", y_label, text_fmt), use_container_width=True)
                else:
                    st.warning("Sharp book data missing.")
            
            with col_chart_2:
                st.subheader("Over vs Under")
                side_stats = closed_bets.groupby('Bet Side')['profit'].agg(agg_func).reset_index()
                st.plotly_chart(plot_metric_bar(side_stats, 'Bet Side', 'profit', "", y_label, text_fmt), use_container_width=True)

            if 'league' in closed_bets.columns and 'market' in closed_bets.columns:
                st.subheader(f"🔥 {metric_title} Heatmap")
                heatmap_data = closed_bets.groupby(['league', 'market'])['profit'].agg(agg_func).reset_index()
                fig_heat = px.density_heatmap(
                    heatmap_data, x="market", y="league", z="profit", text_auto=text_fmt,
                    color_continuous_scale="RdYlGn", range_color=[-500 if agg_func=='sum' else -50, 500 if agg_func=='sum' else 50]
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            st.subheader("Bet Type")
            type_stats = closed_bets.groupby('Bet Type')['profit'].agg(agg_func).reset_index()
            st.plotly_chart(plot_metric_bar(type_stats, 'Bet Type', 'profit', "", y_label, text_fmt), use_container_width=True)

    with tab_odds:
        st.subheader("🎲 Profitability by Odds Range")
        if closed_bets.empty:
            st.warning("No graded bets to analyze.")
        else:
            odds_stats = closed_bets.groupby('Odds Bucket').agg(
                Total_Profit=('profit', 'sum'),
                Bet_Count=('profit', 'count')
            ).reset_index()
            
            odds_stats['ROI'] = (odds_stats['Total_Profit'] / (odds_stats['Bet_Count'] * UNIT_SIZE)) * 100
            
            sort_order = ["Less than -750", "-750 to -300", "-300 to -150", "-150 to +150", "+150 to +300", "+300 to +750", "+750 and Higher"]
            odds_stats['Odds Bucket'] = pd.Categorical(odds_stats['Odds Bucket'], categories=sort_order, ordered=True)
            odds_stats = odds_stats.sort_values('Odds Bucket')
            
            target_metric = 'Total_Profit' if metric_mode == "Total Profit ($)" else 'ROI'
            fig_odds = plot_metric_bar(odds_stats, 'Odds Bucket', target_metric, "Performance by Odds Bucket", y_label, text_fmt)
            st.plotly_chart(fig_odds, use_container_width=True)
            
            display_odds = odds_stats.copy()
            display_odds['ROI'] = display_odds['ROI'].map('{:.1f}%'.format)
            display_odds['Total_Profit'] = display_odds['Total_Profit'].map('${:,.2f}'.format)
            st.dataframe(display_odds, use_container_width=True)

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

    with tab_sim:
        st.subheader("💰 Bankroll Growth Simulator")
        if closed_bets.empty:
            st.warning("No graded bets to simulate.")
        else:
            col_sim1, col_sim2 = st.columns(2)
            start_bankroll = col_sim1.number_input("Starting Bankroll ($)", value=10000, step=500)
            pct_stake = col_sim2.slider("Percentage Staking Strategy (%)", 0.5, 5.0, 2.0, step=0.5) / 100.0
            
            sim_df = closed_bets.sort_values('timestamp').copy()
            sim_df['Flat_Bankroll'] = start_bankroll + sim_df['profit'].cumsum()
            
            def get_multiplier(row):
                if row['result'] == 'Won':
                    odds = parse_odds_val(row['play_odds'])
                    return (odds / 100.0) if odds > 0 else (100.0 / abs(odds))
                elif row['result'] == 'Lost': return -1.0
                return 0.0
            
            sim_df['multiplier'] = sim_df.apply(get_multiplier, axis=1)
            sim_df['growth_factor'] = 1 + (pct_stake * sim_df['multiplier'])
            sim_df['Pct_Bankroll'] = start_bankroll * sim_df['growth_factor'].cumprod()
            
            fig_sim = px.line(sim_df, x='timestamp', y=['Flat_Bankroll', 'Pct_Bankroll'], 
                              title="Flat vs. Compounding Growth",
                              labels={'value': 'Bankroll ($)', 'variable': 'Strategy'})
            st.plotly_chart(fig_sim, use_container_width=True)
            
            final_flat = sim_df['Flat_Bankroll'].iloc[-1]
            final_pct = sim_df['Pct_Bankroll'].iloc[-1]
            c1, c2 = st.columns(2)
            c1.metric("Final Bankroll (Flat)", f"${final_flat:,.0f}", delta=f"${final_flat - start_bankroll:,.0f}")
            c2.metric(f"Final Bankroll ({pct_stake*100}%)", f"${final_pct:,.0f}", delta=f"${final_pct - start_bankroll:,.0f}")

    with tab_grader:
        render_manual_grader(df)

    with st.expander("🛠️ Debug"):
        st.write("Current Data Shape:", df.shape)
        st.write("Cache Info:", st.cache_data)
        if 'profit' in df.columns:
            st.write("Profit Column Type:", df['profit'].dtype)
