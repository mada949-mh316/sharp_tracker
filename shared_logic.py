"""
shared_logic.py — Single source of truth for Smart Money Tracker.

Imported by both tracker_v8.py (Discord bot) and dashboard_v4.py (Streamlit).
Any change to tier logic, classification, or thresholds happens here only.
"""

import re
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

UNIT_SIZE   = 100
CSV_PATH    = "data/bets.csv"
SHEET_NAME  = "Smart Money Bets"
CREDS_FILE  = "creds.json"

DFS_BOOKS     = ['PrizePicks', 'Betr', 'Dabble', 'Underdog', 'Sleeper', 'Draftkings6']
VALID_LEAGUES = ['NBA', 'NFL', 'NHL', 'NCAAB', 'NCAAF', 'Tennis', 'UFC']

# Tier scoring thresholds
GOOD_ODDS_MIN     = -150
GOOD_ODDS_MAX     = 499
BAD_ODDS_MIN      = 500
BAD_ODDS_MAX      = 999
GOOD_LIQ_MIN      = 2000
GOOD_LIQ_MAX      = 10000
PRIME_HOURS       = {6, 13, 22}
CONSENSUS_THRESHOLD = 3

# Tier display metadata
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

# Discord embed colors (hex int)
TIER_DISCORD_COLOR = {
    'DIAMOND': 0x00BFFF,
    'GOLD':    0xD4AF37,
    'SILVER':  0xC0C0C0,
    'STANDARD':0x2ECC71,
    'WATCH':   0x95A5A6,
    'FADE':    0xE74C3C,
}

# Sharp book signal quality
SHARP_BOOK_MARKET_WHITELIST = {
    ("NoVigApp", "Player Prop"),
    ("Prophet",  "Player Prop"),
    ("Pinnacle", "Point Spread"),
    ("Pinnacle", "Total"),
    ("Polymarket","Moneyline"),
    ("NoVigApp", "Total"),
    ("NoVigApp", "Moneyline"),
    ("Prophet",  "Total"),
}

SHARP_BOOK_MARKET_BLACKLIST = {
    ("Kalshi",   "Point Spread"),
    ("Kalshi",   "NFL"),
    ("NoVigApp", "Tennis"),
    ("Prophet",  "Tennis"),
    ("Pinnacle", "Tennis"),
    ("Prophet",  "NCAAF"),
    ("NoVigApp", "NCAAF"),
    ("NoVigApp", "Total Games"),
}

LOW_CONFIDENCE_BOOKS = {"Kalshi"}

LEAGUE_SHARP_BOOK_SUPPRESS = {
    "Tennis": {"NoVigApp", "Prophet", "Pinnacle"},
    "NCAAF":  {"Prophet",  "NoVigApp"},
}

# Tier Discord descriptions (DIAMOND is built dynamically — see build_diamond_description)
TIER_DESCRIPTIONS = {
    'GOLD':    "Strong signal: either 3+ sharp books in consensus, a player prop Under with good "
               "liquidity, or a prime-time spot with solid market depth. 13% ROI historically.",
    'SILVER':  "Player prop Under in a profitable odds range, or a prime-time alert. "
               "Our core edge — 10.6% ROI across ~3,800 historical bets.",
    'STANDARD':"Sharp money detected. Passes ROI filters but no additional edge flags. "
               "Volume play — follow at standard unit size.",
    'WATCH':   "Flagged for review: Fanatics line or odds in the 500–999 range, both historically "
               "negative. Proceed with caution or skip.",
    'FADE':    "Sharp signal is historically negative here. Data suggests betting the opposite side.",
}

# Odds bucket display order
ODDS_BUCKET_ORDER = ["< -750", "-750 to -300", "-300 to -150", "-150 to +150",
                     "+150 to +300", "+300 to +750", "> +750"]


# ─────────────────────────────────────────────────────────────
# ODDS / MATH HELPERS
# ─────────────────────────────────────────────────────────────

def parse_odds_val(val):
    """Parse American odds string/float → float. Handles −, 'even', +/- prefixes."""
    if pd.isna(val): return 0.0
    s = str(val).lower().replace('−', '-')
    if 'even' in s: return 100.0
    m = re.search(r'([-+]?\d+)', s)
    if m:
        try: return float(m.group(1))
        except: return 0.0
    return 0.0

def get_decimal_odds(american_odds):
    if pd.isna(american_odds) or american_odds == 0: return 0.0
    if american_odds > 0: return 1 + (american_odds / 100.0)
    return 1 + (100.0 / abs(american_odds))

def get_odds_bucket(val):
    if val < -750: return "< -750"
    if -750 <= val < -300: return "-750 to -300"
    if -300 <= val < -150: return "-300 to -150"
    if -150 <= val <= 150:  return "-150 to +150"
    if 150  < val <= 300:   return "+150 to +300"
    if 300  < val <= 750:   return "+300 to +750"
    return "> +750"

def calculate_arb_percent(play_odds, sharp_odds):
    play, sharp = parse_odds_val(play_odds), parse_odds_val(sharp_odds)
    if play == 0 or sharp == 0: return 0.0
    dp = get_decimal_odds(play); ds = get_decimal_odds(sharp)
    if dp == 0 or ds == 0: return 0.0
    total_imp = (1 / dp) + (1 / ds)
    return ((1 / total_imp) - 1) * 100 if total_imp else 0.0

def calculate_profit(odds_val, result):
    """Flat UNIT_SIZE profit calculation."""
    try: odds = float(odds_val)
    except: return 0.0
    if result == "Won":
        return UNIT_SIZE * (odds / 100.0) if odds > 0 else UNIT_SIZE * (100.0 / abs(odds))
    if result == "Lost":
        return -float(UNIT_SIZE)
    return 0.0


# ─────────────────────────────────────────────────────────────
# BET CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def categorize_bet(market, selection):
    """
    Classify a bet into Moneyline / Point Spread / Player Prop / Total.
    Player Prop is checked BEFORE Total to prevent 'Total Points' / 'Total Goals'
    being stolen by the 'points'/'goals' keyword list.
    """
    m, s = str(market).lower(), str(selection).lower()
    if "moneyline" in m: return "Moneyline"
    if "spread" in m or "handicap" in m or "run line" in m or "puck line" in m: return "Point Spread"
    if "player" in m or "milestone" in m or "props" in m: return "Player Prop"
    if any(x in m for x in ["shots", "sog", "assists", "rebounds", "threes", "touchdowns"]):
        return "Player Prop"
    if "total" in m or "over/under" in m: return "Total"
    if "to score" in s or re.search(r'\d+\+', s): return "Player Prop"
    if "over" in s or "under" in s: return "Total"
    return "Moneyline"

def get_bet_side(selection):
    s = str(selection).lower()
    if re.search(r'\bover\b', s):  return "Over"
    if re.search(r'\bunder\b', s): return "Under"
    return "Other"

def extract_prop_category(market):
    """
    Extract prop stat category from market name.
    Combined categories (PRA, Pts+Reb, etc.) are checked BEFORE singles
    to prevent 'Points + Rebounds' being returned as 'Points'.
    """
    m = str(market).lower().replace("player ", "").replace("alternate ", "").replace("alt ", "")
    if "milestone" in m: return "Milestone"
    # Combined first
    if "points" in m and "rebounds" in m and "assists" in m: return "PRA"
    if "points" in m and "rebounds" in m: return "Pts+Reb"
    if "points" in m and "assists"  in m: return "Pts+Ast"
    if "rebounds" in m and "assists" in m: return "Reb+Ast"
    if "blocks"   in m and "steals"  in m: return "Blk+Stl"
    # Singles
    if "points"    in m: return "Points"
    if "rebounds"  in m: return "Rebounds"
    if "assists"   in m: return "Assists"
    if "threes"    in m or "3-point" in m or "3pt" in m: return "Threes"
    if "blocks"    in m: return "Blocks"
    if "steals"    in m: return "Steals"
    if "turnovers" in m: return "Turnovers"
    if "shots"     in m or "sog"    in m: return "Shots on Goal"
    if "saves"     in m: return "Saves"
    if "goals"     in m: return "Goals"
    if "passing"   in m: return "Passing"
    if "rushing"   in m: return "Rushing"
    if "receiving" in m or "receptions" in m: return "Receiving"
    if "touchdown" in m or "score" in m: return "Touchdowns"
    if "double"    in m: return "Double Double"
    if "triple"    in m: return "Triple Double"
    return "Other"


# ─────────────────────────────────────────────────────────────
# SHARP BOOK SIGNAL QUALITY
# ─────────────────────────────────────────────────────────────

def evaluate_sharp_signal(sharp_book_str, bet_type, league):
    """
    Returns (is_whitelisted, is_blacklisted, is_low_confidence, is_suppressed, reason_str).
    """
    books = [b.strip().strip('"') for b in str(sharp_book_str).split(',') if b.strip()]
    is_whitelisted = is_blacklisted = is_low_confidence = is_suppressed = False
    reasons = []

    for book in books:
        if (book, bet_type) in SHARP_BOOK_MARKET_WHITELIST:
            is_whitelisted = True
        if (book, bet_type) in SHARP_BOOK_MARKET_BLACKLIST:
            is_blacklisted = True
            reasons.append(f"{book} blacklisted on {bet_type}")
        if (book, league) in SHARP_BOOK_MARKET_BLACKLIST:
            is_blacklisted = True
            reasons.append(f"{book} blacklisted on {league}")
        if book in LOW_CONFIDENCE_BOOKS:
            is_low_confidence = True
        if league in LEAGUE_SHARP_BOOK_SUPPRESS and book in LEAGUE_SHARP_BOOK_SUPPRESS[league]:
            is_suppressed = True
            reasons.append(f"{book} suppressed on {league}")

    return is_whitelisted, is_blacklisted, is_low_confidence, is_suppressed, "; ".join(reasons)


# ─────────────────────────────────────────────────────────────
# TIER CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def classify_tier(bet_data):
    """
    Classify a bet dict (or DataFrame row) into a tier + flags dict.
    Returns (tier_string, flags_dict).

    Works for both the tracker (dict from scraper) and the dashboard
    (row from DataFrame — pass row.to_dict() or just the row directly
    since .get() works on both).
    """
    s           = str(bet_data.get('play_selection', '')).lower()
    is_under    = 'under' in s
    books       = [b.strip().strip('"') for b in str(bet_data.get('sharp_book', '')).split(',') if b.strip()]
    consensus   = len(books)
    is_fanatics = 'fanatics' in str(bet_data.get('play_book', '')).lower()

    bet_type      = categorize_bet(bet_data.get('market', ''), bet_data.get('play_selection', ''))
    is_prop_under = is_under and bet_type == 'Player Prop'

    try:    odds = float(str(bet_data.get('play_odds', '0')).replace('+', ''))
    except: odds = 0
    good_odds = GOOD_ODDS_MIN <= odds <= GOOD_ODDS_MAX
    bad_odds  = BAD_ODDS_MIN  <= odds <= BAD_ODDS_MAX

    try:    liq = float(bet_data.get('liquidity', 0))
    except: liq = 0
    good_liq = GOOD_LIQ_MIN <= liq <= GOOD_LIQ_MAX

    prime_time = False
    try:
        ts = bet_data.get('timestamp', '')
        if ts:
            # Handle both datetime objects and strings
            dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).strip())
            prime_time = dt.hour in PRIME_HOURS
    except Exception:
        pass

    flags = {
        'is_under':      is_under,
        'is_prop_under': is_prop_under,
        'consensus':     consensus,
        'good_odds':     good_odds,
        'bad_odds':      bad_odds,
        'good_liq':      good_liq,
        'prime_time':    prime_time,
        'is_fanatics':   is_fanatics,
    }

    if (bad_odds or is_fanatics) and consensus < CONSENSUS_THRESHOLD and not is_prop_under:
        return 'WATCH', flags
    if consensus >= CONSENSUS_THRESHOLD and good_odds and is_prop_under: return 'DIAMOND', flags
    if consensus >= CONSENSUS_THRESHOLD and good_odds and good_liq:      return 'DIAMOND', flags
    if consensus >= CONSENSUS_THRESHOLD and good_odds:                   return 'GOLD',    flags
    if is_prop_under and good_liq and good_odds:                         return 'GOLD',    flags
    if prime_time    and good_liq and good_odds:                         return 'GOLD',    flags
    if is_prop_under and good_odds:                                      return 'SILVER',  flags
    if prime_time    and good_odds:                                      return 'SILVER',  flags
    return 'STANDARD', flags


# ─────────────────────────────────────────────────────────────
# DISCORD MESSAGE BUILDERS  (tracker only, but kept here for single source of truth)
# ─────────────────────────────────────────────────────────────

def build_diamond_description(flags):
    """Dynamic DIAMOND description — avoids hardcoding 'player prop Under' for non-prop bets."""
    consensus     = flags.get('consensus', 0)
    is_prop_under = flags.get('is_prop_under', False)
    good_liq      = flags.get('good_liq', False)

    if consensus >= CONSENSUS_THRESHOLD and is_prop_under:
        return "3+ sharp books agree on a player prop Under — our highest-conviction combo. 14.5% ROI historically."
    if consensus >= CONSENSUS_THRESHOLD and good_liq:
        return "3+ sharp books agree and liquidity is in the sweet spot. 13.5% ROI historically."
    if consensus >= CONSENSUS_THRESHOLD:
        return "3+ sharp books agree on this line. High consensus is our strongest signal — 14–15% ROI historically."
    if is_prop_under and good_liq:
        return "Player prop Under with liquidity in the sweet spot. 13.0% ROI historically."
    return "Multiple edge factors aligned. Historically our strongest signal tier — 14.5% ROI."


def build_signal_summary(tier, flags, alert_type, is_low_confidence, is_suppressed, signal_reason):
    if alert_type == "FADE":
        return TIER_DESCRIPTIONS['FADE']

    reasons = []
    if flags.get('consensus', 0) >= CONSENSUS_THRESHOLD:
        reasons.append(f"{flags['consensus']} sharp books agree on this line")
    elif flags.get('consensus', 0) == 2:
        reasons.append("2 sharp books agree on this line")
    if flags.get('is_prop_under'):
        reasons.append("it's a player prop Under (historically +13.4% ROI vs −2.7% for prop Overs)")
    if flags.get('good_liq'):
        reasons.append("liquidity is in the sweet spot ($2k–$10k)")
    if flags.get('prime_time'):
        reasons.append("this appeared during a historically profitable time window")
    if flags.get('good_odds'):
        reasons.append("odds are in the profitable range (−150 to +499)")

    base = build_diamond_description(flags) if tier == 'DIAMOND' else TIER_DESCRIPTIONS.get(tier, "Sharp money detected.")
    why  = ("Flagged because: " + ", ".join(reasons) + ".") if reasons else ""

    notes = []
    if is_low_confidence:                       notes.append("⚠️ Low-confidence sharp source — verify before betting.")
    if is_suppressed and signal_reason:         notes.append(f"🔶 Note: {signal_reason}.")
    if flags.get('is_fanatics'):                notes.append("⚠️ Fanatics line — historically underperforms other books.")
    if flags.get('bad_odds'):                   notes.append("⚠️ Odds in 500–999 range — historically negative ROI.")

    return "\n".join(p for p in [why, base, " ".join(notes)] if p)


def build_flag_bar(flags):
    parts = []
    if flags.get('is_prop_under'):                          parts.append('🔽 PROP UNDER')
    if flags.get('consensus', 1) >= CONSENSUS_THRESHOLD:   parts.append(f"🤝 {flags['consensus']}x CONSENSUS")
    elif flags.get('consensus', 1) == 2:                   parts.append('🤝 2x CONSENSUS')
    if flags.get('good_odds'):   parts.append('✅ GOOD ODDS')
    if flags.get('bad_odds'):    parts.append('⚠️ BAD ODDS')
    if flags.get('good_liq'):    parts.append('💧 GOOD LIQ')
    if flags.get('prime_time'):  parts.append('⏰ PRIME TIME')
    if flags.get('is_fanatics'): parts.append('⚠️ FANATICS')
    return '  |  '.join(parts) if parts else '—'


# ─────────────────────────────────────────────────────────────
# DATA CLEANING  (shared between dashboard Pull and any batch re-import)
# ─────────────────────────────────────────────────────────────

def clean_raw_df(df):
    """
    Strip malformed rows that sneak in from Google Sheets import errors.
    Normalises column names, filters to valid leagues/books, parses profit.
    Returns cleaned DataFrame.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Drop rows where league is not a real league (catches ' Prophet"', '1004.93', etc.)
    if 'league' in df.columns:
        before = len(df)
        df = df[df['league'].isin(VALID_LEAGUES)]
        dropped = before - len(df)
        if dropped:
            print(f"clean_raw_df: dropped {dropped} rows with invalid league values")

    # Drop DFS books
    if 'play_book' in df.columns:
        df = df[~df['play_book'].isin(DFS_BOOKS)]
        df = df[df['play_book'].notna()]

    # Normalise profit column to float
    if 'profit' in df.columns:
        df['profit'] = pd.to_numeric(
            df['profit'].astype(str)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False),
            errors='coerce'
        ).fillna(0.0)

    # Parse timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    return df


def add_derived_columns(df):
    """
    Add all classification/derived columns used by the dashboard.
    Call this after clean_raw_df().
    """
    df = df.copy()

    df['odds_val']      = df['play_odds'].apply(parse_odds_val)
    df['odds_bucket']   = df['odds_val'].apply(get_odds_bucket)
    df['bet_type']      = df.apply(lambda r: categorize_bet(r.get('market',''), r.get('play_selection','')), axis=1)
    df['bet_side']      = df['play_selection'].apply(get_bet_side)
    df['prop_cat']      = df.apply(lambda r: extract_prop_category(r.get('market','')) if r['bet_type'] == 'Player Prop' else '', axis=1)
    df['consensus']     = df['sharp_book'].astype(str).str.split(',').str.len().fillna(1).astype(int)
    df['primary_sharp'] = df['sharp_book'].astype(str).str.split(',').str[0].str.strip().str.strip('"')
    df['is_prop_under'] = (df['bet_type'] == 'Player Prop') & (df['bet_side'] == 'Under')
    df['arb_pct']       = df.apply(lambda r: calculate_arb_percent(r.get('play_odds', 0), r.get('sharp_odds', 0)), axis=1)

    # Tier — classify_tier returns (tier, flags); we only need tier for the df column
    tiers = df.apply(lambda r: classify_tier(r.to_dict())[0], axis=1)
    df['tier'] = tiers

    # Combo label for leaderboard
    def _combo(r):
        league = str(r.get('league', ''))
        bt, side, pc = r['bet_type'], r['bet_side'], r['prop_cat']
        if bt == 'Player Prop': return f"{side} {league} {pc}"
        if bt == 'Total':       return f"{side} {league} Game Total"
        return f"{league} {bt}"
    df['combo'] = df.apply(_combo, axis=1)

    return df