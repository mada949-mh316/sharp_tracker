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

DFS_BOOKS     = ['PrizePicks', 'Betr', 'Dabble', 'Underdog', 'Sleeper', 'Draftkings6', 'DraftKings6']
VALID_LEAGUES = ['NBA', 'NFL', 'NHL', 'NCAAB', 'NCAAF', 'Tennis', 'UFC', 'MLB']

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
TIER_ORDER  = ['DIAMOND', 'GOLD', 'SILVER', 'STANDARD_PLUS', 'STANDARD', 'WATCH']
TIER_COLORS = {
    'DIAMOND': '#00BFFF',
    'GOLD':    '#D4AF37',
    'SILVER':  '#C0C0C0',
    'STANDARD_PLUS': '#27AE60',   # deeper green — distinct from STANDARD
    'STANDARD':'#2ECC71',
    'WATCH':   '#95A5A6',
}
TIER_EMOJI = {
    'DIAMOND': '💎', 'GOLD': '🥇', 'SILVER': '🥈',
    'STANDARD_PLUS': '⭐', 'STANDARD': '🔥', 'WATCH': '👁️',
}

# Discord embed colors (hex int)
TIER_DISCORD_COLOR = {
    'DIAMOND': 0x00BFFF,
    'GOLD':    0xD4AF37,
    'SILVER':  0xC0C0C0,
    'STANDARD_PLUS': 0x27AE60,
    'STANDARD':0x1ABC9C,   # teal — distinct from silver/grey
    'WATCH':   0xE67E22,   # orange — clearly "caution" tone
    'FADE':    0xE74C3C,
    'HIGH_CONVICTION': 0x9B59B6,  # purple — score 55+
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

# Prop categories that are structurally negative within specific leagues.
# Keyed as (league, prop_category). Bets matching these are demoted to WATCH.
# Data basis: NHL Shots on Goal Under = -4.8% ROI (N=75), Saves Under = -40% (N=5).
# The general "prop Under = edge" signal breaks down for NHL goalie/shot stats.
PROP_CATEGORY_LEAGUE_BLACKLIST = {
    ("NBA", "Double Double"),   # -85.7% ROI (N=28), near-0% WR — almost never hits
    ("NFL", "Touchdowns"),      # -85.5% ROI (N=27), 0% WR — never wins
    ("NHL", "Points"),          # -38.3% ROI (N=22)
    ("Tennis", "Total Games"),  # -23.0% ROI (N=106) — added 2026-02-28
    # Removed 2026-02-28 — flipped positive with larger sample:
    # ("NHL", "Shots on Goal") was -4.8% (N=75), now +5.7% (N=279)
    # ("NHL", "Saves") was -40% (N=5), now +2.0% (N=36)
}

LOW_CONFIDENCE_BOOKS = {"Kalshi"}

LEAGUE_SHARP_BOOK_SUPPRESS = {
    "Tennis": {"NoVigApp", "Prophet", "Pinnacle"},
    "NCAAF":  {"Prophet",  "NoVigApp"},
}

# Tier Discord descriptions (DIAMOND is built dynamically — see build_diamond_description)
def _tier_desc_gold():
    r = _SIGNAL_ROI.get('gold_roi', 13.0)
    return f"Strong signal: either 3+ sharp books in consensus, a player prop Under with good liquidity, or a prime-time spot with solid market depth. {r:+.1f}% ROI historically."

def _tier_desc_silver():
    r = _SIGNAL_ROI.get('silver_roi', 10.6)
    return f"Player prop Under in a profitable odds range, or a prime-time alert. Our core edge — {r:+.1f}% ROI historically."

def _tier_desc_std_plus():
    r = _SIGNAL_ROI.get('std_plus_roi', 23.3)
    return f"Selective STANDARD signal: Moneyline on NCAAB/NHL/NFL/NCAAF, or a Point Spread on NFL/NCAAF. These slices run {r:+.1f}% ROI historically. Play at standard unit size."

TIER_DESCRIPTIONS = {
    'GOLD':          None,  # populated dynamically via _tier_desc_gold()
    'SILVER':        None,  # populated dynamically via _tier_desc_silver()
    'STANDARD_PLUS': None,  # populated dynamically via _tier_desc_std_plus()
    'STANDARD':"Sharp money detected. Passes ROI filters but no additional edge flags. "
               "Volume play — follow at standard unit size.",
    'WATCH':   "Flagged for review: Fanatics line or odds in the 500–999 range, both historically "
               "negative. Proceed with caution or skip.",
    'FADE':    "Sharp signal is historically negative here. Data suggests betting the opposite side.",
}

def _get_tier_description(tier):
    if tier == 'GOLD':          return _tier_desc_gold()
    if tier == 'SILVER':        return _tier_desc_silver()
    if tier == 'STANDARD_PLUS': return _tier_desc_std_plus()
    return TIER_DESCRIPTIONS.get(tier, "Sharp money detected.")

# ─────────────────────────────────────────────────────────────
# DYNAMIC SIGNAL ROI CACHE
# Populated at tracker startup via refresh_signal_roi_cache()
# ─────────────────────────────────────────────────────────────
_SIGNAL_ROI = {
    'prop_under_roi':      12.4,
    'prop_over_roi':       -3.0,
    'good_liq_roi':         6.4,
    'good_odds_roi':        3.7,
    'diamond_roi':         14.9,
    'gold_roi':            12.2,
    'silver_roi':          11.4,
    'std_plus_roi':        23.3,
    'diamond_3book_roi':   14.9,
    'diamond_under_roi':    7.5,
    'ncaab_ml_night_roi':   0.5,
    'ncaab_ml_day_roi':     1.7,
}
_SIGNAL_ROI_LAST_REFRESH = None


def refresh_signal_roi_cache(db_url=None, csv_path=None):
    """
    Pull settled bets and recompute all ROI stats used in signal summaries.
    Call at tracker startup and periodically (e.g. daily).
    """
    import os, re as _re
    from datetime import datetime as _dt

    global _SIGNAL_ROI, _SIGNAL_ROI_LAST_REFRESH

    db_url   = db_url   or os.environ.get('DATABASE_URL', '')
    csv_path = csv_path or 'data/bets.csv'

    df = None
    if db_url:
        try:
            import psycopg2, pandas as _pd
            conn = psycopg2.connect(db_url, sslmode='disable')
            df = _pd.read_sql(
                "SELECT play_selection, market, league, play_odds, play_book, "
                "       sharp_book, liquidity, profit, status, timestamp "
                "FROM bets WHERE status IN ('Won','Lost')", conn)
            conn.close()
            df.columns = df.columns.str.lower().str.strip()
        except Exception as e:
            print(f"⚠️  refresh_signal_roi_cache DB error: {e}")

    if df is None:
        try:
            import pandas as _pd
            df = _pd.read_csv(csv_path)
            df.columns = df.columns.str.lower().str.strip()
        except: return

    import pandas as _pd, numpy as _np
    df['profit']    = _pd.to_numeric(df['profit'], errors='coerce').fillna(0)
    df['liquidity'] = _pd.to_numeric(df['liquidity'], errors='coerce').fillna(0)
    UNIT = 100

    def _roi(sub):
        if len(sub) < 10: return None
        return round(float(sub['profit'].sum() / (len(sub) * UNIT) * 100), 1)

    def _odds(v):
        try: return float(str(v).replace('+', ''))
        except: return 0

    df['_odds'] = df['play_odds'].apply(_odds)
    df['_hour'] = _pd.to_datetime(df['timestamp'], errors='coerce').dt.hour

    is_player  = df['market'].str.contains('Player', case=False, na=False)
    is_under   = df['play_selection'].str.contains('Under', case=False, na=False)
    is_over    = df['play_selection'].str.contains('Over',  case=False, na=False)
    is_good_liq  = df['liquidity'].between(2000, 10000)
    is_good_odds = df['_odds'].between(-150, 499)
    is_ncaab_ml  = (df['league'] == 'NCAAB') & df['market'].str.contains('Moneyline', case=False, na=False)
    is_night     = df['_hour'].between(0, 9) | df['_hour'].between(22, 23)

    # Tier classification
    try:
        tiers = []
        for _, row in df.iterrows():
            try: t, _ = classify_tier(row); tiers.append(t)
            except: tiers.append('STANDARD')
        df['_tier'] = tiers
    except: df['_tier'] = 'STANDARD'

    updates = {
        'prop_under_roi':    _roi(df[is_player & is_under]),
        'prop_over_roi':     _roi(df[is_player & is_over]),
        'good_liq_roi':      _roi(df[is_good_liq]),
        'good_odds_roi':     _roi(df[is_good_odds]),
        'diamond_roi':       _roi(df[df['_tier'] == 'DIAMOND']),
        'gold_roi':          _roi(df[df['_tier'] == 'GOLD']),
        'silver_roi':        _roi(df[df['_tier'] == 'SILVER']),
        'std_plus_roi':      _roi(df[df['_tier'] == 'STANDARD_PLUS']),
        'diamond_3book_roi': _roi(df[(df['_tier'] == 'DIAMOND') & (df['sharp_book'].str.count(',') >= 2)]),
        'diamond_under_roi': _roi(df[(df['_tier'] == 'DIAMOND') & is_player & is_under]),
        'ncaab_ml_night_roi': _roi(df[is_ncaab_ml & is_night]),
        'ncaab_ml_day_roi':   _roi(df[is_ncaab_ml & ~is_night]),
    }

    for k, v in updates.items():
        if v is not None:
            _SIGNAL_ROI[k] = v

    _SIGNAL_ROI_LAST_REFRESH = _dt.now()
    print(f"✅ Signal ROI cache refreshed — "
          f"prop_under={_SIGNAL_ROI['prop_under_roi']:+.1f}% "
          f"gold={_SIGNAL_ROI['gold_roi']:+.1f}% "
          f"silver={_SIGNAL_ROI['silver_roi']:+.1f}%")


# ─────────────────────────────────────────────────────────────
# Odds bucket display order

ODDS_BUCKET_ORDER = ["< -750", "-750 to -300", "-300 to -150", "-150 to +150",
                     "+150 to +300", "+300 to +750", "> +750"]


# STANDARD_PLUS filter rules — derived from historical ROI analysis.
# These slices run ~+14-15% ROI; rest of STANDARD runs -1.3%.
# Moneylines on these leagues outperform regardless of sharp book.
STANDARD_PLUS_ML_LEAGUES    = {'NCAAB', 'NHL', 'NFL', 'NCAAF'}
# Spreads on these leagues outperform (Prophet/NoVigApp driven).
STANDARD_PLUS_SPREAD_LEAGUES = {'NFL', 'NCAAF'}
# Sharp books that kill Moneyline ROI — exclude these from STANDARD_PLUS ML
# (Prophet ML = -10.4%, BetMGM play_book ML = -11.5% — handled at play_book level)
STANDARD_PLUS_ML_BAD_SHARPS = {'Prophet'}
# Play books that drag ML down even on good leagues
STANDARD_PLUS_BAD_BOOKS     = {'BetMGM', 'Fliff'}
# Odds range to skip within STANDARD_PLUS (the +300-+750 bucket is -21.5% ROI)
STANDARD_PLUS_BAD_ODDS_MIN  = 301
STANDARD_PLUS_BAD_ODDS_MAX  = 750


# FADE filter rules — derived from historical analysis.
# A bet only qualifies as a FADE if:
#   (a) its league/type/side is in this set (structurally negative all-time), AND
#   (b) the broad 30d ROI for that slice is also <= -5% (confirms current trend), AND
#   (c) n_30d >= 30 (enough data to be meaningful)
# Excluded slices where fading also loses money (both sides have negative EV from vig).
STRUCTURAL_FADES = {
    ('NCAAF', 'Total',       'Under'),   # fade ROI +53.0%  WR 75%
    ('NFL',   'Player Prop', 'Over'),    # fade ROI +41.2%  WR 69%
    # NFL Total Under removed — fade WR only 48% during awake hours (8AM–11PM), below break-even
    ('NHL',   'Player Prop', 'Over'),    # fade ROI +10.7%  WR 55%
    ('NHL',   'Total',       'Under'),   # fade ROI +17.6%  WR 61%
}
FADE_ROI_THRESHOLD  = -5.0   # 30d broad ROI must be at or below this to trigger
FADE_MIN_SAMPLE     = 30     # minimum bets in 30d broad window



# ─────────────────────────────────────────────────────────────

def clean_text(text):
    """Normalise text to lowercase single-spaced string for signature comparison."""
    if not text: return ""
    return re.sub(r'\s+', ' ', str(text).strip()).lower()

def clean_matchup_string(raw_text):
    """Strip UI noise from matchup strings scraped from the tracker page."""
    text = re.sub(r'Open actions menu', '', raw_text, flags=re.IGNORECASE)
    text = re.sub(r'\$\d{1,3}(,\d{3})*(\.\d+)?', '', text)
    text = re.sub(r'(NBA|NFL|NHL|NCAAB|NCAAF|Tennis|UFC).*? at \d{1,2}:\d{2} [AP]M', '', text)
    return text.strip()


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
    if "blocked" in m: return "Blocked Shots"  # must come before generic "shots" check
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

def is_standard_plus(bet_data, bet_type, odds_val, primary_sharp):
    """
    Returns True if a STANDARD-tier bet falls into the historically profitable
    STANDARD_PLUS slice (+14-15% ROI historically).

    Rules derived from slicing 5,929 settled STANDARD bets:
      • Moneyline on NCAAB/NHL/NFL/NCAAF
          AND primary sharp is NOT Prophet (Prophet ML = -10.4%)
          AND play_book is NOT BetMGM or Fliff
          AND odds NOT in +301 to +750 range (-21.5% ROI there)
      • Point Spread on NFL/NCAAF
          AND odds NOT in +301 to +750 range
    """
    league    = str(bet_data.get('league', ''))
    play_book = str(bet_data.get('play_book', '')).strip()

    # Reject bad odds bucket regardless of market
    if STANDARD_PLUS_BAD_ODDS_MIN <= odds_val <= STANDARD_PLUS_BAD_ODDS_MAX:
        return False

    if bet_type == 'Moneyline':
        if league not in STANDARD_PLUS_ML_LEAGUES:           return False
        if primary_sharp in STANDARD_PLUS_ML_BAD_SHARPS:     return False
        if play_book in STANDARD_PLUS_BAD_BOOKS:              return False
        return True

    if bet_type == 'Point Spread':
        if league not in STANDARD_PLUS_SPREAD_LEAGUES:        return False
        return True

    return False


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
    league        = str(bet_data.get('league', ''))

    # Prop category blacklist — demote to WATCH before anything else.
    # Catches structurally negative prop stats (e.g. NHL Shots on Goal, Saves)
    # that aren't caught by the sharp book blacklist.
    if bet_type == 'Player Prop':
        prop_cat = extract_prop_category(bet_data.get('market', ''))
        if (league, prop_cat) in PROP_CATEGORY_LEAGUE_BLACKLIST:
            flags = {
                'is_under': is_under, 'is_prop_under': is_prop_under,
                'is_standard_plus': False, 'consensus': consensus,
                'good_odds': False, 'bad_odds': False, 'good_liq': False,
                'prime_time': False, 'is_fanatics': is_fanatics,
                'prop_blacklisted': True,
            }
            return 'WATCH', flags

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
        'is_under':        is_under,
        'is_prop_under':   is_prop_under,
        'is_standard_plus': False,   # updated below if STANDARD_PLUS is assigned
        'consensus':       consensus,
        'good_odds':       good_odds,
        'bad_odds':        bad_odds,
        'good_liq':        good_liq,
        'prime_time':      prime_time,
        'is_fanatics':     is_fanatics,
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
    # Check if this STANDARD bet falls into a historically positive sub-slice
    primary_sharp = [b.strip().strip('"') for b in str(bet_data.get('sharp_book', '')).split(',') if b.strip()]
    primary_sharp = primary_sharp[0] if primary_sharp else ''
    try:    odds_val = float(str(bet_data.get('play_odds', '0')).replace('+', '').replace('−', '-'))
    except: odds_val = 0

    if is_standard_plus(bet_data, bet_type, odds_val, primary_sharp):
        flags['is_standard_plus'] = True
        # NCAAB ML is overnight-driven: +14.9% all-time but only +4.4% during
        # awake hours (8AM–11PM). Flag so Discord alert can warn the user.
        if league == 'NCAAB' and bet_type == 'Moneyline':
            flags['ncaab_ml_overnight'] = True  # always flag NCAAB ML — message adapts to time
        return 'STANDARD_PLUS', flags

    flags['is_standard_plus'] = False
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
        r = _SIGNAL_ROI.get('diamond_3book_roi', 14.5)
        return f"3+ sharp books agree on a player prop Under — our highest-conviction combo. {r:+.1f}% ROI historically."
    if consensus >= CONSENSUS_THRESHOLD and good_liq:
        return "3+ sharp books agree and liquidity is in the sweet spot. 13.5% ROI historically."
    if consensus >= CONSENSUS_THRESHOLD:
        return "3+ sharp books agree on this line. High consensus is our strongest signal — 14–15% ROI historically."
    if is_prop_under and good_liq:
        r = _SIGNAL_ROI.get('diamond_under_roi', 13.0)
        return f"Player prop Under with liquidity in the sweet spot. {r:+.1f}% ROI historically."
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
        _pu = _SIGNAL_ROI.get('prop_under_roi', 13.4)
        _po = _SIGNAL_ROI.get('prop_over_roi', -2.7)
        reasons.append(f"it's a player prop Under (historically {_pu:+.1f}% ROI vs {_po:+.1f}% for prop Overs)")
    if flags.get('good_liq'):
        reasons.append("liquidity is in the sweet spot ($2k–$10k)")
    if flags.get('prime_time'):
        reasons.append("this appeared during a historically profitable time window")
    if flags.get('good_odds'):
        reasons.append("odds are in the profitable range (−150 to +499)")

    base = build_diamond_description(flags) if tier == 'DIAMOND' else _get_tier_description(tier)
    if tier == 'STANDARD_PLUS':
        sl = []
        if flags.get('bet_type') == 'Moneyline': sl.append('Moneyline')
        if flags.get('bet_type') == 'Point Spread': sl.append('Point Spread')
        base = TIER_DESCRIPTIONS['STANDARD_PLUS']
    why  = ("Flagged because: " + ", ".join(reasons) + ".") if reasons else ""

    notes = []
    if is_low_confidence:                       notes.append("⚠️ Low-confidence sharp source — verify before betting.")
    if is_suppressed and signal_reason:         notes.append(f"🔶 Note: {signal_reason}.")
    if flags.get('is_fanatics'):                notes.append("⚠️ Fanatics line — historically underperforms other books.")
    if flags.get('bad_odds'):                   notes.append("⚠️ Odds in 500–999 range — historically negative ROI.")
    if flags.get('ncaab_ml_overnight'):
        from datetime import datetime as _dt
        _night = _SIGNAL_ROI.get('ncaab_ml_night_roi', 0.5)
        _day   = _SIGNAL_ROI.get('ncaab_ml_day_roi', 1.7)
        _hour  = _dt.now().hour
        _is_night = _hour < 9
        if _night >= _day:
            # Overnight historically better
            if _is_night:
                notes.append(f"🌙 NCAAB Moneyline is in its strongest window — overnight edge {_night:+.1f}% vs {_day:+.1f}% daytime.")
            else:
                notes.append(f"⚠️ NCAAB Moneyline performs best overnight ({_night:+.1f}%) vs now ({_day:+.1f}% daytime). Proceed with awareness.")
        else:
            # Daytime historically better
            if _is_night:
                notes.append(f"⚠️ NCAAB Moneyline performs better during the day ({_day:+.1f}%) than overnight ({_night:+.1f}%). Proceed with awareness.")
            else:
                notes.append(f"☀️ NCAAB Moneyline is in its strongest window — daytime edge {_day:+.1f}% vs {_night:+.1f}% overnight.")

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
    if flags.get('is_fanatics'):      parts.append('⚠️ FANATICS')
    if flags.get('is_standard_plus'):   parts.append('⭐ STD+')
    if flags.get('ncaab_ml_overnight'): parts.append('🌙 OVERNIGHT EDGE')
    if flags.get('prop_blacklisted'):   parts.append('🚫 PROP BLACKLISTED')
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
