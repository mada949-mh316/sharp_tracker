"""
shared_logic.py — Single source of truth for Smart Money Tracker.

Imported by both tracker_v9.py (Discord bot) and dashboard_v5.py (Streamlit).
Any change to tier logic, classification, or thresholds happens here only.

Updated 2026-04-08:
  - Polymarket: Removed Moneyline from Whitelist. Added Tennis & NCAAB to Blacklist.
  - Kalshi: Removed Point Spread from Blacklist. Added Tennis & NCAAB to Blacklist.
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
GOOD_ODDS_MIN       = -150   
GOOD_ODDS_MAX       = 499
GOOD_ODDS_UNDER_MIN = -250   
BAD_ODDS_MIN        = 500
BAD_ODDS_MAX        = 999
GOOD_LIQ_MIN        = 1      
GOOD_LIQ_MAX        = 2000   
PRIME_HOURS         = {1, 2, 3, 4, 5, 8, 12, 13, 16, 18, 19, 22} 
CONSENSUS_THRESHOLD = 3

# Tier display metadata
TIER_ORDER  = ['DIAMOND', 'GOLD', 'SILVER', 'STANDARD_PLUS', 'STANDARD', 'WATCH']
TIER_COLORS = {
    'DIAMOND':       '#00BFFF',
    'GOLD':          '#D4AF37',
    'SILVER':        '#C0C0C0',
    'STANDARD_PLUS': '#27AE60',
    'STANDARD':      '#2ECC71',
    'WATCH':         '#95A5A6',
}
TIER_EMOJI = {
    'DIAMOND': '💎', 'GOLD': '🥇', 'SILVER': '🥈',
    'STANDARD_PLUS': '⭐', 'STANDARD': '🔥', 'WATCH': '👁️',
}

# Discord embed colors (hex int)
TIER_DISCORD_COLOR = {
    'DIAMOND':        0x00BFFF,
    'GOLD':           0xD4AF37,
    'SILVER':         0xC0C0C0,
    'STANDARD_PLUS':  0x27AE60,
    'STANDARD':       0x1ABC9C,
    'WATCH':          0xE67E22,
    'FADE':           0xE74C3C,
    'HIGH_CONVICTION':0x9B59B6,
}

# Sharp book signal quality
SHARP_BOOK_MARKET_WHITELIST = {
    ("NoVigApp",  "Player Prop"),
    ("Prophet",   "Player Prop"),
    ("Pinnacle",  "Point Spread"),
    ("Pinnacle",  "Total"),
    ("NoVigApp",  "Total"),
    ("NoVigApp",  "Moneyline"),
    ("Prophet",   "Total"),
}

SHARP_BOOK_MARKET_BLACKLIST = {
    ("Kalshi",     "NFL"),
    ("NoVigApp",   "Tennis"),
    ("Prophet",    "Tennis"),
    ("Pinnacle",   "Tennis"),
    ("Prophet",    "NCAAF"),
    ("NoVigApp",   "NCAAF"),
    ("NoVigApp",   "Total Games"),
    ("Kalshi",     "Tennis"),
    ("Kalshi",     "NCAAB"),
    ("Polymarket", "Tennis"),
    ("Polymarket", "NCAAB"),
}

# Prop categories that are structurally negative within specific leagues.
PROP_CATEGORY_LEAGUE_BLACKLIST = {
    ("NFL",    "Touchdowns"),     
    ("NHL",    "Points"),         
    ("Tennis", "Total Games"),    
}

LOW_CONFIDENCE_BOOKS = {"Kalshi"}

LEAGUE_SHARP_BOOK_SUPPRESS = {
    "Tennis": {"NoVigApp", "Prophet", "Pinnacle"},
    "NCAAF":  {"Prophet",  "NoVigApp"},
}

# Tier Discord descriptions
def _tier_desc_gold():
    r = _SIGNAL_ROI.get('gold_roi', 13.0)
    return (f"Strong signal: either 3+ sharp books in consensus, a player prop Under "
            f"with good liquidity, or a prime-time spot with solid market depth. "
            f"{r:+.1f}% ROI historically.")

def _tier_desc_silver():
    r = _SIGNAL_ROI.get('silver_roi', 10.6)
    return (f"Player prop Under in a profitable odds range, or a prime-time alert. "
            f"Our core edge — {r:+.1f}% ROI historically.")

def _tier_desc_std_plus():
    r = _SIGNAL_ROI.get('std_plus_roi', 23.3)
    return (f"Selective STANDARD signal: Moneyline on NHL/NFL, or a Point "
            f"Spread on NFL. These slices run {r:+.1f}% ROI historically. "
            f"Play at standard unit size.")

TIER_DESCRIPTIONS = {
    'GOLD':          None,   
    'SILVER':        None,   
    'STANDARD_PLUS': None,   
    'STANDARD': "Sharp money detected. Passes ROI filters but no additional edge flags. "
                "Volume play — follow at standard unit size.",
    'WATCH':    "Flagged for review: Fanatics line or odds in the 500–999 range, both historically "
                "negative. Proceed with caution or skip.",
    'FADE':     "Sharp signal is historically negative here. Data suggests betting the opposite side.",
}

def _get_tier_description(tier):
    if tier == 'GOLD':          return _tier_desc_gold()
    if tier == 'SILVER':        return _tier_desc_silver()
    if tier == 'STANDARD_PLUS': return _tier_desc_std_plus()
    return TIER_DESCRIPTIONS.get(tier, "Sharp money detected.")


# ─────────────────────────────────────────────────────────────
# DYNAMIC SIGNAL ROI CACHE
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
    import os
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
        except:
            return

    import pandas as _pd
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

    is_player    = df['market'].str.contains('Player', case=False, na=False)
    is_under     = df['play_selection'].str.contains('Under', case=False, na=False)
    is_over      = df['play_selection'].str.contains('Over',  case=False, na=False)
    is_good_liq  = df['liquidity'].between(GOOD_LIQ_MIN, GOOD_LIQ_MAX)
    is_good_odds = df['_odds'].between(GOOD_ODDS_MIN, GOOD_ODDS_MAX)
    is_ncaab_ml  = (df['league'] == 'NCAAB') & df['market'].str.contains('Moneyline', case=False, na=False)
    is_night     = df['_hour'].between(0, 9) | df['_hour'].between(22, 23)

    try:
        tiers = []
        for _, row in df.iterrows():
            try:
                t, _ = classify_tier(row)
                tiers.append(t)
            except:
                tiers.append('STANDARD')
        df['_tier'] = tiers
    except:
        df['_tier'] = 'STANDARD'

    updates = {
        'prop_under_roi':     _roi(df[is_player & is_under]),
        'prop_over_roi':      _roi(df[is_player & is_over]),
        'good_liq_roi':       _roi(df[is_good_liq]),
        'good_odds_roi':      _roi(df[is_good_odds]),
        'diamond_roi':        _roi(df[df['_tier'] == 'DIAMOND']),
        'gold_roi':           _roi(df[df['_tier'] == 'GOLD']),
        'silver_roi':         _roi(df[df['_tier'] == 'SILVER']),
        'std_plus_roi':       _roi(df[df['_tier'] == 'STANDARD_PLUS']),
        'diamond_3book_roi':  _roi(df[(df['_tier'] == 'DIAMOND') & (df['sharp_book'].str.count(',') >= 2)]),
        'diamond_under_roi':  _roi(df[(df['_tier'] == 'DIAMOND') & is_player & is_under]),
        'ncaab_ml_night_roi': _roi(df[is_ncaab_ml & is_night]),
        'ncaab_ml_day_roi':   _roi(df[is_ncaab_ml & ~is_night]),
    }

    for k, v in updates.items():
        if v is not None:
            _SIGNAL_ROI[k] = v

    _SIGNAL_ROI_LAST_REFRESH = _dt.now()
    print(f"✅ Signal ROI cache refreshed (Liquidity Range: ${GOOD_LIQ_MIN}-${GOOD_LIQ_MAX})")


# ─────────────────────────────────────────────────────────────
# Odds bucket display order
# ─────────────────────────────────────────────────────────────
ODDS_BUCKET_ORDER = [
    "< -750", "-750 to -300", "-300 to -150", "-150 to +150",
    "+150 to +300", "+300 to +750", "> +750",
]

# STANDARD_PLUS filter rules (NCAA REMOVED)
STANDARD_PLUS_ML_LEAGUES     = {'NHL', 'NFL'}
STANDARD_PLUS_SPREAD_LEAGUES = {'NFL'}
STANDARD_PLUS_ML_BAD_SHARPS  = {'Prophet'}
STANDARD_PLUS_BAD_BOOKS      = {'BetMGM', 'Fliff'}
STANDARD_PLUS_BAD_ODDS_MIN   = 301
STANDARD_PLUS_BAD_ODDS_MAX   = 750

# FADE filter rules
STRUCTURAL_FADES = {
    ('NCAAF', 'Total',       'Under'),   
    ('NFL',   'Player Prop', 'Over'),    
    ('NHL',   'Player Prop', 'Over'),    
    ('NHL',   'Total',       'Under'),   
}
FADE_ROI_THRESHOLD = -5.0
FADE_MIN_SAMPLE    = 30


# ─────────────────────────────────────────────────────────────
# TEXT HELPERS
# ─────────────────────────────────────────────────────────────

def clean_text(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', str(text).strip()).lower()

def clean_matchup_string(raw_text):
    text = re.sub(r'Open actions menu', '', raw_text, flags=re.IGNORECASE)
    text = re.sub(r'\$\d{1,3}(,\d{3})*(\.\d+)?', '', text)
    text = re.sub(r'(NBA|NFL|NHL|NCAAB|NCAAF|Tennis|UFC).*? at \d{1,2}:\d{2} [AP]M', '', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────
# ODDS / MATH HELPERS
# ─────────────────────────────────────────────────────────────

def parse_odds_val(val):
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
    if val < -750:         return "< -750"
    if -750 <= val < -300: return "-750 to -300"
    if -300 <= val < -150: return "-300 to -150"
    if -150 <= val <= 150: return "-150 to +150"
    if  150 <  val <= 300: return "+150 to +300"
    if  300 <  val <= 750: return "+300 to +750"
    return "> +750"

def calculate_arb_percent(play_odds, sharp_odds):
    play, sharp = parse_odds_val(play_odds), parse_odds_val(sharp_odds)
    if play == 0 or sharp == 0: return 0.0
    dp = get_decimal_odds(play); ds = get_decimal_odds(sharp)
    if dp == 0 or ds == 0: return 0.0
    total_imp = (1 / dp) + (1 / ds)
    return ((1 / total_imp) - 1) * 100 if total_imp else 0.0

def calculate_profit(odds_val, result):
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
    m, s = str(market).lower(), str(selection).lower()
    if "moneyline" in m: return "Moneyline"
    if "spread" in m or "handicap" in m or "run line" in m or "puck line" in m: return "Point Spread"
    if "pitcher" in m: return "Player Prop"
    if "player" in m or "milestone" in m or "props" in m: return "Player Prop"
    if any(x in m for x in ["shots", "sog", "assists", "rebounds", "threes", "touchdowns",
                            "hits", "home runs", "strikeouts", "total bases", "earned runs", "rbi"]):
        return "Player Prop"
    if "total runs - 1st inning" in m:  return "Total - 1st Inning"
    if "total runs - 1st 5" in m:       return "Total - 1st 5 Innings"
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
    m = str(market).lower().replace("player ", "").replace("alternate ", "").replace("alt ", "")
    if "milestone" in m: return "Milestone"
    if "points" in m and "rebounds" in m and "assists" in m: return "PRA"
    if "points" in m and "rebounds" in m:                    return "Pts+Reb"
    if "points" in m and "assists"  in m:                    return "Pts+Ast"
    if "rebounds" in m and "assists" in m:                   return "Reb+Ast"
    if "blocks"   in m and "steals"  in m:                   return "Blk+Stl"
    if "points"    in m:                                      return "Points"
    if "rebounds"  in m:                                      return "Rebounds"
    if "assists"   in m:                                      return "Assists"
    if "threes"    in m or "3-point" in m or "3pt" in m:     return "Threes"
    if "blocked"   in m:                                      return "Blocked Shots"
    if "blocks"    in m:                                      return "Blocks"
    if "steals"    in m:                                      return "Steals"
    if "turnovers" in m:                                      return "Turnovers"
    if "shots"     in m or "sog"    in m:                    return "Shots on Goal"
    if "saves"     in m:                                      return "Saves"
    if "goals"     in m:                                      return "Goals"
    if "passing"   in m:                                      return "Passing"
    if "rushing"   in m:                                      return "Rushing"
    if "receiving" in m or "receptions" in m:                 return "Receiving"
    if "touchdown" in m or "score" in m:                      return "Touchdowns"
    if "double"    in m:                                      return "Double Double"
    if "triple"    in m:                                      return "Triple Double"
    if "pitcher strikeout" in m or "strikeouts" in m:        return "Pitcher Strikeouts"
    if "pitcher earned runs" in m or "earned runs" in m:     return "Pitcher Earned Runs"
    if "pitcher hits allowed" in m:                          return "Pitcher Hits Allowed"
    if "pitcher walks" in m or "walks allowed" in m:         return "Pitcher Walks Allowed"
    if "pitcher outs" in m or "outs recorded" in m:          return "Pitcher Outs Recorded"
    if "home runs" in m or "hr" in m:                        return "Home Runs"
    if "total bases" in m:                                   return "Total Bases"
    if "hits" in m and "runs" in m and "rbi" in m:           return "Hits+Runs+RBIs"
    if "hits" in m:                                          return "Hits"
    if "runs" in m:                                          return "Runs"
    if "rbi" in m:                                           return "RBIs"
    if "stolen bases" in m:                                  return "Stolen Bases"
    if "walks" in m:                                         return "Walks"
    if "outs" in m:                                          return "Pitching Outs"
    return "Other"


# ─────────────────────────────────────────────────────────────
# SHARP BOOK SIGNAL QUALITY
# ─────────────────────────────────────────────────────────────

def evaluate_sharp_signal(sharp_book_str, bet_type, league):
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
    league    = str(bet_data.get('league', ''))
    play_book = str(bet_data.get('play_book', '')).strip()

    if STANDARD_PLUS_BAD_ODDS_MIN <= odds_val <= STANDARD_PLUS_BAD_ODDS_MAX:
        return False

    if bet_type == 'Moneyline':
        if league not in STANDARD_PLUS_ML_LEAGUES:       return False
        if primary_sharp in STANDARD_PLUS_ML_BAD_SHARPS: return False
        if play_book in STANDARD_PLUS_BAD_BOOKS:          return False
        return True

    if bet_type == 'Point Spread':
        if league not in STANDARD_PLUS_SPREAD_LEAGUES:   return False
        return True

    return False


def classify_tier(bet_data):
    s           = str(bet_data.get('play_selection', '')).lower()
    is_under    = 'under' in s
    books       = [b.strip().strip('"') for b in str(bet_data.get('sharp_book', '')).split(',') if b.strip()]
    consensus   = len(books)
    is_fanatics = 'fanatics' in str(bet_data.get('play_book', '')).lower()

    bet_type      = categorize_bet(bet_data.get('market', ''), bet_data.get('play_selection', ''))
    is_prop_under = is_under and bet_type == 'Player Prop'
    league        = str(bet_data.get('league', ''))

    if bet_type == 'Player Prop':
        prop_cat = extract_prop_category(bet_data.get('market', ''))
        if (league, prop_cat) in PROP_CATEGORY_LEAGUE_BLACKLIST:
            flags = {
                'is_under': is_under, 'is_prop_under': is_prop_under,
                'is_standard_plus': False, 'consensus': consensus,
                'good_odds': False, 'good_odds_under': False,
                'bad_odds': False, 'good_liq': False,
                'prime_time': False, 'is_fanatics': is_fanatics,
                'prop_blacklisted': True,
            }
            return 'WATCH', flags

    try:
        odds = float(str(bet_data.get('play_odds', '0')).replace('+', '').replace('−', '-'))
    except:
        odds = 0.0

    good_odds = GOOD_ODDS_MIN <= odds <= GOOD_ODDS_MAX
    good_odds_under = GOOD_ODDS_UNDER_MIN <= odds <= GOOD_ODDS_MAX
    bad_odds = BAD_ODDS_MIN <= odds <= BAD_ODDS_MAX

    try:    liq = float(bet_data.get('liquidity', 0))
    except: liq = 0.0
    good_liq = GOOD_LIQ_MIN <= liq <= GOOD_LIQ_MAX

    prime_time = False
    try:
        ts = bet_data.get('timestamp', '')
        if ts:
            dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).strip())
            prime_time = dt.hour in PRIME_HOURS
    except Exception:
        pass

    flags = {
        'is_under':         is_under,
        'is_prop_under':    is_prop_under,
        'is_standard_plus': False,        
        'consensus':        consensus,
        'good_odds':        good_odds,
        'good_odds_under':  good_odds_under,
        'bad_odds':         bad_odds,
        'good_liq':         good_liq,
        'prime_time':       prime_time,
        'is_fanatics':      is_fanatics,
    }

    if bad_odds and consensus < CONSENSUS_THRESHOLD and not is_prop_under:
        return 'WATCH', flags

    if consensus >= CONSENSUS_THRESHOLD and good_odds and is_prop_under:
        return 'DIAMOND', flags
    if consensus >= CONSENSUS_THRESHOLD and good_odds and good_liq:
        return 'DIAMOND', flags
    if consensus >= CONSENSUS_THRESHOLD and good_odds:
        return 'GOLD', flags
    if is_prop_under and good_liq and good_odds_under:
        return 'GOLD', flags
    if prime_time and good_liq and good_odds:
        return 'GOLD', flags
    if is_prop_under and good_odds_under:
        return 'SILVER', flags
    if prime_time and good_odds:
        return 'SILVER', flags

    primary_sharp = books[0] if books else ''
    if is_standard_plus(bet_data, bet_type, odds, primary_sharp):
        flags['is_standard_plus'] = True
        return 'STANDARD_PLUS', flags

    return 'STANDARD', flags


# ─────────────────────────────────────────────────────────────
# DISCORD MESSAGE BUILDERS
# ─────────────────────────────────────────────────────────────

def build_diamond_description(flags):
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
        reasons.append("liquidity is in the sweet spot ($1–$2k)")
    if flags.get('prime_time'):
        reasons.append("this appeared during a historically profitable time window")
    if flags.get('good_odds'):
        reasons.append("odds are in the profitable range (−150 to +499)")
    elif flags.get('good_odds_under') and flags.get('is_prop_under'):
        reasons.append("odds are in the profitable prop Under range (−250 to +499)")

    base = build_diamond_description(flags) if tier == 'DIAMOND' else _get_tier_description(tier)
    why  = ("Flagged because: " + ", ".join(reasons) + ".") if reasons else ""

    notes = []
    if is_low_confidence:               notes.append("⚠️ Low-confidence sharp source — verify before betting.")
    if is_suppressed and signal_reason: notes.append(f"🔶 Note: {signal_reason}.")
    if flags.get('is_fanatics'):        notes.append("⚠️ Fanatics line — historically underperforms other books.")
    if flags.get('bad_odds'):           notes.append("⚠️ Odds in 500–999 range — historically negative ROI.")
    
    return "\n".join(p for p in [why, base, " ".join(notes)] if p)


def build_flag_bar(flags):
    parts = []
    if flags.get('is_prop_under'):                        parts.append('🔽 PROP UNDER')
    if flags.get('consensus', 1) >= CONSENSUS_THRESHOLD:  parts.append(f"🤝 {flags['consensus']}x CONSENSUS")
    elif flags.get('consensus', 1) == 2:                  parts.append('🤝 2x CONSENSUS')
    if flags.get('good_odds'):
        parts.append('✅ GOOD ODDS')
    elif flags.get('good_odds_under') and flags.get('is_prop_under'):
        parts.append('✅ GOOD ODDS (UNDER)')   
    if flags.get('bad_odds'):        parts.append('⚠️ BAD ODDS')
    if flags.get('good_liq'):        parts.append('💧 GOOD LIQ')
    if flags.get('prime_time'):      parts.append('⏰ PRIME TIME')
    if flags.get('is_fanatics'):     parts.append('⚠️ FANATICS')
    if flags.get('is_standard_plus'):    parts.append('⭐ STD+')
    if flags.get('ncaab_ml_overnight'):  parts.append('🌙 OVERNIGHT EDGE')
    if flags.get('prop_blacklisted'):    parts.append('🚫 PROP BLACKLISTED')
    return '  |  '.join(parts) if parts else '—'


# ─────────────────────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────────────────────

def clean_raw_df(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    if 'league' in df.columns:
        df = df[df['league'].isin(VALID_LEAGUES)]

    if 'play_book' in df.columns:
        df = df[~df['play_book'].isin(DFS_BOOKS)]
        df = df[df['play_book'].notna()]

    if 'profit' in df.columns:
        df['profit'] = pd.to_numeric(
            df['profit'].astype(str)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False),
            errors='coerce'
        ).fillna(0.0)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    return df


def add_derived_columns(df):
    df = df.copy()

    df['odds_val']      = df['play_odds'].apply(parse_odds_val)
    df['odds_bucket']   = df['odds_val'].apply(get_odds_bucket)
    df['bet_type']      = df.apply(lambda r: categorize_bet(r.get('market', ''), r.get('play_selection', '')), axis=1)
    df['bet_side']      = df['play_selection'].apply(get_bet_side)
    df['prop_cat']      = df.apply(lambda r: extract_prop_category(r.get('market', '')) if r['bet_type'] == 'Player Prop' else '', axis=1)
    df['consensus']     = df['sharp_book'].astype(str).str.split(',').str.len().fillna(1).astype(int)
    df['primary_sharp'] = df['sharp_book'].astype(str).str.split(',').str[0].str.strip().str.strip('"')
    df['is_prop_under'] = (df['bet_type'] == 'Player Prop') & (df['bet_side'] == 'Under')
    df['arb_pct']       = df.apply(lambda r: calculate_arb_percent(r.get('play_odds', 0), r.get('sharp_odds', 0)), axis=1)

    tiers = df.apply(lambda r: classify_tier(r.to_dict())[0], axis=1)
    df['tier'] = tiers

    def _combo(r):
        league = str(r.get('league', ''))
        bt, side, pc = r['bet_type'], r['bet_side'], r['prop_cat']
        if bt == 'Player Prop': return f"{side} {league} {pc}"
        if bt == 'Total':       return f"{side} {league} Game Total"
        return f"{league} {bt}"
    df['combo'] = df.apply(_combo, axis=1)

    return df
