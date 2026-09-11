"""
shared_logic.py — Single source of truth for Smart Money Tracker.

Imported by both tracker_v9.py (Discord bot) and dashboard_v5.py (Streamlit).
Any change to tier logic, classification, or thresholds happens here only.

Updated 2026-04-26:
  - Applied threshold recalibration data (n=32,617).
  - Adjusted BAD_ODDS, PRIME_HOURS, and STANDARD_PLUS logic.
  - Suppressed NCAAB/Tennis globally and demoted Polymarket confidence.
  - Expanded PROP_CATEGORY_LEAGUE_BLACKLIST and STRUCTURAL_FADES.
"""

import re
import os
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

# 2026-08-20: `bets.timestamp` was mislabeled at the source until this moment --
# tracker.py stamped rows with datetime.now() on its host, which is naive *local*
# time (America/New_York), then that got inserted into a timestamptz column on a
# UTC-session DB connection and was silently stored as if it were already UTC.
# Rows at/before this cutover (Mac wall-clock, the OLD mislabeled convention) are
# old-style -- their raw value IS already ET, no conversion needed. Rows after are
# new-style -- genuine UTC, needs the normal conversion. See project memory
# project_date_anchor_et_utc_gotcha and post_grader.py/signal_review.py's matching
# constant (kept in sync manually -- same literal value, not a shared import, since
# this module has no dependency on either of those files).
BETS_TS_BUG_CUTOVER = pd.Timestamp('2026-08-20 10:08:25')


def bets_ts_to_et_series(ts_series):
    """Vectorized cutover-aware conversion of a `bets.timestamp` Series (e.g. from
    pd.read_sql on the `bets` table) to tz-aware America/New_York. See
    BETS_TS_BUG_CUTOVER / bets_ts_hour_et above. Use .dt.hour / .dt.date on the result
    same as any tz-aware datetime Series.

    2026-08-26: `bets` now also has a precomputed `ts_et` column (backfilled once with
    this exact function, and stamped correctly by tracker.py's write_bets_to_db() for
    every row since) -- for a plain SQL query or a fresh pd.read_sql, just SELECT ts_et
    directly instead of calling this function on `timestamp`. This function still exists
    for any code that only has a raw `timestamp` value/Series and no easy way to
    join/reselect ts_et, and as the source of truth ts_et was derived from -- but new
    code should prefer the column."""
    # The historical column contains both naive and offset-bearing ISO timestamps.
    # pandas otherwise infers one format from the first row and rejects the other.
    raw = pd.to_datetime(ts_series, utc=True, format='mixed').dt.tz_localize(None)
    is_old = raw <= BETS_TS_BUG_CUTOVER
    et_old = raw[is_old].dt.tz_localize(_ET)
    et_new = raw[~is_old].dt.tz_localize('UTC').dt.tz_convert(_ET)
    return pd.concat([et_old, et_new]).sort_index()


def bets_ts_hour_et_series(ts_series):
    """Vectorized cutover-aware ET hour for a `bets.timestamp` Series. See
    bets_ts_to_et_series above."""
    return bets_ts_to_et_series(ts_series).dt.hour


def bets_ts_hour_et(ts):
    """Cutover-aware ET hour for a single bets.timestamp value (str, naive datetime,
    or tz-aware datetime) as actually stored in the `bets` table. Use this instead of
    a raw `.hour` on that column -- see BETS_TS_BUG_CUTOVER above."""
    if ts is None or ts == '':
        return None
    dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).strip())
    naive = dt.replace(tzinfo=None) if dt.tzinfo else dt
    if naive <= BETS_TS_BUG_CUTOVER.to_pydatetime():
        return naive.hour  # old-style: raw value IS already ET
    return naive.replace(tzinfo=ZoneInfo('UTC')).astimezone(_ET).hour


def load_owner_note_cache():
    """Loads owner_note_cache (populated nightly by rebuild_owner_notes.py) into
    {segment_key: {'roi', 'n', 'stable', 'positive'}}. Backs tracker.py's
    owner_market_note()/owner_sharp_source_note() -- those used to be hardcoded
    percentages baked into if/else branches with nothing to catch staleness (see
    rebuild_owner_notes.py's docstring). Returns {} on any failure; callers should
    treat an empty/missing key as 'no stable read yet', not silence."""
    try:
        import psycopg2
        # 2026-08-26: was a hardcoded DB password here as the fallback default -- this file
        # gets pushed to a PUBLIC GitHub repo (app.py's dependency), so a literal credential
        # here would be a real exposure, not just a reliability convenience. Every real
        # runtime (launchd plists, systemd services, Streamlit Cloud secrets) already sets
        # DATABASE_URL, so this degrades to the documented {} return instead of connecting
        # if it's ever missing -- no hardcoded fallback needed.
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            return {}
        conn = psycopg2.connect(db_url, sslmode='disable')
        cur = conn.cursor()
        cur.execute("SELECT segment_key, roi, n, stable, positive FROM owner_note_cache")
        rows = cur.fetchall()
        conn.close()
        return {k: {'roi': float(roi), 'n': int(n), 'stable': bool(stable), 'positive': bool(positive)}
                for k, roi, n, stable, positive in rows}
    except Exception:
        return {}

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

UNIT_SIZE   = 100
CSV_PATH    = "data/bets.csv"
SHEET_NAME  = "Smart Money Bets"
CREDS_FILE  = "creds.json"

DFS_BOOKS     = ['PrizePicks', 'Betr', 'Dabble', 'Underdog', 'Sleeper', 'Draftkings6', 'DraftKings6']
# Temporarily removed 'NCAAB' and 'Tennis' due to heavy negative ROI all-time
VALID_LEAGUES = ['NBA', 'NFL', 'NHL', 'NCAAF', 'UFC', 'MLB', 'WNBA', 'Soccer']

# Tier scoring thresholds
GOOD_ODDS_MIN       = -150   
GOOD_ODDS_MAX       = 499
GOOD_ODDS_UNDER_MIN = -250   
BAD_ODDS_MIN        = 1000   # Lifted from 500 based on +6.8% ROI for the 500-999 range
BAD_ODDS_MAX        = 9999
GOOD_LIQ_MIN        = 1      
GOOD_LIQ_MAX        = 2000   
PRIME_HOURS         = {0, 4, 5, 6, 8, 9, 11, 13, 14, 15, 16, 17, 18} # Updated 2026-05-07: removed 1am (-3%), noon (-3.8%); added 6am (+14.4%), 11am (+16%), 1pm (+12.4%), 2pm (+7.5%)
CONSENSUS_THRESHOLD = 3

# Tier display metadata
TIER_ORDER  = ['GOLD', 'SILVER', 'STANDARD_PLUS', 'STANDARD', 'BRONZE', 'WATCH']
TIER_COLORS = {
    'BRONZE':       '#CD7F32',
    'GOLD':          '#D4AF37',
    'SILVER':        '#C0C0C0',
    'STANDARD_PLUS': '#27AE60',
    'STANDARD':      '#2ECC71',
    'WATCH':         '#95A5A6',
}
TIER_EMOJI = {
    'BRONZE': '🥉', 'GOLD': '🥇', 'SILVER': '🥈',
    'STANDARD_PLUS': '⭐', 'STANDARD': '🔥', 'WATCH': '👁️',
}

# Discord embed colors (hex int)
TIER_DISCORD_COLOR = {
    'BRONZE':        0xCD7F32,
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
    ("NoVigApp",   "Total Games"),
    ("Kalshi",     "Tennis"),
    ("Kalshi",     "NCAAB"),
    ("Polymarket", "Tennis"),
    ("Polymarket", "NCAAB"),
}
# Kalshi/NFL unblocked 2026-08-13 (owner): pre-existing, undated blacklist entry with no
# recorded rationale in this file's history -- owner's read is Kalshi may be sharper this
# NFL season than whenever this was originally added. Same grace-period logic as any other
# fresh-season league now applies: let it through and let the data speak once real volume
# accrues, same treatment the Tennis blacklist got on 2026-08-09 before being partly reversed.
# Tennis (NoVigApp/Prophet/Pinnacle) unblocked 2026-08-09: the 2026-04-26
# blacklist was stale — re-check on smash_score>=50 population (n=3742)
# showed +3.8% ROI combined, STABLE, improving H1 +3.2% -> H2 +4.4%.
# Pinnacle +6.9% (n=1260), Prophet +4.7% (n=766), NoVigApp +2.5% (n=1228),
# all stable/positive individually. Kalshi (-4.7%, n=93) and Polymarket
# (-1.9%, unstable) stay blacklisted for Tennis.
# NCAAF (Prophet/NoVigApp) unblocked 2026-08-27 (owner report: "I see them popping up on
# propprofessor but not in the alerts"): another pre-existing, undated blacklist entry with
# no recorded rationale -- same pattern as the two above. 453 NCAAF bets had been collected
# into `bets` with only 1 ever alerted, entirely from this pair being blacklisted (they're
# the dominant sharp-source books on most cards, same as everywhere else). Re-checked:
# NoVigApp +17.3% ROI (n=73, 60.3% win), Prophet +8.3% ROI (n=73, 51.4% win), both solidly
# positive -- no basis found for the block. Data spans Dec 2025-Aug 2026 (last season +
# this season's early games); revisit once this season's own volume accrues if it decays.

# Prop categories that are structurally negative within specific leagues.
PROP_CATEGORY_LEAGUE_BLACKLIST = {
    ("NFL",    "Touchdowns"),     
    ("NHL",    "Points"),         
    ("Tennis", "Total Games"),
    ("MLB",    "Stolen Bases"),
    ("MLB",    "RBIs"),
    ("MLB",    "Total Bases"),
    ("NHL",    "Blocked Shots"),
    ("NHL",    "Shots on Goal"),
}

# Demoted Polymarket here to reduce weighting influence (-2.8% ROI)
LOW_CONFIDENCE_BOOKS = {"Kalshi"}

# NCAAF entry removed 2026-08-27 (owner-approved) -- see SHARP_BOOK_MARKET_BLACKLIST comment
# above for the re-check that found it stale. Kept as an empty dict, not deleted, since it's
# the hook evaluate_sharp_signal() already checks if a league ever needs this again.
LEAGUE_SHARP_BOOK_SUPPRESS = {}

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
    'WATCH':    "Flagged for review: Odds exceed +1000, historically extremely high variance. "
                "Proceed with caution or skip.",
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
    'bronze_roi':         14.9,
    'gold_roi':            12.2,
    'silver_roi':          11.4,
    'std_plus_roi':        23.3,
    'bronze_3book_roi':   14.9,
    'bronze_under_roi':    7.5,
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
            conn = psycopg2.connect(db_url, sslmode='disable', connect_timeout=8)
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
    df['_hour'] = bets_ts_hour_et_series(df['timestamp'])

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
        'bronze_roi':        _roi(df[df['_tier'] == 'BRONZE']),
        'gold_roi':           _roi(df[df['_tier'] == 'GOLD']),
        'silver_roi':         _roi(df[df['_tier'] == 'SILVER']),
        'std_plus_roi':       _roi(df[df['_tier'] == 'STANDARD_PLUS']),
        'bronze_3book_roi':  _roi(df[(df['_tier'] == 'BRONZE') & (df['sharp_book'].str.count(',') >= 2)]),
        'bronze_under_roi':  _roi(df[(df['_tier'] == 'BRONZE') & is_player & is_under]),
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

# STANDARD_PLUS filter rules
STANDARD_PLUS_ML_LEAGUES     = {'NFL'} # Removed NHL (negative ROI)
STANDARD_PLUS_SPREAD_LEAGUES = {'NFL', 'NCAAF'} # Added NCAAF (+11.5% ROI)
STANDARD_PLUS_ML_BAD_SHARPS  = {'Prophet'}
STANDARD_PLUS_BAD_BOOKS      = {'BetMGM', 'Fliff'}
STANDARD_PLUS_BAD_ODDS_MIN   = 301
STANDARD_PLUS_BAD_ODDS_MAX   = 750

# FADE filter rules
STRUCTURAL_FADES = {
    ('NFL',   'Player Prop', 'Over'),    
    ('NHL',   'Player Prop', 'Over'),    
    ('MLB',   'Total',       'Under'),
    ('NCAAB', 'Point Spread', 'Other'),
    ('Tennis', 'Total',      'Over'),
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

def fair_value_decimal(odds):
    """2026-09-02 (owner request: "anywhere that says fair value, include it in decimal, ie
    .60 for 60 cents"). American odds -> implied probability as a bare decimal fraction (no
    leading 0, e.g. '.60' not '0.60' or '60%'). Same probability math as
    exchange_cents_suffix() above, just a different display convention for wherever a report/
    ping shows a "fair odds" number and the owner wants the probability alongside it. Empty
    string on unparseable/zero odds, so it's always safe to append inline."""
    odds_s = str(odds).strip()
    if odds_s.lower() == 'even':
        o = 100.0
    else:
        try:
            o = float(odds_s.replace('+', '').replace('−', '-'))
        except (TypeError, ValueError):
            return ''
    if o == 0:
        return ''
    p = 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)
    return f"{p:.2f}".lstrip('0')

def cents_to_american(cents):
    """Inverse of exchange_cents_suffix's cents formula: a 0-100 cents price (implied
    probability x 100) -> American odds. None if cents is out of (0, 100) range (a
    contract can't cost $0 or $1+)."""
    try:
        c = float(cents)
    except (TypeError, ValueError):
        return None
    if c <= 0 or c >= 100:
        return None
    p = c / 100.0
    if p > 0.5:
        return round(-100 * p / (1 - p))
    return round(100 * (1 - p) / p)

# Books that natively price in cents (implied probability x 100) rather than American
# odds -- see normalize_odds_val() below. 2026-08-27 (owner report): Novig just switched
# its own app to cents-style pricing ("decimal odds (cents)"), and the owner expects
# ProphetX to follow -- kept as a set (not just Novig) so a future book flip is a one-line
# add here, not a new code path.
CENTS_NATIVE_BOOKS = {'novig', 'novigapp', 'prophet', 'prophetx', 'prophet x',
                       'kalshi', 'polymarket', 'polymarketus'}

def normalize_odds_val(raw, book):
    """Best-effort: return proper American odds (float) for a raw odds value that MAY have
    arrived as a cents/decimal price instead, for a book in CENTS_NATIVE_BOOKS. 2026-08-27
    (owner report): Novig switched its own app to cents pricing; every integration here reads
    Novig/exchange prices through a third-party aggregator (Prop Professor, CNO, KeepBetting)
    that's expected to keep normalizing to American odds on ITS end -- confirmed still doing so
    live as of this date. This is the defensive backstop for when that normalization breaks or
    lags (here, or for ProphetX next): American odds are NEVER in (-100, 100) by construction,
    so any value in that band for a cents-native book is almost certainly a cents price that
    slipped through unconverted, not a real American odds value.

    Handles: '52¢'/'52c' (explicit cents suffix), 0 < value < 1 (a $/probability fraction,
    e.g. 0.52), and 1 <= abs(value) < 100 (a bare cents integer, e.g. 52). Values with
    abs(value) >= 100 are assumed to already be valid American odds and returned unchanged --
    this function is safe to call unconditionally on every row, not just ones known to be
    broken (no historical-data cutover needed; already-correct rows round-trip untouched)."""
    s = str(raw).strip().replace('−', '-')
    is_cents_book = str(book or '').strip().lower().replace(' ', '') in \
        {b.replace(' ', '') for b in CENTS_NATIVE_BOOKS}
    if not is_cents_book:
        return parse_odds_val(raw)
    explicit_cents = re.search(r'([\d.]+)\s*[c¢]\b', s, re.IGNORECASE)
    if explicit_cents:
        american = cents_to_american(float(explicit_cents.group(1)))
        return american if american is not None else parse_odds_val(raw)
    # parse_odds_val's regex only captures the integer part ('0.52' -> 0), which silently
    # breaks the < 1 fractional-price case below -- parse the raw float ourselves first.
    float_m = re.search(r'([-+]?\d*\.?\d+)', s)
    try:
        val = float(float_m.group(1)) if float_m else 0.0
    except (TypeError, ValueError):
        val = 0.0
    if val == 0:
        return 0.0
    if abs(val) >= 100:
        return val   # already valid American odds
    # cents prices are inherently unsigned (cost of a contract, 0-100) -- abs() first so a
    # stray sign on the raw value (shouldn't happen, but seen elsewhere as scraper noise)
    # doesn't fall through unconverted.
    cents = abs(val) * 100.0 if 0 < abs(val) < 1 else abs(val)   # 0.52 -> 52, or bare 52 -> 52
    american = cents_to_american(cents)
    return american if american is not None else val

# Books whose OWN app shows implied probability instead of American odds, in CENTS
# specifically ("52c") -- see exchange_cents_suffix() below. Every other book (including
# traditional sportsbooks with no cents-native app of their own) gets the same number shown
# as a plain percent instead -- see 2026-08-28 note in the docstring.
_CENTS_DISPLAY_BOOKS      = ('kalshi', 'polymarket')   # PolymarketUS matches via 'polymarket'

def exchange_cents_suffix(odds, book):
    """'(NN%)' or '(NNc)' showing this American odds' implied probability, as a
    self-contained parenthetical with no leading space. Cents specifically for
    Kalshi/Polymarket (their own app's native display -- same underlying number, just a
    different symbol). 2026-08-19 (owner request, Kalshi only): Kalshi natively quotes this
    way, not American odds, so every system here converting to American odds for display can
    make the line look like it moved when it hasn't (or vice versa) -- showing both lets the
    owner cross-check against what the book's own app actually shows before placing.
    2026-08-25: extended to Polymarket/PolymarketUS. 2026-08-27: Novig switched its own app to
    this same implied-probability display as a percentage. 2026-08-28 (owner request): "I want
    all american odds to list the cents percentage conversion next to it" -- widened from
    exchange-only to EVERY book, since the owner wants the probability annotation universally,
    not just for books that happen to natively quote this way themselves. 2026-08-28 (owner
    request): "the parentheses [should] be separate ... (+108)(48.1%)" -- previously returned
    a LEADING-SPACE '(NN%)' meant to be concatenated onto the odds string before the whole
    thing got wrapped in one more pair of parens by the caller, e.g. "(+108 (48.1%))"; now
    self-contained with no leading space so callers wrap the odds alone in its own parens and
    append this directly after, e.g. "(+108)" + "(48.1%)" = "(+108)(48.1%)". Empty string only
    for unparseable/zero odds, so it's safe to always append inline."""
    book_l = str(book or '').lower()
    is_cents = any(k in book_l for k in _CENTS_DISPLAY_BOOKS)
    odds_s = str(odds).strip()
    if odds_s.lower() == 'even':
        # 2026-08-28: pre-existing gap, more visible now that this runs on every book instead
        # of just a few exchanges -- "EVEN" (used by some books instead of "+100") never
        # parsed as a float and silently returned ''. Matches parse_odds_val's EVEN->100 rule.
        o = 100.0
    else:
        try:
            o = float(odds_s.replace('+', '').replace('−', '-'))
        except (TypeError, ValueError):
            return ''
    if o == 0:
        return ''
    p = 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)
    # 2026-08-28 (owner request): "to the decimal, ie (51.5) if necessary" -- rounding to a
    # whole number was losing real precision the exchanges themselves show (a book quoting
    # 51.5c/51.5% isn't the same price as 51 or 52). One decimal place, but only shown when
    # it's not a whole number, so a clean 52% still prints as "52%" not "52.0%".
    pct = round(p * 100, 1)
    pct_s = f"{pct:g}"
    return f"({pct_s}¢)" if is_cents else f"({pct_s}%)"

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
    if "total" in m or "over/under" in m: return "Total"
    if "to score" in s or re.search(r'\d+\+', s): return "Player Prop"
    if "over" in s or "under" in s: return "Total"
    return "Moneyline"


def extract_total_subtype(market, league=''):
    """Return a human-readable sub-category for Total bets."""
    m = str(market).lower()
    league = str(league).upper()

    if re.search(r'(1st|first)\s*half', m) or '1h total' in m: return '1st Half'
    if re.search(r'(2nd|second)\s*half', m) or '2h total' in m: return '2nd Half'
    if re.search(r'(1st|first)\s*5', m) or 'f5' in m: return '1st 5 Innings'
    if re.search(r'(1st|first)\s*inning', m): return '1st Inning'
    if 'team total' in m: return 'Team Total'
    if league == 'MLB' and re.search(r'total runs\s*-\s*\w', m): return 'Team Total'
    if re.search(r'(1st|first)\s*quarter|1q\b', m):  return '1st Quarter'
    if re.search(r'(2nd|second)\s*quarter|2q\b', m): return '2nd Quarter'
    if re.search(r'(3rd|third)\s*quarter|3q\b', m):  return '3rd Quarter'
    if re.search(r'(4th|fourth)\s*quarter|4q\b', m): return '4th Quarter'
    if re.search(r'(1st|first)\s*period|1p\b', m):   return '1st Period'
    if re.search(r'(2nd|second)\s*period|2p\b', m):  return '2nd Period'
    if re.search(r'(3rd|third)\s*period|3p\b', m):   return '3rd Period'

    return 'Full Game'

def get_bet_side(selection):
    s = str(selection).lower()
    if re.search(r'\bover\b', s):  return "Over"
    if re.search(r'\bunder\b', s): return "Under"
    return "Other"


# ─────────────────────────────────────────────────────────────
# STABLE MARKET-TYPE EDGES (2026-08-19, owner request)
# ─────────────────────────────────────────────────────────────
# One shared rule table, checked by BOTH dm_worker.py (owner DM tag) and
# post_grader.py (next-day morning report), so the two never drift apart.
# Every rule here passed signal_review.py's stability bar (same sign, within 10 ROI
# points, both halves of the data) at the prop-type / market-type granularity, scoped
# -200/+200 and 9am-9pm ET, smash>=50 (the same precondition every one of these was
# validated under -- see the 2026-08-19 session's per-prop/per-market breakdown).
# `positive=False` rows are STABLE-NEGATIVE -- proven bad, worth a warning, not a bet.
def _implied_prob_american(o):
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    if o == 0:
        return None
    return abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)


def sharp_gap_frac(odds_val, sharp_odds):
    """Mean sharp implied prob - play implied prob, as a fraction (0.05 = 5pp gap)."""
    pi = _implied_prob_american(odds_val)
    if pi is None:
        return None
    nums = re.findall(r'-?\d+\.?\d*', str(sharp_odds))
    sis = [_implied_prob_american(float(n)) for n in nums]
    sis = [x for x in sis if x is not None]
    if not sis:
        return None
    return sum(sis) / len(sis) - pi


# (league, prop_category_or_None, market_type_or_None, test(features)->bool,
#  label, roi_str, n, positive)
#
# 2026-08-20: fully rebuilt. The original 26-rule table (built 2026-08-19) was derived
# from bets.timestamp, which was mislabeled at the source -- tracker.py stamped rows
# with naive local (EDT) time that got silently stored as if it were UTC, so every
# "convert to ET" hour-of-day computation double-shifted by 4-5h and scoped the WRONG
# bets into/out of the 9am-9pm window used to validate these rules. Fixed at the
# source; re-ran the full battery against corrected timestamps with the same STABLE
# bar used everywhere else in this project (same sign both halves AND within 10 ROI
# points, n>=20 with >=5 per half). Of the original 26: 11 remained genuinely STABLE
# (kept below, 2 with their sign flipped -- Rebounds Under and Assists TWROI>0 turned
# out to be stable-NEGATIVE, not positive). The other 15 dropped to UNSTABLE/thin under
# corrected scoping and were removed rather than kept on stale numbers.
STABLE_MARKET_EDGES = [
    # ── MLB Pitcher Props ── (all require smash>=50, their original shared precondition)
    ('MLB', 'Pitcher Strikeouts', None, lambda f: f['smash_ok'] and f['side'] == 'Under',
     'Pitcher Ks Under', '+10.9%', 728, True),
    ('MLB', 'Pitcher Strikeouts', None, lambda f: f['smash_ok'] and f['side'] == 'Over',
     'Pitcher Ks Over', '+1.8%', 517, True),
    ('MLB', 'Pitcher Strikeouts', None, lambda f: f['smash_ok'] and f['bk_twroi'] is not None and f['bk_twroi'] < 0,
     'Pitcher Ks Book-TWROI<0', '+4.3%', 309, True),
    ('MLB', 'Pitcher Strikeouts', None, lambda f: f['smash_ok'] and f['twroi'] is not None and f['twroi'] > 0,
     'Pitcher Ks TWROI>0', '+8.1%', 975, True),
    ('MLB', 'Pitcher Walks Allowed', None, lambda f: f['smash_ok'] and f['side'] == 'Over',
     'Pitcher Walks Allowed Over', '+9.6%', 104, True),

    # ── WNBA Props ──
    ('WNBA', 'Rebounds', None, lambda f: f['smash_ok'] and f['side'] == 'Under',
     'Rebounds Under', '-4.9%', 100, False),
    ('WNBA', 'Assists', None, lambda f: f['smash_ok'] and f['side'] == 'Under',
     'Assists Under', '+7.0%', 113, True),
    ('WNBA', 'Assists', None, lambda f: f['smash_ok'] and f['twroi'] is not None and f['twroi'] > 0,
     'Assists TWROI>0', '-5.3%', 176, False),

    # ── MLB Mainlines ──
    ('MLB', None, 'Total', lambda f: f['smash_ok'] and f['twroi'] is not None and f['twroi'] > 0,
     'MLB Total TWROI>0', '+16.9%', 258, True),

    # ── WNBA Mainlines ──
    ('WNBA', None, 'Total', lambda f: f['smash_ok'] and f['twroi'] is not None and f['twroi'] > 0,
     'WNBA Total TWROI>0', '+16.3%', 119, True),

    # ── Segment markets (2026-08-20, owner report) -- 1st Inning/1st Half/etc. tracked as
    # their own category now, not folded into full-game Total/Spread (see market_type
    # comment in matched_stable_market_edges()). Full stability sweep across every segment
    # found only these two genuinely STABLE; everything else (MLB 1st 5 Innings, NBA/WNBA
    # 1st Half pooled, MLB 1st Inning pooled) was UNSTABLE despite some big-looking numbers.
    ('WNBA', None, 'Total-1st Half', lambda f: f['smash_ok'] and f['side'] == 'Over',
     'WNBA 1st Half Total Over', '+7.0%', 65, True),
    ('MLB', None, 'Total-1st Inning', lambda f: f['smash_ok'] and f['side'] == 'Over',
     'MLB 1st Inning Total Over', '-13.1%', 79, False),

    # ── Tennis Mainlines ── 2026-08-26 (owner request): re-derived from bets.ts_et (the
    # corrected timing column -- see BETS_TS_BUG_CUTOVER) after the owner's earlier "Tennis
    # ML 12pm-5pm" finding turned out to be a timing-bug artifact (corrected: n=753,
    # -1.81% ROI, unstable). Full ET-hour sweep on the corrected data found this window
    # instead: n=370, +12.63% ROI, STABLE (H1 +13.25%/H2 +12.10%, nearly identical). No
    # smash filter -- validated on the unfiltered population; smash>=50 shrinks it to n=130
    # and it stops being stable (H1 -3.32%/H2 +4.48%), so deliberately NOT gated on smash_ok
    # the way the rules above are.
    ('Tennis', None, 'Moneyline', lambda f: f['hour_et'] is not None and 18 <= f['hour_et'] < 22,
     'Tennis ML 6-10pm ET', '+12.63%', 370, True),
]
# 2026-08-20: "MLB Total Book-TWROI<0" removed. It read STABLE +11.2% (n=315) the day this
# table was first built, but that pool was contaminated -- categorize_bet() has no period
# awareness, so "1st Inning Total Runs", "1st Half Total Points", "1st 5 Innings" segment
# markets were silently mixed in with full-game Total. Re-run with segment markets properly
# excluded (see matched_stable_market_edges()'s is_segment check below): clean data is only
# +3.5% (n=159) and UNSTABLE. The other two Total rules survived the same re-check (and got
# updated to their clean numbers) -- this was specifically a Book-TWROI<0 problem, not every
# MLB Total rule.


def matched_stable_market_edges(league, market, selection, odds_val, sharp_odds,
                                 smash_score, catboost_score, twroi, bk_twroi, hour_et=None):
    """Every STABLE_MARKET_EDGES rule this bet matches, as a list of dicts:
    {'label','roi','n','positive'}. Most rules require smash>=50 -- that was originally a
    blanket function-level gate (their shared validation precondition at derivation time),
    but 2026-08-26 that got moved into a per-rule `smash_ok` feature flag instead, since a
    new rule (Tennis ML 6-10pm ET) was validated WITHOUT a smash filter and the blanket gate
    would have silently shrunk it to a much smaller, unstable subset (n=370 -> n=130,
    stability flips). Every pre-existing rule's test() explicitly ANDs f['smash_ok'] to
    preserve its exact original behavior; only rules that opt in by referencing it are
    smash-gated."""
    try:
        smash_ok = smash_score is not None and float(smash_score) >= 50
    except (TypeError, ValueError):
        smash_ok = False
    cat = categorize_bet(market, selection)
    prop_cat = extract_prop_category(market) if cat == 'Player Prop' else None
    # 2026-08-20 (owner report): categorize_bet() returns plain 'Total'/'Point Spread' for
    # ANY market containing those words, with no awareness of period -- "1st Inning Total
    # Runs", "1st Half Total Points", "1st 5 Innings Run Line" were all silently getting
    # counted as full-game MLB/WNBA Total or Spread, alongside genuinely different markets
    # with their own (often much less stable -- confirmed in the PP EV stability sweep the
    # same day) economics. extract_total_subtype()'s regex isn't actually Total-specific
    # despite its docstring/name -- it detects the period pattern in the market string
    # regardless of bet type, so it works for Spread segments too.
    # Segment bets get their own market_type string ("Total-1st Inning", "Spread-1st Half",
    # etc.) instead of the plain 'Total'/'Spread' full-game bets use -- so a segment bet can
    # NEVER accidentally match a full-game rule (different string), and a dedicated
    # segment-specific rule (see STABLE_MARKET_EDGES below) can target it precisely.
    _base_mt = 'Spread' if cat == 'Point Spread' else cat
    if cat == 'Player Prop':
        market_type = None
    elif cat in ('Total', 'Point Spread'):
        _subtype = extract_total_subtype(market, league)
        market_type = _base_mt if _subtype == 'Full Game' else f'{_base_mt}-{_subtype}'
    else:
        market_type = _base_mt
    features = {
        'side': get_bet_side(selection),
        'twroi': twroi, 'bk_twroi': bk_twroi, 'catboost': catboost_score,
        'sharp_gap': sharp_gap_frac(odds_val, sharp_odds),
        'smash_ok': smash_ok, 'hour_et': hour_et,
    }
    out = []
    for lg, pc, mt, test, label, roi, n, positive in STABLE_MARKET_EDGES:
        if lg != league:
            continue
        if pc is not None and pc != prop_cat:
            continue
        if mt is not None and mt != market_type:
            continue
        try:
            if not test(features):
                continue
        except Exception:
            continue
        out.append({'label': label, 'roi': roi, 'n': n, 'positive': positive})
    return out

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
    # Fanatics demotion logic removed

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
                'prime_time': False, 'is_fanatics': False,
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
        h = bets_ts_hour_et(bet_data.get('timestamp', ''))
        prime_time = h is not None and h in PRIME_HOURS
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
        'is_fanatics':      False, # Disabled
    }

    if bad_odds and consensus < CONSENSUS_THRESHOLD and not is_prop_under:
        return 'WATCH', flags

    if consensus >= CONSENSUS_THRESHOLD and good_odds and is_prop_under:
        return 'BRONZE', flags
    if consensus >= CONSENSUS_THRESHOLD and good_odds and good_liq:
        return 'BRONZE', flags
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

def build_bronze_description(flags):
    consensus     = flags.get('consensus', 0)
    is_prop_under = flags.get('is_prop_under', False)
    good_liq      = flags.get('good_liq', False)

    if consensus >= CONSENSUS_THRESHOLD and is_prop_under:
        r = _SIGNAL_ROI.get('bronze_3book_roi', 14.5)
        return f"3+ sharp books agree on a player prop Under — our highest-conviction combo. {r:+.1f}% ROI historically."
    if consensus >= CONSENSUS_THRESHOLD and good_liq:
        return "3+ sharp books agree and liquidity is in the sweet spot. 13.5% ROI historically."
    if consensus >= CONSENSUS_THRESHOLD:
        return "3+ sharp books agree on this line. High consensus is our strongest signal — 14–15% ROI historically."
    if is_prop_under and good_liq:
        r = _SIGNAL_ROI.get('bronze_under_roi', 13.0)
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
        reasons.append("odds are in the profitable range (−150 to +999)")
    elif flags.get('good_odds_under') and flags.get('is_prop_under'):
        reasons.append("odds are in the profitable prop Under range (−250 to +999)")

    base = build_bronze_description(flags) if tier == 'BRONZE' else _get_tier_description(tier)
    why  = ("Flagged because: " + ", ".join(reasons) + ".") if reasons else ""

    notes = []
    if is_low_confidence:               notes.append("⚠️ Low-confidence sharp source (e.g. Polymarket) — verify before betting.")
    if is_suppressed and signal_reason: notes.append(f"🔶 Note: {signal_reason}.")
    if flags.get('bad_odds'):           notes.append("⚠️ Odds exceed +1000 — historically high variance.")
    
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
    if flags.get('is_standard_plus'):    parts.append('⭐ STD+')
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
    df['prop_cat']      = df.apply(
        lambda r: (
            extract_prop_category(r.get('market', ''))   if r['bet_type'] == 'Player Prop'
            else extract_total_subtype(r.get('market', ''), r.get('league', '')) if r['bet_type'] == 'Total'
            else ''
        ), axis=1
    )
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
        if bt == 'Total':       return f"{side} {league} {pc}" if pc else f"{side} {league} Full Game"
        return f"{league} {bt}"
    df['combo'] = df.apply(_combo, axis=1)

    return df
