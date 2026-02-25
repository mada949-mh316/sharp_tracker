"""
db_utils.py
─────────────────────────────────────────────────────────────
Shared database utilities for Smart Money Tracker.
Used by both tracker_v9.py (writes) and app.py (reads).

Set DATABASE_URL in your environment:
  export DATABASE_URL="postgresql://user:password@host:5432/dbname"

For Streamlit Cloud, add it to Streamlit Secrets:
  [database]
  url = "postgresql://..."
"""

import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
from datetime import datetime, timedelta
from contextlib import contextmanager
import streamlit as st

# ─────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────

_pool = None   # module-level connection pool

def _get_database_url():
    """
    Try to get DATABASE_URL from:
    1. Environment variable (tracker, local dev)
    2. Streamlit secrets (dashboard on Streamlit Cloud)
    """
    url = os.environ.get("DATABASE_URL", "")
    if url:
        return url

    # Try Streamlit secrets
    try:
        return st.secrets["database"]["url"]
    except Exception:
        pass

    try:
        return st.secrets["DATABASE_URL"]
    except Exception:
        pass

    raise RuntimeError(
        "DATABASE_URL not found.\n"
        "Set it as an environment variable or add to Streamlit secrets:\n"
        "  [database]\n"
        "  url = 'postgresql://...'"
    )


def get_pool():
    """Get or create a threaded connection pool (max 5 connections)."""
    global _pool
    if _pool is None:
        url = _get_database_url()
        _pool = ThreadedConnectionPool(1, 5, url, sslmode='require')
    return _pool


@contextmanager
def get_conn():
    """Context manager that checks out a connection from the pool."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ─────────────────────────────────────────────────────────────
# WRITE — used by tracker
# ─────────────────────────────────────────────────────────────

INSERT_SQL = """
    INSERT INTO bets (
        id, timestamp, league, matchup, market, play_selection,
        play_odds, play_book, sharp_odds, sharp_book,
        liquidity, wager, result, profit, status
    )
    VALUES %s
    ON CONFLICT ON CONSTRAINT bets_natural_key DO NOTHING
"""

def insert_bet(bet_data: dict):
    """Insert a single bet. Safe to call multiple times — ignores duplicates."""
    insert_bets([bet_data])


def insert_bets(bets_list: list):
    """Bulk insert a list of bet dicts. Ignores duplicates."""
    if not bets_list:
        return

    rows = []
    for b in bets_list:
        ts = b.get('timestamp')
        if isinstance(ts, str):
            try:
                ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                ts = pd.to_datetime(ts, errors='coerce')

        rows.append((
            str(b.get('id', '')),
            ts,
            str(b.get('league', '')),
            str(b.get('matchup', '')),
            str(b.get('market', '')),
            str(b.get('play_selection', '')),
            str(b.get('play_odds', '')),
            str(b.get('play_book', '')),
            str(b.get('sharp_odds', '')),
            str(b.get('sharp_book', '')),
            float(b.get('liquidity', 0) or 0),
            float(b.get('wager', 0) or 0),
            str(b.get('result', 'Pending')),
            float(b.get('profit', 0) or 0),
            str(b.get('status', 'Open')),
        ))

    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, INSERT_SQL, rows, page_size=200)

    print(f"☁️  DB: inserted up to {len(rows)} bets (dupes skipped).")


def update_bet_status(matchup: str, play_selection: str, timestamp,
                      status: str, profit: float, result: str):
    """Update a bet's result after settlement."""
    sql = """
        UPDATE bets
        SET status = %s, profit = %s, result = %s
        WHERE matchup = %s AND play_selection = %s AND timestamp = %s
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (status, profit, result, matchup, play_selection, timestamp))


# ─────────────────────────────────────────────────────────────
# READ — used by dashboard
# ─────────────────────────────────────────────────────────────

def load_bets(
    days_back: int = None,
    leagues: list = None,
    status: list = None,
    limit: int = None,
) -> pd.DataFrame:
    """
    Load bets from the database into a DataFrame.

    Args:
        days_back:  Only load bets from the last N days.
                    None = load everything.
        leagues:    Filter to specific leagues e.g. ['NBA','NFL'].
                    None = all leagues.
        status:     Filter to specific statuses e.g. ['Won','Lost','Push'].
                    None = all statuses including Open/Pending.
        limit:      Cap the number of rows returned (newest first).
                    None = no limit.

    Returns a DataFrame with the same columns as the old CSV,
    so the dashboard code needs zero changes.
    """
    conditions = ["1=1"]
    params     = []

    if days_back is not None:
        conditions.append("timestamp >= NOW() - INTERVAL '%s days'")
        params.append(days_back)

    if leagues:
        conditions.append("league = ANY(%s)")
        params.append(leagues)

    if status:
        conditions.append("status = ANY(%s)")
        params.append(status)

    where = " AND ".join(conditions)
    order = "ORDER BY timestamp DESC"
    lim   = f"LIMIT {int(limit)}" if limit else ""

    sql = f"""
        SELECT id, timestamp, league, matchup, market, play_selection,
               play_odds, play_book, sharp_odds, sharp_book,
               liquidity, wager, result, profit, status
        FROM bets
        WHERE {where}
        {order}
        {lim}
    """

    with get_conn() as conn:
        df = pd.read_sql(sql, conn, params=params)

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
    return df


def load_bets_date_range(start_date, end_date) -> pd.DataFrame:
    """Load bets between two dates (inclusive)."""
    sql = """
        SELECT id, timestamp, league, matchup, market, play_selection,
               play_odds, play_book, sharp_odds, sharp_book,
               liquidity, wager, result, profit, status
        FROM bets
        WHERE timestamp >= %s AND timestamp < %s
        ORDER BY timestamp DESC
    """
    with get_conn() as conn:
        df = pd.read_sql(sql, conn, params=[start_date, end_date + timedelta(days=1)])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
    return df


def get_date_range() -> tuple:
    """Return (min_date, max_date) of all bets in the database."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(timestamp)::date, MAX(timestamp)::date FROM bets")
            row = cur.fetchone()
    return row[0], row[1]


def count_bets() -> int:
    """Quick count of total rows."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM bets")
            return cur.fetchone()[0]
