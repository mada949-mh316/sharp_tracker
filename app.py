import pandas as pd
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
from datetime import datetime, timedelta
from fuzzywuzzy import process, fuzz

# --- CONFIGURATION ---
CSV_PATH = "data/bets.csv"
SHEET_NAME = "Smart Money Bets"
CREDS_FILE = "creds.json"
MAX_LOOKAHEAD_DAYS = 5
DEBUG_MODE = True

# 🔑 RAPIDAPI KEY (Get free key from https://rapidapi.com/jjrm365-kIFr3Nx_odV/api/tennis-api-atp-wta-itf)
# REPLACE THIS WITH YOUR ACTUAL KEY
RAPID_API_KEY = "1bf7a5d882msh85ac03ceafed66bp1ee0f6jsn724aa12348d2" 

# API MAP
LEAGUE_MAP = {
    "NBA":   {"sport": "basketball", "league": "nba", "group": None},
    "NFL":   {"sport": "football", "league": "nfl", "group": None},
    "NHL":   {"sport": "hockey", "league": "nhl", "group": None},
    "NCAAF": {"sport": "football", "league": "college-football", "group": "80"},
    "NCAAB": {"sport": "basketball", "league": "mens-college-basketball", "group": "50"},
    "Tennis": {"sport": "tennis", "league": "atp", "group": None},
}

# STATS TRANSLATION
STAT_MAP = {
    "Points": ["pts", "points", "score"],
    "Rebounds": ["reb", "rebounds", "totalrebounds"],
    "Assists": ["ast", "assists"],
    "Threes Made": ["3pt", "threepointfieldgoalsmade", "3pm", "3ptm"],
    "Blocks": ["blk", "blocks"],
    "Steals": ["stl", "steals"],
    "Turnovers": ["to", "turnovers"],
    "Passing Yards": ["passingyards", "yds"],
    "Rushing Yards": ["rushingyards", "yds"],
    "Receiving Yards": ["receivingyards", "yds"],
    "Passing Touchdowns": ["passingtouchdowns", "td"],
    "Rushing Touchdowns": ["rushingtouchdowns", "td"],
    "Receiving Touchdowns": ["receivingtouchdowns", "td"],
    "Receptions": ["receptions", "rec"],
    "Passing Completions": ["completions", "completedpasses"],
    "Goals": ["goals", "g"],
    "Saves": ["saves"],
    "Shots": ["shots", "sog", "shotsongoal"],
    "Blocked Shots": ["blockedshots", "blk"]
}

# --- GOOGLE SHEETS SYNC ---
def batch_sync_to_cloud(df):
    print(f"☁️ Syncing to '{SHEET_NAME}'...")
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).sheet1
        sheet.clear()
        set_with_dataframe(sheet, df)
        print(f"✅ Success! Uploaded {len(df)} rows.")
    except Exception as e:
        print(f"⚠️ Sync Error: {e}")

# --- LOCAL DATABASE ---
def load_db():
    try:
        return pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        return pd.DataFrame()

def save_db(df):
    df.to_csv(CSV_PATH, index=False)
    print("💾 Local Database saved.")

# --- HELPER: PARSE TENNIS SCORES ---
def parse_tennis_score_string(score_str):
    """
    Parses '6-4, 3-6, 7-6(4)' into game/set counts
    Returns: (p1_sets, p2_sets, p1_games, p2_games)
    """
    p1_sets, p2_sets = 0, 0
    p1_games, p2_games = 0, 0
    
    # Remove things like "RET" (Retired) or spaces
    clean_score = score_str.replace("RET", "").strip()
    sets = clean_score.split(',')
    
    for s in sets:
        s = s.strip()
        if not s: continue
        # Remove tiebreak info like (4)
        if '(' in s: s = s.split('(')[0]
        
        try:
            parts = s.split('-')
            if len(parts) == 2:
                g1 = int(parts[0])
                g2 = int(parts[1])
                p1_games += g1
                p2_games += g2
                if g1 > g2: p1_sets += 1
                elif g2 > g1: p2_sets += 1
        except:
            pass # Skip weird formatting
            
    return p1_sets, p2_sets, p1_games, p2_games

# --- API 1: RAPIDAPI (BACKUP FOR TENNIS) ---
def check_rapidapi_backup(matchup, bet_timestamp):
    if not RAPID_API_KEY or "YOUR_KEY" in RAPID_API_KEY:
        print("   [BACKUP] Skipped (No API Key)")
        return None
        
    print(f"   🌍 Checking RapidAPI (Matchstat) for: {matchup}...")
    
    # Extract names
    names = matchup.replace(" vs ", "|").replace(" @ ", "|").split("|")
    if len(names) < 2: return None
    p1_name = names[0].strip()
    p2_name = names[1].strip()
    
    url = f"https://tennis-api-atp-wta-itf.p.rapidapi.com/api/v1/h2h/{p1_name}/{p2_name}"
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "tennis-api-atp-wta-itf.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if 'matches' in data:
            # Sort by date, newest first
            # Format usually: "2025-01-02T00:00:00"
            matches = data['matches']
            
            # Find the match closest to our bet date
            bet_date = pd.to_datetime(bet_timestamp).date()
            
            for m in matches:
                m_date_str = m.get('date', '').split('T')[0]
                m_date = datetime.strptime(m_date_str, "%Y-%m-%d").date()
                
                # Check if match is within 2 days of bet (Timezones can be wild)
                delta = abs((m_date - bet_date).days)
                
                if delta <= 2:
                    # FOUND IT!
                    result_str = m.get('result', '') # e.g. "6-4, 6-2"
                    s1, s2, g1, g2 = parse_tennis_score_string(result_str)
                    
                    print(f"   ✅ Found in RapidAPI: {result_str}")
                    
                    return {
                        "id": f"rapid_{m.get('id')}",
                        "date": m_date_str,
                        "home": {"name": p1_name, "score": s1, "games": g1},
                        "away": {"name": p2_name, "score": s2, "games": g2}
                    }
    except Exception as e:
        print(f"   [BACKUP ERROR] {e}")
        
    return None

# --- API 2: ESPN (PRIMARY) ---
def get_scoreboard(league_key, date_str):
    if league_key == "Tennis":
        events = []
        for l in ['atp', 'wta']:
            url = f"http://site.api.espn.com/apis/site/v2/sports/tennis/{l}/scoreboard"
            try:
                resp = requests.get(url, params={'dates': date_str, 'limit': 1000})
                if resp.status_code == 200:
                    events.extend(resp.json().get('events', []))
            except: pass
        return events

    info = LEAGUE_MAP.get(league_key)
    if not info: return []
    sport = info['sport']
    if league_key == 'NHL': sport = 'hockey'
    
    url = f"http://site.api.espn.com/apis/site/v2/sports/{sport}/{info['league']}/scoreboard"
    params = {'dates': date_str, 'limit': 1000}
    if info['group']: params['groups'] = info['group']
    try:
        return requests.get(url, params=params).json().get('events', [])
    except:
        return []

def get_game_summary(league_key, game_id):
    info = LEAGUE_MAP.get(league_key)
    if league_key == 'Tennis':
        url = f"http://site.api.espn.com/apis/site/v2/sports/tennis/atp/summary" 
    else:
        if not info: return None
        sport = info['sport']
        if league_key == 'NHL': sport = 'hockey'
        url = f"http://site.api.espn.com/apis/site/v2/sports/{sport}/{info['league']}/summary"
        
    try:
        return requests.get(url, params={'event': game_id}).json()
    except:
        return None

# --- PARSING ---
def parse_player_stats(summary_json):
    player_data = {}
    try:
        boxscore = summary_json.get('boxscore', {})
        teams = boxscore.get('players', [])
        for team in teams:
            statistics = team.get('statistics', [])
            for category in statistics:
                keys = [k.lower() for k in category.get('keys', [])]
                for athlete_entry in category.get('athletes', []):
                    athlete = athlete_entry.get('athlete', {})
                    name = athlete.get('displayName', "Unknown")
                    stats_list = athlete_entry.get('stats', [])
                    if name not in player_data: player_data[name] = {}
                    for i, stat_val in enumerate(stats_list):
                        if i < len(keys):
                            try:
                                s_val = str(stat_val)
                                if "-" in s_val:
                                    made = float(s_val.split("-")[0])
                                    player_data[name][keys[i]] = made
                                else:
                                    player_data[name][keys[i]] = float(stat_val)
                            except: continue
        return player_data
    except: return {}

def extract_game_info(event):
    try:
        if 'competitions' not in event or not event['competitions']: return None
        
        status = event['status']['type']['state']
        if status not in ['post', 'final']: return None 
        
        comp = event['competitions'][0]
        if 'competitors' not in comp: return None
        
        c1_data, c2_data = None, None
        
        def parse_competitor(c):
            linescores = c.get('linescores', [])
            total_games = 0 
            score_1h = 0
            if linescores:
                score_1h = sum([x.get('value', 0) for x in linescores[:2]])
                total_games = sum([x.get('value', 0) for x in linescores])
            
            return {
                "name": c['team']['displayName'],
                "short": c['team'].get('shortDisplayName', ""),
                "abbr": c['team'].get('abbreviation', ""),
                "score": int(c['score']),
                "games": total_games,
                "score_1h": score_1h
            }

        for c in comp['competitors']:
            side = c.get('homeAway', 'unknown')
            if side == 'home': c1_data = parse_competitor(c)
            elif side == 'away': c2_data = parse_competitor(c)

        if not c1_data or not c2_data:
            if len(comp['competitors']) >= 2:
                c1_data = parse_competitor(comp['competitors'][0])
                c2_data = parse_competitor(comp['competitors'][1])

        if not c1_data or not c2_data: return None

        return {
            "id": event['id'],
            "date": event['date'],
            "home": c1_data,
            "away": c2_data
        }
    except: return None

# --- GRADING LOGIC ---
def grade_bet(row, games_cache, boxscore_cache):
    try:
        ts = pd.to_datetime(row['timestamp'])
    except: return None, None

    found_game = None

    # 1. SEARCH ESPN API (Primary)
    for day_offset in range(MAX_LOOKAHEAD_DAYS + 1):
        check_date = (ts + timedelta(days=day_offset)).strftime('%Y%m%d')
        if check_date not in games_cache.get(row['league'], {}):
            events = get_scoreboard(row['league'], check_date)
            if row['league'] not in games_cache: games_cache[row['league']] = {}
            daily = []
            for e in events:
                info = extract_game_info(e)
                if info: daily.append(info)
            games_cache[row['league']][check_date] = daily
        
        daily_games = games_cache[row['league']][check_date]
        tracker_teams = row['matchup'].replace(" vs ", "|").replace(" @ ", "|").split("|")
        if len(tracker_teams) < 2: continue
        t1_raw, t2_raw = tracker_teams[0].strip(), tracker_teams[1].strip()

        for game in daily_games:
            game_names = [
                game['home']['name'], game['home']['short'], game['home']['abbr'],
                game['away']['name'], game['away']['short'], game['away']['abbr']
            ]
            t1_found = any(fuzz.partial_ratio(t1_raw.lower(), n.lower()) > 80 for n in game_names)
            t2_found = any(fuzz.partial_ratio(t2_raw.lower(), n.lower()) > 80 for n in game_names)
            if t1_found and t2_found:
                found_game = game
                break
        if found_game: break
    
    # 2. SEARCH RAPID API (Backup for Tennis)
    if not found_game and row['league'] == 'Tennis':
        found_game = check_rapidapi_backup(row['matchup'], ts)

    if not found_game: return None, None

    result = "Pending"
    
    try:
        # PLAYER PROP
        if "Player" in row['market']:
            # RapidAPI doesn't support Player Props in this basic script, skipping
            if "rapid_" in str(found_game['id']):
                print(f"   ⚠️ RapidAPI found game but cannot grade Player Prop: {row['market']}")
                return None, None

            game_id = found_game['id']
            if game_id not in boxscore_cache:
                summary = get_game_summary(row['league'], game_id)
                boxscore_cache[game_id] = parse_player_stats(summary)
            player_stats_db = boxscore_cache[game_id]
            target_name = row['play_selection'].split(" Over ")[0].split(" Under ")[0].strip()
            
            best_match = process.extractOne(target_name, player_stats_db.keys())
            if not best_match or best_match[1] < 85: return None, None
            player_name_found = best_match[0]
            stats = player_stats_db[player_name_found]
            
            market_lower = row['market'].lower()
            market_type = row['market'].replace("Player ", "").strip()
            actual_val = 0.0
            
            def get_stat_val(category):
                for k in STAT_MAP.get(category, []):
                    if k in stats: return stats[k]
                return 0.0

            if "pra" in market_lower:
                p, r, a = get_stat_val("Points"), get_stat_val("Rebounds"), get_stat_val("Assists")
                actual_val = p + r + a
            elif "+" in market_lower:
                if "points" in market_lower or "pts" in market_lower: actual_val += get_stat_val("Points")
                if "rebounds" in market_lower or "rebs" in market_lower: actual_val += get_stat_val("Rebounds")
                if "assists" in market_lower or "ast" in market_lower: actual_val += get_stat_val("Assists")
            else:
                actual_val = get_stat_val(market_type)

            line = float(row['play_selection'].split()[-1])
            if "Over" in row['play_selection']: result = "Won" if actual_val > line else "Lost"
            elif "Under" in row['play_selection']: result = "Won" if actual_val < line else "Lost"
            if actual_val == line: result = "Push"

        # TEAM/MATCH LOGIC
        else:
            sel = row['play_selection']
            
            home_names = [found_game['home']['name']]
            # Add ESPN short names if available (RapidAPI object structure is simpler)
            if 'short' in found_game['home']: home_names.append(found_game['home']['short'])

            is_home = any(fuzz.partial_ratio(sel, n) > 80 for n in home_names)
            
            use_games = (row['league'] == 'Tennis' and ("Spread" in row['market'] or "Total" in row['market']))
            
            if use_games:
                score_us = found_game['home']['games'] if is_home else found_game['away']['games']
                score_them = found_game['away']['games'] if is_home else found_game['home']['games']
            else:
                score_us = found_game['home']['score'] if is_home else found_game['away']['score']
                score_them = found_game['away']['score'] if is_home else found_game['home']['score']
            
            if "Total" in row['market']:
                line = float(sel.split()[-1])
                if row['league'] == 'Tennis':
                    total = found_game['home']['games'] + found_game['away']['games']
                else:
                    total = found_game['home']['score'] + found_game['away']['score']
                    
                if "Over" in sel: result = "Won" if total > line else "Lost"
                elif "Under" in sel: result = "Won" if total < line else "Lost"
                if total == line: result = "Push"

            elif ("Spread" in row['market'] or "Moneyline" in row['market']):
                line = float(sel.split()[-1]) if "Spread" in row['market'] else 0.0
                if score_us + line > score_them: result = "Won"
                elif score_us + line < score_them: result = "Lost"
                else: result = "Push"

    except: return None, None

    profit = 0.0
    unit = 100
    if result == "Won":
        odds = int(row['play_odds'])
        profit = unit * (odds/100) if odds > 0 else unit * (100/abs(odds))
    elif result == "Lost":
        profit = -unit

    return result, round(profit, 2)

# --- MAIN ---
def main():
    print("🎓 Grader running (Hybrid: ESPN + RapidAPI Backup)...")
    df = load_db()
    
    if 'Pending' in df['status'].values:
        df.loc[df['status'] == 'Pending', 'status'] = 'Open'
        
    open_mask = df['status'] == 'Open'
    games_cache = {}
    boxscore_cache = {}
    graded_count = 0
    
    print(f"Checking {open_mask.sum()} open bets...")
    for index, row in df[open_mask].iterrows():
        result, profit = grade_bet(row, games_cache, boxscore_cache)
        if result and result != "Pending":
            print(f"✅ {row['play_selection']}: {result} (${profit})")
            df.at[index, 'status'] = result
            df.at[index, 'profit'] = profit
            df.at[index, 'result'] = result
            graded_count += 1
            
    if graded_count > 0: save_db(df)
    batch_sync_to_cloud(df)

if __name__ == "__main__":
    main()