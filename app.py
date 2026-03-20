import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import anthropic, requests, io, re, unicodedata

st.set_page_config(page_title="Club Fit Finder", page_icon="🏟️", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;600;700;800&family=Barlow+Condensed:wght@700;800&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Barlow',sans-serif;background:#07090f;color:#e2e8f0;}
.stApp{background:#07090f;}
section[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid #1e2d42;}
label,p,span,div,[data-testid="stWidgetLabel"] p,.stMarkdown p{color:#e2e8f0!important;}
.stSelectbox>div>div,.stMultiSelect>div>div{background:#111827!important;border-color:#1e2d42!important;}
input,textarea{background:#111827!important;color:#e2e8f0!important;border-color:#1e2d42!important;}
/* Multiselect & selectbox dropdown — black text, dark checkmarks */
[data-baseweb="popover"],[data-baseweb="popover"] *{color:#000000!important;}
[data-baseweb="popover"] ul li{background:#ffffff!important;}
[data-baseweb="popover"] ul li:hover{background:#dbeafe!important;}
[data-baseweb="menu"],[data-baseweb="menu"] *{color:#000000!important;background:#ffffff!important;}
[data-baseweb="menu"] [aria-selected="true"]{background:#dbeafe!important;}
[data-baseweb="menu"] svg{fill:#000000!important;color:#000000!important;}
/* Selected tags — blue background, white text */
[data-baseweb="tag"]{background:#2563eb!important;border:none!important;}
[data-baseweb="tag"] span,[data-baseweb="tag"] *{color:#ffffff!important;font-weight:600!important;}
/* Style pill tags in main area */
.pill{display:inline-block;background:#2563eb!important;border:1px solid #3b82f6!important;border-radius:5px;padding:2px 8px;font-size:.75rem;color:#ffffff!important;margin:2px;}
.stButton>button{background:#3b82f6!important;color:#fff!important;border:none!important;font-weight:700!important;border-radius:6px!important;}
.stButton>button:hover{opacity:.85!important;}
div[data-testid="stExpander"]{background:#111827;border-color:#1e2d42;border-radius:8px;}
.head{font-family:'Barlow Condensed',sans-serif;font-size:2.4rem;font-weight:800;color:#fff;border-bottom:2px solid #3b82f6;padding-bottom:8px;margin-bottom:20px;}
.pcard{background:#111827;border:1px solid #1e2d42;border-left:3px solid #3b82f6;border-radius:10px;padding:14px 18px;margin-bottom:16px;}
.pcard .nm{font-family:'Barlow Condensed',sans-serif;font-size:1.4rem;font-weight:800;color:#fff;}
.pcard .mt{color:#64748b;font-size:.82rem;margin-top:3px;}
.sec{font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;color:#3b82f6;letter-spacing:.08em;text-transform:uppercase;border-bottom:1px solid #1e2d42;padding-bottom:5px;margin:18px 0 10px 0;}
.acard{background:#0d1117;border:1px solid #1e2d42;border-left:3px solid #f59e0b;border-radius:8px;padding:12px 16px;margin-bottom:10px;}
.acard h4{font-family:'Barlow Condensed',sans-serif;font-size:.95rem;font-weight:700;color:#f59e0b;margin:0 0 6px 0;}
.acard p{color:#e2e8f0;font-size:.86rem;line-height:1.6;margin:0;}
.pill{display:inline-block;background:#1e2d42;border:1px solid #1e3a5f;border-radius:5px;padding:2px 8px;font-size:.75rem;color:#94a3b8;margin:2px;}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────
LEAGUE_STRENGTHS = {
    'England 1.':100,'Spain 1.':98,'Germany 1.':96,'Italy 1.':94,'France 1.':88,
    'Portugal 1.':82,'Netherlands 1.':80,'Belgium 1.':78,'Scotland 1.':70,
    'Turkey 1.':72,'Russia 1.':68,'Ukraine 1.':66,'Austria 1.':64,
    'Switzerland 1.':62,'Greece 1.':60,'Czech 1.':58,'Denmark 1.':56,
    'Sweden 1.':54,'Poland 1.':52,'Norway 1.':50,'Croatia 1.':49,
    'Serbia 1.':47,'Romania 1.':45,'Slovakia 1.':43,'Slovenia 1.':41,
    'Bulgaria 1.':39,'Hungary 1.':37,'Finland 1.':35,'Israel 1.':60,
    'Cyprus 1.':33,'England 2.':78,'England 3.':58,'England 4.':45,
    'Germany 2.':72,'Spain 2.':62,'Italy 2.':60,'France 2.':58,
    'Portugal 2.':52,'Netherlands 2.':50,'Scotland 2.':38,
    'Venezuela 1.':35,'Uruguay 1.':52,'Japan 1.':55,'Egypt 1.':40,
}

POS_MAP = {
    'GK':  ['GK'],
    'CB':  ['CB','LCB','RCB','WCB'],
    'FB':  ['LB','RB','LWB','RWB','WB'],
    'CM':  ['CM','DMF','CMF','AMF','DM','CAM','CDM'],
    'ATT': ['LW','RW','LWF','RWF','LAMF','RAMF'],
    'CF':  ['CF','ST','SS','LCF','RCF'],
}

CF_FEATURES = {
    'GK':['Exits per 90','Aerial duels per 90','Aerial duels won, %','Save rate, %',
          'Prevented goals per 90','Passes per 90','Accurate passes, %',
          'Long passes per 90','Accurate long passes, %'],
    'CB':['Defensive duels per 90','Defensive duels won, %','Aerial duels per 90',
          'Aerial duels won, %','Shots blocked per 90','PAdj Interceptions',
          'Dribbles per 90','Successful dribbles, %','Progressive runs per 90',
          'Accelerations per 90','Passes per 90','Accurate passes, %',
          'Forward passes per 90','Accurate forward passes, %','Long passes per 90',
          'Accurate long passes, %','Passes to final third per 90',
          'Accurate passes to final third, %','Progressive passes per 90',
          'Accurate progressive passes, %'],
    'FB':['Defensive duels per 90','Defensive duels won, %','Aerial duels per 90',
          'Aerial duels won, %','PAdj Interceptions','Non-penalty goals per 90',
          'xG per 90','Shots per 90','Dribbles per 90','Successful dribbles, %',
          'Offensive duels per 90','Offensive duels won, %','Touches in box per 90',
          'Progressive runs per 90','Accelerations per 90','Passes per 90',
          'Accurate passes, %','xA per 90','Smart passes per 90',
          'Passes to final third per 90','Accurate passes to final third, %',
          'Passes to penalty area per 90','Accurate passes to penalty area, %',
          'Deep completions per 90'],
    'CM':['Defensive duels per 90','Defensive duels won, %','Aerial duels per 90',
          'Aerial duels won, %','Shots blocked per 90','PAdj Interceptions',
          'Non-penalty goals per 90','xG per 90','Shots per 90','Dribbles per 90',
          'Successful dribbles, %','Offensive duels per 90','Offensive duels won, %',
          'Touches in box per 90','Progressive runs per 90','Accelerations per 90',
          'Passes per 90','Accurate passes, %','Forward passes per 90',
          'Accurate forward passes, %','Long passes per 90','Accurate long passes, %',
          'xA per 90','Smart passes per 90','Key passes per 90',
          'Passes to final third per 90','Accurate passes to final third, %',
          'Passes to penalty area per 90','Accurate passes to penalty area, %',
          'Deep completions per 90','Progressive passes per 90'],
    'ATT':['Defensive duels per 90','Aerial duels per 90','Aerial duels won, %',
           'PAdj Interceptions','Non-penalty goals per 90','xG per 90',
           'Shots per 90','Shots on target, %','Crosses per 90','Accurate crosses, %',
           'Dribbles per 90','Successful dribbles, %','Offensive duels per 90',
           'Offensive duels won, %','Touches in box per 90','Progressive runs per 90',
           'Accelerations per 90','Passes per 90','Accurate passes, %',
           'Forward passes per 90','Accurate forward passes, %','Long passes per 90',
           'Accurate long passes, %','xA per 90','Smart passes per 90',
           'Key passes per 90','Passes to final third per 90',
           'Accurate passes to final third, %','Passes to penalty area per 90',
           'Accurate passes to penalty area, %','Deep completions per 90',
           'Progressive passes per 90'],
    'CF':['Defensive duels per 90','Aerial duels per 90','Aerial duels won, %',
          'PAdj Interceptions','Non-penalty goals per 90','xG per 90',
          'Shots per 90','Shots on target, %','Crosses per 90','Accurate crosses, %',
          'Dribbles per 90','Successful dribbles, %','Offensive duels per 90',
          'Offensive duels won, %','Touches in box per 90','Progressive runs per 90',
          'Accelerations per 90','Passes per 90','Accurate passes, %',
          'xA per 90','Smart passes per 90','Key passes per 90',
          'Passes to final third per 90','Passes to penalty area per 90',
          'Accurate passes to penalty area, %','Deep completions per 90',
          'Progressive passes per 90'],
}

DEFAULT_WEIGHTS = {
    'GK':{'Aerial duels won, %':3,'Passes per 90':2,'Accurate passes, %':2,
          'Long passes per 90':2,'Accurate long passes, %':2},
    'CB':{'Passes per 90':2,'Accurate passes, %':2,'Progressive passes per 90':2,
          'Defensive duels per 90':2,'Defensive duels won, %':2,'Dribbles per 90':2,
          'PAdj Interceptions':1,'Progressive runs per 90':2,
          'Aerial duels per 90':2,'Aerial duels won, %':3},
    'FB':{'Passes per 90':2,'Accurate passes, %':2,'Dribbles per 90':2,
          'Non-penalty goals per 90':2,'Shots per 90':2,'Successful dribbles, %':2,
          'Aerial duels won, %':2,'xA per 90':2,'xG per 90':2,'Touches in box per 90':2},
    'CM':{'Passes per 90':3,'Passes to penalty area per 90':2,'Dribbles per 90':2,
          'xA per 90':2,'Progressive passes per 90':3,'Defensive duels per 90':2,
          'Forward passes per 90':3,'PAdj Interceptions':2,
          'Aerial duels won, %':2,'Touches in box per 90':2},
    'ATT':{'Passes per 90':2,'Progressive runs per 90':2,'Progressive passes per 90':2,
           'Dribbles per 90':2,'xA per 90':2,'Touches in box per 90':2,
           'Accurate passes, %':2,'Aerial duels won, %':2,
           'Passes to penalty area per 90':2,'Defensive duels per 90':2},
    'CF':{'Passes per 90':2,'Accurate passes, %':2,'Dribbles per 90':3,
          'Non-penalty goals per 90':2,'Shots per 90':2,'Successful dribbles, %':2,
          'Aerial duels won, %':2,'xA per 90':2,'xG per 90':2,
          'Touches in box per 90':2,'Passes to final third per 90':2,
          'Passes to penalty area per 90':2},
}

STYLE_BLENDS = {
    "Possession":               {"Possession %":1.0,"Pass Accuracy %":0.8,"Passes p90":0.7},
    "Pressing":                 {"PPDA":-1.0,"Defensive Duels p90":0.6},
    "Direct":                   {"Long Passes p90":1.0,"Aerial Duels p90":0.7},
    "Passing Based":            {"Passes to Final Third p90":1.0,"Progressive Passes p90":0.9,"Pass Accuracy %":0.6},
    "High Attacking Territory": {"xG p90":1.0,"Touches in Box p90":0.9,"Shots p90":0.7},
    "Cross Heavy":              {"Crosses p90":1.0},
}

def detect_pos(pos_str):
    pos_str = str(pos_str)
    primary = pos_str.split(",")[0].strip().upper()
    for grp, tags in POS_MAP.items():
        if primary in tags: return grp
    for grp, tags in POS_MAP.items():
        if any(t in pos_str.upper() for t in tags): return grp
    return 'CM'

def fmt_mv(v):
    try:
        v = float(v)
        if v >= 1e6: return f"£{v/1e6:.1f}m"
        if v >= 1e3: return f"£{v/1e3:.0f}k"
        return f"£{v:.0f}"
    except: return "—"

def score_col(s):
    s = float(s)
    if s >= 78: return "#22c55e"
    if s >= 62: return "#86efac"
    if s >= 48: return "#f59e0b"
    return "#ef4444"

# ── LEAGUE PRESETS & REGION MAP (mirrors team_hq.py) ─────────────────
PRESET_LEAGUES = {
    "Top 5 Europe":    {"England 1.","Spain 1.","Germany 1.","Italy 1.","France 1."},
    "Top 20 Europe":   {
        "England 1.","Spain 1.","Germany 1.","Italy 1.","France 1.",
        "England 2.","Portugal 1.","Belgium 1.","Turkey 1.","Germany 2.",
        "Spain 2.","France 2.","Netherlands 1.","Austria 1.","Switzerland 1.",
        "Denmark 1.","Croatia 1.","Italy 2.","Czech 1.","Norway 1.",
    },
    "EFL (England 2–4)": {"England 2.","England 3.","England 4."},
}

COUNTRY_TO_REGION = {
    "England":"Europe","Spain":"Europe","Germany":"Europe","Italy":"Europe","France":"Europe",
    "Belgium":"Europe","Portugal":"Europe","Netherlands":"Europe","Croatia":"Europe",
    "Switzerland":"Europe","Norway":"Europe","Sweden":"Europe","Denmark":"Europe",
    "Czech":"Europe","Greece":"Europe","Austria":"Europe","Hungary":"Europe",
    "Romania":"Europe","Scotland":"Europe","Slovenia":"Europe","Slovakia":"Europe",
    "Ukraine":"Europe","Bulgaria":"Europe","Serbia":"Europe","Poland":"Europe",
    "Russia":"Europe","Turkey":"Asia","Israel":"Europe","Cyprus":"Europe",
    "Finland":"Europe","Ireland":"Europe","Albania":"Europe","Bosnia":"Europe",
    "Kosovo":"Europe","Armenia":"Europe","Georgia":"Europe","Iceland":"Europe",
    "Latvia":"Europe","Montenegro":"Europe","Estonia":"Europe",
    "Northern Ireland":"Europe","Wales":"Europe","Kazakhstan":"Europe",
    "Moldova":"Europe","Lithuania":"Europe","Malta":"Europe",
    "Australia":"Oceania","Japan":"Asia","Korea":"Asia","China":"Asia",
    "Azerbaijan":"Asia","Brazil":"South America","Argentina":"South America",
    "Colombia":"South America","Ecuador":"South America","Uruguay":"South America",
    "Chile":"South America","Venezuela":"South America",
    "USA":"North America","Mexico":"North America",
    "Morocco":"Africa","Tunisia":"Africa","South Africa":"Africa","Egypt":"Africa",
}

def _league_country(lg: str) -> str:
    import re as _re
    return _re.sub(r"\s*\d+\.?\s*$", "", str(lg)).strip().rstrip(".")

def _league_region(lg: str) -> str:
    return COUNTRY_TO_REGION.get(_league_country(lg), "Other")

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:1.3rem;font-weight:800;color:#3b82f6;padding:10px 0 2px 0">🏟️ CLUB FIT</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:.76rem;margin-bottom:14px">Football Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown("**Upload Data**")
    player_files = st.file_uploader("Player CSVs", type="csv", accept_multiple_files=True)
    team_file    = st.file_uploader("Team Stats CSV", type="csv")
    st.markdown("---")
    api_key = st.text_input("Anthropic API Key (AI Analysis)", type="password")
    st.markdown("---")

    # ── Candidate league filtering ────────────────────────────────────
    st.markdown("**Candidate Leagues**")
    st.caption("Controls which clubs appear in results. Player can be from any league.")

    # We need player_df to build league list — load it early for sidebar use
    # (cached so no performance cost when called again below)
    @st.cache_data(show_spinner=False)
    def _get_leagues(files):
        if not files: return []
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if 'League' in df.columns:
                    dfs.append(df[['League']].dropna())
            except: pass
        if not dfs: return []
        return sorted(pd.concat(dfs)['League'].unique().tolist())

    _all_lgs = _get_leagues(tuple(player_files) if player_files else ())

    # Regions
    _all_regions = sorted({_league_region(lg) for lg in _all_lgs}) if _all_lgs else []
    sel_regions  = st.multiselect("Regions", _all_regions, default=_all_regions, key="cf_regions")
    # Fall back to all leagues if user clears regions
    _region_lgs  = [lg for lg in _all_lgs if _league_region(lg) in sel_regions] or _all_lgs

    # Presets
    st.markdown("##### League Presets")
    _pc1, _pc2, _pc3 = st.columns(3)
    use_top5  = _pc1.checkbox("Top 5",  False, key="cf_top5")
    use_top20 = _pc2.checkbox("Top 20", False, key="cf_top20")
    use_efl   = _pc3.checkbox("EFL",    False, key="cf_efl")

    # Normalised preset sets — no dots, lowercase, matches any format
    _PRESET_NORM = {
        "top5":  {"england 1","spain 1","germany 1","italy 1","france 1"},
        "top20": {"england 1","spain 1","germany 1","italy 1","france 1",
                  "england 2","portugal 1","belgium 1","turkey 1","germany 2",
                  "spain 2","france 2","netherlands 1","austria 1","switzerland 1",
                  "denmark 1","croatia 1","italy 2","czech 1","norway 1"},
        "efl":   {"england 2","england 3","england 4"},
    }
    def _lg_norm(s): return str(s).strip().rstrip('.').strip().lower()

    _active_norm = set()
    if use_top5:  _active_norm |= _PRESET_NORM["top5"]
    if use_top20: _active_norm |= _PRESET_NORM["top20"]
    if use_efl:   _active_norm |= _PRESET_NORM["efl"]

    # Leagues in data that match the active preset
    _preset_matched = [lg for lg in _region_lgs if _lg_norm(lg) in _active_norm]

    # What to show selected: preset match if any preset on, else everything
    _preset_sig = (tuple(sorted(sel_regions)), use_top5, use_top20, use_efl)
    if st.session_state.get("cf_preset_sig") != _preset_sig:
        st.session_state["cf_preset_sig"]  = _preset_sig
        st.session_state["cf_leagues_val"] = _preset_matched if _active_norm else _region_lgs

    if "cf_leagues_val" not in st.session_state:
        st.session_state["cf_leagues_val"] = _region_lgs

    # Clamp to what's currently available
    _current_val = [lg for lg in st.session_state["cf_leagues_val"] if lg in _region_lgs]
    if not _current_val:
        _current_val = _preset_matched if _active_norm else _region_lgs

    cand_leagues = st.multiselect("Candidate leagues", _region_lgs, default=_current_val)
    st.session_state["cf_leagues_val"] = cand_leagues

    st.markdown("---")
    st.markdown("**Scoring Weights**")
    league_weight = st.slider("League quality weight", 0.0, 1.0, 0.20, 0.05)
    market_weight = st.slider("Market value weight",   0.0, 1.0, 0.10, 0.05)

    st.markdown("**Filters**")
    min_ls   = st.slider("Min league strength", 0, 101, 0)
    max_ls   = st.slider("Max league strength", 0, 101, 101)
    min_mins = st.slider("Min minutes (candidates)", 0, 3000, 500, 100)
    top_n    = st.number_input("Results to show", 5, 50, 10, 5)

    st.markdown("---")
    st.markdown("**Image Export**")
    img_theme  = st.radio("Theme",  ["Light", "Dark"], index=0, horizontal=True, key="img_theme")
    img_format = st.radio("Format", ["Standard", "1920×1080"], index=0, horizontal=True, key="img_format")

# ── LOAD DATA ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_players(files):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'Position' in df.columns:
                df['_pg'] = df['Position'].apply(detect_pos)
            dfs.append(df)
        except: pass
    if not dfs: return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=['Player','Team'], keep='first')
    for c in ['Age','Minutes played','Market value']:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

# ── TEAM CSV COLUMN NORMALISATION (mirrors team_hq.py COL_MAP) ────────
_TEAM_COL_MAP = {
    "Team":                          ["team"],
    "League":                        ["league"],
    "Matches":                       ["matches"],
    "Possession %":                  ["possession %", "possession", "possession_pct", "poss %", "poss"],
    "Pass Accuracy %":               ["pass accuracy %", "passing accuracy %", "pass_accuracy_pct",
                                      "accurate passes %", "pass acc %", "passing acc %"],
    "Passes p90":                    ["passes p90", "passes per 90", "passes_p90"],
    "PPDA":                          ["ppda"],
    "Defensive Duels p90":           ["defensive duels p90", "defensive duels per 90", "defensive_duels_p90",
                                      "def duels p90"],
    "Long Passes p90":               ["long passes p90", "long passes per 90", "long_passes_p90"],
    "Aerial Duels p90":              ["aerial duels p90", "aerial duels per 90", "aerial_duels_p90"],
    "Passes to Final Third p90":     ["passes to final third p90", "passes to final 3rd p90",
                                      "passes_to_final_third_p90", "passes to final third per 90"],
    "Progressive Passes p90":        ["progressive passes p90", "progressive passes per 90",
                                      "progressive_passes_p90"],
    "xG p90":                        ["xg p90", "xg per 90", "xg_p90"],
    "Touches in Box p90":            ["touches in box p90", "touches in box per 90", "touches_in_box_p90"],
    "Shots p90":                     ["shots p90", "shots per 90", "shots_p90"],
    "Crosses p90":                   ["crosses p90", "crosses per 90", "crosses_p90"],
    "xG Against p90":                ["xg against p90", "xga p90", "xg_against_p90", "xg against per 90"],
    "Goals Against p90":             ["goals against p90", "goals conceded p90", "goals_against_p90"],
    "Goals p90":                     ["goals p90", "goals per 90", "goals_p90", "goals scored p90"],
    "Expected Points":               ["expected points", "xpoints", "x points", "expected_points", "xpts"],
    "Points":                        ["points", "pts"],
}

def _normalise_team_cols(df: pd.DataFrame) -> pd.DataFrame:
    existing_lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for canonical, aliases in _TEAM_COL_MAP.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in existing_lower:
                rename[existing_lower[alias]] = canonical
                break
    return df.rename(columns=rename)

@st.cache_data(show_spinner=False)
def load_teams(f):
    try:
        df = pd.read_csv(f)
        df = _normalise_team_cols(df)
        # coerce all known numeric columns
        for c in [k for k in _TEAM_COL_MAP if k not in ("Team", "League")]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except:
        return pd.DataFrame()

player_df = load_players(player_files) if player_files else pd.DataFrame()
team_df   = load_teams(team_file)      if team_file   else pd.DataFrame()

# ── HEADER ────────────────────────────────────────────────────────────
st.markdown('<div class="head">🏟️ CLUB FIT <span style="color:#3b82f6">FINDER</span></div>', unsafe_allow_html=True)

if player_df.empty:
    st.info("Upload player CSV files in the sidebar to get started.")
    st.stop()

# ── PLAYER SELECTION ──────────────────────────────────────────────────
c1, c2 = st.columns([3,1])
search_q  = c1.text_input("Search player", placeholder="Name or team…")
pos_filt  = c2.selectbox("Position", ["All"]+list(CF_FEATURES.keys()))

pool = player_df.copy()
if search_q:
    m = (pool['Player'].astype(str).str.contains(search_q, case=False, na=False) |
         pool.get('Team', pd.Series(dtype=str)).astype(str).str.contains(search_q, case=False, na=False))
    pool = pool[m]
if pos_filt != "All" and '_pg' in pool.columns:
    pool = pool[pool['_pg'] == pos_filt]

opts = []
for _, row in pool.head(200).iterrows():
    opts.append(f"{row.get('Player','?')}  ·  {row.get('Team','?')}  ·  {row.get('League','?')}")

if not opts:
    st.warning("No players found.")
    st.stop()

sel_label = st.selectbox("Select player", opts)
sel_name  = sel_label.split("  ·  ")[0].strip()

tgt_rows = player_df[player_df['Player'] == sel_name]
if tgt_rows.empty:
    st.error("Player not found.")
    st.stop()

tgt = tgt_rows.sort_values('Minutes played', ascending=False).iloc[0]
pg  = detect_pos(tgt.get('Position','CM'))
feats = [f for f in CF_FEATURES.get(pg, CF_FEATURES['CM']) if f in player_df.columns]

if len(feats) < 3:
    st.error(f"Not enough feature columns for {pg}. Check your CSV.")
    st.stop()

tgt_team   = str(tgt.get('Team','—'))
tgt_league = str(tgt.get('League','—'))

def _ls_lookup(league_name):
    key = str(league_name).strip()
    if key in LEAGUE_STRENGTHS:
        return LEAGUE_STRENGTHS[key]
    key2 = key.rstrip('.')
    for k, v in LEAGUE_STRENGTHS.items():
        if k.rstrip('.').lower() == key2.lower():
            return v
    return 50.0

tgt_ls = _ls_lookup(tgt_league)

st.markdown(f"""
<div class="pcard">
  <div class="nm">{sel_name}</div>
  <div class="mt">{tgt_team} · {tgt_league} · {tgt.get('Position','—')} · Age {tgt.get('Age','—')} · {tgt.get('Minutes played','—')} mins · MV {fmt_mv(tgt.get('Market value'))} · League Strength {tgt_ls:.0f}</div>
</div>
""", unsafe_allow_html=True)

# ── STYLE BLEND ───────────────────────────────────────────────────────
st.markdown('<div class="sec">Style Blend</div>', unsafe_allow_html=True)
all_styles = ["Similar to Current System","Possession","Pressing","Direct",
              "Passing Based","High Attacking Territory","Cross Heavy"]
sel_styles = st.multiselect("Select styles (combine any)", all_styles,
                             default=["Similar to Current System"])

# ── ADVANCED ──────────────────────────────────────────────────────────
with st.expander("⚙️ Advanced Settings", expanded=False):
    ac1, ac2 = st.columns(2)
    with ac1:
        max_age  = st.slider("Max candidate age", 16, 45, 40)
    with ac2:
        mv_override      = st.number_input("MV override (£)", 0, 100_000_000, 0, 500_000)
        style_blend_w    = st.slider("Team style blend weight", 0.0, 1.0, 0.43, 0.05)

    st.markdown("**Metric weights**")
    wc = st.columns(3)
    defs = DEFAULT_WEIGHTS.get(pg, {})
    custom_w = {}
    for i, f in enumerate(feats):
        with wc[i % 3]:
            custom_w[f] = st.slider(f[:28], 0, 5, defs.get(f, 1), key=f"w_{f}")

# ── RUN ───────────────────────────────────────────────────────────────
run = st.button("🔍  Find Club Fits", type="primary", use_container_width=True)

# ── COMPUTE ───────────────────────────────────────────────────────────
if run:
    with st.spinner("Computing…"):

        cand = player_df.copy()
        if cand_leagues:
            cand = cand[cand['League'].isin(cand_leagues)]
        # Exclude the player's own current club from candidate results
        cand = cand[cand['Team'] != tgt_team]
        if '_pg' in cand.columns:
            cand = cand[cand['_pg'] == pg]
        cand = cand[cand['Minutes played'].fillna(0) >= min_mins]
        cand = cand[cand['Age'].fillna(99)            <= max_age]
        cand = cand.dropna(subset=feats).reset_index(drop=True)

        if cand.empty:
            st.error("No candidates after filters.")
            st.stop()

        for f in feats:
            cand[f] = pd.to_numeric(cand[f], errors='coerce').fillna(0)
        tgt_vec = np.array([pd.to_numeric(tgt.get(f, 0), errors='coerce') or 0 for f in feats], dtype=float)

        w = np.array([custom_w.get(f, 1) for f in feats], dtype=float)
        w = w / (w.sum() or 1.0)

        tgt_df = player_df[player_df['Player'] == sel_name].copy()
        for f in feats:
            tgt_df[f] = pd.to_numeric(tgt_df[f], errors='coerce').fillna(0)

        # ── Percentile similarity (within-league) ─────────────────────
        ref = pd.concat([cand[feats + ['League']], tgt_df[feats + ['League']].head(1)],
                        ignore_index=True)
        pct_mat = ref.groupby('League')[feats].rank(pct=True).fillna(0.5)

        n_cand   = len(cand)
        tgt_pct  = pct_mat.iloc[n_cand:].mean(axis=0).values
        cand_pct = pct_mat.iloc[:n_cand].values

        pct_dist = np.sum(np.abs(cand_pct - tgt_pct) * w, axis=1)
        sim_pct  = np.exp(-2.8 * pct_dist) * 100.0

        # ── Z-score similarity (within-league) ────────────────────────
        tgt_league_label = str(tgt.get('League', ''))
        act_dist = np.zeros(n_cand)

        all_leagues = cand['League'].unique().tolist()
        if tgt_league_label not in all_leagues:
            all_leagues = [tgt_league_label] + all_leagues

        for lg in all_leagues:
            cand_mask = cand['League'] == lg
            cand_idx  = np.where(cand_mask)[0]
            if len(cand_idx) == 0:
                continue
            lg_cand = cand.loc[cand_mask, feats].values
            if lg == tgt_league_label:
                lg_all = np.vstack([lg_cand, tgt_vec.reshape(1, -1)])
            else:
                lg_all = lg_cand
            if lg_all.shape[0] < 2:
                global_sc = StandardScaler().fit(np.vstack([cand[feats].values, tgt_vec.reshape(1, -1)]))
                c_std_lg  = global_sc.transform(lg_cand)
                t_std_lg  = global_sc.transform(tgt_vec.reshape(1, -1))
            else:
                lg_sc    = StandardScaler().fit(lg_all)
                c_std_lg = lg_sc.transform(lg_cand)
                t_std_lg = lg_sc.transform(tgt_vec.reshape(1, -1))
            act_dist[cand_idx] = np.sum(np.abs(c_std_lg - t_std_lg) * w, axis=1)

        sim_act = np.exp(-0.6 * act_dist) * 100.0
        cand['_sim'] = sim_pct * 0.5 + sim_act * 0.5

        if 'Market value' in cand.columns:
            mv_series = pd.to_numeric(cand['Market value'], errors='coerce')
            median_mv = float(mv_series.median() or 2_000_000)
            cand['_mv'] = mv_series.fillna(median_mv)
        else:
            cand['_mv'] = 2_000_000

        club_agg = cand.groupby('Team').agg(
            League  = ('League', lambda x: x.mode().iloc[0]),
            SimPct  = ('_sim',   'mean'),
            AvgMV   = ('_mv',    'mean'),
        ).reset_index()
        club_agg['AvgMV']  = club_agg['AvgMV'].fillna(2_000_000)
        club_agg['SimPct'] = club_agg['SimPct'].fillna(0)

        if club_agg.empty:
            st.error("No clubs with complete data.")
            st.stop()

        club_agg['LS'] = club_agg['League'].apply(_ls_lookup)
        club_agg = club_agg[(club_agg['LS'] >= min_ls) & (club_agg['LS'] <= max_ls)].reset_index(drop=True)
        if club_agg.empty:
            st.error("No clubs after league filter.")
            st.stop()

        has_team    = not team_df.empty and 'Team' in team_df.columns
        style_scores = np.full(len(club_agg), 50.0)

        if has_team and sel_styles:
            tdf = team_df.drop_duplicates(subset=['Team']).set_index('Team')
            team_league_lookup = {}
            if 'League' in team_df.columns:
                team_league_lookup = team_df.drop_duplicates(subset=['Team'])\
                                            .set_index('Team')['League'].to_dict()
            parts = []

            if "Similar to Current System" in sel_styles and tgt_team in tdf.index:
                num_cols = [c for c in tdf.columns if c != 'Team' and pd.api.types.is_numeric_dtype(tdf[c])]
                if num_cols:
                    tdf_num = tdf[num_cols].apply(pd.to_numeric, errors='coerce')
                    tdf_num = tdf_num.fillna(tdf_num.mean())
                    sc2    = StandardScaler().fit(tdf_num)
                    t_std2 = sc2.transform(tdf_num.loc[[tgt_team]])[0]
                    all_d  = np.linalg.norm(sc2.transform(tdf_num) - t_std2, axis=1)
                    rng    = float(all_d.max() - all_d.min()) or 1.0
                    part   = []
                    for team_name in club_agg['Team']:
                        if team_name in tdf_num.index:
                            d = float(np.linalg.norm(sc2.transform(tdf_num.loc[[team_name]])[0] - t_std2))
                            part.append(float((1 - (d - all_d.min()) / rng) * 100))
                        else:
                            part.append(50.0)
                    parts.append(np.array(part))

            for sname in sel_styles:
                if sname == "Similar to Current System":
                    continue
                blend      = STYLE_BLENDS.get(sname, {})
                col_parts  = []
                style_pass = np.ones(len(club_agg), dtype=bool)
                for col, direction in blend.items():
                    if col not in tdf.columns:
                        continue
                    part = []
                    for j, team_name in enumerate(club_agg['Team']):
                        if team_name not in tdf.index:
                            part.append(50.0); continue
                        v = pd.to_numeric(tdf.loc[team_name, col], errors='coerce')
                        if pd.isna(v):
                            part.append(50.0); continue
                        club_lg = team_league_lookup.get(team_name, None)
                        if club_lg and 'League' in team_df.columns:
                            lg_teams = team_df[team_df['League'] == club_lg]['Team'].tolist()
                            col_vals = pd.to_numeric(tdf.loc[tdf.index.isin(lg_teams), col], errors='coerce').dropna()
                        else:
                            col_vals = pd.to_numeric(tdf[col], errors='coerce').dropna()
                        if col_vals.empty:
                            part.append(50.0); continue
                        pct          = float((col_vals <= float(v)).mean() * 100)
                        effective_pct = (100 - pct) if direction < 0 else pct
                        if effective_pct < 60.0:
                            style_pass[j] = False
                        part.append(effective_pct)
                    col_parts.append(np.array(part) * abs(direction))
                if col_parts:
                    style_arr = np.mean(col_parts, axis=0)
                    style_arr[~style_pass] = 0.0
                    parts.append(style_arr)

            if parts:
                style_scores = np.mean(parts, axis=0)

        if sel_styles and has_team:
            combined = club_agg['SimPct'].values * (1 - style_blend_w) + style_scores * style_blend_w
        else:
            combined = club_agg['SimPct'].values.copy()

        club_agg['StyleFit'] = combined
        ratio    = (club_agg['LS'] / tgt_ls).clip(0.5, 1.2)
        adj      = club_agg['StyleFit'] * (1 - league_weight) + club_agg['StyleFit'] * ratio * league_weight
        gap      = (club_agg['LS'] - tgt_ls).clip(lower=0)
        adj     *= (1 - gap / 100).clip(lower=0.7)

        tgt_mv   = float(mv_override) if mv_override > 0 else float(pd.to_numeric(tgt.get('Market value', 0), errors='coerce') or 2_000_000)
        tgt_mv   = max(tgt_mv, 1.0)
        mv_ratio = (club_agg['AvgMV'] / tgt_mv).clip(0.5, 1.5)
        mv_score = (1 - abs(1 - mv_ratio)) * 100

        club_agg['FinalFit'] = (adj * (1 - market_weight) + mv_score * market_weight).round(1)

        results = (
            club_agg[['Team','League','LS','SimPct','FinalFit','AvgMV']]
            .sort_values('FinalFit', ascending=False)
            .reset_index(drop=True)
            .head(int(top_n))
        )
        results['FinalFit'] = results['FinalFit'].fillna(0).round(1)
        results['SimPct']   = results['SimPct'].fillna(0).round(1)
        results['League']   = results['League'].fillna('Unknown')
        results.insert(0, 'Rank', range(1, len(results) + 1))

        st.session_state['cf_results']    = results
        st.session_state['cf_sel_name']   = sel_name
        st.session_state['cf_sel_styles'] = sel_styles
        st.session_state['cf_tgt']        = dict(tgt)
        st.session_state['cf_pg']         = pg
        st.session_state['cf_tgt_league'] = tgt_league
        st.session_state['cf_img_params'] = dict(
            pg=pg, tgt_league=tgt_league, tgt_ls=tgt_ls,
            league_weight=league_weight, market_weight=market_weight,
            min_mins=min_mins, top_n=int(top_n), sel_styles=sel_styles,
        )

# ── Restore from session state — always reached on every rerun ───────
if 'cf_results' not in st.session_state:
    st.stop()

results    = st.session_state['cf_results']
sel_name   = st.session_state['cf_sel_name']
sel_styles = st.session_state['cf_sel_styles']
_ip        = st.session_state['cf_img_params']
_tgt       = st.session_state.get('cf_tgt', {})
_pg        = st.session_state.get('cf_pg', '')
_tgt_league = st.session_state.get('cf_tgt_league', '')

# ── DISPLAY ───────────────────────────────────────────────────────────
st.markdown(f'<div class="sec">Top {int(top_n)} Club Fits — {sel_name}</div>', unsafe_allow_html=True)
if sel_styles:
    st.markdown("Active styles: " + "".join(f'<span class="pill">{s}</span>' for s in sel_styles),
                unsafe_allow_html=True)

# Diagnostic: warn if scores look wrong
_max_fit = results['FinalFit'].max()
_max_sim = results['SimPct'].max()
if _max_fit < 1:
    st.warning(f"⚠️ Scores look wrong (max FinalFit={_max_fit:.2f}, max SimPct={_max_sim:.2f}). "
               f"Check that your player CSV league names match the LEAGUE_STRENGTHS keys "
               f"(e.g. 'England 1.' not 'Premier League'). "
               f"Target league detected as: **{tgt_league}** → strength {tgt_ls:.0f}")

# ── Badge fetching ────────────────────────────────────────────────────
try:
    from team_fotmob_urls import FOTMOB_TEAM_URLS as _FU
except Exception:
    _FU = {}

@st.cache_data(show_spinner=False)
def _badge(team):
    raw = (_FU.get(team) or "").strip()
    if not raw: return None
    m2 = re.search(r"/teams/(\d+)/", raw)
    if not m2: return None
    url = f"https://images.fotmob.com/image_resources/logo/teamlogo/{m2.group(1)}.png"
    try:
        r = requests.get(url, timeout=5); r.raise_for_status()
        return plt.imread(io.BytesIO(r.content))
    except: return None

# ── Ranking image ─────────────────────────────────────────────────────
def make_ranking_img(df_show, player_name, active_styles, theme="Light", export_mode="Standard (auto)",
                     pg="", tgt_league="", tgt_ls=50, league_weight=0.35, market_weight=0.20,
                     min_mins=500, top_n=10, sel_styles=None):
    if df_show.empty: return None

    # ── Theme palettes ────────────────────────────────────────────────
    if theme == "Dark":
        BG        = "#0a0f1c"
        ROW_A     = "#0f1628"
        ROW_B     = "#0d1422"
        TXT       = "#ffffff"
        SUB       = "#b8c0cf"
        FOOT      = "#9aa6bd"
        DIV       = "#23304a"
        BAR_BG    = "#1a2540"
        BAR_FG    = "#6b7cff"
        RANK_BG   = "#111a2e"
        RANK_EDGE = "#2b3a5a"
        HDR_ACCENT = "#3b82f6"
    else:  # Light — exact match to team_hq.py
        BG        = "#ffffff"
        ROW_A     = "#f7f7f7"
        ROW_B     = "#ffffff"
        TXT       = "#111111"
        SUB       = "#777777"
        FOOT      = "#9b9b9b"
        DIV       = "#e2e2e2"
        BAR_BG    = "#e1e1e1"
        BAR_FG    = "#bfbfbf"   # grey bar, no colour coding on light
        RANK_BG   = "#f3f3f3"
        RANK_EDGE = "#c0c0c0"
        HDR_ACCENT = "#111111"  # player name in black on light

    GOLD = "#f59e0b"
    N    = len(df_show)
    max_v = float(df_show['FinalFit'].max()) or 1.0
    style_str = "  ·  ".join(active_styles) if active_styles else ""

    # ── 1920×1080 banner ──────────────────────────────────────────────
    if export_mode == "1920×1080 (banner)":
        DPI = 100
        fig = plt.figure(figsize=(19.2, 10.8), dpi=DPI)
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, color=BG, zorder=0))
        LEFT, RIGHT = 0.045, 0.955
        ax.text(LEFT, 0.972, "CLUB FIT FINDER", fontsize=48, fontweight="bold", color=TXT, ha="left", va="top")
        ax.text(LEFT, 0.912, player_name.upper(), fontsize=34, fontweight="bold", color=HDR_ACCENT, ha="left", va="top")
        if style_str:
            ax.text(LEFT, 0.870, style_str, fontsize=20, color=SUB, ha="left", va="top")
        ax.plot([LEFT, RIGHT], [0.835, 0.835], color=DIV, lw=2.2)
        ax.plot([LEFT, RIGHT], [0.040, 0.040], color=DIV, lw=2.2)
        ax.text(LEFT, 0.022, "Final Fit % = Player Similarity · League Quality · Market Value", fontsize=13, color=FOOT, ha="left", va="top")
        ROW_TOP = 0.813; ROW_BOT = 0.050
        row_gap = (ROW_TOP - ROW_BOT) / float(N)
        row_h   = row_gap * 0.92
        RANK_X = LEFT+0.024; CREST_X = LEFT+0.105; NAME_X = LEFT+0.175
        BAR_L  = LEFT+0.62;  BAR_R   = RIGHT-0.14
        BAR_W  = BAR_R-BAR_L; BAR_H2 = row_h*0.26; VAL_X = RIGHT-0.025
        for i, (_, row) in enumerate(df_show.iterrows()):
            y = ROW_TOP - (i + 0.5) * row_gap
            ax.add_patch(Rectangle((LEFT, y-row_h/2), RIGHT-LEFT, row_h,
                                    color=(ROW_A if i%2==0 else ROW_B), zorder=1))
            edge_col = GOLD if i < 3 else RANK_EDGE
            rank_col = GOLD if i < 3 else TXT
            ax.scatter([RANK_X], [y], s=1320, facecolor=RANK_BG, edgecolor=edge_col, linewidths=2.2, zorder=4)
            ax.text(RANK_X, y, str(i+1), fontsize=16, fontweight="bold", color=rank_col, ha="center", va="center", zorder=5)
            bdg = _badge(str(row['Team']))
            if bdg is not None:
                h, w2 = bdg.shape[:2]; z = 52.0/max(h, w2)
                ax.add_artist(AnnotationBbox(OffsetImage(bdg, zoom=z), (CREST_X, y), frameon=False, zorder=5))
            ax.text(NAME_X, y+row_h*0.18, str(row['Team']).upper(), fontsize=28, fontweight="bold", color=TXT, ha="left", va="center", zorder=6)
            ax.text(NAME_X, y-row_h*0.22, str(row['League']), fontsize=19, color=SUB, ha="left", va="center", zorder=6)
            frac = max(0.0, min(1.0, float(row['FinalFit']) / 100.0))
            ax.add_patch(Rectangle((BAR_L, y-BAR_H2/2), BAR_W, BAR_H2, color=BAR_BG, zorder=2))
            ax.add_patch(Rectangle((BAR_L, y-BAR_H2/2), BAR_W*frac, BAR_H2, color=BAR_FG, zorder=3))
            fc = score_col(float(row['FinalFit']))
            ax.text(VAL_X, y, f"{row['FinalFit']:.0f}", fontsize=29, fontweight="bold", color=fc, ha="right", va="center", zorder=6)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor=BG)
        plt.close(fig); buf.seek(0)
        return buf.getvalue()

    # ── Standard — exact team_hq.py layout (normalised 0–1 coords) ───
    ROW_H    = 0.82
    HEADER_H = 1.70
    FOOT_H   = 0.80
    TOTAL_H  = HEADER_H + N * ROW_H + FOOT_H
    DPI     = 220

    fig = plt.figure(figsize=(8.3, TOTAL_H), dpi=DPI)
    ax  = fig.add_axes([0, 0, 1, 1.0])
    ax.set_xlim(0, 1.0); ax.set_ylim(0, TOTAL_H); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1.0, TOTAL_H, color=BG, zorder=0))

    # Header (top-left, same position as team_hq)
    title_y = TOTAL_H - 0.25
    ax.text(0.04, title_y,       "CLUB FIT FINDER",   fontsize=19, fontweight="bold", color=TXT, ha="left", va="top")
    ax.text(0.04, title_y-0.34,  player_name.upper(), fontsize=14, fontweight="bold", color=HDR_ACCENT, ha="left", va="top")

    # Subtitle — Role and League only
    subtitle_parts = []
    if pg:         subtitle_parts.append(f"Role: {pg}")
    if tgt_league: subtitle_parts.append(f"League: {tgt_league}")
    subtitle_line = "  ·  ".join(subtitle_parts)
    if subtitle_line:
        ax.text(0.04, title_y-0.62, subtitle_line, fontsize=9, color=SUB, ha="left", va="top")

    base_y = TOTAL_H - HEADER_H
    ax.plot([0.04, 0.96], [base_y + ROW_H/2 + 0.02]*2, color=DIV, lw=1.1, zorder=2)

    # Column layout — exactly matching team_hq
    LEFT, RIGHT = 0.04, 0.96
    crest_x = 0.14
    BAR_L, BAR_R = 0.66, 0.82
    BAR_W = BAR_R - BAR_L
    BAR_H2 = 0.14
    VAL_X = 0.94

    for i, (_, row) in enumerate(df_show.iterrows()):
        y = base_y - i * ROW_H
        ax.add_patch(Rectangle((LEFT, y-ROW_H/2), RIGHT-LEFT, ROW_H,
                                color=(ROW_A if i%2==0 else ROW_B), zorder=1))

        # Rank badge — scatter circle, same as team_hq
        edge_col  = GOLD if i < 3 else RANK_EDGE
        rank_color = GOLD if i < 3 else TXT
        ax.scatter([0.07], [y], s=520, facecolor=RANK_BG,
                   edgecolor=edge_col, linewidths=1.2, zorder=4)
        ax.text(0.07, y, str(i+1), fontsize=10, fontweight="bold",
                color=rank_color, ha="center", va="center", zorder=5)

        # Club badge at crest_x=0.14
        bdg = _badge(str(row['Team']))
        if bdg is not None:
            h, w2 = bdg.shape[:2]
            z = 40.0 / max(h, w2)
            ax.add_artist(AnnotationBbox(OffsetImage(bdg, zoom=z),
                          (crest_x, y), frameon=False, zorder=5))

        # Team name at 0.21, league below — exactly team_hq
        ax.text(0.21, y+0.12, str(row['Team']).upper(),
                fontsize=16, fontweight="bold", color=TXT, ha="left", va="center", zorder=5)
        ax.text(0.21, y-0.10, str(row['League']),
                fontsize=12, color=SUB, ha="left", va="center", zorder=5)

        # Fit bar — out of 100 so bars are comparable across searches
        frac = max(0.0, min(1.0, float(row['FinalFit']) / 100.0))
        ax.add_patch(Rectangle((BAR_L, y-BAR_H2/2), BAR_W, BAR_H2, color=BAR_BG, zorder=2))
        ax.add_patch(Rectangle((BAR_L, y-BAR_H2/2), BAR_W*frac, BAR_H2, color=BAR_FG, zorder=3))

        # Score — black numbers on light, colour-coded on dark
        fc = "#111111" if theme == "Light" else score_col(float(row['FinalFit']))
        ax.text(VAL_X, y, f"{row['FinalFit']:.0f}",
                fontsize=16, fontweight="bold", color=fc, ha="right", va="center", zorder=6)

    # ── Footer: anchored just below the last row ─────────────────────
    # Last row bottom edge = base_y - (N-1)*ROW_H - ROW_H/2
    last_row_bottom = base_y - (N - 1) * ROW_H - ROW_H / 2
    div_y    = last_row_bottom - 0.10   # divider line
    line1_y  = div_y - 0.08
    line2_y  = line1_y - 0.20
    line3_y  = line2_y - 0.20

    ax.plot([LEFT, RIGHT], [div_y]*2, color=DIV, lw=0.9, zorder=2)

    _named_styles = [s for s in (sel_styles or []) if s != "Similar to Current System"]
    _lw_pct   = int(round(league_weight * 100))
    _mv_pct   = int(round(market_weight * 100))
    _fit_pct  = 100 - _lw_pct - _mv_pct          # StyleFit share
    _sty_pct  = int(round(_fit_pct * style_blend_w)) if 'style_blend_w' in dir() else int(round(_fit_pct * 0.43))
    _play_pct = _fit_pct - _sty_pct

    summary_line1 = (f"Ranked by: {_play_pct}% player profile ({pg}), "
                     f"{_sty_pct}% team style, "
                     f"{_lw_pct}% league quality, "
                     f"{_mv_pct}% market value. "
                     f"{min_mins}+ mins only.")
    summary_line2 = f"League quality vs {tgt_league} (strength {tgt_ls:.0f}/100)."
    summary_line3 = (f"Style filters ({', '.join(_named_styles)}): top 40% within own league required."
                     if _named_styles else "")

    ax.text(LEFT, line1_y, summary_line1, fontsize=7.5, color=FOOT, ha="left", va="top", zorder=4)
    ax.text(LEFT, line2_y, summary_line2, fontsize=7.5, color=FOOT, ha="left", va="top", zorder=4)
    if summary_line3:
        ax.text(LEFT, line3_y, summary_line3, fontsize=7.5, color=FOOT, ha="left", va="top", zorder=4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf.getvalue()


rank_img = make_ranking_img(
    results, sel_name, sel_styles,
    theme=img_theme,
    export_mode="1920×1080 (banner)" if img_format == "1920×1080" else "Standard (auto)",
    pg=_ip['pg'],
    tgt_league=_ip['tgt_league'],
    tgt_ls=_ip['tgt_ls'],
    league_weight=_ip['league_weight'],
    market_weight=_ip['market_weight'],
    min_mins=_ip['min_mins'],
    top_n=_ip['top_n'],
    sel_styles=_ip['sel_styles'],
)
# Cache image so download button rerun doesn't regenerate or lose it
if rank_img:
    st.session_state['cf_rank_img'] = rank_img
if st.session_state.get('cf_rank_img'):
    st.image(st.session_state['cf_rank_img'], use_column_width=True)
    st.download_button("⬇️ Download Ranking Image",
                       st.session_state['cf_rank_img'],
                       f"club_fit_{sel_name.replace(' ','_')}.png", "image/png")

csv_out = results.rename(columns={
    'SimPct':'Similarity %','FinalFit':'Final Fit %','AvgMV':'Avg MV','LS':'League Strength'
}).to_csv(index=False).encode()
st.session_state['cf_csv'] = csv_out
st.download_button("⬇️ Download CSV",
    st.session_state['cf_csv'],
    f"club_fit_{sel_name.replace(' ','_')}.csv", "text/csv")


# ── AI SQUAD ANALYSIS ─────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec">🤖 AI Squad Analysis</div>', unsafe_allow_html=True)

if not api_key:
    st.info("Add your Anthropic API key in the sidebar to unlock squad analysis for each club.")
else:
    st.caption("Expand any club below to generate a full AI scouting report.")
    client_ai = anthropic.Anthropic(api_key=api_key)

    # Role metrics used to compute a proxy performance score per player
    ROLE_SCORE_WEIGHTS = {
        'GK':  ['Save rate, %','Accurate passes, %','Accurate long passes, %'],
        'CB':  ['Defensive duels won, %','Aerial duels won, %','Progressive passes per 90',
                'Progressive runs per 90','Accurate passes, %'],
        'FB':  ['Defensive duels won, %','xA per 90','Dribbles per 90',
                'Touches in box per 90','Accurate passes, %'],
        'CM':  ['Passes per 90','Accurate passes, %','Progressive passes per 90',
                'xA per 90','Defensive duels per 90'],
        'ATT': ['xA per 90','Dribbles per 90','Touches in box per 90',
                'Progressive runs per 90','Passes to penalty area per 90'],
        'CF':  ['Non-penalty goals per 90','xG per 90','Touches in box per 90',
                'Dribbles per 90','xA per 90'],
    }

    def _player_role_score(player_row, pg_key):
        metrics = ROLE_SCORE_WEIGHTS.get(pg_key, [])
        scores  = [float(pd.to_numeric(player_row.get(m, np.nan), errors='coerce'))
                   for m in metrics if pd.notna(pd.to_numeric(player_row.get(m, np.nan), errors='coerce'))]
        return round(float(np.mean(scores)), 1) if scores else 0.0


    def _squad_depth_html(squad, pg_key):
        """Build styled HTML player cards + plain text for AI prompt."""
        pos_players = squad[squad['_pg'] == pg_key].copy()
        if pos_players.empty:
            return "", f"No {pg_key}s in dataset."
        pos_players = pos_players.sort_values('Minutes played', ascending=False)
        cc = next((c for c in squad.columns if 'contract' in c.lower()), None)
        import re as _rec
        html_rows, text_rows = [], []
        for _, p in pos_players.head(6).iterrows():
            name    = str(p.get('Player', '?'))
            mins    = int(p.get('Minutes played', 0) or 0)
            matches = int(pd.to_numeric(p.get('Matches played', p.get('Matches', 0)), errors='coerce') or 0)
            goals   = int(pd.to_numeric(p.get('Goals', 0), errors='coerce') or 0)
            mv      = fmt_mv(p.get('Market value', 0))
            age     = int(p.get('Age', 0) or 0)
            rscore  = _player_role_score(p, pg_key)
            # Contract colour
            contract_raw = str(p.get(cc, '')) if cc else ''
            cy_m = _rec.search(r'(\d{4})', contract_raw)
            cy   = int(cy_m.group(1)) if cy_m else 9999
            contract_txt = contract_raw[:7] if contract_raw else '—'
            if cy <= 2026:   cc_col = '#ef4444'
            elif cy <= 2027: cc_col = '#f59e0b'
            else:            cc_col = '#64748b'
            # Role score colour
            rs_col = '#22c55e' if rscore >= 0.6 else ('#f59e0b' if rscore >= 0.35 else '#ef4444')
            html_rows.append(
                f'<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;'
                f'background:#0f172a;border:1px solid #1e2d42;border-radius:7px;margin-bottom:4px;flex-wrap:wrap;">'
                f'<span style="font-weight:800;color:#fff;min-width:130px;font-size:.88rem;">{name}</span>'
                f'<span style="color:#64748b;font-size:.78rem;">Age {age}</span>'
                f'<span style="background:#1e2d42;color:#cbd5e1;padding:2px 6px;border-radius:4px;font-size:.78rem;">{matches} apps · {mins}min</span>'
                f'<span style="background:#052e16;color:#22c55e;padding:2px 6px;border-radius:4px;font-size:.78rem;font-weight:700;">{goals}G</span>'
                f'<span style="background:{rs_col}18;color:{rs_col};border:1px solid {rs_col}44;padding:2px 6px;border-radius:4px;font-size:.78rem;font-weight:700;">Score {rscore:.2f}</span>'
                f'<span style="color:#64748b;font-size:.78rem;">{mv}</span>'
                f'<span style="background:{cc_col}18;color:{cc_col};border:1px solid {cc_col}44;padding:2px 6px;border-radius:4px;font-size:.76rem;margin-left:auto;">📄 {contract_txt}</span>'
                f'</div>'
            )
            text_rows.append(f"{name} (Age {age}, {matches} apps, {mins}min, {goals}G, Score {rscore:.2f}, MV {mv}, Contract {contract_txt})")
        return "".join(html_rows), "\n".join(text_rows)

    def run_ai_analysis(ai_team, fit_pct, sim_pct, ls_val, avg_mv):
        squad = player_df[player_df['Team'] == ai_team].copy()
        if '_pg' not in squad.columns:
            squad['_pg'] = squad['Position'].apply(detect_pos)
        squad['Market value']   = pd.to_numeric(squad.get('Market value', 0), errors='coerce').fillna(0)
        squad['Age']            = pd.to_numeric(squad.get('Age', 0), errors='coerce').fillna(0)
        squad['Minutes played'] = pd.to_numeric(squad.get('Minutes played', 0), errors='coerce').fillna(0)
        for col in ROLE_SCORE_WEIGHTS.get(_pg, []):
            if col in squad.columns:
                squad[col] = pd.to_numeric(squad[col], errors='coerce').fillna(0)

        depth_html, depth_text = _squad_depth_html(squad, _pg)
        st.session_state[f'_depth_html_{ai_team}'] = depth_html

        avg_age      = round(float(squad['Age'].mean()), 1) if not squad.empty else "—"
        u23          = int((squad['Age'] < 23).sum())
        o28          = int((squad['Age'] > 28).sum())
        pos_count    = squad.groupby('_pg').size().to_dict()
        high_val     = squad.sort_values('Market value', ascending=False).head(4)
        high_val_str = "; ".join(f"{r.get('Player','?')} {fmt_mv(r.get('Market value',0))}" for _, r in high_val.iterrows())
        tgt_mv_val   = float(pd.to_numeric(_tgt.get('Market value', 0), errors='coerce') or 0)
        league_gap   = abs(ls_val - _ls_lookup(_tgt_league))
        dest_league  = squad['League'].iloc[0] if not squad.empty else '?'

        prompt = f"""You are a senior football recruitment analyst writing a structured scouting report.

TARGET: {sel_name} | {_pg} | {_tgt_league} | Age {_tgt.get('Age','?')} | MV {fmt_mv(_tgt.get('Market value'))}
CLUB: {ai_team} | {dest_league} | Strength {ls_val:.0f}/100 | Fit {fit_pct:.0f}% | Similarity {sim_pct:.0f}% | Squad avg MV {fmt_mv(avg_mv)}

{_pg} PLAYERS AT {ai_team} (sorted by minutes):
{depth_text}

SQUAD: Avg age {avg_age} | U23: {u23} | Over 28: {o28} | {pos_count}
Top value players: {high_val_str}
League gap vs {_tgt_league}: {league_gap:.0f} pts

Write EXACTLY 4 sections. Plain text only, no markdown, no bold, no asterisks. Header in CAPS followed by colon.

SQUAD DEPTH: 1-2 sentences. Is there a vacancy or competition at {_pg}? Reference role scores and contract situations — do NOT list all players again, just summarise what they mean.

BREAKDOWN: 2-3 sentences. The {sim_pct:.0f}% similarity score means there are players here with comparable statistical profiles to {sel_name} — name the key metrics that drove the match (e.g. similar xG output, dribbling rate, passing volume). Note where they differ and what the {ls_val:.0f}/100 league strength means for this ranking.

STYLE: 2 sentences max. First: {ai_team}'s tactical identity in {dest_league} in plain terms (pressing, possession, direct, transition). Second: one sentence on whether that suits {sel_name} as a {_pg}.

FIT VERDICT: SIGN / MONITOR / PASS. 2 sentences. League gap {league_gap:.0f} pts, MV {fmt_mv(tgt_mv_val)} vs squad avg {fmt_mv(avg_mv)} — is the price realistic and is this step up / lateral / step down?

SUMMARY LINE: Max 15 words. Start with SIGN / MONITOR / PASS."""

        resp = client_ai.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=700,
            messages=[{"role": "user", "content": prompt}])
        return resp.content[0].text.strip()

    def parse_and_render(text, ai_team, fit_pct):
        import re as _re
        sections  = {}
        clean     = _re.sub(r'\*+', '', text)
        KEYS      = ["SQUAD DEPTH","BREAKDOWN","STYLE","FIT VERDICT","SUMMARY LINE"]
        cur_key, cur_lines = None, []
        for line in clean.split('\n'):
            line = line.strip()
            if not line: continue
            matched = False
            for k in KEYS:
                if _re.match(rf'^{k}[\s:.\-–]*', line, _re.IGNORECASE):
                    if cur_key: sections[cur_key] = " ".join(cur_lines).strip()
                    cur_key   = k
                    rest      = _re.sub(rf'^{k}[\s:.\-–]*', '', line, flags=_re.IGNORECASE).strip()
                    cur_lines = [rest] if rest else []
                    matched   = True
                    break
            if not matched and cur_key:
                cur_lines.append(line)
        if cur_key: sections[cur_key] = " ".join(cur_lines).strip()

        if not any(sections.values()):
            st.markdown(f"### {ai_team} — Scouting Report")
            st.write(text)
            return

        icons  = {"SQUAD DEPTH":"👥","BREAKDOWN":"📐","STYLE":"🎨",
                  "FIT VERDICT":"✅","SUMMARY LINE":"⚡"}
        colors = {"SQUAD DEPTH":"#3b82f6","BREAKDOWN":"#8b5cf6","STYLE":"#06b6d4",
                  "FIT VERDICT":"#f59e0b","SUMMARY LINE":"#22c55e"}

        fc = score_col(float(fit_pct))
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">'
            f'<span style="font-family:\'Barlow Condensed\',sans-serif;font-size:1.4rem;'
            f'font-weight:800;color:#fff;">{ai_team} — Scouting Report</span>'
            f'<span style="background:{fc};color:#000;font-weight:800;padding:4px 12px;'
            f'border-radius:6px;font-size:1rem;">{fit_pct:.0f}%</span></div>',
            unsafe_allow_html=True)

        for k in KEYS:
            content = sections.get(k, "")
            if not content: continue
            bc = colors[k]
            if k == "SQUAD DEPTH":
                depth_html = st.session_state.get(f'_depth_html_{ai_team}', '')
                st.markdown(
                    f'<div class="acard" style="border-left-color:{bc};margin-bottom:10px;">'
                    f'<h4 style="color:{bc};margin:0 0 10px 0;">{icons[k]} {k}</h4>'
                    f'{depth_html}'
                    f'<p style="margin:10px 0 0 0;color:#94a3b8;font-size:.85rem;line-height:1.6;font-style:italic;">'
                    f'{content}</p></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="acard" style="border-left-color:{bc};margin-bottom:10px;">'
                    f'<h4 style="color:{bc};margin:0 0 6px 0;">{icons[k]} {k}</h4>'
                    f'<p style="margin:0;line-height:1.7;">{content}</p></div>',
                    unsafe_allow_html=True)

    # ── Per-club expanders ────────────────────────────────────────────
    for _, row in results.iterrows():
        team_name = row['Team']
        fit_score = row['FinalFit']
        cache_key = f"ai_v2_{team_name}_{sel_name}"

        with st.expander(
            f"#{int(row['Rank'])}  {team_name}  ·  {row['League']}  ·  Fit {fit_score:.0f}%",
            expanded=False):

            if cache_key in st.session_state:
                parse_and_render(st.session_state[cache_key], team_name, fit_score)
                if st.button(f"🔄 Regenerate", key=f"regen_{team_name}"):
                    del st.session_state[cache_key]
                    st.rerun()
            else:
                if st.button(f"🔍 Generate Scouting Report — {team_name}", key=f"ai_{team_name}"):
                    with st.spinner(f"Analysing {team_name}…"):
                        try:
                            ai_text = run_ai_analysis(
                                team_name, fit_score,
                                row['SimPct'], row['LS'], row['AvgMV'])
                            st.session_state[cache_key] = ai_text
                            parse_and_render(ai_text, team_name, fit_score)
                        except Exception as e:
                            st.error(f"AI error: {e}")

    # ── Final Summary Report ──────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec">📋 Final Summary Report</div>', unsafe_allow_html=True)

    import re as _re3
    all_summaries = {}
    for _, row in results.iterrows():
        ck = f"ai_v2_{row['Team']}_{sel_name}"
        if ck in st.session_state:
            m = _re3.search(r'SUMMARY LINE[\s:.\-–]*(.*)', st.session_state[ck], _re3.IGNORECASE)
            if m:
                all_summaries[row['Team']] = (m.group(1).strip(), row['FinalFit'])

    analysed = len(all_summaries)
    total    = len(results)

    if analysed == 0:
        st.info(f"Generate scouting reports above to build the summary ({total} clubs available).")
    else:
        st.caption(f"{analysed}/{total} clubs analysed.")
        for team, (summary, score) in sorted(all_summaries.items(), key=lambda x: -x[1][1]):
            fc = score_col(float(score))
            verdict = ("🟢 SIGN" if "SIGN" in summary.upper()
                       else ("🟡 MONITOR" if "MONITOR" in summary.upper() else "🔴 PASS"))
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;padding:8px 12px;'
                f'background:#111827;border:1px solid #1e2d42;border-radius:8px;margin-bottom:6px;">'
                f'<span style="background:{fc};color:#000;font-weight:800;padding:2px 8px;'
                f'border-radius:4px;min-width:36px;text-align:center;">{score:.0f}</span>'
                f'<span style="font-weight:700;color:#fff;min-width:160px;">{team}</span>'
                f'<span style="color:#94a3b8;font-size:.85rem;">{verdict} — {summary}</span>'
                f'</div>', unsafe_allow_html=True)

        if st.button("🤖 Generate Full Board Summary", type="primary", key="gen_final_summary"):
            summary_lines = "\n".join(
                f"• {team} (Fit {score:.0f}%): {summary}"
                for team, (summary, score) in sorted(all_summaries.items(), key=lambda x: -x[1][1]))
            final_prompt = f"""You are a head of recruitment writing a board-ready summary.

TARGET PLAYER: {sel_name} | {_pg} | {_tgt_league} | Age {_tgt.get('Age','?')} | MV {fmt_mv(_tgt.get('Market value'))}

CLUB VERDICTS:
{summary_lines}

Write a concise board summary (150-200 words). Cover:
1. Top recommendation and why
2. Best value option if different
3. Any deals to avoid and why
4. One clear action point

Plain text, no headers, no markdown. Write as if presenting verbally to a Director of Football."""

            with st.spinner("Generating board summary…"):
                try:
                    resp = client_ai.messages.create(
                        model="claude-haiku-4-5-20251001", max_tokens=500,
                        messages=[{"role": "user", "content": final_prompt}])
                    st.session_state['cf_final_summary'] = resp.content[0].text.strip()
                except Exception as e:
                    st.error(f"Summary error: {e}")

        if 'cf_final_summary' in st.session_state:
            st.markdown(
                f'<div class="acard" style="border-left-color:#22c55e;margin-top:16px;">'
                f'<h4 style="color:#22c55e;margin:0 0 8px 0;">⚡ Board Summary — {sel_name}</h4>'
                f'<p style="margin:0;line-height:1.8;font-size:.95rem;">'
                f'{st.session_state["cf_final_summary"]}</p></div>',
                unsafe_allow_html=True)
