import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import anthropic
import os

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Club Fit Finder",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Barlow:wght@400;600;700;800&family=Barlow+Condensed:wght@600;700;800&display=swap');

:root {
  --bg:      #07090f;
  --surface: #0d1117;
  --card:    #111827;
  --border:  #1e2d42;
  --accent:  #3b82f6;
  --gold:    #f59e0b;
  --green:   #22c55e;
  --red:     #ef4444;
  --white:   #ffffff;
  --off:     #e2e8f0;
  --muted:   #64748b;
  --dim:     #374151;
}

html, body, [class*="css"] {
  font-family: 'Barlow', sans-serif;
  background: var(--bg);
  color: var(--off);
}

.stApp { background: var(--bg); }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}

/* Header */
.cf-header {
  padding: 28px 0 12px 0;
  border-bottom: 2px solid var(--accent);
  margin-bottom: 24px;
}
.cf-header h1 {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 2.6rem;
  font-weight: 800;
  color: var(--white);
  letter-spacing: -0.02em;
  margin: 0;
}
.cf-header p {
  color: var(--muted);
  font-size: 0.9rem;
  margin: 4px 0 0 0;
}
.cf-accent { color: var(--accent); }
.cf-gold   { color: var(--gold);   }

/* Player card */
.player-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 10px;
  padding: 16px 20px;
  margin-bottom: 16px;
}
.player-card .name {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--white);
}
.player-card .meta {
  color: var(--muted);
  font-size: 0.8rem;
  margin-top: 4px;
}

/* Style toggle pills */
.style-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 12px 0;
}

/* Result cards */
.result-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 18px;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 16px;
  transition: border-color 0.2s;
}
.result-card:hover { border-color: var(--accent); }
.result-rank {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.8rem;
  font-weight: 800;
  color: var(--dim);
  min-width: 36px;
}
.result-rank.top { color: var(--gold); }
.result-team {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--white);
  flex: 1;
}
.result-league {
  font-size: 0.78rem;
  color: var(--muted);
}
.result-score {
  font-family: 'DM Mono', monospace;
  font-size: 1.5rem;
  font-weight: 500;
  min-width: 70px;
  text-align: right;
}
.score-bar-wrap {
  width: 100px;
  height: 6px;
  background: var(--dim);
  border-radius: 3px;
  overflow: hidden;
}
.score-bar { height: 100%; border-radius: 3px; }

/* Section header */
.section-head {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin: 20px 0 10px 0;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

/* AI box */
.ai-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--gold);
  border-radius: 8px;
  padding: 14px 18px;
  margin-bottom: 12px;
}
.ai-card h4 {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1rem;
  font-weight: 700;
  color: var(--gold);
  margin: 0 0 8px 0;
}
.ai-card p { color: var(--off); font-size: 0.88rem; line-height: 1.6; margin: 0; }

/* Metric pill */
.metric-pill {
  display: inline-block;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 0.78rem;
  color: var(--muted);
  margin: 2px;
}

/* Streamlit overrides — force white labels everywhere */
.stSelectbox > div > div { background: var(--card) !important; border-color: var(--border) !important; color: var(--off) !important; }
.stMultiSelect > div > div { background: var(--card) !important; border-color: var(--border) !important; }
div[data-testid="stExpander"] { background: var(--card); border-color: var(--border); border-radius: 8px; }
.stButton > button {
  background: var(--accent) !important;
  color: white !important;
  border: none !important;
  border-radius: 6px !important;
  font-family: 'Barlow', sans-serif !important;
  font-weight: 600 !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
/* All labels, sliders, checkboxes, radio, text — white */
label, .stCheckbox > label, .stRadio > label, .stSlider > label,
.stSelectbox > label, .stMultiSelect > label, .stTextInput > label,
.stNumberInput > label, .stFileUploader > label,
div[data-testid="stWidgetLabel"] > label,
div[data-testid="stWidgetLabel"] p,
.stMarkdown p, .stMarkdown li,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
.stSlider span, .stSlider p,
[data-testid="stCaptionContainer"] p,
p, span { color: var(--off) !important; }
/* Muted is still ok for subtitles */
.result-league, .player-card .meta { color: #94a3b8 !important; }
/* Slider track */
.stSlider > div { color: var(--off) !important; }
/* Expander header */
div[data-testid="stExpander"] summary span,
div[data-testid="stExpander"] summary p { color: var(--off) !important; }
/* Multiselect tags */
.stMultiSelect span[data-baseweb="tag"] { background: var(--accent) !important; }
/* Input fields */
input, textarea { background: var(--card) !important; color: var(--off) !important; border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

LEAGUE_STRENGTHS = {
    'England 1.':100.0,'Spain 1.':98.04,'Germany 1.':96.08,'Italy 1.':94.12,
    'France 1.':88.24,'Portugal 1.':82.35,'Netherlands 1.':80.39,'Belgium 1.':78.43,
    'Scotland 1.':70.59,'Turkey 1.':72.55,'Russia 1.':68.63,'Ukraine 1.':66.67,
    'Austria 1.':64.71,'Switzerland 1.':62.75,'Greece 1.':60.78,'Czech 1.':58.82,
    'Denmark 1.':56.86,'Sweden 1.':54.90,'Poland 1.':52.94,'Norway 1.':50.98,
    'Croatia 1.':49.02,'Serbia 1.':47.06,'Romania 1.':45.10,'Slovakia 1.':43.14,
    'Slovenia 1.':41.18,'Bulgaria 1.':39.22,'Hungary 1.':37.25,'Finland 1.':35.29,
    'Israel 1.':60.00,'Cyprus 1.':33.33,
    'England 2.':78.43,'England 3.':58.82,'England 4.':45.10,
    'Germany 2.':72.55,'Spain 2.':62.75,'Italy 2.':60.78,'France 2.':58.82,
    'Portugal 2.':52.94,'Netherlands 2.':50.98,
    'Scotland 2.':38.63,'Venezuela 1.':35.00,'Uruguay 1.':52.00,
    'Japan 1.':55.00,'Egypt 1.':40.00,
}

# Per-position feature sets — exactly as defined in the position-specific snippets
CF_FEATURES = {
    'GK': [
        'Exits per 90','Aerial duels per 90','Aerial duels won, %',
        'Save rate, %','Prevented goals per 90','Passes per 90',
        'Accurate passes, %','Long passes per 90','Accurate long passes, %',
    ],
    'CB': [  # Doc 8
        'Defensive duels per 90','Defensive duels won, %',
        'Aerial duels per 90','Aerial duels won, %','Shots blocked per 90',
        'PAdj Interceptions','Dribbles per 90','Successful dribbles, %',
        'Progressive runs per 90','Accelerations per 90','Passes per 90',
        'Accurate passes, %','Forward passes per 90','Accurate forward passes, %',
        'Long passes per 90','Accurate long passes, %',
        'Passes to final third per 90','Accurate passes to final third, %',
        'Progressive passes per 90','Accurate progressive passes, %',
    ],
    'FB': [  # Doc 11
        'Defensive duels per 90','Defensive duels won, %','Aerial duels per 90',
        'Aerial duels won, %','PAdj Interceptions',
        'Non-penalty goals per 90','xG per 90','Shots per 90',
        'Dribbles per 90','Successful dribbles, %',
        'Offensive duels per 90','Offensive duels won, %',
        'Touches in box per 90','Progressive runs per 90','Accelerations per 90',
        'Passes per 90','Accurate passes, %',
        'xA per 90','Smart passes per 90',
        'Passes to final third per 90','Accurate passes to final third, %',
        'Passes to penalty area per 90','Accurate passes to penalty area, %',
        'Deep completions per 90',
    ],
    'CM': [  # Doc 9
        'Defensive duels per 90','Defensive duels won, %','Aerial duels per 90',
        'Aerial duels won, %','Shots blocked per 90','PAdj Interceptions',
        'Non-penalty goals per 90','xG per 90','Shots per 90',
        'Dribbles per 90','Successful dribbles, %','Offensive duels per 90',
        'Offensive duels won, %','Touches in box per 90','Progressive runs per 90',
        'Accelerations per 90','Passes per 90','Accurate passes, %',
        'Forward passes per 90','Accurate forward passes, %','Long passes per 90',
        'Accurate long passes, %','xA per 90','Smart passes per 90',
        'Key passes per 90','Passes to final third per 90','Accurate passes to final third, %',
        'Passes to penalty area per 90','Accurate passes to penalty area, %',
        'Deep completions per 90','Progressive passes per 90',
    ],
    'ATT': [  # Doc 10
        'Defensive duels per 90','Aerial duels per 90','Aerial duels won, %',
        'PAdj Interceptions','Non-penalty goals per 90','xG per 90',
        'Shots per 90','Shots on target, %','Crosses per 90','Accurate crosses, %',
        'Dribbles per 90','Successful dribbles, %','Offensive duels per 90',
        'Offensive duels won, %','Touches in box per 90','Progressive runs per 90',
        'Accelerations per 90','Passes per 90','Accurate passes, %',
        'Forward passes per 90','Accurate forward passes, %','Long passes per 90',
        'Accurate long passes, %','xA per 90','Smart passes per 90',
        'Key passes per 90','Passes to final third per 90','Accurate passes to final third, %',
        'Passes to penalty area per 90','Accurate passes to penalty area, %',
        'Deep completions per 90','Progressive passes per 90',
    ],
    'CF': [  # From main app
        'Defensive duels per 90','Aerial duels per 90','Aerial duels won, %',
        'PAdj Interceptions','Non-penalty goals per 90','xG per 90',
        'Shots per 90','Shots on target, %','Crosses per 90','Accurate crosses, %',
        'Dribbles per 90','Successful dribbles, %','Offensive duels per 90',
        'Offensive duels won, %','Touches in box per 90','Progressive runs per 90',
        'Accelerations per 90','Passes per 90','Accurate passes, %',
        'xA per 90','Smart passes per 90','Key passes per 90',
        'Passes to final third per 90','Passes to penalty area per 90',
        'Accurate passes to penalty area, %','Deep completions per 90',
        'Progressive passes per 90',
    ],
}

# Default weights per position — directly from each snippet's _DEFAULT_W_CF
DEFAULT_WEIGHTS = {
    'GK': {
        'Aerial duels won, %': 3,'Passes per 90': 2,'Accurate passes, %': 2,
        'Long passes per 90': 2,'Accurate long passes, %': 2,
    },
    'CB': {  # Doc 8
        'Passes per 90': 2,'Accurate passes, %': 2,'Progressive passes per 90': 2,
        'Defensive duels per 90': 2,'Defensive duels won, %': 2,'Dribbles per 90': 2,
        'PAdj Interceptions': 1,'Progressive runs per 90': 2,
        'Aerial duels per 90': 2,'Aerial duels won, %': 3,
    },
    'FB': {  # Doc 11
        'Passes per 90': 2,'Accurate passes, %': 2,'Dribbles per 90': 2,
        'Non-penalty goals per 90': 2,'Shots per 90': 2,'Successful dribbles, %': 2,
        'Aerial duels won, %': 2,'xA per 90': 2,'xG per 90': 2,
        'Touches in box per 90': 2,
    },
    'CM': {  # Doc 9
        'Passes per 90': 3,'Passes to penalty area per 90': 2,'Dribbles per 90': 2,
        'xA per 90': 2,'Progressive passes per 90': 3,'Defensive duels per 90': 2,
        'Forward passes per 90': 3,'PAdj Interceptions': 2,
        'Aerial duels won, %': 2,'Touches in box per 90': 2,
    },
    'ATT': {  # Doc 10
        'Passes per 90': 2,'Progressive runs per 90': 2,'Progressive passes per 90': 2,
        'Dribbles per 90': 2,'xA per 90': 2,'Touches in box per 90': 2,
        'Accurate passes, %': 2,'Aerial duels won, %': 2,
        'Passes to penalty area per 90': 2,'Defensive duels per 90': 2,
    },
    'CF': {  # From main app
        'Passes per 90': 2,'Accurate passes, %': 2,'Dribbles per 90': 3,
        'Non-penalty goals per 90': 2,'Shots per 90': 2,'Successful dribbles, %': 2,
        'Aerial duels won, %': 2,'xA per 90': 2,'xG per 90': 2,
        'Touches in box per 90': 2,'Passes to final third per 90': 2,
        'Passes to penalty area per 90': 2,
    },
}

# Position detection
POS_MAP = {
    'GK':  ['GK'],
    'CB':  ['CB','LCB','RCB','WCB'],
    'FB':  ['LB','RB','LWB','RWB','WB'],
    'CM':  ['CM','DMF','CMF','AMF','DM','CAM','CDM'],
    'ATT': ['LW','RW','LWF','RWF','LAMF','RAMF','AMF'],
    'CF':  ['CF','ST','SS','LCF','RCF'],
}

def detect_position_group(pos_str):
    pos_str = str(pos_str)
    primary = pos_str.split(",")[0].strip().upper()
    for grp, tags in POS_MAP.items():
        if primary in tags:
            return grp
    for grp, tags in POS_MAP.items():
        if any(t in pos_str.upper() for t in tags):
            return grp
    return 'CM'

# Style blend definitions — which team stats columns each style maps to
# and direction (high = good for that style)
STYLE_BLENDS = {
    "Possession":            {"Possession %": 1.0, "Pass Accuracy %": 0.8, "Passes p90": 0.7},
    "Pressing":              {"PPDA": -1.0, "Defensive Duels p90": 0.6},  # low PPDA = high press
    "Direct":                {"Long Passes p90": 1.0, "Aerial Duels p90": 0.7},
    "Passing Based":         {"Passes to Final Third p90": 1.0, "Progressive Passes p90": 0.9, "Pass Accuracy %": 0.6},
    "High Attacking Territory": {"xG p90": 1.0, "Touches in Box p90": 0.9, "Shots p90": 0.7},
    "Cross Heavy":           {"Crosses p90": 1.0},
}

# Team stat columns available
TEAM_STAT_COLS = [
    'Possession %','PPDA','Passes p90','Pass Accuracy %',
    'xG p90','xG Against p90','Shots p90','Crosses p90',
    'Long Passes p90','Long Pass Accuracy %','Progressive Passes p90',
    'Progressive Runs p90','Aerial Duels p90','Aerial Duels Won %',
    'Defensive Duels p90','Defensive Duels Won %','Touches in Box p90',
    'Passes to Final Third p90',
]

def score_color(s):
    if s >= 80: return "#22c55e"
    if s >= 65: return "#86efac"
    if s >= 50: return "#f59e0b"
    return "#ef4444"

def fmt_mv(v):
    try:
        v = float(v)
        if v >= 1_000_000: return f"£{v/1_000_000:.1f}m"
        if v >= 1_000:     return f"£{v/1_000:.0f}k"
        return f"£{v:.0f}"
    except: return "—"

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — DATA UPLOAD
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:1.4rem;font-weight:800;color:#3b82f6;padding:12px 0 4px 0;">🏟️ CLUB FIT</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:0.78rem;margin-bottom:16px;">Football Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown("**Data Files**")
    player_files = st.file_uploader(
        "Player CSVs (upload all positions)",
        type="csv", accept_multiple_files=True,
        help="Upload one or more player CSV files (GK, CB, FB, CM, ATT, CF)"
    )
    team_file = st.file_uploader(
        "Team Stats CSV",
        type="csv",
        help="WORLD_team_stats CSV with PPDA, possession etc."
    )

    st.markdown("---")
    st.markdown("**API Key**")
    api_key = st.text_input("Anthropic API Key", type="password",
                             help="Required for AI Squad Analysis")

    st.markdown("---")
    st.markdown("**League Filters**")
    min_ls = st.slider("Min league strength", 0, 100, 0)
    max_ls = st.slider("Max league strength", 0, 100, 100)

    st.markdown("---")
    st.markdown("**Realism**")
    league_weight  = st.slider("League quality weight", 0.0, 1.0, 0.35, 0.05)
    market_weight  = st.slider("Market value weight",   0.0, 1.0, 0.20, 0.05)
    min_minutes    = st.slider("Min minutes (candidates)", 0, 3000, 500, 100)

    top_n = st.number_input("Results to show", 5, 50, 20, 5)

# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════
@st.cache_data
def load_players(files):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'Position' in df.columns:
                df['_pos_group'] = df['Position'].apply(detect_position_group)
            dfs.append(df)
        except Exception as e:
            st.warning(f"Could not load {f.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['Player','Team'], keep='first')
    for col in ['Age','Minutes played','Market value']:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')
    return combined

@st.cache_data
def load_teams(f):
    try:
        df = pd.read_csv(f)
        return df
    except:
        return pd.DataFrame()

player_df = load_players(player_files) if player_files else pd.DataFrame()
team_df   = load_teams(team_file) if team_file else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="cf-header">
  <h1>🏟️ CLUB FIT <span class="cf-accent">FINDER</span></h1>
  <p>Player-to-club stylistic similarity · Team data blended · AI squad intelligence</p>
</div>
""", unsafe_allow_html=True)

if player_df.empty:
    st.info("Upload player CSV files in the sidebar to get started. You can upload multiple position files at once.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════
# PLAYER SELECTION
# ══════════════════════════════════════════════════════════════════════
col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    search_q = st.text_input("Search player", placeholder="Type name or team...")

with col_sel2:
    pos_filter_sel = st.selectbox("Filter by position group",
        ["All"] + list(CF_FEATURES.keys()))

# Filter options
pdf_search = player_df.copy()
if search_q:
    mask = (pdf_search['Player'].astype(str).str.contains(search_q, case=False, na=False) |
            pdf_search.get('Team', pd.Series()).astype(str).str.contains(search_q, case=False, na=False))
    pdf_search = pdf_search[mask]
if pos_filter_sel != "All" and '_pos_group' in pdf_search.columns:
    pdf_search = pdf_search[pdf_search['_pos_group'] == pos_filter_sel]

player_options = []
if not pdf_search.empty:
    for _, row in pdf_search.head(200).iterrows():
        label = f"{row.get('Player','?')}  ·  {row.get('Team','?')}  ·  {row.get('League','?')}"
        player_options.append((label, row.get('Player')))

if not player_options:
    st.warning("No players found. Try a different search.")
    st.stop()

selected_label = st.selectbox("Select target player",
    [x[0] for x in player_options])
selected_name = dict(player_options)[selected_label]

# Get player row
target_rows = player_df[player_df['Player'] == selected_name]
if target_rows.empty:
    st.error("Player not found in dataset.")
    st.stop()
target_row = target_rows.sort_values('Minutes played', ascending=False).iloc[0]
pos_group  = detect_position_group(target_row.get('Position','CM'))
features   = CF_FEATURES.get(pos_group, CF_FEATURES['CM'])
avail_feats = [f for f in features if f in player_df.columns]

# Player card
age     = target_row.get('Age','—')
team    = target_row.get('Team','—')
league  = target_row.get('League','—')
mins    = target_row.get('Minutes played','—')
mv      = fmt_mv(target_row.get('Market value'))
pos     = target_row.get('Position','—')
ls      = LEAGUE_STRENGTHS.get(str(league), 50.0)
contract = target_row.get('Contract expires','—')

st.markdown(f"""
<div class="player-card">
  <div class="name">{selected_name}</div>
  <div class="meta">
    {team} · {league} · {pos} · Age {age} · {mins} mins · MV {mv}
    {f' · Contract: {contract}' if str(contract) not in ('nan','—','') else ''}
    · <span style="color:#3b82f6">League Strength: {ls:.0f}</span>
  </div>
</div>
""", unsafe_allow_html=True)

if len(avail_feats) < 3:
    st.error(f"Insufficient feature columns for position group {pos_group}. Check your CSV.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════
# STYLE BLEND TOGGLES
# ══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Style Blend</div>', unsafe_allow_html=True)
st.caption("Select any combination of styles to blend team-level tactical signals into the fit score.")

style_names = ["Similar to Current System", "Possession", "Pressing", "Direct",
               "Passing Based", "High Attacking Territory", "Cross Heavy"]
selected_styles = st.multiselect(
    "Active styles",
    style_names,
    default=["Similar to Current System"],
    help="Multiple styles are blended equally. 'Similar to Current System' compares against your player's actual club tactics."
)
style_active = {s: (s in selected_styles) for s in style_names}

# ══════════════════════════════════════════════════════════════════════
# CANDIDATE POOL + SETTINGS
# ══════════════════════════════════════════════════════════════════════
with st.expander("⚙️ Advanced Settings", expanded=False):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        all_leagues = sorted(player_df['League'].dropna().unique().tolist())
        selected_leagues = st.multiselect("Candidate leagues",
            all_leagues, default=all_leagues)
        max_age_cand = st.slider("Max candidate age", 16, 45, 40)

    with adv_col2:
        mv_override = st.number_input("Market value override (£)", 0, 100_000_000, 0, 500_000)
        team_blend_weight = st.slider(
            "Team stats blend weight",
            0.0, 1.0, 0.30, 0.05,
            help="How much team tactical stats influence fit vs pure player similarity"
        )

    st.markdown("**Per-metric weights** (advanced)")
    weight_cols = st.columns(3)
    custom_weights = {}
    pos_defaults = DEFAULT_WEIGHTS.get(pos_group, {})
    for i, feat in enumerate(avail_feats):
        with weight_cols[i % 3]:
            default_w = pos_defaults.get(feat, 1)
            custom_weights[feat] = st.slider(
                feat[:30], 0, 5, default_w, key=f"w_{feat}")

# ══════════════════════════════════════════════════════════════════════
# RUN BUTTON
# ══════════════════════════════════════════════════════════════════════
run_fit = st.button("🔍  Find Club Fits", type="primary", use_container_width=True)

if not run_fit:
    st.stop()

# ══════════════════════════════════════════════════════════════════════
# COMPUTE
# ══════════════════════════════════════════════════════════════════════
with st.spinner("Computing club fits..."):

    # ── Candidate pool
    cand = player_df.copy()
    if selected_leagues:
        cand = cand[cand['League'].isin(selected_leagues)]
    if '_pos_group' in cand.columns:
        cand = cand[cand['_pos_group'] == pos_group]
    cand = cand[cand['Minutes played'].fillna(0) >= min_minutes]
    cand = cand[cand['Age'].fillna(99) <= max_age_cand]
    cand = cand.dropna(subset=avail_feats)

    if cand.empty:
        st.error("No candidates after filters. Widen league selection or relax filters.")
        st.stop()

    # ── Player similarity — vectorised percentile + z-score blend
    w_vec = np.array([custom_weights.get(f, 1) for f in avail_feats], dtype=float)
    w_vec = w_vec / (w_vec.sum() or 1.0)

    cand = cand.reset_index(drop=True)
    for f in avail_feats:
        cand[f] = pd.to_numeric(cand[f], errors='coerce')

    target_in_pool = player_df[player_df['Player'] == selected_name].copy()
    full_pool_sim  = pd.concat([cand, target_in_pool], ignore_index=True)\
                      .drop_duplicates(subset=['Player','Team'])\
                      .reset_index(drop=True)
    for f in avail_feats:
        full_pool_sim[f] = pd.to_numeric(full_pool_sim[f], errors='coerce')

    # Single groupby percentile rank call
    pct_df  = full_pool_sim.groupby('League')[avail_feats].rank(pct=True)
    tgt_mask = full_pool_sim['Player'] == selected_name
    tgt_pct  = pct_df.loc[tgt_mask].mean(axis=0).fillna(0.5).values
    cand_pct = pct_df.loc[list(range(len(cand)))].fillna(0.5).values

    pct_dist    = np.sum(np.abs(cand_pct - tgt_pct) * w_vec, axis=1)
    sim_pct_arr = np.exp(-2.8 * pct_dist) * 100.0

    cand_vals_raw = cand[avail_feats].fillna(0)
    tgt_vals_raw  = pd.to_numeric(
        pd.Series(target_row[avail_feats].values, index=avail_feats),
        errors='coerce').fillna(0).values.reshape(1, -1)
    all_fit = pd.concat([
        cand_vals_raw,
        pd.DataFrame(tgt_vals_raw, columns=avail_feats)
    ], ignore_index=True)
    scaler_sim  = StandardScaler().fit(all_fit)
    cand_std    = scaler_sim.transform(cand_vals_raw)
    tgt_std     = scaler_sim.transform(tgt_vals_raw)
    act_dist    = np.sum(np.abs(cand_std - tgt_std) * w_vec, axis=1)
    sim_act_arr = np.exp(-0.6 * act_dist) * 100.0

    cand = cand.copy()
    cand['_raw_sim'] = sim_pct_arr * 0.5 + sim_act_arr * 0.5

    # Club profile: aggregate to team level
    club_profiles = cand.groupby('Team')[avail_feats].mean().reset_index()
    team_league_map_local = (cand.groupby('Team')['League']
                       .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]))
    team_mv_map   = cand.groupby('Team')['Market value'].mean()
    team_sim_map  = cand.groupby('Team')['_raw_sim'].mean()

    club_profiles['League']  = club_profiles['Team'].map(team_league_map_local)
    club_profiles['Avg MV']  = club_profiles['Team'].map(team_mv_map)
    club_profiles['Player Sim %'] = club_profiles['Team'].map(team_sim_map).round(2)
    club_profiles = club_profiles.dropna(subset=['Avg MV'])

    # ── Team style score (vectorised — no row-by-row loops)
    team_style_score = np.full(len(club_profiles), 50.0)
    team_data_available = not team_df.empty and 'Team' in team_df.columns

    active_non_system = [s for s, v in style_active.items()
                         if v and s != "Similar to Current System"]

    if team_data_available and (active_non_system or style_active.get("Similar to Current System")):
        avail_team_cols = [c for c in TEAM_STAT_COLS if c in team_df.columns]
        tdf = team_df.copy()
        for c in avail_team_cols:
            tdf[c] = pd.to_numeric(tdf[c], errors='coerce')

        # Map each club_profile team → its team stats row
        tdf_indexed = tdf.set_index('Team')
        teams_in_results = club_profiles['Team'].values
        style_parts = []

        # ── "Similar to Current System" — vectorised distance from target club
        if style_active.get("Similar to Current System"):
            tts = tdf[tdf['Team'] == team]
            if not tts.empty and avail_team_cols:
                sim_cols = avail_team_cols
                all_v = tdf[sim_cols].fillna(0)
                ts = StandardScaler().fit(all_v)
                t_std = ts.transform(
                    pd.to_numeric(tts.iloc[0][sim_cols], errors='coerce')
                    .fillna(0).values.reshape(1,-1)
                )[0]
                # Get vector for each result team
                club_team_vals = pd.DataFrame(
                    [tdf_indexed.loc[t, sim_cols].fillna(0).values.astype(float)
                     if t in tdf_indexed.index else np.zeros(len(sim_cols), dtype=float)
                     for t in teams_in_results],
                    columns=sim_cols, dtype=float
                )
                c_std_all = ts.transform(club_team_vals.values)
                dists = np.linalg.norm(c_std_all - t_std, axis=1)
                all_dists_full = np.linalg.norm(
                    ts.transform(all_v.fillna(0)) - t_std, axis=1)
                rng2 = float(all_dists_full.max() - all_dists_full.min()) or 1.0
                sys_score = (1 - (dists - all_dists_full.min()) / rng2) * 100
                style_parts.append(np.clip(sys_score, 0, 100))

        # ── Tactical style blends — vectorised percentile per column
        for sname in active_non_system:
            blend_def = STYLE_BLENDS.get(sname, {})
            col_scores = []
            for col, direction in blend_def.items():
                if col not in tdf.columns: continue
                col_series = tdf[col].dropna()
                if col_series.empty: continue
                # Get value for each result team
                vals = np.array([
                    float(tdf_indexed.loc[t, col])
                    if t in tdf_indexed.index and pd.notna(tdf_indexed.loc[t, col])
                    else np.nan
                    for t in teams_in_results
                ])
                # Vectorised percentile rank
                pct = np.array([
                    float((col_series <= v).mean() * 100) if not np.isnan(v) else 50.0
                    for v in vals
                ])
                if direction < 0:
                    pct = 100 - pct
                col_scores.append(pct * abs(direction))
            if col_scores:
                style_parts.append(np.mean(col_scores, axis=0))

        if style_parts:
            team_style_score = np.mean(style_parts, axis=0)

    # ── Blend player sim + team style
    any_style_active = any(style_active.values())
    if any_style_active and team_data_available:
        combined_style = (club_profiles['Player Sim %'].values * (1 - team_blend_weight) +
                          team_style_score * team_blend_weight)
    else:
        combined_style = club_profiles['Player Sim %'].values.copy()

    club_profiles['Style Fit %'] = combined_style.round(2)

    # ── League quality adjustment
    club_profiles['LS'] = club_profiles['League'].map(LEAGUE_STRENGTHS).fillna(50.0)
    club_profiles = club_profiles[
        (club_profiles['LS'] >= min_ls) & (club_profiles['LS'] <= max_ls)]

    if club_profiles.empty:
        st.error("No teams remain after league strength filter.")
        st.stop()

    ratio = (club_profiles['LS'] / ls).clip(0.5, 1.2)
    adj   = (club_profiles['Style Fit %'] * (1 - league_weight) +
             club_profiles['Style Fit %'] * ratio * league_weight)
    gap   = (club_profiles['LS'] - ls).clip(lower=0)
    adj  *= (1 - gap / 100).clip(lower=0.7)
    club_profiles['Adj Fit %'] = adj

    # ── Market value realism
    target_mv = (float(mv_override) if mv_override > 0
                 else float(target_row.get('Market value') or 2_000_000))
    mv_ratio  = (club_profiles['Avg MV'] / target_mv).clip(0.5, 1.5)
    mv_score  = (1 - abs(1 - mv_ratio)) * 100

    club_profiles['Final Fit %'] = (
        club_profiles['Adj Fit %'] * (1 - market_weight) +
        mv_score * market_weight
    ).round(2)

    # ── Sort + top N
    results = club_profiles[['Team','League','LS','Player Sim %','Style Fit %','Final Fit %','Avg MV']]\
        .sort_values('Final Fit %', ascending=False)\
        .reset_index(drop=True)\
        .head(int(top_n))
    results.insert(0, 'Rank', range(1, len(results)+1))

# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests, io as _io

st.markdown(f'<div class="section-head">Top {int(top_n)} Club Fits — {selected_name}</div>',
            unsafe_allow_html=True)

active_styles = [s for s, v in style_active.items() if v]
if active_styles:
    tags = "".join(f'<span class="metric-pill">{s}</span>' for s in active_styles)
    st.markdown(f'<div style="margin-bottom:12px">Active styles: {tags}</div>',
                unsafe_allow_html=True)

# ── Load fotmob badge helper
try:
    from team_fotmob_urls import FOTMOB_TEAM_URLS as _FOTMOB_URLS
except Exception:
    _FOTMOB_URLS = {}

@st.cache_data(show_spinner=False)
def _load_img(url):
    try:
        r = requests.get(url, timeout=6); r.raise_for_status()
        return plt.imread(_io.BytesIO(r.content))
    except: return None

def _badge_url(team):
    raw = (_FOTMOB_URLS.get(team) or "").strip()
    if not raw: return ""
    import re as _re2
    m = _re2.search(r"/teams/(\d+)/", raw)
    return f"https://images.fotmob.com/image_resources/logo/teamlogo/{m.group(1)}.png" if m else ""

def _get_badge(team):
    url = _badge_url(team)
    return _load_img(url) if url else None

# ── Ranking image (team_hq style)
def make_ranking_image(df_show, player_name, theme="Dark"):
    if df_show.empty: return None

    BG    = "#0a0e1a" if theme=="Dark" else "#f0f2f5"
    ROW_A = "#0f1628" if theme=="Dark" else "#f7f8fa"
    ROW_B = "#0b1222" if theme=="Dark" else "#ffffff"
    TXT   = "#ffffff" if theme=="Dark" else "#111111"
    SUB   = "#94a3b8" if theme=="Dark" else "#64748b"
    DIV   = "#1e2d42" if theme=="Dark" else "#e2e8f0"
    BAR_BG= "#1a2540" if theme=="Dark" else "#e1e1e1"
    BAR_FG= "#3b82f6"
    GOLD  = "#f59e0b"
    RANK_BG = "#111a2e" if theme=="Dark" else "#f3f3f3"
    RANK_EDGE="#2b3a5a" if theme=="Dark" else "#c0c0c0"

    N = len(df_show)
    ROW_H = 0.90; HEADER_H = 1.80; FOOT_H = 0.50
    TOTAL_H = HEADER_H + N * ROW_H + FOOT_H

    fig = plt.figure(figsize=(8.5, TOTAL_H), dpi=200)
    ax  = fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,TOTAL_H); ax.axis("off")
    ax.add_patch(Rectangle((0,0),1,TOTAL_H,color=BG,zorder=0))

    # Header
    ty = TOTAL_H - 0.22
    ax.text(0.04, ty,      "CLUB FIT FINDER",  fontsize=18, fontweight="bold", color=TXT, ha="left", va="top")
    ax.text(0.04, ty-0.34, player_name.upper(), fontsize=13, fontweight="bold", color=BAR_FG, ha="left", va="top")
    if active_styles:
        ax.text(0.04, ty-0.62, "  ·  ".join(active_styles), fontsize=10, color=SUB, ha="left", va="top")
    base_y = TOTAL_H - HEADER_H
    ax.plot([0.04,0.96],[base_y+ROW_H/2+0.02]*2, color=DIV, lw=1.1, zorder=2)

    vals = df_show['Final Fit %'].values
    max_v = float(vals.max()) if vals.max() > 0 else 1.0

    for i, (_, row) in enumerate(df_show.iterrows()):
        y = base_y - i * ROW_H
        bg = ROW_A if i % 2 == 0 else ROW_B
        ax.add_patch(Rectangle((0.04, y-ROW_H/2), 0.92, ROW_H, color=bg, zorder=1))

        # Rank badge
        is_top3 = i < 3
        rank_c = GOLD if is_top3 else RANK_EDGE
        ax.scatter([0.08],[y], s=550, facecolor=RANK_BG,
                   edgecolor=rank_c, linewidths=1.5, zorder=4)
        ax.text(0.08, y, str(i+1), fontsize=10, fontweight="bold",
                color=GOLD if is_top3 else TXT, ha="center", va="center", zorder=5)

        # Badge
        badge = _get_badge(str(row.get('Team','')))
        if badge is not None:
            h,w = badge.shape[0], badge.shape[1]
            z = 38.0 / max(h,w)
            ax.add_artist(AnnotationBbox(OffsetImage(badge,zoom=z),(0.155,y),frameon=False,zorder=5))

        # Team name + league
        ax.text(0.22, y+0.14, str(row.get('Team','')).upper(),
                fontsize=15, fontweight="bold", color=TXT, ha="left", va="center", zorder=5)
        ax.text(0.22, y-0.12, f"{row.get('League','')}  ·  Strength {row['LS']:.0f}  ·  Avg MV {fmt_mv(row['Avg MV'])}",
                fontsize=9.5, color=SUB, ha="left", va="center", zorder=5)

        # Bar
        frac = max(0.0, min(1.0, float(row['Final Fit %']) / max_v))
        BAR_L, BAR_W_MAX, BAR_H = 0.62, 0.22, 0.16
        ax.add_patch(Rectangle((BAR_L, y-BAR_H/2), BAR_W_MAX, BAR_H, color=BAR_BG, zorder=2))
        ax.add_patch(Rectangle((BAR_L, y-BAR_H/2), BAR_W_MAX*frac, BAR_H, color=BAR_FG, zorder=3))

        # Sim score small
        ax.text(0.877, y+0.14, f"sim {row['Player Sim %']:.0f}%",
                fontsize=8, color=SUB, ha="right", va="center", zorder=5)
        # Fit score
        fc = score_color(float(row['Final Fit %']))
        ax.text(0.94, y, f"{row['Final Fit %']:.0f}",
                fontsize=16, fontweight="bold", color=fc, ha="right", va="center", zorder=6)

    # Footer
    ax.plot([0.04,0.96],[0.82]*2, color=DIV, lw=0.9, zorder=2)
    ax.text(0.04, 0.60, "Final Fit % = Player Similarity · League Quality · Market Value realism", fontsize=8.5, color=SUB, va="top")

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

img_theme = st.radio("Image theme", ["Dark","Light"], horizontal=True, key="img_theme")
rank_img = make_ranking_image(results, selected_name, theme=img_theme)
if rank_img:
    st.image(rank_img, use_column_width=True)
    st.download_button("⬇️ Download Ranking Image", rank_img,
                       f"club_fit_{selected_name.replace(' ','_')}.png", "image/png")

# ── Card view (detailed)
with st.expander("📋 Detailed cards + table", expanded=False):
    for _, row in results.iterrows():
        rank    = int(row['Rank'])
        rteam   = row['Team']
        rleague = row['League']
        rls     = row['LS']
        sim_pct = row['Player Sim %']
        fit_pct = row['Final Fit %']
        avg_mv  = fmt_mv(row['Avg MV'])
        fc      = score_color(fit_pct)
        rank_cls= "top" if rank <= 3 else ""
        bar_w   = int(fit_pct)

        st.markdown(f"""
<div class="result-card">
  <div class="result-rank {rank_cls}">#{rank}</div>
  <div style="flex:1">
    <div class="result-team">{rteam}</div>
    <div class="result-league">{rleague} · Strength {rls:.0f} · Avg MV {avg_mv}</div>
    <div style="margin-top:6px">
      <div class="score-bar-wrap">
        <div class="score-bar" style="width:{bar_w}%;background:{fc}"></div>
      </div>
    </div>
  </div>
  <div>
    <div style="font-size:0.72rem;color:#64748b;text-align:right">Similarity</div>
    <div style="font-size:0.85rem;color:#94a3b8;text-align:right">{sim_pct:.0f}%</div>
  </div>
  <div class="result-score" style="color:{fc}">{fit_pct:.0f}<span style="font-size:0.9rem;color:#64748b">%</span></div>
</div>
""", unsafe_allow_html=True)

    st.dataframe(results.style.format({
        'Final Fit %': '{:.1f}', 'Player Sim %': '{:.1f}',
        'Style Fit %': '{:.1f}', 'LS': '{:.0f}',
    }), use_container_width=True)

# Download CSV
csv_out = results.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Results CSV", csv_out,
                   f"club_fit_{selected_name.replace(' ','_')}.csv",
                   "text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# AI SQUAD ANALYSIS BOX
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-head">🤖 AI Squad Analysis</div>', unsafe_allow_html=True)
st.caption("Select a club from the results to get AI-powered squad depth analysis, age profile, contract risk, and player fit narrative.")

if not api_key:
    st.warning("Add your Anthropic API key in the sidebar to enable AI Squad Analysis.")
else:
    top_teams = results['Team'].tolist()[:10]
    ai_team = st.selectbox("Analyse squad for:", top_teams)
    run_ai  = st.button("🧠  Generate AI Squad Analysis", use_container_width=True)

    if run_ai:
        # Build squad data for selected team
        squad = player_df[player_df['Team'] == ai_team].copy()
        squad['_pos_group'] = squad['Position'].apply(detect_position_group)
        squad['Market value'] = pd.to_numeric(squad.get('Market value', 0), errors='coerce').fillna(0)
        squad['Age'] = pd.to_numeric(squad.get('Age', 0), errors='coerce')
        squad['Minutes played'] = pd.to_numeric(squad.get('Minutes played', 0), errors='coerce').fillna(0)

        # Position depth
        depth = squad.groupby('_pos_group').size().to_dict()

        # Age profile
        avg_age_squad = squad['Age'].mean()
        u23 = (squad['Age'] < 23).sum()
        o28 = (squad['Age'] > 28).sum()

        # High performers (top minutes in each pos group)
        top_performers = []
        for pg in squad['_pos_group'].unique():
            pg_squad = squad[squad['_pos_group'] == pg].sort_values('Minutes played', ascending=False)
            if not pg_squad.empty:
                p = pg_squad.iloc[0]
                top_performers.append(f"{p.get('Player','?')} ({pg}, {p.get('Minutes played',0):.0f} mins, MV {fmt_mv(p.get('Market value',0))})")

        # Contract info
        contract_col = next((c for c in squad.columns if 'contract' in c.lower() or 'expir' in c.lower()), None)
        contract_info = ""
        if contract_col:
            expiring = squad[pd.to_numeric(squad[contract_col].astype(str).str[:4], errors='coerce').fillna(2099) <= 2026]
            if not expiring.empty:
                contract_info = "Expiring 2026: " + ", ".join(expiring['Player'].head(5).tolist())

        # Team stats
        team_stats_str = ""
        if team_data_available:
            trow = team_df[team_df['Team'] == ai_team]
            if not trow.empty:
                tr = trow.iloc[0]
                team_stats_str = (
                    f"PPDA: {tr.get('PPDA','—')}, Possession: {tr.get('Possession %','—')}%, "
                    f"xG p90: {tr.get('xG p90','—')}, Passes p90: {tr.get('Passes p90','—')}"
                )

        # Club fit rank
        club_rank = results[results['Team'] == ai_team]
        fit_score = club_rank['Final Fit %'].iloc[0] if not club_rank.empty else "—"

        prompt = f"""You are a head of recruitment at a Premier League club presenting a concise squad analysis.

TARGET PLAYER: {selected_name}
Position: {pos_group} | League: {league} | Age: {age} | MV: {mv}

DESTINATION CLUB: {ai_team}
Club Fit Score: {fit_score}% (for {selected_name})
{f'Team Stats: {team_stats_str}' if team_stats_str else ''}

SQUAD DATA:
Position depth: {depth}
Avg squad age: {avg_age_squad:.1f} | U23 players: {u23} | Over 28: {o28}
Key players by minutes: {'; '.join(top_performers[:6])}
{contract_info}

Write a concise 4-section analysis using EXACTLY this structure:

SQUAD DEPTH: One sentence on whether {ai_team} has a vacancy or logjam at the {pos_group} position. Name the current incumbent if obvious.

AGE PROFILE: One sentence on the squad's age shape and what that means for signing {selected_name} (development runway vs immediate need).

DEPARTURE RISK: One sentence identifying which high-value player(s) at their position could leave, creating an opening. Be specific about names and values.

FIT VERDICT: One decisive sentence. SIGN / MONITOR / PASS and the single clearest reason related to the {fit_score}% fit score.

Be specific, use the data, no filler."""

        with st.spinner(f"Analysing {ai_team} squad..."):
            try:
                client = anthropic.Anthropic(api_key=api_key)
                resp   = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=600,
                    messages=[{"role": "user", "content": prompt}]
                )
                analysis = resp.content[0].text.strip()

                sections = {}
                current_key = None
                current_lines = []
                for line in analysis.split('\n'):
                    line = line.strip()
                    if not line: continue
                    for key in ["SQUAD DEPTH", "AGE PROFILE", "DEPARTURE RISK", "FIT VERDICT"]:
                        if line.upper().startswith(key):
                            if current_key:
                                sections[current_key] = " ".join(current_lines)
                            current_key = key
                            rest = line[len(key):].lstrip(':').strip()
                            current_lines = [rest] if rest else []
                            break
                    else:
                        if current_key:
                            current_lines.append(line)
                if current_key:
                    sections[current_key] = " ".join(current_lines)

                icons = {
                    "SQUAD DEPTH": "👥",
                    "AGE PROFILE": "📊",
                    "DEPARTURE RISK": "⚠️",
                    "FIT VERDICT": "✅"
                }
                border_colors = {
                    "SQUAD DEPTH": "#3b82f6",
                    "AGE PROFILE": "#8b5cf6",
                    "DEPARTURE RISK": "#f59e0b",
                    "FIT VERDICT": "#22c55e"
                }

                st.markdown(f"### {ai_team} — Squad Analysis")
                for key in ["SQUAD DEPTH", "AGE PROFILE", "DEPARTURE RISK", "FIT VERDICT"]:
                    content = sections.get(key, "Data not available.")
                    bc = border_colors.get(key, "#3b82f6")
                    icon = icons.get(key, "")
                    st.markdown(f"""
<div class="ai-card" style="border-left-color:{bc}">
  <h4>{icon} {key}</h4>
  <p>{content}</p>
</div>
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"AI analysis failed: {e}")
