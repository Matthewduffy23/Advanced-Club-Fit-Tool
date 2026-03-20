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
    min_ls = st.slider("Min league strength", 0, 100, 0)
    max_ls = st.slider("Max league strength", 0, 100, 100)
    league_weight = st.slider("League quality weight", 0.0, 1.0, 0.35, 0.05)
    market_weight = st.slider("Market value weight",   0.0, 1.0, 0.20, 0.05)
    min_mins      = st.slider("Min minutes (candidates)", 0, 3000, 500, 100)
    top_n         = st.number_input("Results to show", 5, 50, 10, 5)
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

@st.cache_data(show_spinner=False)
def load_teams(f):
    try: return pd.read_csv(f)
    except: return pd.DataFrame()

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
tgt_ls     = float(LEAGUE_STRENGTHS.get(tgt_league, 50))

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
        all_lgs  = sorted(player_df['League'].dropna().unique().tolist())
        cand_lgs = st.multiselect("Candidate leagues", all_lgs, default=all_lgs)
        max_age  = st.slider("Max candidate age", 16, 45, 40)
    with ac2:
        mv_override      = st.number_input("MV override (£)", 0, 100_000_000, 0, 500_000)
        style_blend_w    = st.slider("Team style blend weight", 0.0, 1.0, 0.30, 0.05)

    st.markdown("**Metric weights**")
    wc = st.columns(3)
    defs = DEFAULT_WEIGHTS.get(pg, {})
    custom_w = {}
    for i, f in enumerate(feats):
        with wc[i % 3]:
            custom_w[f] = st.slider(f[:28], 0, 5, defs.get(f, 1), key=f"w_{f}")

# ── RUN ───────────────────────────────────────────────────────────────
run = st.button("🔍  Find Club Fits", type="primary", use_container_width=True)
if not run:
    st.stop()

# ── COMPUTE ───────────────────────────────────────────────────────────
with st.spinner("Computing…"):

    cand = player_df.copy()
    if cand_lgs:
        cand = cand[cand['League'].isin(cand_lgs)]
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

    ref = pd.concat([cand[feats + ['League']], tgt_df[feats + ['League']].head(1)],
                    ignore_index=True)
    pct_mat = ref.groupby('League')[feats].rank(pct=True).fillna(0.5)

    n_cand = len(cand)
    tgt_pct = pct_mat.iloc[n_cand:].mean(axis=0).values
    cand_pct = pct_mat.iloc[:n_cand].values

    pct_dist = np.sum(np.abs(cand_pct - tgt_pct) * w, axis=1)
    sim_pct  = np.exp(-2.8 * pct_dist) * 100.0

    X_all = np.vstack([cand[feats].values, tgt_vec.reshape(1,-1)])
    sc    = StandardScaler().fit(X_all)
    c_std = sc.transform(cand[feats].values)
    t_std = sc.transform(tgt_vec.reshape(1,-1))
    act_dist = np.sum(np.abs(c_std - t_std) * w, axis=1)
    sim_act  = np.exp(-0.6 * act_dist) * 100.0

    cand['_sim'] = sim_pct * 0.5 + sim_act * 0.5

    grp = cand.groupby('Team')
    club = grp[feats].mean().reset_index()
    club['League'] = grp['League'].agg(lambda x: x.mode().iloc[0])
    if 'Market value' in cand.columns:
        mv_series = pd.to_numeric(cand['Market value'], errors='coerce')
        median_mv = mv_series.median() or 2_000_000
        cand['_mv'] = mv_series.fillna(median_mv)
    else:
        cand['_mv'] = 2_000_000
    club['AvgMV'] = cand.groupby('Team')['_mv'].mean()
    club['AvgMV'] = club['AvgMV'].fillna(2_000_000)
    club['SimPct'] = cand.groupby('Team')['_sim'].mean()

    if club.empty:
        st.error("No clubs with complete data.")
        st.stop()

    has_team = not team_df.empty and 'Team' in team_df.columns
    style_scores = np.full(len(club), 50.0)

    if has_team and sel_styles:
        tdf = team_df.drop_duplicates(subset=['Team']).set_index('Team')
        parts = []

        if "Similar to Current System" in sel_styles and tgt_team in tdf.index:
            num_cols = [c for c in tdf.columns if tdf[c].dtype in [float, int] or
                        pd.api.types.is_numeric_dtype(tdf[c])]
            num_cols = [c for c in num_cols if c != 'Team']
            if num_cols:
                tdf_num = tdf[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                sc2 = StandardScaler().fit(tdf_num)
                t_row = tdf_num.loc[tgt_team].values.reshape(1,-1)
                t_std2 = sc2.transform(t_row)[0]
                part = []
                for team_name in club['Team']:
                    if team_name in tdf_num.index:
                        c_row = tdf_num.loc[team_name].values.reshape(1,-1)
                        c_std2 = sc2.transform(c_row)[0]
                        d = float(np.linalg.norm(c_std2 - t_std2))
                        all_d = np.linalg.norm(sc2.transform(tdf_num) - t_std2, axis=1)
                        rng = float(all_d.max() - all_d.min()) or 1.0
                        part.append(float((1 - (d - all_d.min()) / rng) * 100))
                    else:
                        part.append(50.0)
                parts.append(np.array(part))

        for sname in sel_styles:
            if sname == "Similar to Current System": continue
            blend = STYLE_BLENDS.get(sname, {})
            col_parts = []
            for col, direction in blend.items():
                if col not in tdf.columns: continue
                col_vals = pd.to_numeric(tdf[col], errors='coerce').dropna()
                if col_vals.empty: continue
                part = []
                for team_name in club['Team']:
                    if team_name in tdf.index:
                        v = pd.to_numeric(tdf.loc[team_name, col], errors='coerce')
                        if pd.isna(v):
                            part.append(50.0)
                        else:
                            pct = float((col_vals <= float(v)).mean() * 100)
                            part.append(100 - pct if direction < 0 else pct)
                    else:
                        part.append(50.0)
                col_parts.append(np.array(part) * abs(direction))
            if col_parts:
                parts.append(np.mean(col_parts, axis=0))

        if parts:
            style_scores = np.mean(parts, axis=0)

    if sel_styles and has_team:
        combined = (club['SimPct'].values * (1 - style_blend_w) +
                    style_scores * style_blend_w)
    else:
        combined = club['SimPct'].values.copy()

    club['StyleFit'] = combined

    club['LS'] = club['League'].map(LEAGUE_STRENGTHS).fillna(50.0)
    club = club[(club['LS'] >= min_ls) & (club['LS'] <= max_ls)]
    if club.empty:
        st.error("No clubs after league filter.")
        st.stop()

    ratio = (club['LS'] / tgt_ls).clip(0.5, 1.2)
    adj   = club['StyleFit'] * (1 - league_weight) + club['StyleFit'] * ratio * league_weight
    gap   = (club['LS'] - tgt_ls).clip(lower=0)
    adj  *= (1 - gap / 100).clip(lower=0.7)

    tgt_mv = float(mv_override) if mv_override > 0 else float(pd.to_numeric(tgt.get('Market value', 0), errors='coerce') or 2_000_000)
    tgt_mv = max(tgt_mv, 1.0)
    mv_ratio = (club['AvgMV'] / tgt_mv).clip(0.5, 1.5)
    mv_score = (1 - abs(1 - mv_ratio)) * 100

    club['FinalFit'] = (adj * (1 - market_weight) + mv_score * market_weight).round(1)

    results = club[['Team','League','LS','SimPct','FinalFit','AvgMV']]\
        .sort_values('FinalFit', ascending=False)\
        .reset_index(drop=True).head(int(top_n))
    results['FinalFit'] = results['FinalFit'].fillna(0).round(1)
    results['SimPct']   = results['SimPct'].fillna(0).round(1)
    results.insert(0,'Rank', range(1, len(results)+1))

# ── DISPLAY ───────────────────────────────────────────────────────────
st.markdown(f'<div class="sec">Top {int(top_n)} Club Fits — {sel_name}</div>', unsafe_allow_html=True)
if sel_styles:
    st.markdown("Active styles: " + "".join(f'<span class="pill">{s}</span>' for s in sel_styles),
                unsafe_allow_html=True)

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
def make_ranking_img(df_show, player_name, active_styles, theme="Light", export_mode="Standard (auto)"):
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
    else:  # Light
        BG        = "#ffffff"
        ROW_A     = "#f8fafc"
        ROW_B     = "#ffffff"
        TXT       = "#0f172a"
        SUB       = "#475569"
        FOOT      = "#94a3b8"
        DIV       = "#cbd5e1"
        BAR_BG    = "#e2e8f0"
        BAR_FG    = "#3b82f6"
        RANK_BG   = "#f1f5f9"
        RANK_EDGE = "#94a3b8"
        HDR_ACCENT = "#2563eb"

    GOLD = "#f59e0b"
    N    = len(df_show)

    # ── 1920×1080 banner ──────────────────────────────────────────────
    if export_mode == "1920×1080 (banner)":
        DPI = 100
        fig = plt.figure(figsize=(19.2, 10.8), dpi=DPI)
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, color=BG, zorder=0))
        if theme == "Light":
            ax.add_patch(Rectangle((0, 0), 1, 1,
                                    fill=False, edgecolor="#cbd5e1", linewidth=2, zorder=10))

        LEFT, RIGHT = 0.045, 0.955

        # Header
        ax.text(LEFT, 0.972, "CLUB FIT FINDER",
                fontsize=48, fontweight="bold", color=TXT, ha="left", va="top")
        ax.text(LEFT, 0.912, player_name.upper(),
                fontsize=34, fontweight="bold", color=HDR_ACCENT, ha="left", va="top")
        style_str = "  ·  ".join(active_styles) if active_styles else ""
        if style_str:
            ax.text(LEFT, 0.870, style_str, fontsize=20, color=SUB, ha="left", va="top")
        ax.plot([LEFT, RIGHT], [0.835, 0.835], color=DIV, lw=2.2)
        ax.plot([LEFT, RIGHT], [0.040, 0.040], color=DIV, lw=2.2)
        ax.text(LEFT, 0.022,
                "Final Fit % = Player Similarity · League Quality · Market Value",
                fontsize=13, color=FOOT, ha="left", va="top")

        ROW_TOP = 0.813; ROW_BOT = 0.050
        row_gap = (ROW_TOP - ROW_BOT) / float(N)
        row_h   = row_gap * 0.92
        RANK_X  = LEFT + 0.024; CREST_X = LEFT + 0.105; NAME_X = LEFT + 0.175
        BAR_L   = LEFT + 0.62;  BAR_R   = RIGHT - 0.14
        BAR_W   = BAR_R - BAR_L; BAR_H2  = row_h * 0.26; VAL_X  = RIGHT - 0.025
        max_v   = float(df_show['FinalFit'].max()) or 1.0

        for i, (_, row) in enumerate(df_show.iterrows()):
            y  = ROW_TOP - (i + 0.5) * row_gap
            ax.add_patch(Rectangle((LEFT, y - row_h / 2), RIGHT - LEFT, row_h,
                                    color=(ROW_A if i % 2 == 0 else ROW_B), zorder=1))

            # Rank badge
            is_top3   = i < 3
            edge_col  = GOLD if is_top3 else RANK_EDGE
            rank_col  = GOLD if is_top3 else TXT
            ax.scatter([RANK_X], [y], s=1320, facecolor=RANK_BG,
                       edgecolor=edge_col, linewidths=2.2, zorder=4)
            ax.text(RANK_X, y, str(i + 1), fontsize=16, fontweight="bold",
                    color=rank_col, ha="center", va="center", zorder=5)

            bdg = _badge(str(row['Team']))
            if bdg is not None:
                h, w2 = bdg.shape[:2]; z = 52.0 / max(h, w2)
                ax.add_artist(AnnotationBbox(OffsetImage(bdg, zoom=z),
                              (CREST_X, y), frameon=False, zorder=5))

            ax.text(NAME_X, y + row_h * 0.18, str(row['Team']).upper(),
                    fontsize=28, fontweight="bold", color=TXT, ha="left", va="center", zorder=6)
            ax.text(NAME_X, y - row_h * 0.22, str(row['League']),
                    fontsize=19, color=SUB, ha="left", va="center", zorder=6)

            frac = max(0.0, min(1.0, float(row['FinalFit']) / max_v))
            ax.add_patch(Rectangle((BAR_L, y - BAR_H2 / 2), BAR_W, BAR_H2, color=BAR_BG, zorder=2))
            ax.add_patch(Rectangle((BAR_L, y - BAR_H2 / 2), BAR_W * frac, BAR_H2, color=BAR_FG, zorder=3))

            fc = score_col(float(row['FinalFit']))
            ax.text(VAL_X, y, f"{row['FinalFit']:.0f}",
                    fontsize=29, fontweight="bold", color=fc, ha="right", va="center", zorder=6)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor=BG)
        plt.close(fig); buf.seek(0)
        return buf.getvalue()

    # ── Standard (auto-height) ────────────────────────────────────────
    W_IN  = 7.2
    ROW_H = 1.10
    HDR_H = 1.65
    FT_H  = 0.55
    TOT_H = HDR_H + N * ROW_H + FT_H
    DPI   = 180

    fig = plt.figure(figsize=(W_IN, TOT_H), dpi=DPI)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_IN); ax.set_ylim(0, TOT_H); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W_IN, TOT_H, color=BG, zorder=0))
    # Subtle border so light image doesn't bleed into white page
    if theme == "Light":
        ax.add_patch(Rectangle((0, 0), W_IN, TOT_H,
                                fill=False, edgecolor="#cbd5e1", linewidth=1.5, zorder=10))

    # Header
    ty = TOT_H - 0.20
    ax.text(0.30, ty, "CLUB FIT FINDER",
            fontsize=16, fontweight="bold", color=TXT, ha="left", va="top")
    ax.text(0.30, ty - 0.36, player_name.upper(),
            fontsize=13, fontweight="bold", color=HDR_ACCENT, ha="left", va="top")
    style_str = "  ·  ".join(active_styles) if active_styles else ""
    if style_str:
        ax.text(0.30, ty - 0.65, style_str, fontsize=9, color=SUB, ha="left", va="top")

    base_y = TOT_H - HDR_H
    ax.plot([0.18, W_IN - 0.18], [base_y + ROW_H * 0.08] * 2, color=DIV, lw=0.9)

    BAR_X = W_IN - 2.50; BAR_W = 1.55; VAL_X = W_IN - 0.25
    ax.text(BAR_X + BAR_W / 2, base_y - 0.04, "FIT",
            fontsize=8, color=SUB, ha="center", va="top")
    ax.text(VAL_X, base_y - 0.04, "SCORE",
            fontsize=8, color=SUB, ha="right", va="top")

    max_v = float(df_show['FinalFit'].max()) or 1.0

    for i, (_, row) in enumerate(df_show.iterrows()):
        y  = base_y - (i + 0.5) * ROW_H
        bg = ROW_A if i % 2 == 0 else ROW_B
        ax.add_patch(Rectangle((0.14, y - ROW_H / 2), W_IN - 0.28, ROW_H,
                                color=bg, zorder=1))

        # Rank badge
        is_top3   = i < 3
        edge_col  = GOLD if is_top3 else RANK_EDGE
        rank_color = GOLD if is_top3 else TXT
        circle = plt.Circle((0.42, y), 0.32, color=RANK_BG, zorder=3)
        ax.add_patch(circle)
        theta = np.linspace(0, 2 * np.pi, 120)
        ax.plot(0.42 + 0.32 * np.cos(theta), y + 0.32 * np.sin(theta),
                color=edge_col, lw=1.6, zorder=4)
        ax.text(0.42, y, str(i + 1), fontsize=11, fontweight="bold",
                color=rank_color, ha="center", va="center", zorder=5)

        # Club badge
        bdg = _badge(str(row['Team']))
        if bdg is not None:
            h, w2 = bdg.shape[:2]
            zoom = 0.46 / max(h, w2) * DPI
            ax.add_artist(AnnotationBbox(OffsetImage(bdg, zoom=zoom),
                          (1.02, y), frameon=False, zorder=5))

        # Team name & league
        name_x = 1.42
        ax.text(name_x, y + 0.20, str(row['Team']).upper(),
                fontsize=12.5, fontweight="bold", color=TXT,
                ha="left", va="center", zorder=5)
        ax.text(name_x, y - 0.18, str(row['League']),
                fontsize=8.5, color=SUB, ha="left", va="center", zorder=5)

        # Fit bar
        BAR_H2 = 0.16
        frac   = max(0.0, min(1.0, float(row['FinalFit']) / max_v))
        ax.add_patch(Rectangle((BAR_X, y - BAR_H2 / 2), BAR_W, BAR_H2, color=BAR_BG, zorder=2))
        ax.add_patch(Rectangle((BAR_X, y - BAR_H2 / 2), BAR_W * frac, BAR_H2, color=BAR_FG, zorder=3))

        # Score
        fc = score_col(float(row['FinalFit']))
        ax.text(VAL_X, y, f"{row['FinalFit']:.0f}",
                fontsize=20, fontweight="bold", color=fc,
                ha="right", va="center", zorder=6)

    # Footer
    ax.plot([0.18, W_IN - 0.18], [FT_H + 0.10] * 2, color=DIV, lw=0.7)
    ax.text(0.30, FT_H - 0.02,
            "Final Fit % = Player Similarity · League Quality · Market Value",
            fontsize=8, color=FOOT, ha="left", va="top")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf.getvalue()


rank_img = make_ranking_img(results, sel_name, sel_styles,
                             theme=img_theme,
                             export_mode="1920×1080 (banner)" if img_format == "1920×1080" else "Standard (auto)")
if rank_img:
    st.image(rank_img, use_column_width=True)
    st.download_button("⬇️ Download Ranking Image", rank_img,
                       f"club_fit_{sel_name.replace(' ','_')}.png", "image/png")

csv_out = results.rename(columns={
    'SimPct':'Similarity %','FinalFit':'Final Fit %','AvgMV':'Avg MV','LS':'League Strength'
}).to_csv(index=False).encode()
st.download_button("⬇️ Download CSV", csv_out,
    f"club_fit_{sel_name.replace(' ','_')}.csv", "text/csv")

# ── AI SQUAD ANALYSIS ─────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec">🤖 AI Squad Analysis</div>', unsafe_allow_html=True)

if not api_key:
    st.info("Add your Anthropic API key in the sidebar to unlock squad analysis for each club.")
else:
    st.caption("Expand any club below to generate an AI squad analysis.")
    client_ai = anthropic.Anthropic(api_key=api_key)

    def run_ai_analysis(ai_team, fit_pct):
        squad = player_df[player_df['Team'] == ai_team].copy()
        if '_pg' not in squad.columns:
            squad['_pg'] = squad['Position'].apply(detect_pos)
        squad['Market value']   = pd.to_numeric(squad.get('Market value',   0), errors='coerce').fillna(0)
        squad['Age']            = pd.to_numeric(squad.get('Age',            0), errors='coerce')
        squad['Minutes played'] = pd.to_numeric(squad.get('Minutes played', 0), errors='coerce').fillna(0)

        depth   = squad.groupby('_pg').size().to_dict()
        avg_age = round(float(squad['Age'].mean()), 1) if not squad.empty else "—"
        u23     = int((squad['Age'] < 23).sum())
        o28     = int((squad['Age'] > 28).sum())

        top_players = []
        for g in squad['_pg'].unique():
            gs = squad[squad['_pg']==g].sort_values('Minutes played', ascending=False)
            if not gs.empty:
                p = gs.iloc[0]
                top_players.append(
                    f"{p.get('Player','?')} ({g}, {p.get('Minutes played',0):.0f}mins, "
                    f"MV {fmt_mv(p.get('Market value',0))})")

        high_val = squad.sort_values('Market value', ascending=False).head(5)
        high_val_str = "; ".join(
            f"{r.get('Player','?')} {fmt_mv(r.get('Market value',0))}"
            for _, r in high_val.iterrows())

        cc = next((c for c in squad.columns if 'contract' in c.lower()), None)
        expiring = ""
        if cc:
            exp = squad[pd.to_numeric(squad[cc].astype(str).str[:4],
                                      errors='coerce').fillna(2099) <= 2026]
            if not exp.empty:
                expiring = "Expiring ≤2026: " + ", ".join(exp['Player'].head(5).tolist())

        prompt = f"""You are head of recruitment at a data-driven club presenting to the board.

TARGET PLAYER: {sel_name} | {pg} | {tgt_league} | Age {tgt.get('Age','?')} | MV {fmt_mv(tgt.get('Market value'))}
DESTINATION CLUB: {ai_team} | Club Fit Score: {fit_pct}%

SQUAD DATA:
- Position depth: {depth}
- Avg age: {avg_age} | U23 players: {u23} | Over 28: {o28}
- Key players by minutes: {'; '.join(top_players[:6])}
- Highest value players: {high_val_str}
- {expiring}

Write exactly 4 sections, one sentence each. Be specific, use player names, no filler:

SQUAD DEPTH: Is there a vacancy or competition at {pg}? Name the current incumbent if clear.
AGE PROFILE: What does the squad's age shape mean for signing {sel_name} now vs waiting?
DEPARTURE RISK: Which high-value player is most likely to leave, opening a spot or budget?
FIT VERDICT: SIGN / MONITOR / PASS with one decisive reason tied to the {fit_pct}% fit score."""

        resp = client_ai.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=500,
            messages=[{"role":"user","content":prompt}])
        return resp.content[0].text.strip()

    def parse_and_render(text, ai_team):
        sections = {}
        cur_key, cur_lines = None, []
        for line in text.split('\n'):
            line = line.strip()
            if not line: continue
            matched = False
            for k in ["SQUAD DEPTH","AGE PROFILE","DEPARTURE RISK","FIT VERDICT"]:
                if line.upper().startswith(k):
                    if cur_key: sections[cur_key] = " ".join(cur_lines)
                    cur_key = k
                    rest = line[len(k):].lstrip(':').strip()
                    cur_lines = [rest] if rest else []
                    matched = True
                    break
            if not matched and cur_key:
                cur_lines.append(line)
        if cur_key: sections[cur_key] = " ".join(cur_lines)

        icons  = {"SQUAD DEPTH":"👥","AGE PROFILE":"📊","DEPARTURE RISK":"⚠️","FIT VERDICT":"✅"}
        colors = {"SQUAD DEPTH":"#3b82f6","AGE PROFILE":"#8b5cf6",
                  "DEPARTURE RISK":"#f59e0b","FIT VERDICT":"#22c55e"}
        st.markdown(f"### {ai_team} — Squad Intelligence")
        for k in ["SQUAD DEPTH","AGE PROFILE","DEPARTURE RISK","FIT VERDICT"]:
            content = sections.get(k, "—")
            bc = colors[k]
            st.markdown(
                f'<div class="acard" style="border-left-color:{bc}">'
                f'<h4>{icons[k]} {k}</h4><p>{content}</p></div>',
                unsafe_allow_html=True)

    for _, row in results.iterrows():
        team_name = row['Team']
        fit_score = row['FinalFit']
        with st.expander(
            f"#{int(row['Rank'])}  {team_name}  ·  {row['League']}  ·  "
            f"Fit {fit_score:.0f}%", expanded=False):
            if st.button(f"Generate AI Analysis — {team_name}",
                         key=f"ai_{team_name}"):
                with st.spinner(f"Analysing {team_name}…"):
                    try:
                        ai_text = run_ai_analysis(team_name, fit_score)
                        parse_and_render(ai_text, team_name)
                        st.session_state[f"ai_cache_{team_name}"] = ai_text
                    except Exception as e:
                        st.error(f"AI error: {e}")
            elif f"ai_cache_{team_name}" in st.session_state:
                parse_and_render(st.session_state[f"ai_cache_{team_name}"], team_name)
