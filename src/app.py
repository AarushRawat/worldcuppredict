import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from itertools import combinations
from src.features import load_data, get_team_stats, get_fifa_rank, load_rankings

st.set_page_config(page_title="WorldCupPredict", page_icon="🏆", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
a{text-decoration:none!important;color:inherit!important}
html,body,[class*="css"],[data-testid="stAppViewContainer"]{font-family:'Inter',sans-serif!important;background:#080808!important;color:#c8c8c8!important}
[data-testid="stAppViewContainer"]{background:#080808!important}
[data-testid="stHeader"],[data-testid="stToolbar"],footer,#MainMenu,section[data-testid="stSidebar"]{display:none!important}
.block-container{padding:0!important;max-width:100%!important}
.nb{display:flex;align-items:center;justify-content:space-between;padding:0 48px;height:56px;background:rgba(8,8,8,0.95);border-bottom:1px solid #141414}
.nb-brand{font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:#f0f0f0;letter-spacing:-0.02em}
.nb-brand span{color:#22c55e}
.nb-links{display:flex;gap:32px}
.nb-links a{font-size:13px;font-weight:400;color:#555!important;text-decoration:none!important;letter-spacing:0.02em}
.nb-links a:hover{color:#22c55e!important}
.nb-dot{width:6px;height:6px;border-radius:50%;background:#22c55e;box-shadow:0 0 8px rgba(34,197,94,0.6)}
.hero{position:relative;width:100%;min-height:420px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:72px 24px 60px;overflow:hidden;background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(34,197,94,0.08) 0%,transparent 70%),radial-gradient(ellipse 60% 40% at 80% 100%,rgba(14,165,233,0.05) 0%,transparent 60%),#080808}
.hero-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,0.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,0.018) 1px,transparent 1px);background-size:60px 60px;mask-image:radial-gradient(ellipse 70% 80% at 50% 0%,black 0%,transparent 100%)}
.hero-eyebrow{font-family:'Inter',sans-serif;font-size:11px;font-weight:500;letter-spacing:3px;text-transform:uppercase;color:#22c55e;margin-bottom:20px;opacity:0.8}
.hero-title{font-family:'Syne',sans-serif;font-size:clamp(36px,5vw,64px);font-weight:800;color:#f0f0f0;line-height:1.05;letter-spacing:-0.03em;margin-bottom:16px}
.hero-title em{color:#22c55e;font-style:normal}
.hero-sub{font-size:16px;color:#888;font-weight:300;margin-bottom:8px;letter-spacing:0.01em}
.hero-sub2{font-size:13px;color:#444;font-style:italic;margin-bottom:44px}
.stButton>button{font-family:'Inter',sans-serif!important;font-size:14px!important;font-weight:500!important;border-radius:8px!important;height:44px!important;transition:all 0.2s ease!important;letter-spacing:0.01em!important}
.stButton>button[kind="primary"]{background:#16a34a!important;border:none!important;color:#f0fdf4!important;box-shadow:0 0 20px rgba(34,197,94,0.2)!important}
.stButton>button[kind="primary"]:hover{background:#15803d!important;box-shadow:0 0 32px rgba(34,197,94,0.35)!important;transform:translateY(-1px)!important}
.stButton>button:not([kind="primary"]){background:transparent!important;border:1px solid #222!important;color:#888!important}
.stButton>button:not([kind="primary"]):hover{border-color:#333!important;color:#ccc!important;transform:translateY(-1px)!important}
.sec-head{font-family:'Syne',sans-serif;font-size:13px;font-weight:600;letter-spacing:2.5px;text-transform:uppercase;color:#555;margin-bottom:20px;display:flex;align-items:center;gap:10px}
.sec-head::after{content:'';flex:1;height:1px;background:#111}
@keyframes champIn{from{opacity:0;transform:translateY(20px) scale(0.97)}to{opacity:1;transform:none}}
.champ-wrap{display:flex;justify-content:center;padding:40px 0}
.champ-card{background:linear-gradient(135deg,rgba(22,163,74,0.08),rgba(8,8,8,0.8));border:1px solid rgba(34,197,94,0.15);border-radius:20px;padding:48px 56px;text-align:center;max-width:440px;width:100%;animation:champIn 0.7s cubic-bezier(0.16,1,0.3,1) forwards;box-shadow:0 0 60px rgba(34,197,94,0.06),0 20px 60px rgba(0,0,0,0.5)}
.champ-lbl{font-size:10px;letter-spacing:3px;text-transform:uppercase;color:#22c55e;opacity:0.6;margin-bottom:16px}
.champ-trophy{font-size:56px;margin-bottom:12px}
.champ-name{font-family:'Syne',sans-serif;font-size:38px;font-weight:800;color:#f0f0f0;letter-spacing:-0.03em;text-shadow:0 0 40px rgba(240,180,40,0.25);margin-bottom:8px}
.champ-prob{font-size:13px;color:#555}
.champ-prob span{color:#22c55e;font-weight:500}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.cont-card{background:#0d0d0d;border:1px solid #141414;border-radius:14px;padding:24px 20px;text-align:center;animation:fadeUp 0.5s ease forwards;opacity:0;transition:border-color 0.2s,box-shadow 0.2s}
.cont-card:hover{border-color:#1e1e1e;box-shadow:0 8px 32px rgba(0,0,0,0.4)}
.cont-card-gold{border-color:rgba(240,180,40,0.2)!important;box-shadow:0 0 24px rgba(240,180,40,0.06)!important}
.cont-medal{font-size:22px;margin-bottom:10px}
.cont-team{font-size:14px;font-weight:400;color:#888;margin-bottom:10px;line-height:1.4}
.cont-pct{font-family:'Syne',sans-serif;font-size:32px;font-weight:700;color:#22c55e;letter-spacing:-0.02em}
.cont-pct-gold{color:#f0b429!important}
.cont-card:nth-child(1){animation-delay:0.05s}
.cont-card:nth-child(2){animation-delay:0.12s}
.cont-card:nth-child(3){animation-delay:0.19s}
.ptbl{width:100%;border-collapse:collapse;font-size:13px}
.ptbl th{font-family:'Inter',sans-serif;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#555;padding:10px 16px;text-align:left;border-bottom:1px solid #111}
.ptbl td{padding:9px 16px;color:#888;border-bottom:1px solid #0d0d0d}
.ptbl tr:hover td{background:rgba(255,255,255,0.015)}
.ptbl tr:last-child td{border-bottom:none}
.td-rank{color:#333!important;font-size:11px}
.td-gold{color:#f0b429!important;font-weight:600}
.td-green{color:#22c55e!important;font-weight:600}
.td-dim{color:#666!important;font-weight:500}
.insight-box{background:#0a0a0a;border:1px solid #111;border-radius:14px;padding:24px 28px;animation:fadeUp 0.5s ease forwards 0.2s;opacity:0}
.ins-lbl{font-size:10px;letter-spacing:2.5px;text-transform:uppercase;color:#333;margin-bottom:16px}
.ins-line{font-size:14px;color:#666;line-height:1.8;margin-bottom:4px}
.ins-hl-g{color:#22c55e}
.ins-hl-o{color:#f0b429}
.ins-hl-b{color:#60a5fa}
.ins-hl-w{color:#888}
.grp-card{background:#0a0a0a;border:1px solid #111;border-radius:12px;padding:16px 18px;margin-bottom:12px;animation:fadeUp 0.4s ease forwards;opacity:0}
.grp-title{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;color:#555;letter-spacing:1px;margin-bottom:10px;text-transform:uppercase}
.grp-row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #0f0f0f}
.grp-row:last-child{border-bottom:none}
.grp-team{font-size:12px;color:#888}
.grp-pts{font-size:12px;color:#666;font-weight:500}
.grp-qual{color:#22c55e!important}
.bracket-outer{overflow-x:auto;padding:16px 0}
.bracket-wrap{display:flex;flex-direction:row;align-items:stretch;width:100%;gap:0}
.bracket-round{display:flex;flex-direction:column;flex:1;min-width:150px;max-width:200px;position:relative}
.bracket-round-inner{display:flex;flex-direction:column;justify-content:space-around;flex:1;padding:0 6px}
.bracket-lbl{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#555;text-align:center;margin-bottom:10px;font-family:'Inter',sans-serif;flex-shrink:0}
.mcard{background:#0a0a0a;border:1px solid #111;border-radius:9px;padding:9px 12px;transition:border-color 0.15s,box-shadow 0.15s;cursor:default}
.mcard:hover{border-color:#1a1a1a;box-shadow:0 4px 16px rgba(0,0,0,0.4)}
.mcard-div{border-top:1px solid #0f0f0f;margin:4px 0}
.mcard-team{font-size:11px;margin:0 0 3px}
.mcard-winner{color:#22c55e;font-weight:500;text-shadow:0 0 8px rgba(34,197,94,0.2)}
.mcard-loser{color:#2a2a2a}
.conn{width:20px;flex-shrink:0;display:flex;flex-direction:column;justify-content:space-around;align-self:stretch;border-left:1px solid #161616}
.story-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.story-card{background:#0a0a0a;border:1px solid #111;border-radius:12px;padding:18px 16px;animation:fadeUp 0.5s ease forwards;opacity:0}
.story-card:nth-child(1){animation-delay:0.05s}
.story-card:nth-child(2){animation-delay:0.15s}
.story-card:nth-child(3){animation-delay:0.25s}
.story-lbl{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#333;margin-bottom:10px}
.story-tm{font-size:13px;font-weight:500;color:#888;margin-bottom:4px}
.story-dt{font-size:12px;color:#555;line-height:1.6}
.br-r32{animation:fadeUp 0.3s ease forwards;opacity:0;animation-delay:0.0s}
.br-r16{animation:fadeUp 0.3s ease forwards;opacity:0;animation-delay:0.06s}
.br-qf{animation:fadeUp 0.3s ease forwards;opacity:0;animation-delay:0.12s}
.br-sf{animation:fadeUp 0.3s ease forwards;opacity:0;animation-delay:0.18s}
.br-fin{animation:fadeUp 0.3s ease forwards;opacity:0;animation-delay:0.24s}
[data-testid="metric-container"]{background:#0a0a0a;border:1px solid #111;border-radius:10px;padding:16px!important}
[data-testid="stMetricValue"]{font-family:'Syne',sans-serif!important;color:#f0f0f0!important}
[data-testid="stMetricLabel"]{color:#555!important;font-size:12px!important}
div[data-testid="stDataFrame"]{border-radius:10px;overflow:hidden}
.stTabs [data-baseweb="tab-list"]{background:#0a0a0a;border-bottom:1px solid #111}
.stTabs [data-baseweb="tab"]{color:#555!important;font-family:'Inter',sans-serif}
.stTabs [aria-selected="true"]{color:#888!important}
div[data-testid="stExpander"]{background:#0a0a0a;border:1px solid #111;border-radius:12px}
[data-baseweb="tab-highlight"]{background:#22c55e!important}
[data-baseweb="tab"]:hover{color:#22c55e!important}
[data-baseweb="tab"][aria-selected="true"]{color:#22c55e!important}
[data-baseweb="tab-border"]{background:#111!important}
[data-testid="stCheckbox"] [data-testid="stWidgetLabel"]{color:#888!important}
[data-testid="stCheckbox"] input:checked+div{background:#22c55e!important;border-color:#22c55e!important}
[data-baseweb="select"] [data-baseweb="select-control"]{background:#0a0a0a!important;border-color:#1a1a1a!important}
[data-baseweb="select"]:focus-within [data-baseweb="select-control"]{border-color:#22c55e!important;box-shadow:0 0 0 1px #22c55e22!important}
[data-baseweb="select"] [data-baseweb="select-control"]:hover{border-color:#22c55e44!important}
[data-baseweb="tag"]{background:#0d2818!important;border:1px solid #22c55e44!important;color:#22c55e!important}
[data-baseweb="tag"] span{color:#22c55e!important}
[data-baseweb="tag"] [role="button"]{color:#22c55e88!important}
[data-baseweb="tag"] [role="button"]:hover{color:#22c55e!important}
[data-baseweb="option"]:hover{background:#0d1f10!important}
[data-baseweb="option"][aria-selected="true"]{background:#0d2818!important;color:#22c55e!important}
[data-baseweb="menu"]{background:#0a0a0a!important;border:1px solid #1a1a1a!important}
[data-baseweb="popover"] [data-baseweb="menu"]{background:#0a0a0a!important}
[data-testid="stMultiSelect"] [data-baseweb="select-control"]{background:#0a0a0a!important;border-color:#1a1a1a!important}
[data-testid="stMultiSelect"]:focus-within [data-baseweb="select-control"]{border-color:#22c55e!important;box-shadow:0 0 0 1px #22c55e22!important}
[data-testid="stSelectbox"] [data-baseweb="select-control"]{background:#0a0a0a!important;border-color:#1a1a1a!important}
[data-testid="stSelectbox"]:focus-within [data-baseweb="select-control"]{border-color:#22c55e!important}
input[type="text"],input[type="number"],[data-baseweb="input"]{background:#0a0a0a!important;border-color:#1a1a1a!important;color:#888!important}
input:focus,[data-baseweb="input"]:focus-within{border-color:#22c55e!important;box-shadow:0 0 0 1px #22c55e22!important}
[data-testid="stExpander"] summary:hover{color:#22c55e!important}
[data-testid="stExpander"] svg{fill:#22c55e!important}
.stProgress [data-testid="stProgress"]>div{background:#22c55e!important}
div[data-testid="stProgressBar"]>div{background:linear-gradient(90deg,#16a34a,#22c55e)!important}
[data-testid="stCheckbox"] label:hover{color:#22c55e!important}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='nb'>
    <div class='nb-brand'>World<span>Cup</span>Predict</div>
    <div class='nb-links'>
        <a href='#'>Simulator</a>
        <a href='#'>Probabilities</a>
        <a href='#'>Playground</a>
    </div>
    <div class='nb-dot'></div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
    pipeline, cols = joblib.load(model_path)
    return pipeline, cols

@st.cache_data
def load_match_data():    return load_data()
@st.cache_data
def load_ranking_data():  return load_rankings()

@st.cache_data
def get_cached_team_stats(team, before_date_str, n=10):
    return get_team_stats(load_match_data(), team, pd.Timestamp(before_date_str), n)

@st.cache_data
def get_cached_fifa_rank(team, before_date_str):
    return get_fifa_rank(load_ranking_data(), team, pd.Timestamp(before_date_str))

pipeline, COLS = load_pipeline()
df          = load_match_data()
rankings_df = load_ranking_data()
teams       = sorted(df['home_team'].unique().tolist())

FLAGS = {
    "Brazil":"🇧🇷","Argentina":"🇦🇷","France":"🇫🇷","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Germany":"🇩🇪",
    "Spain":"🇪🇸","Portugal":"🇵🇹","Netherlands":"🇳🇱","Italy":"🇮🇹","Belgium":"🇧🇪",
    "Croatia":"🇭🇷","Uruguay":"🇺🇾","Mexico":"🇲🇽","United States":"🇺🇸","Japan":"🇯🇵",
    "South Korea":"🇰🇷","Morocco":"🇲🇦","Senegal":"🇸🇳","Nigeria":"🇳🇬","Canada":"🇨🇦",
    "Australia":"🇦🇺","Denmark":"🇩🇰","Switzerland":"🇨🇭","Poland":"🇵🇱","Colombia":"🇨🇴",
    "Chile":"🇨🇱","Ecuador":"🇪🇨","Peru":"🇵🇪","Tunisia":"🇹🇳","Cameroon":"🇨🇲",
    "Ghana":"🇬🇭","Iran":"🇮🇷","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Turkey":"🇹🇷","Sweden":"🇸🇪",
    "Norway":"🇳🇴","Austria":"🇦🇹","Paraguay":"🇵🇾","Saudi Arabia":"🇸🇦","Egypt":"🇪🇬",
    "Algeria":"🇩🇿","Czech Republic":"🇨🇿","Qatar":"🇶🇦","Panama":"🇵🇦",
    "South Africa":"🇿🇦","Ivory Coast":"🇨🇮",
}
def flag(t): return FLAGS.get(t,"🏳️")+" "+t

def make_prediction(home_t, away_t, is_neutral=True):
    ts = pd.Timestamp.now().strftime('%Y-%m-%d')
    hs  = get_cached_team_stats(home_t, ts)
    as_ = get_cached_team_stats(away_t, ts)
    hr, hp = get_cached_fifa_rank(home_t, ts)
    ar, ap = get_cached_fifa_rank(away_t, ts)
    wrd=hs['win_rate']-as_['win_rate']; rnd=ar-hr; gdd=hs['avg_goal_diff']-as_['avg_goal_diff']
    abr=abs(hr-ar); abg=abs(gdd); ic=int(abr<10)
    X=pd.DataFrame([[hs['win_rate'],as_['win_rate'],wrd,hs['avg_goals_for'],as_['avg_goals_for'],
        hs['avg_goals_against'],as_['avg_goals_against'],hs['avg_goal_diff'],as_['avg_goal_diff'],
        gdd,hr,ar,rnd,hp,ap,int(is_neutral),wrd*rnd,gdd*rnd,abr,abg,ic]],columns=COLS)
    proba=pipeline.predict_proba(X)[0]
    res={c:p for c,p in zip(pipeline.classes_,proba)}
    for k in res: res[k]=res[k]**1.3
    t=sum(res.values()); res={k:v/t for k,v in res.items()}
    hs_=0.6*hs['win_rate']+0.4*max(hs['avg_goal_diff'],0.01)
    as__=0.6*as_['win_rate']+0.4*max(as_['avg_goal_diff'],0.01)
    d=hs_-as__; res[0]=res.get(0,0)+0.1*d; res[2]=res.get(2,0)-0.1*d
    if hr<=10:res[0]+=0.03
    if hr>=30:res[0]-=0.03
    if ar<=10:res[2]+=0.03
    if ar>=30:res[2]-=0.03
    res[0]=np.clip(res[0],0.05,0.9); res[2]=np.clip(res[2],0.05,0.9)
    t=sum(res.values()); res={k:v/t for k,v in res.items()}
    draw=res.get(1,0)
    if abr<8:draw*=1.15
    elif abr>25:draw*=0.80
    draw=float(np.clip(draw,0.15,0.35)); res[1]=draw
    t=sum(res.values()); res={k:v/t for k,v in res.items()}
    res={k:float(np.clip(v,0.1,0.8)) for k,v in res.items()}
    t=sum(res.values()); res={k:v/t for k,v in res.items()}
    return res,hs,as_,hr,ar

def get_verdict(res):
    top=max(res.values())
    return "Strong favorite" if top>0.55 else "Close match" if top>=0.45 else "Slight edge"

def get_explanation(hs,as_,hr,ar,home,away):
    r=[]
    if hs['win_rate']>as_['win_rate']+0.1: r.append(f"Better recent form ({hs['win_rate']:.0%} vs {as_['win_rate']:.0%})")
    elif as_['win_rate']>hs['win_rate']+0.1: r.append(f"Better recent form ({as_['win_rate']:.0%} vs {hs['win_rate']:.0%})")
    if hr<ar-10: r.append(f"Higher FIFA ranking (#{int(hr)} vs #{int(ar)})")
    elif ar<hr-10: r.append(f"Higher FIFA ranking (#{int(ar)} vs #{int(hr)})")
    if hs['avg_goals_for']>as_['avg_goals_for']+0.3: r.append(f"Stronger attack ({hs['avg_goals_for']:.1f} vs {as_['avg_goals_for']:.1f} g/game)")
    elif as_['avg_goals_for']>hs['avg_goals_for']+0.3: r.append(f"Stronger attack ({as_['avg_goals_for']:.1f} vs {hs['avg_goals_for']:.1f} g/game)")
    return r[:3] if r else ["Teams are closely matched"]

def simulate_group(group_teams):
    s={t:{"Team":t,"P":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"Pts":0} for t in group_teams}
    res=[]
    for ht,at in combinations(group_teams,2):
        r,hs,as_,_,_=make_prediction(ht,at,is_neutral=True)
        hw=r.get(0,0);d=r.get(1,0);aw=r.get(2,0)
        out=np.random.choice([0,1,2],p=[hw,d,aw])
        hg=max(0,int(np.random.poisson(hs['avg_goals_for']))); ag=max(0,int(np.random.poisson(as_['avg_goals_for'])))
        if out==0:
            if hg<=ag:hg=ag+1
            s[ht]["W"]+=1;s[ht]["Pts"]+=3;s[at]["L"]+=1;lbl="Home win"
        elif out==2:
            if ag<=hg:ag=hg+1
            s[at]["W"]+=1;s[at]["Pts"]+=3;s[ht]["L"]+=1;lbl="Away win"
        else:
            hg=ag=min(hg,ag);s[ht]["D"]+=1;s[ht]["Pts"]+=1;s[at]["D"]+=1;s[at]["Pts"]+=1;lbl="Draw"
        s[ht]["P"]+=1;s[ht]["GF"]+=hg;s[ht]["GA"]+=ag
        s[at]["P"]+=1;s[at]["GF"]+=ag;s[at]["GA"]+=hg
        res.append({"Match":f"{ht} vs {at}","Score":f"{hg} - {ag}","Result":lbl})
    sdf=pd.DataFrame(s.values()); sdf["GD"]=sdf["GF"]-sdf["GA"]
    sdf=sdf[["Team","P","W","D","L","GF","GA","GD","Pts"]]
    sdf=sdf.sort_values(["Pts","GD","GF"],ascending=False).reset_index(drop=True); sdf.index+=1
    return sdf,pd.DataFrame(res)

def simulate_knockout(bracket):
    results,winners=[],[]
    for ht,at in bracket:
        if ht==at: winners.append(ht);results.append({"Match":f"{ht} vs {at}","Winner":ht});continue
        r,_,_,_,_=make_prediction(ht,at,is_neutral=True)
        out=np.random.choice([0,1,2],p=[r.get(0,0),r.get(1,0),r.get(2,0)])
        w=ht if out==0 else at if out==2 else (ht if random.random()>0.5 else at)
        results.append({"Match":f"{ht} vs {at}","Winner":w}); winners.append(w)
    return results,winners

def simulate_wc2026():
    G={
        "A":["Mexico","South Korea","South Africa","Czech Republic"],
        "B":["Canada","Switzerland","Qatar","Bosnia and Herzegovina"],
        "C":["Brazil","Morocco","Scotland","Haiti"],
        "D":["United States","Paraguay","Australia","Turkey"],
        "E":["Germany","Ecuador","Ivory Coast","Sweden"],
        "F":["Netherlands","Japan","Tunisia","Sweden"],
        "G":["Belgium","Iran","Egypt","New Zealand"],
        "H":["Spain","Uruguay","Saudi Arabia","Cape Verde"],
        "I":["France","Senegal","Norway","Iraq"],
        "J":["Argentina","Austria","Algeria","Jordan"],
        "K":["Portugal","Colombia","Uzbekistan","Ghana"],
        "L":["England","Croatia","Panama","Cameroon"],
    }
    all_s,gr,tp={},{},[]
    for gn,gt in G.items():
        v=[t for t in gt if t in teams]
        while len(v)<4: v.append(v[-1])
        sdf,_=simulate_group(v); all_s["Group "+gn]=sdf; gr[gn]=sdf["Team"].tolist()
        th=sdf.iloc[2]; tp.append({"team":th["Team"],"pts":th["Pts"],"gd":th["GD"],"gf":th["GF"]})
    best3=[t["team"] for t in sorted(tp,key=lambda x:(x["pts"],x["gd"],x["gf"]),reverse=True)[:8]]
    q={gn:{"1st":gt[0],"2nd":gt[1]} for gn,gt in gr.items()}
    r32=[(q["A"]["1st"],q["B"]["2nd"]),(q["C"]["1st"],q["D"]["2nd"]),(q["E"]["1st"],q["F"]["2nd"]),
         (q["G"]["1st"],q["H"]["2nd"]),(q["I"]["1st"],q["J"]["2nd"]),(q["K"]["1st"],q["L"]["2nd"]),
         (q["B"]["1st"],q["A"]["2nd"]),(q["D"]["1st"],q["C"]["2nd"]),(q["F"]["1st"],q["E"]["2nd"]),
         (q["H"]["1st"],q["G"]["2nd"]),(q["J"]["1st"],q["I"]["2nd"]),(q["L"]["1st"],q["K"]["2nd"])]
    r32+=[(best3[i],best3[i+1]) for i in range(0,min(len(best3)-1,8),2)]
    r32r,r32w=simulate_knockout(r32)
    r16b=[(r32w[i],r32w[i+1]) for i in range(0,len(r32w)-1,2)]; r16r,r16w=simulate_knockout(r16b)
    qfb=[(r16w[i],r16w[i+1]) for i in range(0,len(r16w)-1,2)]; qfr,qfw=simulate_knockout(qfb)
    sfr,sfw=simulate_knockout([(qfw[0],qfw[1]),(qfw[2],qfw[3])])
    fr,champ=simulate_knockout([(sfw[0],sfw[1])])
    return {"all_standings":all_s,"best_third":best3,
            "r32":{"results":r32r,"winners":r32w},"r16":{"results":r16r,"winners":r16w},
            "qf":{"results":qfr,"winners":qfw},"sf":{"results":sfr,"winners":sfw},
            "final":{"results":fr,"winner":champ[0]}}

def monte_carlo_wc2026(runs=100):
    champs={}
    prog=st.progress(0,text="Running simulations…  0 / "+str(runs))
    for i in range(runs):
        try:
            r=simulate_wc2026(); c=r["final"]["winner"]; champs[c]=champs.get(c,0)+1
        except Exception as e: print(e)
        prog.progress((i+1)/runs,text=f"Running simulations…  {i+1} / {runs}")
    prog.empty()
    return dict(sorted({t:round((c/runs)*100,1) for t,c in champs.items()}.items(),key=lambda x:x[1],reverse=True))

def monte_carlo_group(gt,runs=100):
    cnt={t:0 for t in gt}
    for _ in range(runs):
        sdf,_=simulate_group(gt)
        for t in sdf.head(2)["Team"].tolist(): cnt[t]+=1
    return dict(sorted({t:round((c/runs)*100,1) for t,c in cnt.items()}.items(),key=lambda x:x[1],reverse=True))

def tournament_story(result):
    bu=None;lp=1.0
    for rk in ["r32","r16","qf","sf","final"]:
        for r in result[rk]["results"]:
            h,a=r["Match"].split(" vs ");w=r["Winner"];l=a if w==h else h
            res,_,_,_,_=make_prediction(w,l,is_neutral=True)
            wp=res.get(0,0) if w==h else res.get(2,0)
            if wp<lp: lp=wp;bu={"match":r["Match"],"winner":w,"prob":wp}
    bt=None;bp=-999
    for gn,sdf in result["all_standings"].items():
        row=sdf.iloc[0];sc=row["Pts"]*10+row["GD"]
        if sc>bp: bp=sc;bt={"team":row["Team"],"pts":row["Pts"],"gd":row["GD"]}
    sf_teams=result["sf"]["results"]
    semi=([r["Match"].split(" vs ")[0] for r in sf_teams]+[r["Match"].split(" vs ")[1] for r in sf_teams])
    surp=None;lr=0
    for t in semi:
        _,_,_,rank,_=make_prediction(t,"Brazil" if t!="Brazil" else "Argentina",is_neutral=True)
        if rank>lr: lr=rank;surp={"team":t,"rank":int(rank)}
    return bu,bt,surp

def mcard(match, winner):
    h,a=match.split(" vs ")
    hc="mcard-winner" if winner==h else "mcard-loser"
    ac="mcard-winner" if winner==a else "mcard-loser"
    return (f"<div class='mcard'>"
            f"<p class='mcard-team {hc}'>{flag(h)}</p>"
            f"<div class='mcard-div'></div>"
            f"<p class='mcard-team {ac}'>{flag(a)}</p></div>")

CARD_H = 54   # px — approximate rendered height of one mcard
CARD_GAP = 6  # px — gap between cards in the densest round (R32)

def round_col(title, results, css_class, gap_px, margin_top_px):
    cards = "".join([mcard(r["Match"], r["Winner"]) for r in results])
    return (
        f"<div class='{css_class}' style='"
        f"display:flex;flex-direction:column;"
        f"flex:1;min-width:148px;max-width:185px;"
        f"margin-top:{margin_top_px}px;align-self:flex-start'>"
        f"<p class='bracket-lbl'>{title}</p>"
        f"<div style='display:flex;flex-direction:column;gap:{gap_px}px;padding:0 4px'>"
        + cards +
        "</div></div>"
    )

def render_bracket(result):
    slot   = CARD_H + CARD_GAP
    n32    = len(result["r32"]["results"])

    off_r32 = 0
    off_r16 = slot / 2
    off_qf  = off_r16 + slot
    off_sf  = off_qf  + slot * 2
    off_fin = off_sf  + slot * 4

    gap_r32 = CARD_GAP
    gap_r16 = slot
    gap_qf  = slot * 2
    gap_sf  = slot * 4

    container_h = int(off_fin + CARD_H * 3 + gap_sf)

    r32h = round_col("Round of 32",   result["r32"]["results"], "br-r32", gap_r32, off_r32)
    r16h = round_col("Round of 16",   result["r16"]["results"], "br-r16", gap_r16, off_r16)
    qfh  = round_col("Quarters",      result["qf"]["results"],  "br-qf",  gap_qf,  off_qf)
    sfh  = round_col("Semis",         result["sf"]["results"],  "br-sf",  gap_sf,  off_sf)
    fh   = round_col("Final",         result["final"]["results"],"br-fin", 0,       off_fin)

    conn = "<div style='width:18px;flex-shrink:0;border-left:1px solid #161616'></div>"

    st.markdown(
        f"<div style='overflow:visible'>"
        f"<div style='"
        f"display:flex;flex-direction:row;align-items:flex-start;"
        f"width:100%;min-height:{container_h}px;overflow:visible;gap:0'>"
        + r32h + conn + r16h + conn + qfh + conn + sfh + conn + fh +
        "</div></div>",
        unsafe_allow_html=True
    )
def render_champion(name, prob=None):
    prob_html=f"<p class='champ-prob'>Appeared in <span>{prob}%</span> of simulations</p>" if prob else ""
    st.markdown(
        f"<div class='champ-wrap'><div class='champ-card'>"
        f"<p class='champ-lbl'>World Cup 2026 Champion</p>"
        f"<div class='champ-trophy'>🏆</div>"
        f"<p class='champ-name'>{flag(name)}</p>"
        f"{prob_html}</div></div>",unsafe_allow_html=True)

def render_contenders(champ_pct):
    top3=list(champ_pct.items())[:3]; medals=["🥇","🥈","🥉"]
    c1,c2,c3=st.columns(3)
    for col,(team,pct),medal in zip([c1,c2,c3],top3,medals):
        gold="cont-card-gold" if medal=="🥇" else ""
        pgold="cont-pct-gold" if medal=="🥇" else ""
        with col:
            st.markdown(
                f"<div class='cont-card {gold}'>"
                f"<div class='cont-medal'>{medal}</div>"
                f"<p class='cont-team'>{flag(team)}</p>"
                f"<p class='cont-pct {pgold}'>{pct}%</p></div>",
                unsafe_allow_html=True)

def render_prob_table(champ_pct):
    rows=""
    for i,(team,pct) in enumerate(list(champ_pct.items())[:10]):
        bg="background:rgba(240,180,40,0.04);" if i==0 else "background:rgba(34,197,94,0.02);" if i<3 else ""
        pc="td-gold" if i==0 else "td-green" if i<3 else "td-dim"
        rows+=(f"<tr style='{bg}'>"
               f"<td class='td-rank'>#{i+1}</td>"
               f"<td>{flag(team)}</td>"
               f"<td class='{pc}'>{pct}%</td></tr>")
    st.markdown(
        f"<table class='ptbl'><thead><tr>"
        f"<th>#</th><th>Team</th><th>Win %</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>",
        unsafe_allow_html=True)

def render_bar(champ_pct):
    top10=list(champ_pct.items())[:10]
    lbls=[flag(t) for t,_ in top10]; vals=[p for _,p in top10]
    clrs=["#f0b429" if i==0 else "#166534" if i<3 else "#111" for i in range(len(top10))]
    fig=go.Figure(go.Bar(
        x=vals,y=lbls,orientation='h',marker_color=clrs,marker_line_width=0,
        text=[f"{v}%" for v in vals],textposition='outside',
        textfont=dict(color='#555',size=11)))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False,showgrid=False,zeroline=False,range=[0,max(vals)+10]),
        yaxis=dict(tickfont=dict(color='#888',size=11),autorange="reversed"),
        margin=dict(l=0,r=50,t=0,b=0),height=260,showlegend=False)
    st.plotly_chart(fig,use_container_width=True)

def render_insights(champ_pct):
    st_=list(champ_pct.items())
    if len(st_)<5: return
    w=st_[0]; dh=next((t for t in st_ if 0<t[1]<5),st_[-1])
    under=next((t for t in st_ if t[0] in ["Brazil","Germany","England"] and t[1]<10),None)
    lines=[
        f"<span class='ins-hl-g'>{flag(w[0])}</span> wins <span class='ins-hl-w'>{w[1]}%</span> of all simulated tournaments — the clear favourite.",
        f"<span class='ins-hl-b'>{flag(dh[0])}</span> only shows up at <span class='ins-hl-w'>{dh[1]}%</span> — quietly dangerous.",
    ]
    if under: lines.append(f"<span class='ins-hl-o'>{flag(under[0])}</span> underperforms at just <span class='ins-hl-w'>{under[1]}%</span> — expectations vs reality.")
    items="".join([f"<p class='ins-line'>— {l}</p>" for l in lines])
    st.markdown(
        f"<div class='insight-box'><p class='ins-lbl'>Read the data</p>{items}</div>",
        unsafe_allow_html=True)

def render_story(result):
    bu,bt,surp=tournament_story(result)
    u=(f"<div class='story-card'><p class='story-lbl'>⚡ Biggest upset</p>"
       f"<p class='story-tm' style='color:#22c55e'>{flag(bu['winner'])}</p>"
       f"<p class='story-dt'>{bu['match']}</p>"
       f"<p class='story-dt'>Won with only <span style='color:#22c55e'>{bu['prob']:.0%}</span> probability</p></div>") if bu else ""
    b=(f"<div class='story-card'><p class='story-lbl'>🏅 Group dominance</p>"
       f"<p class='story-tm' style='color:#f0b429'>{flag(bt['team'])}</p>"
       f"<p class='story-dt'><span style='color:#f0b429'>{bt['pts']} pts</span> · GD {bt['gd']:+d}</p></div>") if bt else ""
    s=(f"<div class='story-card'><p class='story-lbl'>🌟 Surprise run</p>"
       f"<p class='story-tm' style='color:#60a5fa'>{flag(surp['team'])}</p>"
       f"<p class='story-dt'>Ranked #{surp['rank']} globally · reached the semis</p></div>") if surp else ""
    if u or b or s:
        st.markdown(f"<div class='story-grid'>{u}{b}{s}</div>",unsafe_allow_html=True)

def render_groups(all_standings):
    keys=list(all_standings.keys()); cols=st.columns(3)
    for i,gname in enumerate(keys):
        sdf=all_standings[gname]
        rows=""
        for j,row in sdf.iterrows():
            qual=j<=2; qc="grp-qual" if qual else ""
            rows+=f"<div class='grp-row'><span class='grp-team {qc}'>{'🟢 ' if qual else ''}{row['Team']}</span><span class='grp-pts {qc}'>{row['Pts']}pt</span></div>"
        with cols[i%3]:
            st.markdown(
                f"<div class='grp-card' style='animation-delay:{i*0.04}s'>"
                f"<p class='grp-title'>{gname}</p>{rows}</div>",
                unsafe_allow_html=True)

def sec(label):
    st.markdown(f"<p class='sec-head'>{label}</p>",unsafe_allow_html=True)

def gap(h="1.5rem"):
    st.markdown(f"<div style='height:{h}'></div>",unsafe_allow_html=True)

st.markdown("""
<div class='hero'>
    <div class='hero-grid'></div>
    <p class='hero-eyebrow'>FIFA Data · 49,000 Matches · ML Model</p>
    <h1 class='hero-title'>World Cup 2026<br><em>Simulator</em></h1>
    <p class='hero-sub'>Run thousands of simulations. Find the real champion.</p>
    <p class='hero-sub2'>Who actually wins if we play it 1000 times?</p>
</div>
""",unsafe_allow_html=True)

_,ctr,_ = st.columns([1,1.6,1])
with ctr:
    ca,cb=st.columns(2)
    with ca: run_single=st.button("⚽  Simulate Tournament",type="primary",use_container_width=True)
    with cb: run_100   =st.button("📊  View Probabilities",use_container_width=True)

gap("2rem")

if run_single:
    with st.spinner(""):
        result=simulate_wc2026()
    render_champion(result["final"]["winner"])
    gap()
    _,inner,_=st.columns([0.5,9,0.5])
    with inner:
        sec("Group Stage")
        render_groups(result["all_standings"])
        gap()
        if result["best_third"]:
            sec("Best Third-Place Teams")
            b3cols=st.columns(4)
            for i,t in enumerate(result["best_third"]):
                with b3cols[i%4]:
                    st.markdown(f"<p style='font-size:12px;color:#555;margin:4px 0'>🟡 {flag(t)}</p>",unsafe_allow_html=True)
            gap()
        sec("Knockout Bracket")
        render_bracket(result)
        gap()
        sec("Tournament Story")
        render_story(result)

if run_100:
    champ_pct=monte_carlo_wc2026(100)
    render_champion(list(champ_pct.keys())[0], list(champ_pct.values())[0])
    gap()
    _,inner,_=st.columns([0.5,9,0.5])
    with inner:
        sec("Top Contenders")
        render_contenders(champ_pct)
        gap()
        left,right=st.columns([1,1])
        with left:
            sec("Win Probability")
            render_prob_table(champ_pct)
        with right:
            sec("Distribution")
            render_bar(champ_pct)
        gap()
        render_insights(champ_pct)

gap("2rem")
_,inner,_=st.columns([0.5,9,0.5])
with inner:
    with st.expander("⚙️  Custom Predictions & Group Simulator",expanded=False):
        tab1,tab2=st.tabs(["Match Predictor","Group Stage"])
        with tab1:
            c1,c2=st.columns(2)
            with c1: home=st.selectbox("Home",teams,index=teams.index('Brazil'),key="ph")
            with c2: away=st.selectbox("Away",teams,index=teams.index('Argentina'),key="pa")
            neutral=st.checkbox("Neutral venue",key="pn")
            if st.button("Predict",type="primary",use_container_width=True,key="pb"):
                if home==away: st.error("Pick two different teams.")
                else:
                    res,hs,as_,hr,ar=make_prediction(home,away,neutral)
                    mc1,mc2,mc3=st.columns(3)
                    mc1.metric(f"{home} win",f"{res.get(0,0):.0%}")
                    mc2.metric("Draw",f"{res.get(1,0):.0%}")
                    mc3.metric(f"{away} win",f"{res.get(2,0):.0%}")
                    vals=[res.get(0,0),res.get(1,0),res.get(2,0)]
                    lbls=[f"{home} win","Draw",f"{away} win"]
                    clrs=['#22c55e' if v==max(vals) else '#111' for v in vals]
                    fig=go.Figure(go.Bar(x=vals,y=lbls,orientation='h',marker_color=clrs,
                        text=[f"{v:.0%}" for v in vals],textposition='outside',
                        textfont=dict(color='#666',size=11)))
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showticklabels=False,showgrid=False,zeroline=False,range=[0,max(vals)+0.15]),
                        yaxis=dict(tickfont=dict(color='#888',size=11)),
                        margin=dict(l=0,r=0,t=0,b=0),height=130,showlegend=False)
                    st.plotly_chart(fig,use_container_width=True)
                    st.info(f"**{get_verdict(res)}** · {max(res.values()):.0%} top probability")
                    for r in get_explanation(hs,as_,hr,ar,home,away):
                        st.caption(f"→ {r}")
        with tab2:
            dg=["Brazil","Argentina","France","England"]
            gt=st.multiselect("Pick 4 teams",teams,default=dg,key="gs")
            ga,gb=st.columns(2)
            with ga:
                if st.button("Simulate Group",use_container_width=True,key="gsa"):
                    if len(gt)!=4: st.error("Pick exactly 4 teams.")
                    else:
                        sdf,rdf=simulate_group(gt)
                        st.dataframe(sdf,use_container_width=True)
                        for i,row in sdf.iterrows():
                            e="🟢" if i<=2 else "⚫"
                            st.caption(f"{e} {row['Team']} — {row['Pts']} pts")
            with gb:
                if st.button("Run 100 Sims",use_container_width=True,key="gsb"):
                    if len(gt)!=4: st.error("Pick exactly 4 teams.")
                    else:
                        with st.spinner("Simulating…"):
                            pct=monte_carlo_group(gt)
                        for team,prob in pct.items():
                            st.metric(label=team,value=f"{prob}%")

gap("3rem")