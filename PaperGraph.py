# -*- coding: utf-8 -*-
"""
PaperGraph Pro v3.1 (Restored UI & Pro Features)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import json
import zipfile
import re
import firebase_admin
from firebase_admin import credentials, auth, db
import stripe
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.interpolate import make_interp_spline

# ==========================================
# ★ 設定エリア (Secrets利用)
# ==========================================
stripe.api_key = st.secrets["stripe_api_key"]
STRIPE_LINK = st.secrets["stripe_link"]
STRIPE_PORTAL = st.secrets["stripe_portal_link"]
FIREBASE_DB_URL = st.secrets["firebase_db_url"]

# --- 0. Firebase 初期化 ---
if not firebase_admin._apps:
    try:
        key_dict = json.loads(st.secrets["firebase_json_str"])
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred, {
            'databaseURL': FIREBASE_DB_URL
        })
    except Exception as e:
        st.error(f"Firebase接続エラー: {e}")

# --- 1. 定数・設定 ---
PT_TO_MM = 0.3528
MM_TO_INCH = 1 / 25.4
st.set_page_config(page_title="PaperGraph Studio", page_icon="📈", layout="wide")

st.markdown("""
<style>
    /* 1. 普通の入力フォーム（文字や数字） */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border: 1px solid #ccc !important;
        background-color: #f9f9f9 !important;
    }

    /* 2. ファイルアップロードの枠（縦に大きく・青枠維持） */
    [data-testid="stFileUploader"] section {
        border: 2px dashed #4F8BF9 !important;
        background-color: #ffffff !important;
        min-height: 250px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 40px !important;
    }
    
    /* マウスを乗せたとき */
    [data-testid="stFileUploader"] section:hover {
        background-color: #f0f8ff !important;
        border-color: #2E66D8 !important;
    }

    /* 共通デザイン */
    .main-header {font-family: 'Times New Roman', serif; color: #333; text-align: center;}
    .sub-header {font-family: 'Times New Roman', serif; color: #555; text-align: center; margin-bottom: 20px;}
    .locked-box {border: 2px solid #ddd; padding: 20px; border-radius: 10px; background-color: #f9f9f9; text-align: center; color: #666;}
</style>
""", unsafe_allow_html=True)

LINE_STYLE_MAP = {"実線 (-)": "-", "破線 (--)": "--", "点線 (:)": ":", "一点鎖線 (-.)": "-."}
MARKER_OPTIONS = {
    "○ (白抜き円)": {"fmt": "o", "fill": "none"},
    "△ (白抜き三角)": {"fmt": "^", "fill": "none"},
    "□ (白抜き四角)": {"fmt": "s", "fill": "none"},
    "▲ (塗りつぶし三角)": {"fmt": "^", "fill": "full"},
    "■ (塗りつぶし四角)": {"fmt": "s", "fill": "full"},
    "● (塗りつぶし円)": {"fmt": "o", "fill": "full"},
}

# --- セッション状態 ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "is_guest" not in st.session_state: st.session_state.is_guest = False
if "user_email" not in st.session_state: st.session_state.user_email = ""
if "user_plan" not in st.session_state: st.session_state.user_plan = "Free"
if "loaded_config" not in st.session_state: st.session_state.loaded_config = {}

def get_conf(key, default):
    return st.session_state.loaded_config.get(key, default)

current_config = {}

# --- Helper Functions ---
def parse_header(text):
    match = re.search(r"^(.*?)(?:[\s\[\(]+([^\]\)]+)[\]\)])?$", str(text))
    if match: return match.group(1).strip(), (match.group(2).strip() if match.group(2) else "")
    return str(text), ""

def check_stripe_plan(email):
    try:
        customers = stripe.Customer.list(email=email, limit=1)
        if customers.data: return "Pro"
        return "Free"
    except: return "Free"

# --- Firebase Functions ---
def sanitize_email(email): return email.replace(".", "_")

def save_config_cloud(email, name, data):
    try:
        user_key = sanitize_email(email)
        ref = db.reference(f'users/{user_key}/configs/{name}')
        ref.set(data)
        return True
    except Exception as e:
        st.error(f"保存エラー: {e}")
        return False

def get_cloud_config_names(email):
    try:
        user_key = sanitize_email(email)
        ref = db.reference(f'users/{user_key}/configs')
        data = ref.get()
        if data: return list(data.keys())
        return []
    except: return []

def load_config_cloud(email, name):
    try:
        user_key = sanitize_email(email)
        ref = db.reference(f'users/{user_key}/configs/{name}')
        return ref.get()
    except: return None

# --- ログイン画面 ---
def show_login_page():
    st.header("🔑 PaperGraph Studio ログイン")
    if st.button("🚀 登録せずにゲストとして利用する", type="secondary", use_container_width=True):
        st.session_state.logged_in = True
        st.session_state.is_guest = True
        st.session_state.user_email = "Guest User"
        st.session_state.user_plan = "Free"
        st.rerun()

    st.markdown("---")
    tab1, tab2 = st.tabs(["ログイン", "新規登録"])
    with tab1:
        email = st.text_input("メールアドレス", key="l_email")
        password = st.text_input("パスワード", type="password", key="l_pass")
        if st.button("ログイン", type="primary"):
            try:
                auth.get_user_by_email(email)
                st.session_state.logged_in = True
                st.session_state.is_guest = False
                st.session_state.user_email = email
                with st.spinner("プラン確認中..."):
                    st.session_state.user_plan = check_stripe_plan(email)
                st.rerun()
            except: st.error("ログイン失敗。登録はお済みですか？")
    with tab2:
        n_email = st.text_input("登録用メールアドレス")
        n_pass = st.text_input("登録用パスワード", type="password")
        if st.button("アカウント作成"):
            try:
                auth.create_user(email=n_email, password=n_pass)
                st.success("作成完了！ログインしてください")
            except Exception as e: st.error(f"エラー: {e}")

# --- 描画関数 (Times New Roman) ---
def create_figure(line_configs, config_dict):
    fig_w_mm = config_dict.get("fig_w_mm", 120)
    fig_h_mm = config_dict.get("fig_h_mm", 80)
    fig, ax = plt.subplots(figsize=(fig_w_mm * MM_TO_INCH, fig_h_mm * MM_TO_INCH))
    
    use_dual = config_dict.get("use_dual_axis", False)
    ax2 = ax.twinx() if use_dual else None
    
    # 論文用フォント設定
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "xtick.direction": "in", "ytick.direction": "in",
        "axes.linewidth": config_dict.get("axis_width", 0.71)
    })

    scale_type = config_dict.get("scale_type", "Linear")
    if "X" in scale_type or "Log-Log" in scale_type: ax.set_xscale("log")
    if "Y" in scale_type or "Log-Log" in scale_type: ax.set_yscale("log")

    if "Linear" in scale_type:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=config_dict.get("nbins_x", 6)))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=config_dict.get("nbins_y", 6)))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=config_dict.get("minor_div_x", 2)))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=config_dict.get("minor_div_y", 2)))

    f_tick = config_dict.get("f_size_tick", 10)
    # 表示設定に従って目盛りとラベルを制御
    ax.tick_params(axis='x', which='both', bottom=config_dict.get("show_xt", True), labelbottom=config_dict.get("show_xl", True), labelsize=f_tick, pad=config_dict.get("tick_pad", 3.5))
    ax.tick_params(axis='y', which='both', left=config_dict.get("show_ytl", True), labelleft=config_dict.get("show_yll", True), labelsize=f_tick, pad=config_dict.get("tick_pad", 3.5))
    if ax2: ax2.tick_params(axis='y', which='both', right=config_dict.get("show_ytr", True), labelright=config_dict.get("show_ylr", True), labelsize=f_tick, pad=config_dict.get("tick_pad", 3.5))
    
    if config_dict.get("show_minor", False): ax.minorticks_on(); ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=config_dict.get("minor_alpha", 0.15))
    if config_dict.get("show_major", False): ax.grid(which="major", linestyle="-", linewidth=0.6, alpha=config_dict.get("major_alpha", 0.3))

    for config in line_configs:
        target_ax = ax if config["axis"] == "left" or not ax2 else ax2
        x_d, y_d = config["x"], config["y"]
        if config.get("proc_mode") == "移動平均": y_d = y_d.rolling(window=config["proc_param"], center=True).mean()
        elif config.get("proc_mode") == "スプライン補間":
            try:
                mask = ~np.isnan(x_d) & ~np.isnan(y_d)
                if np.sum(mask) > 3:
                    x_c, y_c = x_d[mask], y_d[mask]
                    s_idx = np.argsort(x_c); x_u, u_idx = np.unique(x_c.iloc[s_idx], return_index=True)
                    y_u = y_c.iloc[s_idx].iloc[u_idx]
                    spl = make_interp_spline(x_u, y_u, k=3)
                    x_new = np.linspace(x_u.min(), x_u.max(), config["proc_param"])
                    x_d, y_d = x_new, spl(x_new)
            except: pass
        
        m_face = config["color"] if (config["m_info"] and config["m_info"].get("fill")=="full") else "white"
        # 凡例ラベルは「線」か「プロット」どちらかが有効な場合のみ設定
        lbl = config["label"] if (config["linestyle"]!="None" or config["marker"]!="None") else None
        
        # エラーバー、線、プロットの描画分け
        if config.get("show_err") and config.get("err_data") is not None:
             target_ax.errorbar(x_d, y_d, yerr=config["err_data"], label=lbl, color=config["color"], linewidth=config["lw"], linestyle=config["linestyle"], marker=config["marker"], markersize=config["m_size"], markerfacecolor=m_face, markeredgecolor=config["color"], capsize=3.0, ecolor=config["color"], zorder=10)
        else:
             target_ax.plot(x_d, y_d, label=lbl, color=config["color"], linewidth=config["lw"], linestyle=config["linestyle"], marker=config["marker"], markersize=config["m_size"], markerfacecolor=m_face, markeredgecolor=config["color"], zorder=10)

        if config.get("fit"):
            mask = ~np.isnan(config["x"]) & ~np.isnan(config["y"])
            if np.sum(mask) > 1:
                a, b = np.polyfit(config["x"][mask], config["y"][mask], 1)
                target_ax.plot(config["x"], a*config["x"]+b, color=config["color"], linestyle="--", alpha=0.7)
                if config.get("show_r2"):
                     r2 = 1 - (np.sum((config["y"][mask] - (a*config["x"][mask]+b))**2) / np.sum((config["y"][mask] - np.mean(config["y"][mask]))**2))
                     ax.text(config["r2_pos"][0], config["r2_pos"][1], f"$y={a:.3g}x{b:+.3g}$\n$R^2={r2:.3f}$", transform=ax.transAxes, color=config["color"], ha='left', va='top', bbox=dict(boxstyle="square,pad=0.1", fc="white", alpha=0.6, ec="none"))

    if config_dict.get("unify_origin", False):
        fig.canvas.draw()
        if config_dict.get("show_xl", True): ax.text(config_dict.get("origin_x_mm", -3.5)/fig_w_mm, config_dict.get("origin_y_mm", -1.5)/fig_h_mm, "0", transform=ax.transAxes, ha='right', va='top', fontsize=f_tick, zorder=30)

    f_lab = config_dict.get("f_size_lab", 11)
    x_lbl = f"${config_dict.get('x_name','t')}$" + (f" [{config_dict.get('x_unit')}]" if config_dict.get('x_unit') else "")
    y1_lbl = f"${config_dict.get('y1_name','V')}$" + (f" [{config_dict.get('y1_unit')}]" if config_dict.get('y1_unit') else "")
    
    ax.set_xlabel(x_lbl, fontsize=f_lab)
    ax.set_ylabel(y1_lbl, fontsize=f_lab)
    if ax2:
        y2_lbl = f"${config_dict.get('y2_name','I')}$" + (f" [{config_dict.get('y2_unit')}]" if config_dict.get('y2_unit') else "")
        ax2.set_ylabel(y2_lbl, fontsize=f_lab, rotation=270, labelpad=15)
    
    if config_dict.get("show_legend", True):
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
        if l1+l2:
            ax.legend(l1+l2, lb1+lb2, loc='upper right', bbox_to_anchor=(config_dict.get("leg_x",1.0), config_dict.get("leg_y",1.0)), fontsize=f_lab*0.8)
    
    return fig

# --- メインアプリ ---
def main_app():
    c1, c2 = st.columns([8, 2])
    with c2:
        badge = "orange" if st.session_state.user_plan == "Pro" else "gray"
        st.markdown(f"""<div style="text-align: right;"><span style="background-color: {badge}; color: white; padding: 4px 10px; border-radius: 5px;">{st.session_state.user_plan} Plan</span><br><small>{st.session_state.user_email}</small></div>""", unsafe_allow_html=True)
        if st.session_state.user_plan == "Pro":
             st.markdown(f"""<div style="text-align: right; margin-top:5px;"><a href="{STRIPE_PORTAL}" target="_blank" style="font-size:12px; color:#555;">⚙️ 契約内容の確認・解約</a></div>""", unsafe_allow_html=True)
        if st.button("ログアウト"): 
            st.session_state.logged_in = False; st.session_state.is_guest = False; st.rerun()

    st.title("📈 PaperGraph Studio")
    if st.session_state.is_guest: st.info("👀 ゲストモード中: 設定の保存機能などは制限されています。")

    uploaded_files = st.file_uploader("CSVをドロップ (複数可)", type="csv", accept_multiple_files=True)

    if uploaded_files:
        try:
            dfs = {f.name: pd.read_csv(f) for f in uploaded_files}
            all_opts = []
            for fn, df in dfs.items():
                for c in df.columns[1:]: all_opts.append({"file": fn, "column": c})

            # --- サイドバー: 読み込み ---
            if not st.session_state.is_guest:
                st.sidebar.header("☁️ 設定の読み込み")
                saved_names = get_cloud_config_names(st.session_state.user_email)
                if saved_names:
                    s_saved = st.sidebar.selectbox("保存済み設定", ["-- 選択 --"] + saved_names)
                    if s_saved != "-- 選択 --" and st.sidebar.button("読み込む"):
                        st.session_state.loaded_config = load_config_cloud(st.session_state.user_email, s_saved)
                        st.success(f"読み込み完了: {s_saved}"); st.rerun()
                st.sidebar.markdown("---")

            st.sidebar.header("🎨 グラフ構築")
            st.sidebar.caption("👇 ここで選んだデータで見た目を調整してください") # ★復活！
            sel_idx = []
            for i, opt in enumerate(all_opts):
                if st.sidebar.checkbox(f"{opt['column']} ({opt['file']})", value=(i==0), key=f"c_{i}"): sel_idx.append(i)
            
            # (自動設定ロジック)
            auto_xn, auto_xu = "t", "s"; auto_y1n, auto_y1u = "V", "V"
            if sel_idx:
                to = all_opts[sel_idx[0]]; tdf = dfs[to['file']]
                auto_xn, auto_xu = parse_header(tdf.columns[0])
                auto_y1n, auto_y1u = parse_header(to['column'])

            with st.sidebar.expander("🖼️ 原点・スケール・サイズ", expanded=True):
                fw = st.slider("横幅", 50, 200, get_conf("fig_w_mm", 120))
                fh = st.slider("縦幅", 50, 200, get_conf("fig_h_mm", 80))
                uo = st.checkbox("原点一本化", value=get_conf("unify_origin", False))
                ox = st.slider("調整X", -15.0, 5.0, get_conf("origin_x_mm", -3.5), step=0.1)
                oy = st.slider("調整Y", -15.0, 5.0, get_conf("origin_y_mm", -1.5), step=0.1)
                st_type = st.selectbox("スケール", ["Linear", "Semi-log X", "Semi-log Y", "Log-Log"], index=0)
                current_config.update({"fig_w_mm": fw, "fig_h_mm": fh, "unify_origin": uo, "origin_x_mm": ox, "origin_y_mm": oy, "scale_type": st_type})

            with st.sidebar.expander("📍 ラベル・枠・凡例"):
                xn = st.text_input("X記号", value=get_conf("x_name", auto_xn))
                xu = st.text_input("X単位", value=get_conf("x_unit", auto_xu))
                y1n = st.text_input("左Y記号", value=get_conf("y1_name", auto_y1n))
                y1u = st.text_input("左Y単位", value=get_conf("y1_unit", auto_y1u))
                use_dual = st.checkbox("2軸を使用", get_conf("use_dual_axis", False))
                y2n = st.text_input("右Y記号", get_conf("y2_name", "I")) if use_dual else "I"
                y2u = st.text_input("右Y単位", get_conf("y2_unit", "A")) if use_dual else "A"
                
                c_lk1, c_lk2 = st.columns(2)
                fl = c_lk1.slider("ラベル", 6, 24, get_conf("f_size_lab", 11))
                ft = c_lk2.slider("目盛り", 6, 24, get_conf("f_size_tick", 10))
                
                c_x, c_yl, c_yr = st.columns(3)
                sxt = c_x.checkbox("X軸線", get_conf("show_xt", True)); sxl = c_x.checkbox("X数字", get_conf("show_xl", True))
                sytl = c_yl.checkbox("左Y線", get_conf("show_ytl", True)); syll = c_yl.checkbox("左Y数字", get_conf("show_yll", True))
                sytr = c_yr.checkbox("右Y線", get_conf("show_ytr", True)) if use_dual else True; sylr = c_yr.checkbox("右Y数字", get_conf("show_ylr", True)) if use_dual else True
                
                sl = st.checkbox("凡例表示", get_conf("show_legend", True))
                lx = st.slider("LX", -0.5, 1.5, get_conf("leg_x", 1.0))
                ly = st.slider("LY", -0.5, 1.5, get_conf("leg_y", 1.0))
                tp = st.slider("離隔", 0.0, 10.0, get_conf("tick_pad", 3.5))

                current_config.update({"x_name": xn, "x_unit": xu, "y1_name": y1n, "y1_unit": y1u, "y2_name": y2n, "y2_unit": y2u, "use_dual_axis": use_dual, "f_size_lab": fl, "f_size_tick": ft, "show_legend": sl, "leg_x": lx, "leg_y": ly, "axis_width": 0.71, "tick_pad": tp, "show_xt": sxt, "show_xl": sxl, "show_ytl": sytl, "show_yll": syll, "show_ytr": sytr, "show_ylr": sylr})

            with st.sidebar.expander("📏 グリッド・目盛り密度"):
                c1, c2 = st.columns(2)
                smj = c1.checkbox("主グリッド", get_conf("show_major", False)); mja = c1.slider("主線濃さ", 0.1, 1.0, get_conf("major_alpha", 0.3))
                smn = c2.checkbox("補助グリッド", get_conf("show_minor", False)); mna = c2.slider("補助線濃さ", 0.1, 1.0, get_conf("minor_alpha", 0.15))
                nx = st.slider("X主目盛り", 2, 20, get_conf("nbins_x", 6)); ny = st.slider("Y主目盛り", 2, 20, get_conf("nbins_y", 6))
                mx = st.slider("X補助分割", 1, 10, get_conf("minor_div_x", 2)); my = st.slider("Y補助分割", 1, 10, get_conf("minor_div_y", 2))
                current_config.update({"show_major": smj, "major_alpha": mja, "show_minor": smn, "minor_alpha": mna, "nbins_x": nx, "nbins_y": ny, "minor_div_x": mx, "minor_div_y": my})

            with st.sidebar.expander("💾 保存・画質設定"):
                save_format = st.selectbox("形式", ["png", "pdf", "svg"], index=0)
                
                # ★ ここで画質制限！(DPI)
                max_dpi = 600 if st.session_state.user_plan == "Pro" else 300
                save_dpi = st.slider("DPI (画質)", 100, max_dpi, 300, step=50)
                if st.session_state.user_plan != "Pro":
                    st.caption("🔒 300dpi以上の高画質出力はProプラン限定")

            st.sidebar.header("🖊️ 線の詳細設定")
            line_configs = []; last_s = {}
            for idx in sel_idx:
                o = all_opts[idx]; fname, colname = o["file"], o["column"]; target_df = dfs[fname]
                with st.sidebar.expander(f"{colname} ({fname})"): # ★復活！詳細設定UI
                    ax_sel = st.radio("軸", ["左", "右"], horizontal=True, key=f"a_{idx}") if use_dual else "左"
                    col = st.color_picker("色", key=f"co_{idx}")
                    lbl = st.text_input("凡例名", colname, key=f"l_{idx}")
                    
                    c_sl, c_sm, c_se = st.columns(3)
                    sl_b = c_sl.checkbox("線", True, key=f"sl_{idx}")
                    sm_b = c_sm.checkbox("プロット", False, key=f"sm_{idx}")
                    se_b = c_se.checkbox("誤差", False, key=f"se_{idx}")

                    ls = st.selectbox("線種", list(LINE_STYLE_MAP.keys()), key=f"ls_{idx}") if sl_b else "None"
                    mk = st.selectbox("記号", list(MARKER_OPTIONS.keys()), key=f"mk_{idx}") if sm_b else "None"
                    
                    lw = st.slider("太さ", 0.1, 5.0, 1.1, key=f"lw_{idx}"); ms = st.slider("サイズ", 1.0, 20.0, 6.0, key=f"ms_{idx}")
                    pm = st.selectbox("処理", ["なし", "移動平均", "スプライン補間"], key=f"pm_{idx}")
                    pp = st.slider("Param", 2, 500, 5, key=f"pp_{idx}") if pm != "なし" else 0
                    
                    fit = st.checkbox("近似直線", False, key=f"fit_{idx}")
                    sr2 = st.checkbox("R2", True, key=f"r2_{idx}") if fit else False
                    rp = (st.slider("RX", 0.0,1.0,0.05,key=f"rx_{idx}"), st.slider("RY", 0.0,1.0,0.9,key=f"ry_{idx}")) if sr2 else (0,0)
                    
                    ed = target_df.iloc[:, st.selectbox("ErrCol", range(1, len(target_df.columns)), key=f"ec_{idx}")] if se_b else None

                    # 設定辞書を作成
                    conf = {"x": target_df.iloc[:, 0], "y": target_df[colname], "axis": "left" if ax_sel=="左" else "right", 
                            "color": col, "label": lbl, 
                            "linestyle": LINE_STYLE_MAP.get(ls,"None") if sl_b else "None", 
                            "marker": MARKER_OPTIONS.get(mk,{}).get("fmt") if sm_b else "None", 
                            "m_info": MARKER_OPTIONS.get(mk), "lw": lw, "m_size": ms, 
                            "proc_mode": pm, "proc_param": pp, "fit": fit, "show_r2": sr2, "r2_pos": rp, 
                            "show_err": se_b, "err_data": ed}
                    line_configs.append(conf); last_s = conf.copy()

            # --- 保存ボタンエリア (Pro限定) ---
            st.sidebar.markdown("---")
            st.sidebar.header("☁️ クラウドに保存")
            
            # ★ここでProプラン制限！
            if st.session_state.user_plan == "Pro":
                new_config_name = st.sidebar.text_input("現在の設定に名前をつけて保存", placeholder="例: 卒論用グラフ")
                if st.sidebar.button("クラウドに保存"):
                    if new_config_name:
                        if save_config_cloud(st.session_state.user_email, new_config_name, current_config):
                            st.success(f"保存しました: {new_config_name}"); st.rerun()
                    else: st.warning("名前を入力してください")
            else:
                # 無料会員向け表示
                st.sidebar.info("🔒 設定のクラウド保存はProプラン限定機能です。")
                if not st.session_state.is_guest:
                    st.sidebar.markdown(f"[💳 Proプランにアップグレード]({STRIPE_LINK})")

            # --- プレビュー ---
            st.subheader("📊 プレビュー")
            if line_configs:
                fig = create_figure(line_configs, current_config)
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format=save_format, dpi=save_dpi, bbox_inches='tight')
                st.download_button(f"💾 画像を保存 ({save_format})", buf.getvalue(), f"graph.{save_format}")

            # --- バッチ処理 (Pro限定) ---
            st.markdown("---"); st.subheader("📦 バッチ出力 (一括作成)")
            if st.session_state.user_plan == "Pro":
                b_col = st.number_input("列番号", 1, value=1)
                if st.button("🚀 ZIPダウンロード"):
                    z_buf = io.BytesIO(); prog = st.progress(0)
                    with zipfile.ZipFile(z_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        tot = len(dfs)
                        for i, (fn, df) in enumerate(dfs.items()):
                            prog.progress((i+1)/tot)
                            if len(df.columns) <= b_col: continue
                            bc = last_s.copy() if last_s else {}
                            bc.update({"x": df.iloc[:,0], "y": df.iloc[:,b_col], "label": fn})
                            fb = create_figure([bc], current_config)
                            im = io.BytesIO(); fb.savefig(im, format=save_format, dpi=save_dpi, bbox_inches='tight'); plt.close(fb)
                            zf.writestr(f"graph_{fn}.{save_format}", im.getvalue())
                    st.download_button("📦 ZIP保存", z_buf.getvalue(), "graphs.zip", mime="application/zip")
            else:
                st.markdown(f"""<div class="locked-box"><h3>🔒 Proプラン限定</h3><a href="{STRIPE_LINK}" target="_blank"><button style="background-color:#6772E5;color:white;border:none;padding:10px 20px;border-radius:5px;cursor:pointer;">💳 Proへアップグレード</button></a></div>""", unsafe_allow_html=True)

        except Exception as e: st.error(f"Error: {e}")
    else: st.info("CSVファイルをドロップしてください。")

if not st.session_state.logged_in: show_login_page()
else: main_app()