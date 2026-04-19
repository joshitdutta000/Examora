import os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocessing import preprocess
from src.feature_engineering import build_features

MODELS_DIR = os.path.join(ROOT, "models")

st.set_page_config(
    page_title="Examora · Difficulty Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── App background ── */
.stApp {
    background: linear-gradient(140deg, #0d1117 0%, #161b27 60%, #0d1117 100%);
    color: #e6edf3;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b27 0%, #0d1117 100%);
    border-right: 1px solid rgba(99,102,241,0.15);
}
[data-testid="stSidebar"] .stRadio label {
    color: #ffffff !important;
    font-size: 0.95rem;
    font-weight: 600;
}
[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"]:nth-child(1) label {
    color: #60efff !important;
}
[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"]:nth-child(2) label {
    color: #f472b6 !important;
}
[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"]:nth-child(3) label {
    color: #facc15 !important;
}
[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] label:has(input:checked) {
    opacity: 1;
    text-shadow: 0 0 8px currentColor;
}

/* ── Hide Streamlit chrome ── */
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
header { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
.block-container { padding: 1.8rem 2.5rem 3rem; max-width: 1200px; }

/* ── Headings ── */
h1, h2, h3, h4 { color: #f1f5f9; }

/* ── Cards ── */
.card {
    background: rgba(22, 27, 39, 0.85);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(99,102,241,0.35); }

.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.06));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    height: 100%;
}

.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.6rem;
}

/* ── Gradient text ── */
.gradient-text {
    background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Badge ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.55rem 1.6rem;
    border-radius: 9999px;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    padding: 0.62rem 2rem;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: opacity 0.2s, transform 0.15s;
    width: 100%;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0px); }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: rgba(15,20,30,0.7) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 0.92rem !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}
.stSelectbox > div > div {
    background: rgba(15,20,30,0.7) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(22,27,39,0.7);
    border: 1px dashed rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 0.5rem;
}

/* ── DataFrames ── */
.dataframe { border-radius: 10px !important; }
[data-testid="stDataFrame"] {
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px;
    overflow: hidden;
}

/* ── Alerts ── */
.stSuccess { background: rgba(34,197,94,0.08) !important; border: 1px solid rgba(34,197,94,0.25) !important; border-radius: 10px !important; color: #86efac !important; }
.stWarning { background: rgba(245,158,11,0.08) !important; border: 1px solid rgba(245,158,11,0.25) !important; border-radius: 10px !important; }
.stInfo    { background: rgba(56,189,248,0.06) !important; border: 1px solid rgba(56,189,248,0.2)  !important; border-radius: 10px !important; }
.stError   { background: rgba(239,68,68,0.08)  !important; border: 1px solid rgba(239,68,68,0.25)  !important; border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: rgba(99,102,241,0.12) !important; margin: 1.2rem 0; }

/* ── Label color ── */
label, .stSelectbox label, .stTextInput label,
.stNumberInput label, .stTextArea label, .stSlider label {
    color: #ffffff !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em;
}
</style>
""", unsafe_allow_html=True)

DIFF_COLORS  = {"Easy": "#22c55e", "Medium": "#f59e0b", "Hard": "#ef4444"}
DIFF_EMOJIS  = {"Easy": "✅", "Medium": "⚠️",  "Hard": "🔴"}
DIFF_BG      = {"Easy": "rgba(34,197,94,0.10)", "Medium": "rgba(245,158,11,0.10)", "Hard": "rgba(239,68,68,0.10)"}
SUBJECTS     = ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science"]
Q_TYPES      = ["MCQ", "Short Answer", "Long Answer", "Numerical"]
COG_LEVELS   = ["Remember", "Understand", "Apply", "Analyze", "Evaluate"]

@st.cache_resource(show_spinner=False)
def load_resources():
    try:
        res = {
            "model":  joblib.load(os.path.join(MODELS_DIR, "best_model.pkl")),
            "tfidf":  joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")),
            "scaler": joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")),
            "label_encoder": joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl")),
        }
        with open(os.path.join(MODELS_DIR, "meta.json")) as f:
            res["meta"] = json.load(f)
        return res
    except FileNotFoundError:
        return None

@st.cache_data(show_spinner=False)
def load_results():
    path = os.path.join(MODELS_DIR, "results_summary.json")
    return json.load(open(path)) if os.path.exists(path) else {}

def _infer(row_dict: dict, res: dict) -> tuple[str, float]:
    df = pd.DataFrame([row_dict])
    df["difficulty_label"] = "Easy"
    out = preprocess(df, is_train=False,
                     label_encoder=res["label_encoder"],
                     ohe_columns=res["meta"]["ohe_columns"],
                     topic_freq_map=res["meta"]["topic_freq_map"])
    dp = out["df_processed"]
    if "difficulty_encoded" in dp.columns:
        dp.drop(columns=["difficulty_encoded"], inplace=True, errors="ignore")
    feat = build_features(dp, is_train=False, tfidf=res["tfidf"], scaler=res["scaler"])
    pred = res["model"].predict(feat["X"])[0]
    prob = res["model"].predict_proba(feat["X"])[0].max() if hasattr(res["model"], "predict_proba") else None
    return {0: "Easy", 1: "Medium", 2: "Hard"}.get(int(pred), "Unknown"), prob

def _batch_infer(df_in: pd.DataFrame, res: dict) -> pd.DataFrame:
    df = df_in.copy()
    if "difficulty_label" not in df.columns:
        df["difficulty_label"] = "Easy"
    out = preprocess(df, is_train=False,
                     label_encoder=res["label_encoder"],
                     ohe_columns=res["meta"]["ohe_columns"],
                     topic_freq_map=res["meta"]["topic_freq_map"])
    dp = out["df_processed"]
    if "difficulty_encoded" in dp.columns:
        dp.drop(columns=["difficulty_encoded"], inplace=True, errors="ignore")
    feat = build_features(dp, is_train=False, tfidf=res["tfidf"], scaler=res["scaler"])
    preds = res["model"].predict(feat["X"])
    label_map = {0: "Easy", 1: "Medium", 2: "Hard"}
    df_in["Predicted Difficulty"] = [label_map.get(int(p), "Unknown") for p in preds]
    return df_in

with st.sidebar:
    st.markdown("""
    <div style='padding: 0.4rem 0 1.2rem;'>
        <div style='font-size:1.6rem; font-weight:800; letter-spacing:-0.02em;
                    background:linear-gradient(90deg,#818cf8,#c084fc);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            🎓 Examora
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
    "Navigate",
    ["🔍  Single Predictor", "📋  Batch Upload", "📊  Model Dashboard", "🤖  Agent Assistant"],
    label_visibility="collapsed",
    )

    st.divider()

    results = load_results()
    if results:
        st.markdown("<div class='section-label'>Model Accuracies</div>", unsafe_allow_html=True)
        model_icons = {
            "Logistic Regression": "🔵",
            "Decision Tree":       "🌳",
            "Random Forest":       "🌲",
        }
        for name, m in results.items():
            icon = model_icons.get(name, "◉")
            acc  = m["accuracy"] * 100
            bar_w = int(acc)
            st.markdown(f"""
            <div style='margin-bottom:0.8rem;'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.28rem;'>
                    <span style='color:#cbd5e1;font-size:0.78rem;font-weight:500;'>{icon} {name}</span>
                    <span style='color:#a5b4fc;font-size:0.78rem;font-weight:700;'>{acc:.1f}%</span>
                </div>
                <div style='height:5px;background:rgba(99,102,241,0.12);border-radius:999px;'>
                    <div style='height:5px;width:{bar_w}%;background:linear-gradient(90deg,#6366f1,#8b5cf6);
                                border-radius:999px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    res_obj = load_resources()
    if res_obj:
        best = res_obj["meta"].get("best_model_name", "—")
        st.markdown(f"""
        <div style='color:#94a3b8;font-size:0.75rem;line-height:1.7;'>
            <div>🏆 <b style='color:#a5b4fc;'>Best model:</b> {best}</div>
            <div>🗂 <b style='color:#a5b4fc;'>Dataset:</b> 5,000 rows</div>
            <div>🔑 <b style='color:#a5b4fc;'>Random state:</b> 42</div>
            <div>⚙️ <b style='color:#a5b4fc;'>Framework:</b> Scikit-Learn</div>
        </div>
        """, unsafe_allow_html=True)

resources = load_resources()

if resources is None:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:60vh;gap:1.2rem;text-align:center;'>
        <div style='font-size:3rem;'>⚠️</div>
        <div style='font-size:1.4rem;font-weight:700;color:#f1f5f9;'>Model not found</div>
        <div style='color:#94a3b8;'>Please train the model first.</div>
        <code style='background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.2);
                     padding:0.6rem 1.2rem;border-radius:8px;color:#a5b4fc;font-size:0.9rem;'>
            python src/train.py
        </code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if "Single" in page:
    st.markdown("""
    <div style='margin-bottom:1.8rem;'>
        <div class='section-label'>Analysis Tool</div>
        <h2 style='margin:0;font-size:1.75rem;font-weight:700;letter-spacing:-0.02em;'>
            Single Question <span class='gradient-text'>Difficulty Predictor</span>
        </h2>
        <p style='color:#64748b;margin-top:0.4rem;font-size:0.88rem;'>
            Enter your exam question details below and get an instant difficulty classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='section-label'>Question Content</div>", unsafe_allow_html=True)
        question_text = st.text_area(
            "Question Text",
            height=110,
            placeholder="e.g. Apply the theory of Algebra in real-world scenarios to solve for unknown variables...",
            key="q_text",
            label_visibility="collapsed",
        )

    with st.container():
        st.markdown("<div class='section-label'>Question Metadata</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            subject   = st.selectbox("Subject",         SUBJECTS,   key="subj")
        with c2:
            topic     = st.text_input("Topic",           value="Algebra")
        with c3:
            q_type    = st.selectbox("Question Type",    Q_TYPES,    key="qtype")
        with c4:
            cog_level = st.selectbox("Cognitive Level",  COG_LEVELS, key="coglvl")

    with st.container():
        st.markdown("<div class='section-label'>Response Statistics</div>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            avg_score = st.number_input("Avg Score (0–10)",       value=6.5,  step=0.1, min_value=0.0, max_value=10.0)
        with s2:
            std_dev   = st.number_input("Standard Deviation",     value=1.2,  step=0.1, min_value=0.0)
        with s3:
            disc_idx  = st.number_input("Discrimination Index",   value=0.3,  step=0.05, min_value=-1.0, max_value=1.0)

    col_btn = st.columns([1, 2, 1])[1]
    with col_btn:
        predict_clicked = st.button("✦  Predict Difficulty", key="predict_btn")

    if predict_clicked:
        if not question_text.strip():
            st.warning("Please enter the question text before predicting.")
        else:
            with st.spinner(""):
                row = {
                    "question_text":        question_text,
                    "subject":              subject,
                    "topic":                topic,
                    "question_type":        q_type,
                    "cognitive_level":      cog_level,
                    "avg_score":            float(avg_score),
                    "std_dev":              float(std_dev),
                    "discrimination_index": float(disc_idx),
                }
                label, confidence = _infer(row, resources)

            col, _ = st.columns([1, 0.01])
            with col:
                color = DIFF_COLORS[label]
                emoji = DIFF_EMOJIS[label]
                bg    = DIFF_BG[label]
                conf_str = f"{confidence*100:.1f}% confidence" if confidence else ""
                st.markdown(f"""
                <div style='background:{bg};border:1.5px solid {color}40;border-radius:16px;
                            padding:1.8rem 2rem;margin-top:1rem;text-align:center;'>
                    <div style='color:#94a3b8;font-size:0.8rem;font-weight:600;
                                letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;'>
                        PREDICTED DIFFICULTY
                    </div>
                    <div class='badge' style='background:{color}18;color:{color};border:2px solid {color};
                                              font-size:1.7rem;margin:auto;display:inline-flex;'>
                        {emoji}&nbsp;{label}
                    </div>
                    <div style='color:#64748b;font-size:0.8rem;margin-top:0.8rem;'>{conf_str}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                mc1, mc2, mc3 = st.columns(3)
                tips = {
                    "Easy":   ("Low difficulty. Good for testing basic recall and understanding.", "✅"),
                    "Medium": ("Moderate difficulty. Tests application and analysis skills.",     "⚠️"),
                    "Hard":   ("High difficulty. Requires deep synthesis and evaluation.",         "🔴"),
                }
                msg, ico = tips[label]
                with mc1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size:1.4rem;'>{ico}</div>
                        <div style='color:#94a3b8;font-size:0.72rem;margin-top:0.3rem;'>Level</div>
                        <div style='color:{color};font-weight:700;font-size:1rem;'>{label}</div>
                    </div>""", unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size:1.4rem;'>📊</div>
                        <div style='color:#94a3b8;font-size:0.72rem;margin-top:0.3rem;'>Avg Score</div>
                        <div style='color:#f1f5f9;font-weight:700;font-size:1rem;'>{avg_score:.1f}/10</div>
                    </div>""", unsafe_allow_html=True)
                with mc3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size:1.4rem;'>🧠</div>
                        <div style='color:#94a3b8;font-size:0.72rem;margin-top:0.3rem;'>Cognitive</div>
                        <div style='color:#f1f5f9;font-weight:700;font-size:0.95rem;'>{cog_level}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div style='background:rgba(15,20,30,0.5);border:1px solid rgba(99,102,241,0.15);
                            border-radius:10px;padding:0.9rem 1.2rem;margin-top:0.8rem;
                            color:#94a3b8;font-size:0.85rem;line-height:1.5;'>
                    💡 {msg}
                </div>""", unsafe_allow_html=True)

elif "Batch" in page:
    st.markdown("""
    <div style='margin-bottom:1.8rem;'>
        <div class='section-label'>Bulk Processing</div>
        <h2 style='margin:0;font-size:1.75rem;font-weight:700;letter-spacing:-0.02em;'>
            Batch <span class='gradient-text'>CSV Predictor</span>
        </h2>
        <p style='color:#64748b;margin-top:0.4rem;font-size:0.88rem;'>
            Upload a CSV file with multiple questions and get predictions for all rows at once.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📄  Expected CSV Format", expanded=False):
        st.markdown("""
        Your CSV must include these columns:

        | Column | Type | Example |
        |---|---|---|
        | `question_text` | text | "Apply Algebra in real scenarios" |
        | `subject` | category | Mathematics / Physics / Chemistry / Biology / Computer Science |
        | `topic` | text | Algebra |
        | `question_type` | category | MCQ / Short Answer / Long Answer / Numerical |
        | `cognitive_level` | category | Remember / Understand / Apply / Analyze / Evaluate |
        | `avg_score` | float | 6.5 |
        | `std_dev` | float | 1.2 |
        | `discrimination_index` | float | 0.3 |
        """)

    uploaded = st.file_uploader(
        "Drop your CSV here or click to browse",
        type=["csv"],
        label_visibility="visible",
    )

    if uploaded:
        df_up = pd.read_csv(uploaded)

        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:0.8rem;margin-bottom:0.8rem;'>
            <span style='background:rgba(99,102,241,0.15);color:#a5b4fc;padding:0.25rem 0.7rem;
                         border-radius:6px;font-size:0.78rem;font-weight:600;'>
                📁 {uploaded.name}
            </span>
            <span style='color:#64748b;font-size:0.82rem;'>{len(df_up):,} rows · {len(df_up.columns)} columns</span>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(df_up.head(8), use_container_width=True, height=280)

        col_run, _ = st.columns([1, 2])
        with col_run:
            run_batch = st.button("⚡  Run Batch Prediction", key="run_batch")

        if run_batch:
            with st.spinner("Running predictions on all rows…"):
                df_result = _batch_infer(df_up, resources)

            st.success(f"✅  Predictions complete for **{len(df_result):,}** questions!")

            st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
            st.dataframe(
                df_result.style.apply(
                    lambda col: col.map({
                        "Easy":   "color: #22c55e; font-weight: 600;",
                        "Medium": "color: #f59e0b; font-weight: 600;",
                        "Hard":   "color: #ef4444; font-weight: 600;",
                    }) if col.name == "Predicted Difficulty" else [""] * len(col),
                    axis=0,
                ),
                use_container_width=True,
                height=320,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            dist = df_result["Predicted Difficulty"].value_counts()
            ch1, ch2 = st.columns(2)

            with ch1:
                fig_pie = go.Figure(go.Pie(
                    labels=dist.index.tolist(),
                    values=dist.values.tolist(),
                    hole=0.55,
                    marker_colors=[DIFF_COLORS.get(l, "#6366f1") for l in dist.index],
                    textinfo="label+percent",
                    textfont=dict(size=13, color="#e2e8f0"),
                ))
                fig_pie.update_layout(
                    title=dict(text="Difficulty Distribution", font=dict(color="#e2e8f0", size=14)),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    showlegend=False,
                    margin=dict(t=40, b=10, l=10, r=10),
                    height=280,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with ch2:
                fig_bar = go.Figure(go.Bar(
                    x=dist.index.tolist(),
                    y=dist.values.tolist(),
                    marker_color=[DIFF_COLORS.get(l, "#6366f1") for l in dist.index],
                    text=dist.values.tolist(),
                    textposition="outside",
                    textfont=dict(color="#e2e8f0"),
                ))
                fig_bar.update_layout(
                    title=dict(text="Count per Class", font=dict(color="#e2e8f0", size=14)),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    xaxis=dict(showgrid=False, color="#64748b"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(99,102,241,0.1)", color="#64748b"),
                    margin=dict(t=40, b=10, l=10, r=10),
                    height=280,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            col_dl, _ = st.columns([1, 2])
            with col_dl:
                st.download_button(
                    "⬇️  Download Results CSV",
                    data=df_result.to_csv(index=False).encode("utf-8"),
                    file_name="examora_predictions.csv",
                    mime="text/csv",
                )

elif "Dashboard" in page:
    st.markdown("""
    <div style='margin-bottom:1.8rem;'>
        <div class='section-label'>Analytics</div>
        <h2 style='margin:0;font-size:1.75rem;font-weight:700;letter-spacing:-0.02em;'>
            Model Performance <span class='gradient-text'>Dashboard</span>
        </h2>
        <p style='color:#64748b;margin-top:0.4rem;font-size:0.88rem;'>
            Detailed accuracy, F1 scores, and confusion matrices for all trained classifiers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    results = load_results()
    if not results:
        st.warning("No results found. Run `python src/train.py` first.")
        st.stop()

    cols = st.columns(3)
    icons = {"Logistic Regression": "🔵", "Decision Tree": "🌳", "Random Forest": "🌲"}
    for i, (name, m) in enumerate(results.items()):
        acc = m["accuracy"] * 100
        f1  = m["f1_weighted"]
        ico = icons.get(name, "◉")
        with cols[i]:
            st.markdown(f"""
            <div class='metric-card' style='margin-bottom:1rem;'>
                <div style='font-size:1.8rem;margin-bottom:0.4rem;'>{ico}</div>
                <div style='color:#a5b4fc;font-size:0.85rem;font-weight:600;margin-bottom:0.6rem;'>{name}</div>
                <div style='font-size:2.4rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.03em;'>{acc:.1f}<span style='font-size:1rem;color:#64748b;'>%</span></div>
                <div style='color:#64748b;font-size:0.72rem;margin-bottom:0.6rem;'>Accuracy</div>
                <div style='height:1px;background:rgba(99,102,241,0.15);margin:0.6rem 0;'></div>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='color:#64748b;font-size:0.72rem;'>F1 Score</span>
                    <span style='color:#818cf8;font-weight:700;font-size:0.9rem;'>{f1:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    names = list(results.keys())
    accs  = [results[n]["accuracy"] * 100 for n in names]
    f1s   = [results[n]["f1_weighted"] * 100 for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy (%)", x=names, y=accs,
        marker=dict(color=["#6366f1","#8b5cf6","#06b6d4"],
                    line=dict(color="rgba(255,255,255,0.05)", width=1)),
        text=[f"{a:.1f}%" for a in accs],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=12),
        width=0.3,
    ))
    fig.add_trace(go.Bar(
        name="F1 Score (%)", x=names, y=f1s,
        marker=dict(color=["#34d399","#a78bfa","#38bdf8"], opacity=0.75,
                    line=dict(color="rgba(255,255,255,0.05)", width=1)),
        text=[f"{f:.1f}%" for f in f1s],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=12),
        width=0.3,
    ))
    fig.update_layout(
        title=dict(text="Accuracy vs F1 Score — All Models",
                   font=dict(color="#e2e8f0", size=14), x=0),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter"),
        legend=dict(bgcolor="rgba(22,27,39,0.8)", bordercolor="rgba(99,102,241,0.2)",
                    borderwidth=1, font=dict(size=11)),
        xaxis=dict(showgrid=False, color="#64748b", tickfont=dict(size=12)),
        yaxis=dict(range=[0, 100], gridcolor="rgba(99,102,241,0.1)",
                   color="#64748b", ticksuffix="%"),
        margin=dict(t=40, b=20, l=10, r=10),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("""
    <div style='margin-bottom:1rem;'>
        <div class='section-label'>Confusion Matrices</div>
    </div>
    """, unsafe_allow_html=True)

    model_keys = [
        ("Logistic Regression", "logistic_regression", "🔵"),
        ("Decision Tree",       "decision_tree",       "🌳"),
        ("Random Forest",       "random_forest",       "🌲"),
    ]
    img_cols = st.columns(3)
    for idx, (display, safe, ico) in enumerate(model_keys):
        img_path = os.path.join(MODELS_DIR, f"confusion_matrix_{safe}.png")
        with img_cols[idx]:
            st.markdown(f"""
            <div style='background:rgba(22,27,39,0.7);border:1px solid rgba(99,102,241,0.15);
                        border-radius:12px;padding:0.8rem;margin-bottom:0.5rem;text-align:center;'>
                <span style='color:#a5b4fc;font-size:0.85rem;font-weight:600;'>{ico} {display}</span>
            </div>
            """, unsafe_allow_html=True)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
            else:
                st.info("Run training to generate this image.")

    st.divider()

    best_name = max(results, key=lambda n: results[n]["f1_weighted"])
    best_acc  = results[best_name]["accuracy"] * 100
    best_f1   = results[best_name]["f1_weighted"]
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(34,197,94,0.08),rgba(99,102,241,0.06));
                border:1.5px solid rgba(34,197,94,0.3);border-radius:14px;
                padding:1.2rem 1.6rem;display:flex;align-items:center;gap:1.2rem;'>
        <div style='font-size:2rem;'>🏆</div>
        <div>
            <div style='color:#22c55e;font-weight:700;font-size:1rem;'>Best Performing Model</div>
            <div style='color:#f1f5f9;font-size:1.4rem;font-weight:800;letter-spacing:-0.02em;
                        margin:0.15rem 0;'>{best_name}</div>
            <div style='color:#64748b;font-size:0.82rem;'>
                Accuracy: <b style='color:#86efac;'>{best_acc:.1f}%</b>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                F1 Score: <b style='color:#86efac;'>{best_f1:.4f}</b>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                Target range: <b style='color:#86efac;'>80–90%</b> ✓
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Agent Assistant Page ──────────────────────────────────────────
elif "Agent" in page:
    from src.agent.graph import run_agent

    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <h1 style='font-size:2rem;font-weight:800;background:linear-gradient(90deg,#818cf8,#c084fc);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;'>
            🤖 Agent Assessment Assistant
        </h1>
        <p style='color:#64748b;margin-top:0.4rem;'>
            AI-powered analysis with pedagogy-backed recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("agent_form"):
        st.markdown("### 📝 Question Details")
        col1, col2 = st.columns(2)
        with col1:
            q_text     = st.text_area("Question Text", height=120, placeholder="Enter exam question here...")
            subject    = st.selectbox("Subject", SUBJECTS)
            topic      = st.text_input("Topic", value="Algebra")
        with col2:
            q_type     = st.selectbox("Question Type", Q_TYPES)
            cog_level  = st.selectbox("Cognitive Level", COG_LEVELS)
            avg_score  = st.slider("Avg Score (0–10)", 0.0, 10.0, 6.0, 0.1)
            std_dev    = st.slider("Std Deviation (0–3)", 0.0, 3.0, 1.0, 0.05)
            disc_index = st.slider("Discrimination Index (-1 to 1)", -1.0, 1.0, 0.3, 0.01)

        submitted = st.form_submit_button("🚀 Run Agent Analysis", use_container_width=True)

    if submitted:
        if not q_text.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("🤖 Agent is analyzing your question..."):
                try:
                    result = run_agent({
                        "question_text": q_text,
                        "subject": subject,
                        "topic": topic,
                        "question_type": q_type,
                        "cognitive_level": cog_level,
                        "avg_score": avg_score,
                        "std_dev": std_dev,
                        "discrimination_index": disc_index,
                        "predicted_difficulty": None,
                        "confidence": None,
                        "retrieved_context": None,
                        "reasoning": None,
                        "learning_gaps": None,
                        "recommendations": None,
                        "disclaimer": None,
                        "final_report": None,
                    })

                    # Difficulty badge
                    diff = result.get("predicted_difficulty", "N/A")
                    conf = result.get("confidence", 0)
                    badge_color = {"Easy": "#22c55e", "Medium": "#f59e0b", "Hard": "#ef4444"}.get(diff, "#818cf8")
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03);border:1.5px solid {badge_color}55;
                                border-radius:14px;padding:1.2rem 1.6rem;margin:1rem 0;
                                display:flex;align-items:center;gap:1.2rem;'>
                        <div style='font-size:2.5rem;'>🎯</div>
                        <div>
                            <div style='color:#94a3b8;font-size:0.85rem;font-weight:600;'>PREDICTED DIFFICULTY</div>
                            <div style='color:{badge_color};font-size:2rem;font-weight:800;'>{diff}</div>
                            <div style='color:#64748b;font-size:0.85rem;'>Confidence: <b style='color:{badge_color};'>{conf}%</b></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Report sections
                    tab1, tab2, tab3 = st.tabs(["🧠 Reasoning", "📚 Learning Gaps", "💡 Recommendations"])

                    with tab1:
                        st.markdown(result.get("reasoning", ""))

                    with tab2:
                        st.markdown(result.get("learning_gaps", ""))

                    with tab3:
                        st.markdown(result.get("recommendations", ""))

                    # Download report
                    st.divider()
                    st.download_button(
                        label="📥 Download Full Report",
                        data=result.get("final_report", ""),
                        file_name="examora_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

                    # Disclaimer
                    st.markdown(f"""
                    <div style='background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.2);
                                border-radius:10px;padding:0.8rem 1.2rem;margin-top:1rem;
                                color:#fbbf24;font-size:0.82rem;'>
                        {result.get("disclaimer", "")}
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Agent error: {e}")
