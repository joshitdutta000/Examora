import os
import joblib
import json
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from src.agent.state import AgentState
from src.rag.retriever import retrieve_context

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
# ── Load M1 artifacts ──────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")

model      = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
scaler     = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

with open(os.path.join(MODELS_DIR, "meta.json")) as f:
    meta = json.load(f)

topic_freq_map = meta["topic_freq_map"]
ohe_columns    = meta["ohe_columns"]

LABEL_MAP = {0: "Easy", 1: "Medium", 2: "Hard"}

# ── LLM ───────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# ── Helper ────────────────────────────────────────────────────────
def analyze_node(state: AgentState) -> AgentState:
    from src.preprocessing import preprocess
    from src.feature_engineering import build_features
    import pandas as pd

    row = {
        "question_text": state["question_text"],
        "subject": state["subject"],
        "topic": state["topic"],
        "question_type": state["question_type"],
        "cognitive_level": state["cognitive_level"],
        "avg_score": state["avg_score"],
        "std_dev": state["std_dev"],
        "discrimination_index": state["discrimination_index"],
        "difficulty_label": "Easy",  # placeholder
    }
    df = pd.DataFrame([row])

    result = preprocess(df, is_train=False,
                        topic_freq_map=topic_freq_map,
                        ohe_columns=ohe_columns)
    df_processed = result["df_processed"]
    feat_result = build_features(df_processed, is_train=False,
                                  tfidf=vectorizer,
                                  scaler=scaler)
    X = feat_result["X"]

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = round(float(np.max(proba)) * 100, 2)
    state["predicted_difficulty"] = LABEL_MAP[pred]
    state["confidence"] = confidence
    return state
# ── Node 2: Retrieve ──────────────────────────────────────────────
def retrieve_node(state: AgentState) -> AgentState:
    query = f"{state['subject']} {state['cognitive_level']} {state['predicted_difficulty']} question design"
    state["retrieved_context"] = retrieve_context(query)
    return state


# ── Node 3: Reason ────────────────────────────────────────────────
def reason_node(state: AgentState) -> AgentState:
    prompt = f"""You are an expert educational assessment designer.

A student exam question has been analyzed:
- Question: {state['question_text']}
- Subject: {state['subject']} | Topic: {state['topic']}
- Type: {state['question_type']} | Cognitive Level: {state['cognitive_level']}
- Predicted Difficulty: {state['predicted_difficulty']} (Confidence: {state['confidence']}%)
- Avg Score: {state['avg_score']} | Std Dev: {state['std_dev']} | Discrimination Index: {state['discrimination_index']}

Relevant pedagogy guidelines:
{state['retrieved_context']}

1. Explain in 3-4 sentences WHY this question is {state['predicted_difficulty']}.
2. Identify 2-3 specific learning gaps this question may reveal in students.

Be concise and specific."""

    response = llm.invoke([HumanMessage(content=prompt)])
    output = response.content.strip()
    parts = output.split("\n\n", 1)
    state["reasoning"] = parts[0] if len(parts) > 0 else output
    state["learning_gaps"] = parts[1] if len(parts) > 1 else "See reasoning above."
    return state


# ── Node 4: Recommend ─────────────────────────────────────────────
def recommend_node(state: AgentState) -> AgentState:
    prompt = f"""You are an expert educational assessment designer.

Original Question: {state['question_text']}
Difficulty: {state['predicted_difficulty']} | Subject: {state['subject']} | Cognitive Level: {state['cognitive_level']}
Learning Gaps Identified: {state['learning_gaps']}

Pedagogy guidelines:
{state['retrieved_context']}

Provide:
1. 3 specific, actionable improvements to this question
2. A rewritten improved version of the question
3. One pedagogical reference that supports your suggestions

Be specific and practical."""

    response = llm.invoke([HumanMessage(content=prompt)])
    state["recommendations"] = response.content.strip()
    state["disclaimer"] = (
        "⚠️ This analysis is AI-generated for educational support purposes only. "
        "Always consult qualified educators before making assessment decisions."
    )
    return state


# ── Node 5: Report ────────────────────────────────────────────────
def report_node(state: AgentState) -> AgentState:
    report = f"""
# 📊 Examora Assessment Report

## Question
{state['question_text']}

## Metadata
- **Subject:** {state['subject']} | **Topic:** {state['topic']}
- **Type:** {state['question_type']} | **Cognitive Level:** {state['cognitive_level']}
- **Avg Score:** {state['avg_score']} | **Std Dev:** {state['std_dev']} | **Discrimination Index:** {state['discrimination_index']}

## Difficulty Prediction
- **Predicted:** {state['predicted_difficulty']}
- **Confidence:** {state['confidence']}%

## Reasoning
{state['reasoning']}

## Learning Gaps
{state['learning_gaps']}

## Recommendations
{state['recommendations']}

---
{state['disclaimer']}
""".strip()

    state["final_report"] = report
    return state