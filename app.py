from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "tuned_random_forest_model.pkl"

FEATURE_COLUMNS = [
    "sem_eval_lec_test_1_mark",
    "sem_eval_lab_test_1_mark",
    "semester_evaluation_mid_mark",
    "sem_eval_lec_test_2_mark",
    "sem_eval_lab_test_2_mark",
    "semester_evaluation_pre_gtu_mark",
    "semester_evaluation_internal_mark",
    "attendance_percentage",
]

FIELD_LABELS = {
    "sem_eval_lec_test_1_mark": "Lecture Test 1 Mark",
    "sem_eval_lab_test_1_mark": "Lab Test 1 Mark",
    "semester_evaluation_mid_mark": "Mid-Sem Evaluation Mark",
    "sem_eval_lec_test_2_mark": "Lecture Test 2 Mark",
    "sem_eval_lab_test_2_mark": "Lab Test 2 Mark",
    "semester_evaluation_pre_gtu_mark": "Pre-GTU Evaluation Mark",
    "semester_evaluation_internal_mark": "Internal Evaluation Mark",
    "attendance_percentage": "Attendance Percentage",
}

SHORT_LABELS = {
    "sem_eval_lec_test_1_mark": "Lec Test 1",
    "sem_eval_lab_test_1_mark": "Lab Test 1",
    "semester_evaluation_mid_mark": "Mid Eval",
    "sem_eval_lec_test_2_mark": "Lec Test 2",
    "sem_eval_lab_test_2_mark": "Lab Test 2",
    "semester_evaluation_pre_gtu_mark": "Pre GTU",
    "semester_evaluation_internal_mark": "Internal",
    "attendance_percentage": "Attendance %",
}

PERFORMANCE_INFO = {
    "Fail": {
        "range": "0-50",
        "advice": [
            "Focus on core concepts first and revise every day.",
            "Start with solved examples before moving to practice questions.",
            "Track progress weekly with short quizzes.",
        ],
    },
    "Pass": {
        "range": "50-60",
        "advice": [
            "Increase consistency in lecture and lab preparation.",
            "Practice mixed-difficulty problems to improve confidence.",
            "Review mistakes and create a quick correction notebook.",
        ],
    },
    "Good": {
        "range": "60-75",
        "advice": [
            "Use timed tests to push speed and accuracy.",
            "Strengthen weaker units while maintaining strong areas.",
            "Discuss complex topics with peers or mentors.",
        ],
    },
    "Excellent": {
        "range": "75-100",
        "advice": [
            "Maintain momentum with advanced practice sets.",
            "Help peers to reinforce your understanding.",
            "Prepare for competitive or advanced assessments.",
        ],
    },
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def calculate_attendance_percentage(sem_present_count: int, sem_absent_count: int) -> float:
    total = sem_present_count + sem_absent_count
    if total == 0:
        return 0.0
    return (sem_present_count / total) * 100


def build_feature_frame(values: dict) -> pd.DataFrame:
    frame = pd.DataFrame([values])
    return frame[FEATURE_COLUMNS]


def apply_styles() -> None:
    st.set_page_config(page_title="Student Performance Predictor", page_icon="📊", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fraunces:opsz,wght@9..144,600&display=swap');

        :root {
            --paper: #f5f2ea;
            --ink: #1f2522;
            --panel: #fffdfa;
            --accent: #0f766e;
            --accent-soft: #ccece8;
            --line: #e6dfd0;
        }

        .stApp {
            background:
              radial-gradient(circle at 15% 12%, #d4edea 0%, rgba(212, 237, 234, 0) 30%),
              radial-gradient(circle at 88% 8%, #f2dcc5 0%, rgba(242, 220, 197, 0) 28%),
              var(--paper);
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .title {
            font-family: 'Fraunces', serif;
            font-size: 2.6rem;
            line-height: 1.15;
            margin-bottom: 0.3rem;
            color: #14312d;
        }

        .subtitle {
            color: #33413f;
            font-size: 1.05rem;
            margin-bottom: 1.1rem;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(20, 49, 45, 0.08);
            margin-bottom: 1rem;
        }

        /* Force readable labels and values regardless of Streamlit theme */
        [data-testid="stWidgetLabel"] p {
            color: #1f2522 !important;
            font-size: 0.98rem !important;
            font-weight: 650 !important;
            letter-spacing: 0.1px;
        }

        [data-testid="stNumberInput"] input {
            background: #ffffff !important;
            color: #10231f !important;
            border: 1px solid #c7d8d5 !important;
            border-radius: 10px !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            min-height: 44px;
        }

        [data-testid="stNumberInput"] button {
            color: #153c37 !important;
        }

        [data-testid="stMetricValue"] {
            color: #0c4f47 !important;
            font-weight: 700 !important;
        }

        [data-testid="stDataFrame"] div, [data-testid="stTable"] div {
            color: #15201e !important;
            font-size: 0.96rem !important;
        }

        .stButton > button {
            background: linear-gradient(90deg, #0f766e, #1b9a90) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            min-height: 44px;
        }

        .result-card {
            margin-top: 10px;
            border: 1px solid #8ec6be;
            border-radius: 12px;
            padding: 16px;
            background: linear-gradient(135deg, var(--accent-soft), #f9fffe);
            animation: floatup 0.45s ease-out;
        }

        .hint {
            color: #3a4d4a;
            font-size: 0.92rem;
            margin-top: -0.3rem;
            margin-bottom: 0.7rem;
        }

        @keyframes floatup {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(model) -> None:
    classes = list(getattr(model, "classes_", []))
    st.markdown('<div class="title">Student Performance Prediction App</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">High-visibility version: clear labels, readable inputs, and strict feature ordering for your trained model.</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Model file: tuned_random_forest_model.pkl\n\n"
        f"Detected classes in this model: {classes if classes else 'N/A'}"
    )


def render_form() -> tuple[pd.DataFrame, float]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Enter Student Inputs")
    st.markdown('<div class="hint">Use marks from 0-100. Attendance is computed from present and absent counts.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        sem_eval_lec_test_1_mark = st.number_input(FIELD_LABELS["sem_eval_lec_test_1_mark"], min_value=0.0, max_value=100.0, value=20.0)
        sem_eval_lab_test_1_mark = st.number_input(FIELD_LABELS["sem_eval_lab_test_1_mark"], min_value=0.0, max_value=100.0, value=20.0)
        semester_evaluation_mid_mark = st.number_input(
            FIELD_LABELS["semester_evaluation_mid_mark"], min_value=0.0, max_value=100.0, value=25.0
        )

    with c2:
        sem_eval_lec_test_2_mark = st.number_input(FIELD_LABELS["sem_eval_lec_test_2_mark"], min_value=0.0, max_value=100.0, value=22.0)
        sem_eval_lab_test_2_mark = st.number_input(FIELD_LABELS["sem_eval_lab_test_2_mark"], min_value=0.0, max_value=100.0, value=21.0)
        semester_evaluation_pre_gtu_mark = st.number_input(
            FIELD_LABELS["semester_evaluation_pre_gtu_mark"], min_value=0.0, max_value=100.0, value=24.0
        )

    with c3:
        semester_evaluation_internal_mark = st.number_input(
            FIELD_LABELS["semester_evaluation_internal_mark"], min_value=0.0, max_value=100.0, value=23.0
        )
        sem_present_count = st.number_input("Present Count", min_value=0, value=90, step=1)
        sem_absent_count = st.number_input("Absent Count", min_value=0, value=10, step=1)

    attendance_percentage = calculate_attendance_percentage(int(sem_present_count), int(sem_absent_count))

    metric_col1, metric_col2 = st.columns([1, 2])
    with metric_col1:
        st.metric("Calculated Attendance", f"{attendance_percentage:.2f}%")
    with metric_col2:
        st.progress(min(max(attendance_percentage / 100.0, 0.0), 1.0), text="Attendance Progress")

    values = {
        "sem_eval_lec_test_1_mark": float(sem_eval_lec_test_1_mark),
        "sem_eval_lab_test_1_mark": float(sem_eval_lab_test_1_mark),
        "semester_evaluation_mid_mark": float(semester_evaluation_mid_mark),
        "sem_eval_lec_test_2_mark": float(sem_eval_lec_test_2_mark),
        "sem_eval_lab_test_2_mark": float(sem_eval_lab_test_2_mark),
        "semester_evaluation_pre_gtu_mark": float(semester_evaluation_pre_gtu_mark),
        "semester_evaluation_internal_mark": float(semester_evaluation_internal_mark),
        "attendance_percentage": float(attendance_percentage),
    }

    input_df = build_feature_frame(values)
    st.markdown("#### Model Input Preview")
    preview_df = pd.DataFrame(
        {
            "Feature": [FIELD_LABELS[col] for col in FEATURE_COLUMNS],
            "Value": [float(input_df.iloc[0][col]) for col in FEATURE_COLUMNS],
        }
    )
    st.table(preview_df)
    st.markdown('</div>', unsafe_allow_html=True)
    return input_df, attendance_percentage


def render_prediction(model, input_df: pd.DataFrame) -> None:
    if not st.button("Predict Performance", type="primary", use_container_width=True):
        return

    prediction = model.predict(input_df)[0]
    prediction = str(prediction)
    info = PERFORMANCE_INFO.get(prediction, {"range": "Unknown", "advice": ["No guidance available."]})

    st.markdown(
        f"""
        <div class="result-card">
            <h4 style="margin:0; color:#0a4a43;">Predicted Performance Category</h4>
            <p style="font-size:1.45rem; margin:6px 0 2px 0;"><b>{prediction}</b></p>
            <p style="margin:0;">Expected marks band: {info['range']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        labels = list(model.classes_)
        proba_df = pd.DataFrame({"Category": labels, "Probability": probs}).sort_values(
            by="Probability", ascending=False
        )
        st.markdown("### Prediction Confidence")

        top_prob = float(proba_df.iloc[0]["Probability"])
        second_prob = float(proba_df.iloc[1]["Probability"]) if len(proba_df) > 1 else 0.0
        margin = max(top_prob - second_prob, 0.0)

        st.info(
            f"This means the model is most confident in {prediction} ({top_prob * 100:.1f}%). "
            f"Confidence margin over second choice is {margin * 100:.1f}%."
        )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            bars = ax.barh(proba_df["Category"], proba_df["Probability"], color="#1b9a90")
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_title("Confidence by Category")
            for bar, p in zip(bars, proba_df["Probability"]):
                ax.text(min(p + 0.02, 0.96), bar.get_y() + bar.get_height() / 2, f"{p * 100:.1f}%", va="center")
            st.pyplot(fig, use_container_width=True)

        with chart_col2:
            fig2, ax2 = plt.subplots(figsize=(5.6, 3.8))
            colors = ["#0f766e", "#55b7ad", "#9cdad3", "#c7ece8"]
            ax2.pie(
                proba_df["Probability"],
                labels=proba_df["Category"],
                autopct=lambda pct: f"{pct:.1f}%" if pct >= 1 else "",
                startangle=90,
                wedgeprops={"width": 0.42, "edgecolor": "white"},
                colors=colors[: len(proba_df)],
            )
            ax2.set_title("Confidence Split")
            st.pyplot(fig2, use_container_width=True)

    st.markdown("### Input Score Breakdown")
    breakdown_cols = [
        "sem_eval_lec_test_1_mark",
        "sem_eval_lab_test_1_mark",
        "semester_evaluation_mid_mark",
        "sem_eval_lec_test_2_mark",
        "sem_eval_lab_test_2_mark",
        "semester_evaluation_pre_gtu_mark",
        "semester_evaluation_internal_mark",
        "attendance_percentage",
    ]
    breakdown_df = pd.DataFrame(
        {
            "Metric": [SHORT_LABELS[c] for c in breakdown_cols],
            "Value": [float(input_df.iloc[0][c]) for c in breakdown_cols],
        }
    )

    left_col, right_col = st.columns([2, 1])
    with left_col:
        fig3, ax3 = plt.subplots(figsize=(8.0, 4.6))
        bars = ax3.barh(breakdown_df["Metric"], breakdown_df["Value"], color="#2d8d84")
        ax3.invert_yaxis()
        ax3.set_xlim(0, 100)
        ax3.set_xlabel("Score")
        ax3.set_title("Marks and Attendance Profile")
        for bar, v in zip(bars, breakdown_df["Value"]):
            ax3.text(min(v + 1.5, 96.5), bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center")
        st.pyplot(fig3, use_container_width=True)

    with right_col:
        academic_cols = [
            "sem_eval_lec_test_1_mark",
            "sem_eval_lab_test_1_mark",
            "semester_evaluation_mid_mark",
            "sem_eval_lec_test_2_mark",
            "sem_eval_lab_test_2_mark",
            "semester_evaluation_pre_gtu_mark",
            "semester_evaluation_internal_mark",
        ]
        academic_avg = float(input_df[academic_cols].iloc[0].mean())
        attendance = float(input_df.iloc[0]["attendance_percentage"])
        st.metric("Average Academic Score", f"{academic_avg:.1f}")
        st.metric("Attendance", f"{attendance:.1f}%")
        st.caption("These indicators help you compare exam trend vs attendance trend.")

    st.markdown("### Suggested Next Steps")
    for line in info["advice"]:
        st.write(f"- {line}")


def main() -> None:
    apply_styles()

    if not MODEL_PATH.exists():
        st.error("tuned_random_forest_model.pkl not found in the project folder.")
        st.stop()

    model = load_model()
    render_header(model)
    input_df, _ = render_form()
    render_prediction(model, input_df)


if __name__ == "__main__":
    main()
