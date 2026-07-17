from langgraph.graph import StateGraph, START, END
from detection import *
from nodes import *

graph = StateGraph(State)
graph.add_node(defect_description_node, 'defect_description_node')
graph.add_node(steps_to_correct_node, 'steps_to_correct_node')
graph.add_node(financial_feasibility_node, 'financial_feasibility_node')

graph.add_edge(START, 'defect_description_node')
graph.add_edge('defect_description_node', 'steps_to_correct_node')
graph.add_edge('steps_to_correct_node', 'financial_feasibility_node')
graph.add_edge('financial_feasibility_node', END)

agent = graph.compile()


# =============================================================================
# STREAMLIT DASHBOARD
# Run with:  streamlit run graph.py
# =============================================================================

import os
import tempfile
import streamlit as st

st.set_page_config(page_title="SurfaceIQ", page_icon="🔩", layout="wide")


def css(text: str) -> str:
    """Flatten every line to column 0 before handing to st.markdown, so
    nested CSS indentation never gets misread by Markdown as a code block."""
    return "\n".join(line.strip() for line in text.strip("\n").splitlines())


st.markdown(
    css("""
    <style>
    .stApp { background: #101418; color: #E9E9E9; }
    section[data-testid="stSidebar"] { background: #171B20; }

    h1, h2, h3 { font-family: 'Trebuchet MS', sans-serif; }

    .step-card {
        background: #171B20;
        border: 1px solid #2A2F36;
        border-left: 4px solid #4A90A4;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 10px 0;
    }
    .step-card.pending { opacity: 0.4; }
    .step-card.financial { border-left-color: #E8825D; }

    .step-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .step-body {
        font-size: 0.92rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }

    .arrow {
        text-align: center;
        font-size: 1.4rem;
        color: #4A90A4;
        margin: -4px 0;
    }

    .badge {
        display: inline-block;
        background: #1F262C;
        border: 1px solid #2A2F36;
        border-radius: 16px;
        padding: 5px 12px;
        margin: 3px 6px 3px 0;
        font-size: 0.82rem;
    }
    </style>
    """),
    unsafe_allow_html=True,
)

st.title("🔩 SurfaceIQ")
st.caption("Upload a metal surface image — YOLOv8 detects the defect, then three agents analyze it step by step.")

with st.sidebar:
    st.subheader("Settings")
    conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)

uploaded_file = st.file_uploader("Upload a metal surface image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, width=420)
    run_clicked = st.button("▶ Run Inspection")
else:
    run_clicked = False

if run_clicked:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # ---- Step 0: Detection ----
    with st.spinner("Running YOLOv8 detection..."):
        results = model.predict(tmp_path, conf=conf_threshold)
        defects_readable = get_defect(tmp_path)

    if not defects_readable:
        st.warning("No anomalies detected above the confidence threshold. Try lowering it in the sidebar.")
    else:
        st.subheader("🎯 Detected Anomalies")
        badges = "".join(f'<span class="badge">{d}</span>' for d in defects_readable)
        st.markdown(badges, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Agent Pipeline")

        state = State(
            defect_names=defects_readable,
            description=[],
            steps_to_correct=[],
            financial_feasibility=[],
        )

        # Placeholders for the three flowchart steps, filled in one at a time
        step1 = st.empty()
        arrow1 = st.empty()
        step2 = st.empty()
        arrow2 = st.empty()
        step3 = st.empty()

        step1.markdown(
            '<div class="step-card pending"><div class="step-title">🔍 1. Diagnostician</div>'
            '<div class="step-body">Waiting...</div></div>',
            unsafe_allow_html=True,
        )
        arrow1.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)
        step2.markdown(
            '<div class="step-card pending"><div class="step-title">🛠️ 2. Solutions Engineer</div>'
            '<div class="step-body">Waiting...</div></div>',
            unsafe_allow_html=True,
        )
        arrow2.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)
        step3.markdown(
            '<div class="step-card pending"><div class="step-title">💰 3. Financial Analyst</div>'
            '<div class="step-body">Waiting...</div></div>',
            unsafe_allow_html=True,
        )

        # --- Step 1: Diagnostician ---
        with st.spinner("Diagnosing defects..."):
            state = defect_description_node(state)
        step1.markdown(
            css(f"""
            <div class="step-card">
                <div class="step-title">🔍 1. Diagnostician</div>
                <div class="step-body">{" | ".join(state.description)}</div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        # --- Step 2: Solutions Engineer ---
        with st.spinner("Working out remediation steps..."):
            state = steps_to_correct_node(state)
        step2.markdown(
            css(f"""
            <div class="step-card">
                <div class="step-title">🛠️ 2. Solutions Engineer</div>
                <div class="step-body">{" | ".join(state.steps_to_correct)}</div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        # --- Step 3: Financial Analyst ---
        with st.spinner("Assessing financial feasibility..."):
            state = financial_feasibility_node(state)
        step3.markdown(
            css(f"""
            <div class="step-card financial">
                <div class="step-title">💰 3. Financial Analyst</div>
                <div class="step-body">{" | ".join(state.financial_feasibility)}</div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    os.unlink(tmp_path)