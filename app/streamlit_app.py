"""Streamlit experience for the sinGes-mini live demo."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from src.inference import Prediction, SignRecognizer, SentenceGenerator, load_display_labels
from src.utils import load_config, resolve_path

st.set_page_config(page_title="sinGes-mini Demo", page_icon="🧠", layout="wide")

LOGGER = logging.getLogger(__name__)
TEMP_UPLOAD_DIR = Path("tmp") / "web_uploads"


def _normalize_query(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


@st.cache_resource(show_spinner=True)
def load_models() -> Dict[str, object]:
    """Initialise heavy model objects once per session."""

    config = load_config()
    recognition_ckpt_dir = resolve_path(config["paths"]["recognition_checkpoint_dir"])
    recognition_ckpt = recognition_ckpt_dir / "baseline_cnn_lstm.pt"
    sign_recognizer = SignRecognizer(recognition_ckpt)

    transformer_ckpt_dir = resolve_path(config["paths"]["transformer_checkpoint_dir"])
    sentence_generator: Optional[SentenceGenerator] = None
    if transformer_ckpt_dir.exists() and any(transformer_ckpt_dir.iterdir()):
        try:
            sentence_generator = SentenceGenerator(transformer_ckpt_dir)
        except FileNotFoundError:
            LOGGER.warning("Transformer checkpoint incomplete; disabling sentence generation.")

    labels = load_display_labels(config["paths"]["dataset_root"])

    return {
        "recognizer": sign_recognizer,
        "sentence_generator": sentence_generator,
        "labels": labels,
        "raw_labels": sign_recognizer.class_names,
        "config": config,
        "paths": {
            "recognition_ckpt": recognition_ckpt,
            "transformer_ckpt_dir": transformer_ckpt_dir,
            "dataset_root": resolve_path(config["paths"]["dataset_root"]),
        },
    }


def render_sidebar(labels: List[str], dataset_root: Path) -> Dict[str, object]:
    st.sidebar.title("Control Panel")
    max_top_k = max(1, min(10, len(labels)))
    default_top_k = min(5, max_top_k)
    top_k = st.sidebar.slider("Top-K predictions", min_value=1, max_value=max_top_k, value=default_top_k)
    st.sidebar.metric("Vocabulary size", len(labels))
    st.sidebar.caption(f"Dataset root: `{dataset_root}`")
    if st.sidebar.checkbox("Show vocabulary list"):
        st.sidebar.write(", ".join(labels))
    page = st.sidebar.radio(
        "Navigate",
        options=["🎬 Live Demo", "📚 Vocabulary", "🧠 Model Insights"],
        index=0,
    )
    return {"top_k": top_k, "page": page}


def render_hero(model_data: Dict[str, object]) -> None:
    recognizer: SignRecognizer = model_data["recognizer"]  # type: ignore[assignment]
    vocab_size = len(model_data["labels"])  # type: ignore[arg-type]
    sentence_ready = bool(model_data["sentence_generator"])

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #6a5acd, #48b1bf); padding: 2.5rem; border-radius: 1.5rem; color: white;">
            <h1 style="margin-bottom: 0.25rem;">🧠 sinGes-mini Live Lab</h1>
            <p style="font-size: 1.1rem; max-width: 720px;">
                Explore real-time Indian Sign Language recognition powered by transfer learning and transformer-based sentence generation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Classes detected", len(recognizer.class_names))
    with col2:
        st.metric("Vocabulary curated", vocab_size)
    with col3:
        status = "Ready" if sentence_ready else "Fine-tune to enable"
        st.metric("Sentence generator", status)


def render_prediction_cards(predictions: List[Prediction]) -> None:
    if not predictions:
        st.info("No predictions generated yet — upload a video to begin.")
        return

    grid_cols = st.columns(min(3, len(predictions)))
    for idx, pred in enumerate(predictions):
        column = grid_cols[idx % len(grid_cols)]
        with column:
            st.markdown(
                f"""
                <div style="background: #1f2933; color: white; padding: 1.25rem; border-radius: 1rem; margin-bottom: 0.75rem;">
                    <h3 style="margin: 0; font-size: 1.1rem;">{pred.display_label}</h3>
                    <p style="margin: 0.5rem 0 0; font-size: 2rem; font-weight: 700;">{pred.score * 100:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, float(pred.score)))


def render_live_demo(model_data: Dict[str, object], settings: Dict[str, int]) -> None:
    recognizer: SignRecognizer = model_data["recognizer"]  # type: ignore[assignment]
    sentence_generator: Optional[SentenceGenerator] = model_data["sentence_generator"]  # type: ignore[assignment]

    st.subheader("🎥 Upload & Predict")
    st.write("Upload a short sign video (2–4 seconds). We'll run frame sampling, recognition, and optional sentence generation.")

    uploaded_file = st.file_uploader(
        "Drop a sign video or browse",
        type=["mp4", "mov", "avi", "mkv"],
        key="sign_video",
    )

    if uploaded_file is not None:
        TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = TEMP_UPLOAD_DIR / uploaded_file.name
        with temp_path.open("wb") as file:
            file.write(uploaded_file.read())
        st.session_state["temp_video_path"] = str(temp_path)
        st.session_state["uploaded_filename"] = uploaded_file.name

    temp_path_value = st.session_state.get("temp_video_path")
    if temp_path_value:
        temp_path = Path(temp_path_value)
        if temp_path.exists():
            st.video(str(temp_path))

            with st.spinner("Running recognizer..."):
                predictions = recognizer.predict(temp_path, top_k=settings["top_k"])
            st.session_state["latest_predictions"] = predictions
            render_prediction_cards(predictions)

            chip_html = " ".join(
                f"<span style='background:#edf2ff;color:#444;padding:0.35rem 0.7rem;border-radius:999px;margin-right:0.35rem;font-weight:600;'>"
                f"{pred.display_label}</span>"
                for pred in predictions
            )
            if chip_html:
                st.markdown(f"<div style='margin:0.5rem 0 1.5rem;'>{chip_html}</div>", unsafe_allow_html=True)

            if sentence_generator and predictions:
                words = [pred.display_label for pred in predictions]
                with st.spinner("Generating sentence..."):
                    sentence = sentence_generator.generate(words)
                st.markdown("### 🗣️ Generated Sentence")
                st.success(sentence)
            elif not sentence_generator:
                st.warning("Fine-tune the transformer (src/transformer_finetune.py) to enable sentence synthesis.")

            if st.button("Clear uploaded video", type="primary"):
                temp_path.unlink(missing_ok=True)
                st.session_state["sign_video"] = None
                st.session_state.pop("temp_video_path", None)
                st.session_state.pop("uploaded_filename", None)
                st.session_state.pop("latest_predictions", None)
                st.rerun()
        else:
            st.info("Upload a video to start the demo.")
    else:
        st.info("Upload a video to start the demo.")


def render_vocabulary_tab(model_data: Dict[str, object]) -> None:
    st.subheader("📚 Vocabulary Explorer")
    st.write("Browse the words currently recognised by the model.")

    raw_labels: List[str] = model_data["raw_labels"]  # type: ignore[assignment]
    query = st.text_input("Search vocabulary", placeholder="e.g. Hello, Thank you, Friend, tshirt")
    query_normalized = _normalize_query(query.strip())

    table_rows = []
    for idx, raw in enumerate(raw_labels, start=1):
        display_name = raw.split(".", 1)[1].strip() if "." in raw else raw
        label_normalized = _normalize_query(raw)
        display_normalized = _normalize_query(display_name)
        if query_normalized and query_normalized not in label_normalized and query_normalized not in display_normalized:
            continue
        table_rows.append({"#": idx, "Dataset Label": raw, "Display Name": display_name})

    if table_rows:
        st.dataframe(table_rows, hide_index=True, use_container_width=True)
        st.caption(f"Showing {len(table_rows)} of {len(raw_labels)} classes.")
    else:
        st.info("No vocabulary entries match your search.")


def render_model_tab(model_data: Dict[str, object]) -> None:
    st.subheader("🧠 Model Insights")
    st.write(
        "Key checkpoints and configuration powering the demo. Repeat fine-tuning to refresh checkpoints and this dashboard."
    )

    paths: Dict[str, Path] = model_data["paths"]  # type: ignore[assignment]
    recognition_ckpt: Path = paths["recognition_ckpt"]
    transformer_dir: Path = paths["transformer_ckpt_dir"]

    def format_timestamp(path: Path) -> str:
        if not path.exists():
            return "Not found"
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Recognition checkpoint", "baseline_cnn_lstm.pt")
        st.caption(f"Last updated: {format_timestamp(recognition_ckpt)}")
        st.caption(f"Location: `{recognition_ckpt}`")
    with col2:
        status = "Available" if transformer_dir.exists() and any(transformer_dir.iterdir()) else "Missing"
        st.metric("Transformer checkpoint", status)
        st.caption(f"Folder: `{transformer_dir}`")

    st.markdown("### Training configuration snapshot")
    st.json(model_data["config"]["training"])  # type: ignore[index]

    st.markdown("### Quick commands")
    st.code(
        """pyenv activate miniGes
pyenv exec python tmp/list_dataset_words.py
PYTHONPATH=$PWD pyenv exec python src/transformer_finetune.py
PYTHONPATH=$PWD pyenv exec streamlit run app/streamlit_app.py""",
        language="bash",
    )


def main() -> None:
    model_data = load_models()
    sidebar_state = render_sidebar(model_data["labels"], model_data["paths"]["dataset_root"])  # type: ignore[index]
    settings = {"top_k": int(sidebar_state["top_k"]) }
    page = sidebar_state["page"]

    render_hero(model_data)

    if page == "🎬 Live Demo":
        render_live_demo(model_data, settings)
    elif page == "📚 Vocabulary":
        render_vocabulary_tab(model_data)
    else:
        render_model_tab(model_data)


if __name__ == "__main__":
    main()
