"""Streamlit experience for the sinGes-mini live demo."""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
import os
import sys
from typing import Dict, List, Optional, Tuple

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    os.chdir(PROJECT_ROOT)
except OSError:
    pass

from src.inference import Prediction, SignRecognizer, SentenceGenerator, load_display_labels
from src.utils import load_config, resolve_path, strip_label_prefix

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency guard
    pd = None  # type: ignore

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
    if recognition_ckpt_dir.exists():
        candidates = sorted(
            recognition_ckpt_dir.glob("*.pt"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            recognition_ckpt = candidates[0]
    if not recognition_ckpt.exists():
        raise FileNotFoundError(
            f"Recognition checkpoint not found. Expected at least one .pt file in {recognition_ckpt_dir}."
        )

    sign_recognizer = SignRecognizer(recognition_ckpt)

    transformer_ckpt_dir = resolve_path(config["paths"]["transformer_checkpoint_dir"])
    sentence_generator: Optional[SentenceGenerator] = None
    if transformer_ckpt_dir.exists() and any(transformer_ckpt_dir.iterdir()):
        try:
            sentence_generator = SentenceGenerator(transformer_ckpt_dir)
        except FileNotFoundError:
            LOGGER.warning("Transformer checkpoint incomplete; disabling sentence generation.")

    dataset_root = resolve_path(config["paths"]["dataset_root"])
    labels = load_display_labels(config["paths"]["dataset_root"])

    vocabulary_records: List[Dict[str, str]] = []
    vocab_csv_path = dataset_root / "ISL_words_full.csv"
    if vocab_csv_path.exists():
        try:
            with vocab_csv_path.open("r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    vocabulary_records.append(row)
        except Exception as exc:  # pragma: no cover - resilience for malformed CSV
            LOGGER.warning("Unable to load vocabulary CSV: %s", exc)

    supported_exts = {".mp4", ".mov", ".avi", ".mkv"}
    dataset_index: Dict[str, List[Path]] = {}
    if dataset_root.exists():
        for category_dir in sorted(dataset_root.iterdir()):
            if not category_dir.is_dir():
                continue
            for label_dir in sorted(category_dir.iterdir()):
                if not label_dir.is_dir():
                    continue
                videos = sorted(path for path in label_dir.iterdir() if path.suffix.lower() in supported_exts)
                if videos:
                    dataset_index[label_dir.name] = videos[:20]

    analysis_summary = None
    analysis_path = Path("logs/analysis/latest_training_summary.json")
    if analysis_path.exists():
        try:
            analysis_summary = json.loads(analysis_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Unable to parse %s; ignoring training analysis.", analysis_path)

    return {
        "recognizer": sign_recognizer,
        "sentence_generator": sentence_generator,
        "labels": labels,
        "raw_labels": sign_recognizer.class_names,
        "config": config,
        "analysis": analysis_summary,
        "vocabulary": vocabulary_records,
        "dataset_index": dataset_index,
        "paths": {
            "recognition_ckpt": recognition_ckpt,
            "transformer_ckpt_dir": transformer_ckpt_dir,
            "dataset_root": dataset_root,
            "analysis_path": analysis_path if analysis_summary else None,
            "vocab_csv": vocab_csv_path if vocabulary_records else None,
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
        options=["🎬 Live Demo", "📚 Vocabulary", "✅ Dataset Validation", "🧠 Model Insights"],
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
    st.write("Browse or download the words the recogniser was trained on.")

    vocabulary: List[Dict[str, str]] = model_data.get("vocabulary", [])  # type: ignore[assignment]
    raw_labels: List[str] = model_data["raw_labels"]  # type: ignore[assignment]

    query = st.text_input("Search vocabulary", placeholder="e.g. Hello, Thank you, Friend, tshirt")
    query_normalized = _normalize_query(query.strip())

    rows: List[Dict[str, str]] = []
    records = vocabulary if vocabulary else [
        {"category": raw.split(".", 1)[0] if "." in raw else "unknown", "raw_label": raw, "word": strip_label_prefix(raw)}
        for raw in raw_labels
    ]

    for idx, entry in enumerate(records, start=1):
        raw_label = entry.get("raw_label", "")
        cleaned = entry.get("word", strip_label_prefix(raw_label))
        normalized_raw = _normalize_query(raw_label)
        normalized_cleaned = _normalize_query(cleaned)
        if query_normalized and query_normalized not in normalized_raw and query_normalized not in normalized_cleaned:
            continue
        rows.append({
            "#": idx,
            "Category": entry.get("category", "unknown"),
            "Dataset Label": raw_label,
            "Word": cleaned,
        })

    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption(f"Showing {len(rows)} of {len(records)} classes.")
    else:
        st.info("No vocabulary entries match your search.")

    vocab_csv_path: Optional[Path] = model_data["paths"].get("vocab_csv")  # type: ignore[index]
    if vocab_csv_path and vocab_csv_path.exists():
        st.download_button(
            label="Download vocabulary CSV",
            data=vocab_csv_path.read_bytes(),
            file_name=vocab_csv_path.name,
            mime="text/csv",
            key="download_vocab_csv",
        )

    if pd is not None and rows:
        vocab_df = pd.DataFrame(records)
        vocab_df["category"] = vocab_df["category"].fillna("unknown")
        st.markdown("#### Vocabulary distribution by category")
        category_counts = vocab_df["category"].value_counts().sort_values(ascending=False).head(15)
        if not category_counts.empty:
            st.bar_chart(category_counts.rename_axis("Category").to_frame("Count"), use_container_width=True)

        st.markdown("#### Word length distribution")
        length_counts = (
            vocab_df["word"].dropna().str.len().clip(lower=1).value_counts().sort_index()
        )
        if not length_counts.empty:
            st.area_chart(length_counts.rename_axis("Characters").to_frame("Words"), use_container_width=True)


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
        st.metric("Recognition checkpoint", recognition_ckpt.name)
        st.caption(f"Last updated: {format_timestamp(recognition_ckpt)}")
        st.caption(f"Location: `{recognition_ckpt}`")
    with col2:
        status = "Available" if transformer_dir.exists() and any(transformer_dir.iterdir()) else "Missing"
        st.metric("Transformer checkpoint", status)
        st.caption(f"Folder: `{transformer_dir}`")

    analysis = model_data.get("analysis")
    if analysis:
        best_epoch = analysis.get("best_epoch")
        epochs = analysis.get("epochs", [])

        st.markdown("### Latest training summary")
        if best_epoch:
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Best epoch", int(best_epoch.get("epoch", 0)))
            col_b.metric("Val accuracy", f"{best_epoch.get('val_acc', 0.0) * 100:.1f}%")
            col_c.metric("Top-5", f"{best_epoch.get('val_top5', 0.0) * 100:.1f}%")
            col_d.metric("Train acc", f"{best_epoch.get('train_acc', 0.0) * 100:.1f}%")
            st.caption(
                f"Train loss: {best_epoch.get('train_loss', 0.0):.3f} | Val loss: {best_epoch.get('val_loss', 0.0):.3f}"
            )

        if epochs:
            styled_rows = []
            for row in epochs:
                styled_rows.append(
                    {
                        "Epoch": int(row.get("epoch", 0)),
                        "Train Acc": float(row.get("train_acc", 0.0)) * 100,
                        "Val Acc": float(row.get("val_acc", 0.0)) * 100,
                        "Val Top-5": float(row.get("val_top5", 0.0)) * 100,
                        "Train Loss": float(row.get("train_loss", 0.0)),
                        "Val Loss": float(row.get("val_loss", 0.0)),
                    }
                )

            df = pd.DataFrame(styled_rows) if pd else None
            if df is not None:
                df_numeric = df.sort_values("Epoch").set_index("Epoch")
                st.markdown("#### Metrics progression")
                acc_col, loss_col = st.columns(2)
                with acc_col:
                    st.line_chart(df_numeric[["Train Acc", "Val Acc"]], height=260, use_container_width=True)
                with loss_col:
                    st.line_chart(df_numeric[["Train Loss", "Val Loss"]], height=260, use_container_width=True)

                extras_col_left, extras_col_right = st.columns(2)
                with extras_col_left:
                    st.bar_chart(df_numeric[["Val Top-5"]], height=240, use_container_width=True)
                with extras_col_right:
                    df_numeric["Accuracy Gap"] = df_numeric["Val Acc"] - df_numeric["Train Acc"]
                    st.area_chart(df_numeric[["Accuracy Gap"]], height=240, use_container_width=True)

                df_numeric["Train Loss (rolling)"] = df_numeric["Train Loss"].rolling(window=3, min_periods=1).mean()
                df_numeric["Val Loss (rolling)"] = df_numeric["Val Loss"].rolling(window=3, min_periods=1).mean()
                st.markdown("#### Rolling loss (window = 3)")
                st.line_chart(df_numeric[["Train Loss (rolling)", "Val Loss (rolling)"]], height=220, use_container_width=True)

                st.markdown("#### Epoch detail")
                st.dataframe(
                    df_numeric[["Train Acc", "Val Acc", "Val Top-5", "Train Loss", "Val Loss", "Accuracy Gap"]]
                    .round({"Train Acc": 2, "Val Acc": 2, "Val Top-5": 2, "Train Loss": 3, "Val Loss": 3, "Accuracy Gap": 2}),
                    use_container_width=True,
                )

                top_epochs = df_numeric.sort_values("Val Acc", ascending=False).head(5)
                bottom_epochs = df_numeric.sort_values("Val Acc", ascending=True).head(5)
                top_col, bottom_col = st.columns(2)
                with top_col:
                    st.markdown("##### Top validation epochs")
                    st.table(top_epochs[["Val Acc", "Train Acc", "Val Loss"]].round(2))
                with bottom_col:
                    st.markdown("##### Lowest validation epochs")
                    st.table(bottom_epochs[["Val Acc", "Train Acc", "Val Loss"]].round(2))
            else:
                st.dataframe(
                    [
                        {
                            "Epoch": row["Epoch"],
                            "Train Acc (%)": f"{row['Train Acc']:.1f}",
                            "Val Acc (%)": f"{row['Val Acc']:.1f}",
                            "Val Top-5 (%)": f"{row['Val Top-5']:.1f}",
                            "Train Loss": f"{row['Train Loss']:.3f}",
                            "Val Loss": f"{row['Val Loss']:.3f}",
                        }
                        for row in styled_rows
                    ],
                    hide_index=True,
                    use_container_width=True,
                )

        latest_archive = analysis.get("latest_archive")
        if latest_archive:
            st.caption(f"Latest run archive: `{latest_archive}`")
    else:
        st.info("Run `python tmp/training_log_summary.py` after training to populate analytics.")

    st.markdown("### Training configuration snapshot")
    st.json(model_data["config"]["training"]["recognition"])  # type: ignore[index]

    st.markdown("### Quick commands")
    st.code(
        """pyenv activate miniGes
pyenv exec python tmp/list_dataset_words.py
PYTHONPATH=$PWD pyenv exec python src/transformer_finetune.py
PYTHONPATH=$PWD pyenv exec streamlit run app/streamlit_app.py""",
        language="bash",
    )


def render_dataset_validation(model_data: Dict[str, object], settings: Dict[str, int]) -> None:
    st.subheader("✅ Dataset Validation Playground")
    st.write("Pick an existing dataset clip, run the recogniser, and compare against the ground-truth label.")

    recognizer: SignRecognizer = model_data["recognizer"]  # type: ignore[assignment]
    dataset_index: Dict[str, List[Path]] = model_data.get("dataset_index", {})  # type: ignore[assignment]

    if not dataset_index:
        st.info("Dataset index unavailable. Run `python tmp/list_dataset_words.py` and ensure dataset videos are present.")
        return

    label_options: List[Tuple[str, str, int]] = []
    for raw_label, videos in sorted(dataset_index.items()):
        label_options.append((raw_label, strip_label_prefix(raw_label), len(videos)))

    if not label_options:
        st.info("No videos discovered in the dataset root.")
        return

    option_strings = [f"{display} ({count} clips)" for _, display, count in label_options]
    selected_index = st.selectbox("Choose label", list(range(len(option_strings))), format_func=lambda idx: option_strings[idx])
    selected_raw, selected_display, _ = label_options[selected_index]

    video_paths = dataset_index.get(selected_raw, [])
    if not video_paths:
        st.warning("No videos available for the selected label.")
        return

    clip_names = [path.name for path in video_paths]
    default_clip = st.session_state.pop("validation_random", None)
    try:
        default_index = clip_names.index(default_clip) if default_clip else 0
    except ValueError:
        default_index = 0

    chosen_name = st.selectbox("Choose clip", clip_names, index=default_index)
    chosen_path = next(path for path in video_paths if path.name == chosen_name)

    if st.button("Evaluate clip", type="primary"):
        st.video(str(chosen_path))
        with st.spinner("Running recogniser..."):
            predictions = recognizer.predict(chosen_path, top_k=settings["top_k"])

        render_prediction_cards(predictions)

        top1_correct = bool(predictions) and predictions[0].label == selected_raw
        topk_correct = any(pred.label == selected_raw for pred in predictions)

        col1, col2 = st.columns(2)
        col1.metric("Expected", selected_display)
        if predictions:
            col2.metric("Top-1", strip_label_prefix(predictions[0].label))
        else:
            col2.metric("Top-1", "—")

        if top1_correct:
            st.success("Top-1 prediction matched the ground truth!")
        elif topk_correct:
            st.warning("Ground truth present in top-K predictions, but not at rank 1.")
        else:
            st.error("Ground truth not found in current top-K predictions.")

        if st.button("Try another random clip", key="rand_clip"):
            random_path = random.choice(video_paths)
            st.session_state["validation_random"] = random_path.name
            st.experimental_rerun()


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
    elif page == "✅ Dataset Validation":
        render_dataset_validation(model_data, settings)
    else:
        render_model_tab(model_data)


if __name__ == "__main__":
    main()
