"""Gradio interface for sinGes-mini demo."""

from __future__ import annotations

import gradio as gr
from pathlib import Path
from typing import List

from src.inference import Prediction, SignRecognizer, SentenceGenerator, load_display_labels
from src.utils import load_config, resolve_path


def load_models():
    config = load_config()
    recognition_ckpt_dir = resolve_path(config["paths"]["recognition_checkpoint_dir"])
    recognition_ckpt = recognition_ckpt_dir / "baseline_cnn_lstm.pt"
    sign_recognizer = SignRecognizer(recognition_ckpt)

    transformer_ckpt_dir = resolve_path(config["paths"]["transformer_checkpoint_dir"])
    sentence_generator = None
    if transformer_ckpt_dir.exists() and list(transformer_ckpt_dir.iterdir()):
        try:
            sentence_generator = SentenceGenerator(transformer_ckpt_dir)
        except FileNotFoundError:
            sentence_generator = None

    labels = load_display_labels(config["paths"]["dataset_root"])
    return sign_recognizer, sentence_generator, labels


SIGN_RECOGNIZER, SENTENCE_GENERATOR, LABELS = load_models()


def predict(video_file, top_k):
    if video_file is None:
        return [], ""
    video_path = Path(video_file)
    predictions = SIGN_RECOGNIZER.predict(video_path, top_k=top_k)
    prediction_list = [f"{pred.display_label}: {pred.score * 100:.2f}%" for pred in predictions]
    sentence = ""
    if SENTENCE_GENERATOR:
        words = [pred.display_label for pred in predictions]
        sentence = SENTENCE_GENERATOR.generate(words)
    return prediction_list, sentence


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="sinGes-mini Gradio Demo") as demo:
        gr.Markdown("""
        # sinGes-mini: Indian Sign Language Demo

        Upload a short video clip of an Indian Sign Language sign. The model will recognize the sign and optionally generate a sentence using the predicted words.
        """)

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload sign video")
                top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Top-K Predictions")
                run_button = gr.Button("Run Inference")
            with gr.Column():
                prediction_output = gr.List(label="Predictions")
                sentence_output = gr.Textbox(label="Generated Sentence", lines=3)

        run_button.click(
            fn=predict,
            inputs=[video_input, top_k_slider],
            outputs=[prediction_output, sentence_output],
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
