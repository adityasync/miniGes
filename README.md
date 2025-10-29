# sinGes (mini) – Indian Sign Language Recognition

## Project Overview
This project aims to deliver an end-to-end Indian Sign Language (ISL) recognition system that can classify 30 ISL word-level signs from short video clips and convert the recognized word sequence into grammatically correct sentences. A lightweight yet extensible architecture is planned to accommodate iterative improvements across data preprocessing, model training, and user interface layers.

The immediate goals for the project are:

1. Establish a clean, modular project structure that mirrors the roadmap.
2. Build reproducible data preprocessing pipelines for frame extraction and landmark generation.
3. Prototype a baseline sign-recognition model (CNN-LSTM / 3D CNN).
4. Implement a transformer-based sentence generator trained on synthetic sequences derived from the vocabulary.
5. Provide a user-facing Streamlit interface for interactive demonstrations.

## Repository Structure
```text
sinGes(mini)/
├── Roadmap.md
├── dataset/                  # INCLUDE dataset (30 ISL words)
├── data/
│   ├── raw/
│   ├── processed/
│   └── train_test_split/
├── models/
│   ├── checkpoints/
│   │   ├── sign_recognition/
│   │   └── transformer/
│   ├── sign_recognition/
│   └── transformer/
├── src/
│   ├── data_preprocessing.py
│   ├── inference.py
│   ├── train_recognition.py
│   ├── train_transformer.py
│   └── utils.py
├── app/
│   ├── streamlit_app.py
│   └── api.py
├── config/
│   ├── config.yaml
│   └── model_config.yaml
├── notebooks/
│   └── README.md
├── tests/
│   └── test_placeholder.py
├── requirements.txt
└── Dockerfile
```

## Getting Started

### Prerequisites
- Python 3.10 (managed via `pyenv`)
- `miniGes` virtual environment (already configured locally)
- NVIDIA GPU with CUDA 12.x drivers and toolkit properly installed (training is enforced to run on CUDA).

### Setup
```bash
pyenv activate miniGes
pip install -r requirements.txt
```

The pinned versions in `requirements.txt` mirror `mainappRequirement.txt` to ensure parity with the main application stack.

### Environment Variables
Create a `.env` file at the project root if you need to store API keys or other sensitive configuration values for deployment. Do **not** commit the `.env` file to version control.

## Workflow Summary
1. **Data Preparation** – Use `src/data_preprocessing.py` to extract frames or landmarks from the INCLUDE dataset and materialize train/validation/test splits in `data/`.
2. **Model Training** – Configure parameters in `config/model_config.yaml` and run `src/train_recognition.py` to train the sign recognition model.
3. **Sentence Generation** – Train the transformer model with synthetic sequences using `src/train_transformer.py` and persist checkpoints under `models/checkpoints/transformer/`.
4. **Inference Pipeline** – Execute end-to-end predictions through `src/inference.py`, which orchestrates preprocessing, classification, and sentence generation.
5. **User Interface** – Launch `app/streamlit_app.py` for an interactive demo that showcases real-time recognition and sentence formation.

## Next Steps
- Populate preprocessing scripts with MediaPipe/OpenPose feature extraction.
- Implement data loaders and baseline CNN-LSTM training loop.
- Generate synthetic sentence pairs for transformer training.
- Hook trained models into the inference pipeline and Streamlit application.
- Add unit tests and experiment tracking (TensorBoard or Weights & Biases).

## References
- INCLUDE Dataset: Zenodo record 4010759.
- MediaPipe Hands: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- VideoMAE: https://arxiv.org/abs/2203.12602

---
For further guidance, consult `Roadmap.md`, which contains a phased development plan and success criteria.
