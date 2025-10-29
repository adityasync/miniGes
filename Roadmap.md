## VIRTUAL ENV
pyenv activate miniGes

## PROJECT OBJECTIVE
Build a complete end-to-end Indian Sign Language (ISL) recognition application that:
1. ✅ Recognizes 30 ISL word signs from video input
2. 🤖 Uses a transformer model to convert recognized words into grammatically correct sentences
3. 💻 Provides a user-friendly interface for real-time sign language recognition

***

## 📊 DATASET
- **Source**: INCLUDE Dataset (Zenodo record 4010759)
- **Status**: ✅ Already downloaded and ready
- **Structure**: 30 words in category folders (Pronouns, Greetings, Adjectives, Days_and_Time, People, Colours, Animals, Objects_at_Home, Places)
- **Format**: Video files (1920x1080, 25fps, ~2.57 seconds each)
- **Vocabulary**: See provided CSV files with 30 words categorized by priority

***

## 🛠️ TECHNICAL REQUIREMENTS

### 1. Data Preprocessing Module
- Extract frames from ISL videos
- Apply **MediaPipe** or OpenPose for hand landmark detection
- Normalize and augment data (rotation, scaling, brightness)
- Split: 70% train / 15% validation / 15% test
- Handle class imbalance
- Save preprocessed features (HDF5 or NPY format)

### 2. Sign Recognition Model
**Architecture Options** (choose best performing):
- **Option A**: CNN-LSTM (3D convolutions + temporal modeling)
- **Option B**: I3D (Inflated 3D ConvNet)
- **Option C**: Transformer-based (TimeSformer or VideoMAE)

**Requirements**:
- Input: Preprocessed video frames or hand landmarks
- Output: 30-class classification
- Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Model checkpointing

### 3. Sentence Formation Transformer
- **Seq2Seq Transformer**: Word sequence → Grammatical sentence
- Use attention mechanism for word order
- Handle missing words gracefully
- **Training**: Generate synthetic data from 30-word vocabulary combinations

**Example transformations**:
- `["I", "happy", "today"]` → `"I am happy today"`
- `["she", "beautiful"]` → `"She is beautiful"`
- `["hello", "friend"]` → `"Hello, my friend"`

### 4. Real-time Inference Pipeline
- Video capture (webcam or file upload)
- Sliding window recognition
- Accumulate recognized words
- Pass to transformer for sentence generation
- Display with confidence scores

### 5. User Interface (Choose one)
- **Streamlit** ⭐ (recommended for quick deployment)
- **Gradio** (simple ML interface)
- **Flask/FastAPI + React** (production-ready)

**Features**:
- ✅ Video upload or webcam capture
- ✅ Real-time sign recognition display
- ✅ Word sequence visualization
- ✅ Generated sentence output
- ✅ Confidence scores
- ✅ Manual correction option
- ✅ Export recognized text

### 6. Model Training Scripts
- Data preprocessing and feature extraction
- Sign recognition model training + hyperparameter tuning
- Transformer sentence generation training
- Model evaluation and validation
- Use **config files** (YAML/JSON)
- Implement logging (TensorBoard or Weights & Biases)

### 7. Deployment & Packaging
- **Dockerize** the application
- Create `requirements.txt`
- Model versioning and experiment tracking
- Optimize for inference (quantization, pruning)
- API endpoints (optional)

***

## 💻 TECHNOLOGY STACK

### Deep Learning
- PyTorch or TensorFlow/Keras
- Hugging Face Transformers
- OpenCV (video processing)
- MediaPipe or OpenPose

### Data & Utilities
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Albumentations (augmentation)

### Interface & Deployment
- Streamlit or Gradio
- Docker
- FastAPI (optional)
- Git/GitHub

***

## 📁 PROJECT STRUCTURE

```
isl-recognition/
├── data/
│   ├── raw/                    # Original INCLUDE dataset
│   ├── processed/              # Preprocessed features
│   └── train_test_split/       # Train/val/test splits
├── models/
│   ├── sign_recognition/       # CNN-LSTM or I3D models
│   ├── transformer/            # Sentence generation transformer
│   └── checkpoints/            # Saved model weights
├── src/
│   ├── data_preprocessing.py   # Video preprocessing pipeline
│   ├── train_recognition.py    # Train sign recognition model
│   ├── train_transformer.py    # Train sentence transformer
│   ├── inference.py            # Real-time inference engine
│   └── utils.py                # Helper functions
├── notebooks/
│   ├── eda.ipynb              # Exploratory data analysis
│   ├── model_experiments.ipynb # Model experiments
│   └── evaluation.ipynb        # Evaluation and metrics
├── app/
│   ├── streamlit_app.py       # Main UI application
│   └── api.py                 # API endpoints (optional)
├── config/
│   ├── config.yaml            # Training configurations
│   └── model_config.yaml      # Model hyperparameters
├── tests/
│   └── test_*.py              # Unit tests
├── requirements.txt
├── Dockerfile
└── README.md
```

***

## 📋 DELIVERABLES

1. ✅ **Complete Working Application** (all components integrated)
2. 🎯 **Trained Models** (>85% accuracy target)
3. 📖 **Documentation**:
   - README with setup instructions
   - API documentation
   - Model architecture diagrams
   - Training procedures
4. 📓 **Jupyter Notebooks**:
   - Dataset exploration
   - Model training process
   - Results visualization
5. 🎥 **Demo Video** showing application in action

***

## 🗓️ 10-DAY DEVELOPMENT WORKFLOW

### Phase 1: Data Preparation (Day 1-2)
- Load and explore INCLUDE dataset
- Implement preprocessing pipeline
- Extract features (frames or landmarks)
- Create train/val/test splits
- Verify data quality

### Phase 2: Sign Recognition Model (Day 3-5)
- Implement baseline model (CNN-LSTM)
- Train and validate
- Experiment with architectures
- Hyperparameter optimization
- Achieve >85% accuracy

### Phase 3: Transformer Development (Day 6-7)
- Generate synthetic training data
- Implement transformer encoder-decoder
- Train on word sequences → sentences
- Evaluate and fine-tune

### Phase 4: Integration & UI (Day 8-9)
- Build inference pipeline
- Create Streamlit/Gradio interface
- Integrate recognition + transformer
- Real-time video processing
- End-to-end testing

### Phase 5: Testing & Deployment (Day 10)
- Comprehensive testing
- Bug fixes and optimization
- Docker containerization
- Documentation
- Demo preparation

***

## ✅ SUCCESS CRITERIA
- 🎯 Sign recognition accuracy: **>85%** on test set
- 📝 Transformer generates grammatically correct sentences: **>90%**
- ⚡ Real-time processing: **<2 seconds** latency per sign
- 💻 User-friendly interface with clear visualizations
- 📚 Well-documented, reproducible code
- 🐳 Deployable Docker container

***

## 📌 ADDITIONAL NOTES
- Prioritize **CRITICAL** and **HIGH** priority words (17 words) initially
- Use **transfer learning** (pretrained I3D, VideoMAE)
- Implement **data augmentation**
- Add error handling and logging
- Make code modular and maintainable
- Handle edge cases (unclear signs, multiple signs)

## 📂 REFERENCE FILES PROVIDED
- `ISL_30_words_complete.csv` - Complete vocabulary with metadata
- `ISL_30_words_by_priority.csv` - Priority-based grouping
- `ISL_30_words_by_category.csv` - Category-wise distribution

***

**🚀 START**: Set up project structure → Data preprocessing → Build end-to-end pipeline → Optimize components

***

## 🎬 HOW TO USE THIS PROMPT

1. **Copy the entire prompt above**
2. **Open Windsurf AI coding assistant**
3. **Paste the prompt**
4. **Windsurf will**:
   - Create complete project structure
   - Generate all Python scripts
   - Set up configuration files
   - Implement models and pipelines
   - Create UI application
   - Add documentation

5. **Follow the 10-day workflow** for systematic development
