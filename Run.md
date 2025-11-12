To train the full pipeline end to end, run these commands from the project root (after activating your venv and installing [requirements.txt](cci:7://file:///home/aditya/workhub/python/project/sinGes%28mini%29/requirements.txt:0:0-0:0)):

1. Extract MediaPipe landmarks (only needed once, or when raw videos change):
```bash
python src/extract_mediapipe_landmarks.py --config config/config.yaml --skip-existing
```

2. Train the lightweight keypoint model (uses the landmarks above):
```bash
python src/train_keypoint_model.py --config config/config.yaml --num-workers 2
```

3. Train the video recognition model:
```bash
python src/train_recognition.py
```

That sequence will leave you with fresh checkpoints for both the keypoint and video models under `models/checkpoints/…`. After training, you can optionally export/quantize and benchmark using the shell scripts we added earlier.