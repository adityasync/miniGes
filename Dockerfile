FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV, MediaPipe, and scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsm6 \
        libxext6 \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.headless=true", "--server.port=8501", "--server.address=0.0.0.0"]
