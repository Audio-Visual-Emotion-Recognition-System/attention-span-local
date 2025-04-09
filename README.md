# Attention Span Local

This repository is part of the Audio-Visual Emotion Recognition System and focuses on detecting local attention spans using computer vision and deep learning techniques.

## Features

- Detects emotions from visual cues in video input.
- Implements gaze tracking and face landmark detection.
- Uses pre-trained models for emotion recognition.

## File Structure

- `server.py`: Main application file to run the server.
- `requirements.txt`: Python packages required to run the system.
- `haarcascade_frontalface_default.xml`: Face detection model.
- `shape_predictor_68_face_landmarks.dat`: Face landmark detection model.
- `lbfmodel.yaml`: Gaze tracking model.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Audio-Visual-Emotion-Recognition-System/attention-span-local.git
    cd attention-span-local
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the server with:

```bash
python server.py
```
