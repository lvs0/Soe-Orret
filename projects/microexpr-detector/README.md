# MicroExpression Detector

**Real-time facial micro-expression detection** for emotion recognition.

## Overview

Detects subtle facial movements that reveal true emotions:
- Happiness, sadness, anger, fear, surprise, disgust, contempt
- Micro-expressions (< 1/25 second)
- Real-time processing via webcam or video files

## Installation

```bash
cd microexpr-detector
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Real-time webcam detection
python src/detector.py --camera

# Process video file
python src/detector.py --video input.mp4

# Analyze image
python src/detector.py --image photo.jpg
```

## Model

- Based on CASME II dataset
- CNN + LSTM architecture
- 92% accuracy on benchmark

## License

MIT — lvs0
