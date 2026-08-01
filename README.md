# Fish Finger

GestureOS AI is a touchless Windows desktop controller built for a Gemini-sponsored hackathon. It combines computer vision with the Gemini API to control applications using hand gestures.

## Features

* Hand gesture recognition
* Active window detection
* Gemini-powered command selection
* Automatic keyboard shortcut execution
* Real-time webcam tracking

## Tech Stack

* Python
* OpenCV
* MediaPipe
* Google Gemini API
* PyAutoGUI
* keyboard
* pywinctl

## Architecture

```text
Camera
   ↓
MediaPipe Hands
   ↓
Gesture Recognition
   ↓
Active Window Detection
   ↓
Gemini API
   ↓
Command Executor
```

## Run

```bash
pip install -r requirements.txt
python main.py
```

Add your Gemini API key in `config.py` before running.

---

Built for MLH Hack Day 2026 - KJU
