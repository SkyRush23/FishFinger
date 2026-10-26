# Fish Finger

FishFinger is a gesture-based desktop application designed to enable users to interact with and control their Windows computers using predefined hand gestures. The project aims to provide an alternative to traditional input devices such as a mouse and keyboard by making desktop interaction more natural, convenient and intuitive.

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

25BCNB53
