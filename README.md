# AI Virtual Mouse using Hand Tracking in Python

A real-time Computer Vision application that turns your webcam into a touchless **Virtual Mouse** using **OpenCV**, **MediaPipe**, and **PyAutoGUI**. Control your computer with natural hand gestures without touching a physical mouse or trackpad.

---

## Features

- **Touchless Cursor Navigation**: Smooth pointer movement with index finger tracking and exponential moving average (EMA) jitter reduction.
- **Gesture-Based Clicks**:
  - **Left Click**: Pinch Index & Middle finger tips together (< 35px).
  - **Right Click**: Pinch Thumb & Middle finger tips together (< 35px).
  - **Drag & Drop**: Pinch & hold Thumb & Index finger tips while moving.
- **Vertical Page Scrolling**: Raise Index & Middle fingers together and move your hand up/down to scroll up/down.
- **HUD & Diagnostics**: Real-time FPS counter, tracking boundary box, and active gesture visual indicators.
- **Failsafe & Security**: Retains PyAutoGUI safety failsafe (slamming cursor to screen corner stops the process).

---

## Gestures Guide

| Action | Hand Gesture | Visual Feedback |
| :--- | :--- | :--- |
| **Cursor Movement** | Raised **Index Finger** (others down) | Green dot on index fingertip |
| **Left Click** | Pinch **Index + Middle Finger** tips | Green indicator at midpoint |
| **Right Click** | Pinch **Thumb + Middle Finger** tips | Blue indicator at midpoint |
| **Scroll Up/Down** | Raise **Index + Middle** & move vertically | Yellow banner on HUD |
| **Drag & Drop** | Pinch **Thumb + Index Finger** and drag | Orange circle at midpoint |

---

## Installation & Setup

### 1. Prerequisites
- Python 3.8+ installed on your computer.
- A functional webcam.

### 2. Set Up Virtual Environment & Dependencies

Open a command prompt or terminal in the project directory:

```bash
# Create a Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## How to Run

1. Make sure your virtual environment is activated.
2. Launch the Virtual Mouse script:
   ```bash
   python virtual_mouse.py
   ```
3. A window named **"AI Virtual Mouse - Camera Feed"** will appear showing your camera view.
4. Move your index finger inside the **purple boundary box** to control your screen cursor.
5. Press **`q`** key on your keyboard while focusing the video window to quit the application cleanly.

---

## Module Overview

- `hand_tracking_module.py`: A modular `HandDetector` class wrapping MediaPipe Hand Landmark detection, landmark coordinate retrieval, finger status checking, and distance calculations.
- `virtual_mouse.py`: The main controller application integrating camera processing, smoothing filters, screen coordinate interpolation, gesture logic, and OS mouse automation via PyAutoGUI.
- `requirements.txt`: List of dependencies (`opencv-python`, `mediapipe`, `pyautogui`, `numpy`).

---

## Troubleshooting & Tips

- **Cursor moving too fast or jumpy?**
  Adjust the `smoothing_factor` parameter in `virtual_mouse.py` (higher values like `7` or `10` make movement smoother; lower values like `3` make it snappier).
- **Webcam not opening?**
  If using an external camera, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `virtual_mouse.py`.
- **Failsafe Triggered Exception?**
  If PyAutoGUI throws a `FailSafeException`, it means your mouse cursor hit the top-left (0, 0) corner of the screen. Keep your hand inside the active control box or move it back to center.
