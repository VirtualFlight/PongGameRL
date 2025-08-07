# PongGameRL

A reinforcement learning environment for automating and training agents to play Pong using screen capture and keyboard control.

## Features

- Uses OpenAI Gymnasium interface for RL compatibility.
- Captures game frames using `mss` and processes them with `cv2`.
- Controls the game via `pyautogui` for mouse and keyboard automation.
- Detects game end state using OCR (`pytesseract`) on winner/loser text.
- Includes helper functions for mouse position and image display.

## Requirements

- Python 3.10.13 (use a virtual environment)
- `mss`
- `opencv-python`
- `pyautogui`
- `numpy`
- `pytesseract`
- `matplotlib`
- `gymnasium`
- `stable-baselines3`

Install dependencies:

```sh
pip install mss opencv-python pyautogui numpy pytesseract matplotlib gymnasium stable-baselines3
```