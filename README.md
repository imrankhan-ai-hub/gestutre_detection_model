Gesture Detection

Control your Windows PC with hand gestures from a webcam. Hold up an open palm to arm the system, then swipe to switch desktops, point to move the mouse, pinch to click or drag, show a peace sign to scroll, and use a fist or thumbs up/down for media keys.

 webcam ─▶ MediaPipe ─▶ 21 hand points ─▶ gesture name ─▶ actions ─▶ mouse / keyboard
camera.py  processor.py                    gestures.py     actions.py
                          main.py connects everything and draws the preview window

Everything runs locally on your computer. No video is saved or sent anywhere.

Features
Arming gesture: nothing happens until you hold an open palm for 1 second, so normal hand movement never triggers anything.
13 recognised poses: open palm, four, three, peace, point, L shape, rock, shaka, thumbs up, thumbs down, fist, pinch, OK sign.
Swipe left / right / up / down (desktop switching, task view, show desktop).
Cursor control with the index knuckle, smoothed, mapped from a box in the camera image to the whole screen.
Click and drag by pinching thumb and index (with hysteresis so it does not flicker).
Scroll by moving your hand up and down with a peace sign.
Media keys: play/pause, mute, next track, volume up/down.
Dry-run mode (default): prints what would happen without touching your mouse or keyboard.
Live on-screen feedback: mode, detected pose, arming progress bar, cursor target, last action.
Requirements
Windows 10 or 11 (the key bindings are Windows shortcuts)
Python 3.10 or newer (developed with Python 3.12)
A webcam
Internet access once, to download the ~8 MB MediaPipe hand model
Installation
powershell
cd D:\gesture_detection
python -m venv env
.\env\Scripts\Activate.ps1
pip install opencv-python mediapipe pyautogui

(If PowerShell blocks the activation script, run Set-ExecutionPolicy -Scope Process Bypass first.)

Or install from pyproject.toml:

powershell
pip install -e .          # runtime only
pip install -e ".[dev]"   # plus pytest and ruff
Project files
gesture_detection/
├── main.py        # the loop: camera -> processor -> gestures -> actions -> window
├── camera.py      # Webcam class: opens the camera, returns mirrored frames
├── processor.py   # HandProcessor: MediaPipe hand tracking + skeleton drawing
├── gestures.py    # pose recognition (finger states, pinch, classify, smoother)
├── actions.py     # ActionConfig, InputBackend, SwipeDetector, ActionController
├── pyproject.toml
├── README.md
├── ARCHITECTURE.md
├── DEVELOPMENT_PLAN.md
├── STUDENT_NOTES.md
└── hand_landmarker.task   # downloaded automatically on first run
Running it
powershell
python main.py

The first run downloads hand_landmarker.task next to your code. Press q in the video window to quit (or Ctrl+C in the terminal).

Dry-run vs real control

At the top of main.py:

python
DRY_RUN = True    # only prints, never touches the PC

Test with True first and watch the terminal for lines like [dry-run] keys ctrl+win+left. When everything behaves, set it to False to control the real mouse and keyboard.

Cursor movement is silent in dry-run mode. You will see the circle on the video but the real cursor will not move.

How to use it
Show your hand to the camera and hold an open palm (fingers spread, palm facing the camera) for 1 second. A progress bar fills and the label turns green: ARMED.
Perform a gesture from the table below.
Hold a shaka (thumb and pinky out) for 1 second to disarm, or take your hand out of view for 3 seconds.
Gesture	How to make it	What it does
Open palm (hold 1 s)	Flat hand, fingers spread	Arms the system
Swipe left	Open palm, move quickly left	Next desktop (Ctrl+Win+Right)
Swipe right	Open palm, move quickly right	Previous desktop (Ctrl+Win+Left)
Swipe up	Open palm, move quickly up	Task view (Win+Tab)
Swipe down	Open palm, move quickly down	Show desktop (Win+D)
Point / L shape	Index finger up, others curled	Moves the cursor (index knuckle)
Pinch (in cursor mode)	Touch thumb tip to index tip	Hold mouse button: quick = click, moving = drag
Peace	Index + middle up	Scroll: move hand up / down
Fist (hold 0.5 s)	Close all fingers	Play / pause
Rock (hold 0.5 s)	Index + pinky up	Mute
Three (hold 0.5 s)	Index, middle, ring up	Next track
Thumbs up (hold)	Fist, thumb straight up	Volume up (repeats)
Thumbs down (hold)	Fist, thumb straight down	Volume down (repeats)
Shaka (hold 1 s)	Thumb + pinky out	Disarms

The window shows the current mode (DISARMED, ARMED, CURSOR, DRAG, SCROLL), the detected pose, and a grey box. That box is the part of the camera image mapped to your whole screen.

Configuration

All tunable numbers live in ActionConfig (top of actions.py) and a few constants at the top of gestures.py. Common changes:

Problem	Change
Swipes not detected	swipe_distance = 0.12, swipe_min_speed = 0.5
Accidental swipes	raise swipe_distance or swipe_cooldown
Cursor jittery	lower cursor_smoothing (e.g. 0.2)
Cursor laggy	raise cursor_smoothing (e.g. 0.5)
Cannot reach screen corners	smaller region, e.g. (0.3, 0.25, 0.7, 0.65)
Pinch will not click	raise pinch_down to 0.35
Clicks while only pointing	lower pinch_down to 0.18
Scroll too fast / slow	change scroll_gain
A finger is misread	change EXTEND_MARGIN in gestures.py (1.0 to 1.2)

To change what a gesture does, edit swipe_actions or hold_actions in ActionConfig. Key names follow PyAutoGUI ("playpause", "volumeup", "ctrl", "win", ...). To remove a gesture's action, delete its entry.

Troubleshooting
Symptom	Cause and fix
module 'mediapipe' has no attribute 'solutions'	Current MediaPipe removed the old API. This project already uses the new Tasks API. Make sure no file contains mp.solutions.
No matching distribution found for mediapipe==0.10.14	Your Python is too new for that old release. Do not downgrade, use the code as provided.
ValueError: too many values to unpack (expected 2)	processor.py is the old version. It must return (frame, hand).
Arms but nothing else happens	DRY_RUN = True. Set it to False. Check the terminal for [dry-run] lines.
Never arms	Hold the palm flat and spread, facing the camera, in good light. Watch the pose: line.
Desktop does not switch	Create a second virtual desktop with Win+Ctrl+D.
Nothing works in some windows	Windows blocks simulated input to apps running as administrator (e.g. Task Manager).
Cannot open camera	Close Teams / Zoom / the Camera app. Try a different camera index in camera.py.
Volume gestures unreliable	Thumbs up/down depend on thumb detection. Extend the thumb fully with the hand upright.

Quick check that PyAutoGUI works independently of the gesture code:

powershell
python -c "import pyautogui; pyautogui.moveTo(600,400,duration=1); pyautogui.press('volumeup')"
Safety
Actions only run while armed.
The mouse button is released if the hand disappears for 0.3 s, on disarm, and on exit.
The cursor is kept 5 px away from screen edges so PyAutoGUI's corner fail-safe is not triggered by accident.
Emergency stop: press q in the video window or Ctrl+C in the terminal.
Limitations
Windows shortcuts only (edit ActionConfig for other systems).
One hand at a time.
Thumb-based poses (thumbs up/down, L shape, open palm vs four) are the least reliable.
Thumbs up/down assume the hand is roughly upright.
Performance depends on lighting and camera quality.
Tested with scripted hand data and simulated input; real-world thresholds may need tuning for your camera.
More documentation
ARCHITECTURE.md: how the modules fit together
DEVELOPMENT_PLAN.md: milestones and roadmap
STUDENT_NOTES.md: the mathematics and technology explained
License

MIT