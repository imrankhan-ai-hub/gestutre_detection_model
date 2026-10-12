# Architecture

## 1. Goals

1. **Separation of concerns**: each file has exactly one job (see the table below).
2. **Testability**: the decision logic (`gestures.py`, `actions.py`) needs no camera, no MediaPipe and no real mouse, so it can be driven with scripted data.
3. **Safety by default**: arming gesture, hold times, cooldowns, dry-run mode, guaranteed mouse release.
4. **Tunable without code surgery**: thresholds live in `ActionConfig` and a few named constants.

## 2. Components

| File | Class / functions | Responsibility | Depends on |
|---|---|---|---|
| `camera.py` | `Webcam` | Open the camera, return mirrored BGR frames | OpenCV |
| `processor.py` | `HandProcessor` | Run MediaPipe hand tracking, draw the skeleton, return the 21 landmarks | OpenCV, MediaPipe |
| `gestures.py` | `Gesture`, `finger_states`, `pinch_ratio`, `is_pinching`, `thumb_direction`, `classify`, `GestureSmoother` | Turn 21 points into a gesture name (pure maths) | standard library only |
| `actions.py` | `ActionConfig`, `Action`, `InputBackend`, `SwipeDetector`, `ActionController` | Turn gestures and hand motion into keyboard/mouse actions | `gestures.py`, PyAutoGUI (lazy) |
| `main.py` | `main`, `draw_status` | Wire everything together and show the preview window | all of the above |

## 3. Dependency rule

```
                    main.py
          ┌───────────┼───────────────┐
      camera.py   processor.py    actions.py
                       │              │
                       │          gestures.py   (no third-party imports)
                       │
                  MediaPipe + OpenCV
```

- `gestures.py` imports only the Python standard library. It can be tested anywhere.
- `actions.py` imports `gestures.py` and loads PyAutoGUI **lazily**, inside `InputBackend`, so dry-run works even without it.
- Only `processor.py` knows about MediaPipe. Only `InputBackend` knows about PyAutoGUI.
- Files never import `main.py`.

## 4. Data flow for one frame

```
Webcam.read()                      -> frame (BGR image, mirrored)
HandProcessor.process(frame)       -> (frame with skeleton, hand)   hand = 21 landmarks or None
classify(hand)                     -> raw Gesture                    (may flicker)
GestureSmoother.update(raw)        -> stable Gesture                 (majority vote)
ActionController.update(hand, g)   -> list of event strings         (+ real mouse/keyboard)
draw_status(frame, controller, g)  -> HUD drawn on the frame
cv2.imshow / waitKey               -> window, quit on 'q'
```

The loop runs once per camera frame (about 30 times per second).

### Key data types

| Data | Shape | Notes |
|---|---|---|
| Landmark | object with `.x`, `.y`, `.z` | `x`, `y` normalised 0 to 1, `y` grows **downwards**, `z` is relative depth |
| Hand | list of 21 landmarks | index 0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip, 20 pinky tip |
| `Gesture` | `Enum` of strings | `NONE`, `UNKNOWN`, `FIST`, `OPEN_PALM`, `POINT`, `PEACE`, `THREE`, `FOUR`, `ROCK`, `SHAKA`, `L_SHAPE`, `THUMBS_UP`, `THUMBS_DOWN`, `PINCH`, `OK_SIGN` |
| `ActionConfig` | dataclass | All timings, thresholds and key bindings |
| `Action` | frozen dataclass | `label`, `keys`, `repeat` |

## 5. Tracking layer (`processor.py`)

- Uses the **MediaPipe Tasks `HandLandmarker`** in `VIDEO` running mode with `num_hands=1`.
- `VIDEO` mode keeps state between frames, so the slow palm detector runs only when the hand is lost. After that a lightweight tracker follows it.
- Timestamps passed to MediaPipe must be strictly increasing integers in milliseconds, so the code enforces `max(now_ms, last_ms + 1)`.
- The model file `hand_landmarker.task` is downloaded once to the working directory.

## 6. Gesture layer (`gestures.py`)

Three questions answer everything:

```
21 points ─▶ which fingers are straight?      (finger_states)
          ├▶ are thumb and index tips touching? (pinch_ratio)
          └▶ is the thumb above or below the wrist? (thumb_direction)
                              │
                              ▼
                  classify() -> Gesture
```

- All measurements are **ratios** (relative to hand size or to other distances), so they work at any distance from the camera and any hand rotation, except thumbs up/down which assume an upright hand.
- `classify` checks specific gestures before general ones (pinch family first, then four fingers, three, peace, ...).
- `GestureSmoother` keeps the last 7 labels and only switches when one wins 5 of them.

## 7. Action layer (`actions.py`)

### 7.1 Controller states

```
              hold open palm / four 1 s
 DISARMED ───────────────────────────────▶ ARMED ───┐
    ▲                                         │     │ POINT / L SHAPE
    │   shaka held 1 s, or hand away 3 s      │     ▼
    └─────────────────────────────────────────┤   CURSOR ──pinch──▶ DRAG
                                              │     ▲                 │
                                              │     └───un-pinch──────┘
                                              │ PEACE
                                              ▼
                                            SCROLL
```

### 7.2 Priority order while armed (checked every frame)

1. **Shaka** held 1 s: disarm.
2. **Cursor mode**: POINT or L SHAPE starts it, and it continues through the pinch. The pinch has priority so a drag never drops halfway.
3. **Scroll mode**: PEACE.
4. **Swipe** (open palm within the last 0.4 s) and **hold actions** (fist, rock, three, thumbs up/down).

### 7.3 Components

- **`SwipeDetector`**: sliding window of `(time, x, y)` palm positions. A swipe needs distance, speed and straightness. The window empties after each swipe, and a cooldown blocks the return stroke.
- **`InputBackend`**: the only code that presses keys or moves the mouse. In dry-run it prints. It clamps the cursor 5 px inside the screen.
- **`ActionController`**: owns all state (armed, cooldowns, hold timers, smoothing, pinch state, scroll accumulator).

## 8. State kept between frames

| State | Purpose |
|---|---|
| Smoother history (7 labels) | Stabilise the gesture label |
| Swipe samples (0.4 s) | Detect motion |
| Arm / disarm timers | Require sustained poses |
| Cooldown timestamp | Debounce swipes and one-shot actions |
| Hold timer + last-fire time | Trigger after 0.5 s, repeat volume |
| Smoothed cursor position | Remove jitter |
| Pinch-down flag + press position | Hysteresis, click vs drag |
| Scroll accumulator | Convert tiny hand movements to whole scroll units |

## 9. Design decisions

| Decision | Reason |
|---|---|
| Arming gesture | Prevent accidental triggers from everyday hand motion |
| MediaPipe Tasks API | The old `mp.solutions` API is removed from current releases |
| Ratios instead of pixel distances | Scale invariance |
| Index **knuckle** drives the cursor | The fingertip moves when pinching, the knuckle barely does |
| Pinch hysteresis (press < 0.25, release > 0.40) | Stop flicker near the threshold |
| Keep swipe history through dropped frames | Fast motion blurs the hand and the tracker may drop a frame |
| `FOUR` counts as open palm for arming and swipes | Thumb detection is the least reliable measurement |
| Time passed in as a parameter (`now`) | Deterministic testing without sleeping |
| Hold time for actions | Passing through a pose must not fire it |
| `InputBackend` wrapper with dry-run | Safe testing; single place to change for other operating systems |
| Release mouse on hand loss, disarm and exit | Never leave a button stuck |

## 10. Timing and latency

| Stage | Typical delay |
|---|---|
| Camera + tracking | hardware dependent, typically a few tens of ms (not measured) |
| Smoother (needs 5 matching frames) | up to about 165 ms at 30 FPS |
| Hold time before an action | 500 ms (volume, media) |
| Arming | 1000 ms |

Cursor movement uses the raw hand position plus a light exponential smoothing (about 60 ms lag), not the smoothed gesture label, so it feels responsive.

## 11. Error handling and safety

| Risk | Handling |
|---|---|
| Accidental actions | Arming, hold times, cooldowns |
| Mouse stuck down | Released on hand loss (0.3 s), disarm, and in the `finally` block of `main` |
| PyAutoGUI corner fail-safe crash | Cursor clamped 5 px from the edges |
| PyAutoGUI missing in dry-run | Import is optional; default screen size is used |
| Camera frame failure | `main` skips frames where `read()` returns `None` |
| Program exit | `finally` closes controller, tracker, camera and windows |

## 12. Extension points

| Goal | Where |
|---|---|
| New pose | Add to `Gesture`, extend `classify`, optionally add to `hold_actions` |
| New action | Add an `Action` in `ActionConfig` |
| Different OS | Change the key tuples in `swipe_actions` / `hold_actions`, or subclass `InputBackend` |
| Different tracker | Replace `HandProcessor` (keep the `(frame, hand)` return shape) |
| Better smoothing | Replace the cursor EMA in `_do_cursor` with a One-Euro filter |
| Two hands | Set `num_hands=2` and extend `main` / `ActionController` |
| Learned gestures | Replace `classify` with a model (same input and output) |

## 13. Known limitations

- Single hand, Windows key bindings.
- Rule-based thresholds need tuning per camera and lighting.
- Thumb-dependent poses are less stable.
- Hand rotation breaks thumbs up/down.
- No automated test suite yet (see the development plan).
