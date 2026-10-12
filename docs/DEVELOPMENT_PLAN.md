# Development Plan

Project: **Gesture Detection**, a webcam hand-gesture controller for Windows built from five modules: `camera.py`, `processor.py`, `gestures.py`, `actions.py`, `main.py`.

Status legend: **DONE** = written and checked in simulation, **TUNE** = works in principle but needs tuning on a real camera, **TODO** = not started.

## 1. Objectives

| # | Objective | How we measure success |
|---|---|---|
| O1 | Reliable swipe to switch desktops | Most intentional swipes detected; almost no false triggers during normal use |
| O2 | Usable cursor, click and drag | Can hit a normal button and drag a window without fighting the cursor |
| O3 | Smooth and responsive | 25+ FPS in the preview; gesture label stable without feeling slow |
| O4 | Safe | Nothing happens unless armed; mouse can never get stuck down |
| O5 | Understandable and maintainable | One job per file, documented, tunable from one config |

**Targets are goals, not measurements.** None have been measured on real hardware yet. Task M1 below is to measure them.

**Out of scope for now:** multi-hand gestures, sign language, mobile or web versions, cloud processing.

## 2. Current state

| Area | Status | Notes |
|---|---|---|
| Camera capture, mirrored frames | DONE | `camera.py` |
| Hand tracking and skeleton drawing | DONE | `processor.py`, MediaPipe Tasks API, VIDEO mode |
| 13-pose classifier + smoother | DONE / TUNE | `gestures.py`; thumb-based poses are the weakest |
| Arming / disarming | DONE | Open palm 1 s arms; shaka or 3 s absence disarms |
| Swipe (4 directions) | DONE / TUNE | Sliding window, cooldown, dropout tolerance |
| Cursor + click + drag | DONE / TUNE | Index knuckle, EMA smoothing, pinch hysteresis |
| Scroll | DONE / TUNE | Accumulator, dead zone; scroll units are Windows-specific |
| Media keys (play, mute, next, volume) | DONE / TUNE | Hold 0.5 s; volume repeats |
| On-screen HUD | DONE | Mode, pose, arming bar, region box, pointer |
| Dry-run mode | DONE | Default on |
| Automated tests | TODO | Controller logic was verified with a throw-away simulation, not a kept test suite |
| Settings file (outside the code) | TODO | Currently edited in `actions.py` |

## 3. Milestones

### M0: Project setup (DONE)
- [x] Five-module structure
- [x] `pyproject.toml`, README, ARCHITECTURE, DEVELOPMENT_PLAN, STUDENT_NOTES
- [x] Python environment with OpenCV, MediaPipe, PyAutoGUI

### M1: Verify on your machine (TODO, do this first)
- [ ] Run `python main.py` with `DRY_RUN = True`, check the hand skeleton appears
- [ ] Record the FPS shown or measured; aim for 25+
- [ ] Confirm each pose is recognised (watch the `pose:` line): open palm, fist, point, peace, pinch, thumbs up/down
- [ ] Note which poses flicker or are misread
- [ ] Confirm PyAutoGUI works alone (`moveTo`, `press('volumeup')`)
- **Exit criterion:** you can arm reliably and see correct pose labels for at least 8 gestures.

### M2: Tune the gestures (TODO)
- [ ] Adjust `EXTEND_MARGIN`, `PINCH_RATIO`, `THUMB_VERTICAL` in `gestures.py` until poses are stable
- [ ] Tune swipe values (`swipe_distance`, `swipe_min_speed`, `swipe_axis_ratio`)
- [ ] Tune cursor values (`region`, `cursor_smoothing`) and pinch thresholds
- [ ] Remove gestures that are unreliable for you; keep 5 to 8
- **Exit criterion:** 20 attempts per gesture with at most a couple of misses and no accidental triggers over 5 minutes of normal use.

### M3: Real control (TODO)
- [ ] Switch to `DRY_RUN = False`
- [ ] Verify desktop switching (needs two virtual desktops), task view, show desktop
- [ ] Verify cursor reaches all screen areas, click and drag work
- [ ] Verify volume and play/pause work with a media player open
- [ ] Verify the mouse is released when the hand leaves the view mid-drag

### M4: Automated tests (TODO)
- [ ] `tests/test_gestures.py`: synthetic hands for all 13 poses, pinch, thumb direction, smoother
- [ ] `tests/test_actions.py`: scripted frames for arming, swipe, cooldown, dropped frames, cursor, click, drag, scroll, hold, disarm, mouse release (use a fake `InputBackend` that records calls)
- [ ] Run with `pytest`, add `ruff` for linting
- **Exit criterion:** every state transition in ARCHITECTURE.md section 7.1 has a test.

### M5: Move settings out of the code (TODO)
- [ ] Load `ActionConfig` from a `config.json` or TOML file
- [ ] Validate values and reject unknown keys
- [ ] Command-line flags: `--dry-run`, `--camera N`, `--no-window`
- [ ] Save and load per-user calibration (hand size, region)

### M6: Reliability (TODO)
- [ ] Smooth landmarks with a **One-Euro filter** (less jitter when still, less lag when moving fast)
- [ ] Threaded frame grabber so the newest frame is always processed
- [ ] Separate confidence settings for presence and tracking in `processor.py`
- [ ] Optional left/right hand preference
- [ ] Rotation-independent thumbs up/down

### M7: Features (TODO)
- [ ] Right click (e.g. pinch with middle finger) and double click
- [ ] Zoom (two-hand or pinch spread)
- [ ] Sound or on-screen toast when an action fires
- [ ] Per-application profiles (different bindings for browser, media player)
- [ ] Mac and Linux key bindings

### M8: Packaging (TODO)
- [ ] System tray app with on/off toggle
- [ ] Simple settings GUI (sliders for sensitivity)
- [ ] Standalone `.exe` with PyInstaller
- [ ] Startup option

### M9: Stretch (TODO)
- [ ] Record labelled landmark data and train a small classifier for custom gestures
- [ ] Dynamic gesture recognition (circle, tap) with a sequence model
- [ ] Plugin system for user-defined actions

## 4. Suggested timeline (solo student)

| Week | Focus |
|---|---|
| 1 | M0 and M1: run it, understand the pipeline, measure FPS, read STUDENT_NOTES |
| 2 | M2: tune gestures and swipes until stable |
| 3 | M3: real control, fix Windows issues |
| 4 | M4: tests; M5: config file |
| 5 | M6: landmark smoothing, threaded capture |
| 6 | M7: extra gestures and feedback |
| 7+ | M8 and M9 |

## 5. Testing strategy

| Level | What | How |
|---|---|---|
| Unit | Finger states, pinch ratio, classify, smoother, swipe detector | `pytest` with synthetic hands (lists of 21 points with `.x`, `.y`) |
| State machine | Controller transitions, cooldowns, hysteresis, cleanup | Feed scripted `(hand, gesture, time)` sequences with a recording fake `InputBackend` |
| Manual | Real camera, lighting, distance, different people | `DRY_RUN = True`, watch terminal and HUD |
| Safety | Mouse release on hand loss / disarm / exit, no actions while disarmed | Dedicated tests |

Rule: a bug found while using the program becomes a failing test first, then a fix.

## 6. Tuning procedure

1. Set `DRY_RUN = True`.
2. Perform one gesture 20 times, count misses and false triggers.
3. Change **one** value, repeat. Record what helped.
4. Test in dim light, bright light, and with a busy background.
5. Write the final values into `ActionConfig` and `gestures.py` constants.

## 7. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| False triggers from normal hand motion | Annoying, possibly harmful | Arming gesture, hold times, cooldowns, dry-run |
| Poor light or busy background | Missed gestures | Confidence settings, good lighting, tune thresholds |
| Thumb poses misread | Wrong or missing action | `FOUR` counts as open palm; prefer other gestures for important actions |
| MediaPipe API change | Tracker breaks | All MediaPipe code is in `processor.py` only |
| Windows blocks input to admin windows | Actions ignored | Document; run as administrator if needed |
| Stuck mouse button | Frustration, lost work | Release on hand loss, disarm and exit |
| Privacy concerns about the camera | Trust | Everything local, no frames saved |
| Overlapping gestures (point vs L shape, open palm vs four) | Confusion | Smoother, careful pose choice, tune margins |

## 8. Definition of done (per feature)

- [ ] Works on a real camera with `DRY_RUN = False`
- [ ] Covered by an automated test (once M4 exists)
- [ ] New settings added to `ActionConfig` and documented in README
- [ ] ARCHITECTURE.md updated if the data flow or states changed
- [ ] No actions possible while disarmed

## 9. Learning goals

By the end you should be able to explain:

- How the camera image becomes 21 numbers (STUDENT_NOTES chapters 2 and 3)
- Why ratios make gestures work at any distance (chapter 4)
- How a swipe is told apart from ordinary movement (chapter 6)
- Why smoothing, hysteresis and cooldowns exist (chapters 7 and 8)
- How a program presses keys and moves the mouse (chapter 10)
