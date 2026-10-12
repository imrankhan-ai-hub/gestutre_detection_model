# Student Notes: the Mathematics and Technology Behind Gesture Detection

These notes explain **how every part of this project works and why**, with the actual numbers from the code. You do not need advanced maths: distances, ratios, averages and a little algebra are enough.

**Contents**

1. The big picture
2. Digital images and the camera
3. MediaPipe: how a computer finds a hand
4. Geometry: distances, ratios and scale invariance
5. Classifying gestures
6. Motion: speed, windows and swipes
7. Smoothing signals
8. Control logic: states, hysteresis, cooldowns
9. From hand position to screen position
10. How a program presses keys and moves the mouse
11. Python ideas used in this project
12. Performance and latency
13. Exercises (with answers)
14. Glossary
15. Further reading

---

## 1. The big picture

A webcam gives 30 pictures per second. For each picture the program asks four questions:

```
Picture ─▶ 1. Where are the hand's joints?        (MediaPipe, chapter 3)
        ─▶ 2. What pose is the hand making?        (geometry, chapters 4-5)
        ─▶ 3. Is the hand moving, and how?         (kinematics, chapter 6)
        ─▶ 4. What should the computer do?         (state machine + OS input, chapters 8-10)
```

| Question | Technology | Math | File |
|---|---|---|---|
| Find joints | Neural networks (MediaPipe) | Convolutions, regression | `processor.py` |
| Pose | Rule-based classification | Euclidean distance, ratios | `gestures.py` |
| Motion | Sliding window | Velocity = distance / time | `actions.py` |
| Stability | Filters | Averages, voting | `gestures.py`, `actions.py` |
| Act | Operating-system input | None (API calls) | `actions.py` |

---

## 2. Digital images and the camera

### 2.1 What a frame is

A frame is a grid of pixels. OpenCV stores it as a NumPy array with shape `(height, width, 3)`, for example `(480, 640, 3)`: 480 rows, 640 columns, 3 colour values per pixel, each 0 to 255 (`uint8`).

```
frame[row, column] = [Blue, Green, Red]
```

> **BGR, not RGB.** OpenCV orders colours as Blue-Green-Red, but MediaPipe expects Red-Green-Blue. That is why `processor.py` calls `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before detection.

### 2.2 Pixel coordinates

The origin `(0, 0)` is the **top-left** corner. `x` grows to the right, and **`y` grows downwards**. This surprises people used to graphs: moving your hand *up* makes `y` *smaller*. The swipe-up code relies on this:

```python
name = "swipe_down" if dy > 0 else "swipe_up"
```

### 2.3 Normalised coordinates

MediaPipe returns `x` and `y` as fractions between 0 and 1 instead of pixels:

```
x_normalised = x_pixel / image_width
y_normalised = y_pixel / image_height
```

To draw on the image we convert back:

```python
point = (int(lm.x * w), int(lm.y * h))
```

Normalised coordinates make the logic independent of camera resolution. A swipe of `0.2` means "20% of the frame" whether the frame is 640 or 1920 pixels wide.

> **Subtle point.** `x` is divided by the width and `y` by the height. In a 640×480 image, one unit of `x` is 640 pixels but one unit of `y` is 480. Distances in normalised space are therefore slightly stretched along `x`. The project accepts this because it uses ratios and generous thresholds.

### 2.4 Mirroring

`camera.py` flips each frame horizontally so the preview behaves like a mirror. For a flip, a pixel at column `x` moves to column `(w − 1) − x`, and normalised `x` becomes `1 − x`. Without it, moving your hand to your right would move it to the left on screen, which feels wrong. Because we flip *before* tracking, "swipe right" in the code means your hand really moved towards your right.

### 2.5 Frame rate and timing

At 30 frames per second, the time between frames is `1/30 ≈ 33 ms`. All timing in the project uses `time.monotonic()` (seconds as a float), a clock that always moves forward and ignores changes to the system time.

---

## 3. MediaPipe: how a computer finds a hand

### 3.1 Two neural networks, not one

```
image ─▶ [1] Palm detector ─▶ rotated crop ─▶ [2] Hand landmark model ─▶ 21 points
              ▲                                         │
              └── only when the hand is lost ◀──────────┘
```

| | Palm detector | Hand landmark model |
|---|---|---|
| Question | "Where is a hand?" | "Where are the 21 joints?" |
| Input | Whole image, resized | Small, rotated crop around one hand |
| Output | Box (plus rotation info) | 21 × (x, y, z), a presence score, left/right label |
| Cost | Heavier (searches everywhere) | Lighter (looks at a small crop) |
| Runs | First frame, or after losing the hand | Every frame |

**Why detect the palm instead of the whole hand?** A palm is nearly a rigid square, easy to find. A whole hand changes shape constantly as fingers move. So the palm is found first and the fingers are worked out afterwards.

### 3.2 Palm detection ideas

- It is a **single-shot detector (SSD-style)**: a convolutional neural network looks at the whole image once and proposes many candidate boxes at fixed positions and sizes called *anchors*.
- Each candidate gets a confidence score. Many overlapping boxes may point at the same palm, so **non-maximum suppression (NMS)** keeps the best one and discards boxes that overlap it too much.
- Overlap is measured by **IoU (intersection over union)**:

```
IoU = area of overlap / area of union          (0 = no overlap, 1 = identical)
```

### 3.3 Landmark regression

The landmark network is a convolutional neural network that **regresses** (directly predicts as numbers) the positions of 21 joints:

| Index | Landmark | Index | Landmark |
|---|---|---|---|
| 0 | Wrist | 9-12 | Middle finger (MCP, PIP, DIP, TIP) |
| 1-4 | Thumb (CMC, MCP, IP, TIP) | 13-16 | Ring finger |
| 5-8 | Index (MCP, PIP, DIP, TIP) | 17-20 | Pinky |

(MCP = knuckle, PIP = middle joint, DIP = last joint, TIP = fingertip.)

Each landmark has `x` and `y` (0 to 1) and `z`, a relative depth measured with the wrist as the reference. `z` is much less reliable than `x` and `y`, which is why this project ignores it.

The model also outputs a **hand presence score** (is there a hand in the crop?) and handedness (left or right).

The published MediaPipe Hands work reports training the landmark model on roughly 30,000 real-world images annotated with joint positions, plus synthetic rendered hands in many poses. Treat the exact figures as approximate and check the original paper.

### 3.4 Tracking between frames

Running the palm detector on every frame is expensive. Instead:

1. Frame 1: palm detector finds the hand, landmark model gives 21 points.
2. Frame 2 onward: the previous landmarks predict where the hand will be, so that region is cropped directly. The palm detector is skipped.
3. If the presence score drops or the hand moves too far, MediaPipe falls back to the palm detector.

This is why `processor.py` uses `RunningMode.VIDEO`: the library remembers earlier frames. (`IMAGE` mode treats every frame independently and is slower. `LIVE_STREAM` returns results through a callback instead of waiting.)

### 3.5 The three confidence options

| Option | Question it asks | Default |
|---|---|---|
| `min_hand_detection_confidence` | Is this really a hand? (palm detector) | 0.5 |
| `min_hand_presence_confidence` | Is the hand still in the tracked crop? | 0.5 |
| `min_tracking_confidence` | Did the hand box overlap enough with the last frame (IoU)? | 0.5 |

Lower values hold on to the hand through fast, blurry motion but may follow false hands. Higher values are stricter. `processor.py` currently uses only `num_hands=1` and the defaults.

### 3.6 The technology underneath

- Models are **TensorFlow Lite** files. The downloaded `hand_landmarker.task` bundles both networks.
- MediaPipe runs them with optimised CPU code, so no graphics card is needed.
- Timestamps given to `detect_for_video` must be **strictly increasing integer milliseconds**. The code uses `max(int(now * 1000), last_ms + 1)` to guarantee that.

---

## 4. Geometry: distances, ratios and scale invariance

### 4.1 Euclidean distance

The straight-line distance between two points is the Pythagorean theorem:

```
d = √((x₁ − x₂)² + (y₁ − y₂)²)
```

In code (`gestures.py`):

```python
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)
```

`math.hypot` computes exactly that square root of a sum of squares. Only `x` and `y` are used.

### 4.2 The problem: distance depends on how close your hand is

If you move your hand closer to the camera, every distance in the image grows. A rule like "the thumb and index are touching if they are closer than 0.02" works at one distance and fails at another.

### 4.3 The solution: divide by hand size

`hand_size` is the distance from the wrist (0) to the middle-finger knuckle (9). It scales together with everything else on the hand.

```
pinch_ratio = distance(thumb tip, index tip) / hand_size
```

**Worked example** (the synthetic hand used in testing):

| Quantity | Value |
|---|---|
| Wrist (0.50, 0.90), middle knuckle (0.50, 0.65) | hand_size = 0.25 |
| Thumb tip (0.46, 0.43), index tip (0.45, 0.42) | distance = 0.0141 |
| pinch_ratio | 0.0141 / 0.25 = **0.0566** |

Move the same hand to half the distance from the camera: distance becomes 0.00707 and hand_size 0.125, and the ratio is still about **0.0566**. A ratio is unchanged when the whole picture is scaled. This property is called **scale invariance**.

The pinch threshold `PINCH_RATIO = 0.3` means "the tips are closer than 30% of a palm length".

### 4.4 Is a finger straight? (rotation invariance)

For index, middle, ring and pinky, the test compares two distances to the wrist:

```
finger is UP  if   distance(tip, wrist)  >  distance(PIP joint, wrist) × 1.1
```

```
   straight finger            curled finger
        ● tip                     ● PIP
        |                        / ● tip    ← tip is nearer the wrist than the PIP
        ● PIP                   |
        |                       |
        ● wrist                 ● wrist
```

**Worked example** (index finger, synthetic hand, wrist at (0.5, 0.9)):

| Finger | tip–wrist | PIP–wrist | Ratio | Result |
|---|---|---|---|---|
| Straight | 0.4826 | 0.3536 | 1.365 | > 1.1 → **UP** |
| Curled | 0.1868 | 0.3239 | 0.577 | < 1.1 → **down** |

Why compare against the *wrist* and not an up/down direction? **Rotating the hand around the wrist does not change distances to the wrist**, so the test works whether your fingers point up, sideways or diagonally. This is **rotation invariance**.

The `1.1` margin (`EXTEND_MARGIN`) is a safety buffer. With `1.0`, a half-bent finger could count as straight. Raise it to be stricter.

### 4.5 The thumb is different

The thumb folds sideways across the palm, so its tip does not get closer to the wrist when it curls. Instead, `finger_states` compares distances to the **base of the pinky (landmark 17)**:

```
thumb is OUT if  distance(thumb tip, pinky base) > distance(thumb IP joint, pinky base) × 1.1
```

A straight thumb sticks out away from the pinky; a tucked thumb folds towards it.

### 4.6 Thumb up or down

`thumb_direction` measures how far the thumb tip is above or below the wrist, in hand sizes:

```
offset = (wrist.y − thumb_tip.y) / hand_size
offset >  0.5  → "up"      offset < −0.5 → "down"      otherwise → "side"
```

Because `y` grows downwards, `wrist.y − tip.y` is positive when the tip is above the wrist. This is the one measurement that **assumes an upright hand**, so it breaks if you rotate your hand.

### 4.7 The palm centre

For swipe and scroll we need one stable point for the whole hand. The code takes the **average (centroid)** of five points: wrist and the four knuckles:

```
x_center = (x₀ + x₅ + x₉ + x₁₃ + x₁₇) / 5          y_center similarly
```

If each landmark had independent random error of size σ, the average of five would have error about σ/√5, roughly 45% as large. Real errors are partly correlated, so the benefit is smaller, but it is still steadier than any single point. These points also barely move when fingers wiggle.

---

## 5. Classifying gestures

### 5.1 Rule-based classification

`classify()` is a **decision procedure**: it reads five true/false values (which fingers are up) plus pinch and thumb direction, and picks a label by rules, checked in a fixed order.

| Gesture | Thumb | Index | Middle | Ring | Pinky | Extra rule |
|---|---|---|---|---|---|---|
| Open palm | up | up | up | up | up | |
| Four | down | up | up | up | up | |
| Three | any | up | up | up | down | |
| Peace | any | up | up | down | down | |
| Point | down | up | down | down | down | |
| L shape | up | up | down | down | down | |
| Rock | any | up | down | down | up | |
| Shaka | up | down | down | down | up | |
| Thumbs up | up | down | down | down | down | thumb above wrist |
| Thumbs down | up | down | down | down | down | thumb below wrist |
| Fist | any | down | down | down | down | |
| Pinch | any | any | | | | tips touch, not all other fingers up |
| OK sign | any | any | up | up | up | tips touch |

**Order matters.** The pinch family is checked first because an OK sign also has three fingers up. More specific rules come before general ones.

### 5.2 Why the pinch needs another finger up

In a closed fist, the thumb also ends up close to the index finger, which would look like a pinch. Requiring at least one of middle, ring or pinky to be up separates a pinch from a fist.

### 5.3 Why classification is hard

- Every measurement has noise, and a tiny error can flip a finger from "up" to "down".
- Poses overlap: a relaxed point is almost an L shape; an open palm with a tucked thumb is "four".
- Thumbs are the least reliable (they point toward the camera and are often partly hidden). For this reason `FOUR` also counts as an open palm when arming or swiping.

### 5.4 The machine-learning alternative

Instead of hand-written rules, you could record the 21 points (63 numbers) for many examples of each gesture and train a **classifier** (for example k-nearest-neighbours, a random forest or a small neural network) to predict the label. This handles messy poses better but needs labelled data. `classify()` can be replaced by such a model without changing the rest of the program, because its input (21 points) and output (a `Gesture`) stay the same.

---

## 6. Motion: speed, windows and swipes

A pose can be judged from one frame. A **swipe** is a movement, so it needs history.

### 6.1 Velocity

```
velocity = change in position / change in time = Δx / Δt
```

Units here are **frame-fractions per second**. A speed of 1.0 means crossing the whole width of the frame in one second.

### 6.2 The sliding window

`SwipeDetector` keeps `(time, x, y)` samples from the last 0.4 seconds in a `deque`. Old samples are removed from the front, new ones appended at the back:

```
time:   0.00  0.03  0.07  0.10  0.13  0.17  0.20
x:      0.30  0.34  0.38  0.42  0.46  0.50  0.54
        ▲oldest                                  ▲newest

Δx = 0.54 − 0.30 = 0.24      Δt = 0.20 s      speed = 0.24 / 0.20 = 1.2
```

(In the real program the check runs on **every** frame, so this swipe fires earlier, at t = 0.17 s, the moment the distance reaches 0.20 at a speed of about 1.2. It then clears the window so the same movement cannot fire twice.)

A swipe is accepted when **all three** hold:

| Condition | Setting | Meaning |
|---|---|---|
| Distance | `swipe_distance = 0.20` | moved at least 20% of the frame |
| Speed | `swipe_min_speed = 0.8` | at least 0.8 frame-fractions per second |
| Straightness | `swipe_axis_ratio = 1.5` | main direction at least 1.5× the other |

**Interaction between the settings.** Moving 0.20 at speed 0.8 takes at most `0.20 / 0.8 = 0.25 s`. So a swipe must cover a fifth of the frame within a quarter of a second, even though the window is 0.4 s long.

**Worked example:** your hand moves 0.4 in 0.2 s. Speed = 0.4 / 0.2 = **2.0** ✓ (above 0.8). A slow drift of 0.3 over 0.5 s only shows 0.24 inside the 0.4 s window, giving speed 0.24 / 0.4 = 0.6 ✗ (below 0.8), so it is correctly rejected.

### 6.3 Straightness

A swipe counts as horizontal when `|dx| ≥ 1.5 × |dy|`. That allows the path to deviate from the horizontal axis by at most

```
θ = arctan(1 / 1.5) ≈ 33.7°
```

Diagonal movements (more than about 34° off the axis) are rejected as ambiguous.

### 6.4 Direction by sign

```
dx > 0 → right        dx < 0 → left
dy > 0 → down         dy < 0 → up        (because y grows downward)
```

### 6.5 Why a cooldown is needed

After you swipe right, your hand naturally returns left at similar speed, which looks like a left swipe. The 1-second `swipe_cooldown` blocks it. After a swipe, the sample window is also cleared so the same movement cannot fire twice.

### 6.6 Surviving dropped frames

A fast hand blurs and the tracker may lose it for a frame. If the code erased its memory every time that happened, fast swipes would never complete. Instead the history is kept and old samples expire by time. The open-palm requirement also has a 0.4 s grace (`palm_grace`), so one badly classified frame does not cancel a swipe.

### 6.7 Scrolling is also motion

For PEACE the code measures how far the hand moved since the previous frame: `dy = y_now − y_previous` (middle knuckle). Hand up (negative `dy`) scrolls up. See chapter 9.

---

## 7. Smoothing signals

Raw measurements are noisy: a point that is really still jitters by a few pixels. Smoothing trades a little delay for stability. This project uses three techniques.

### 7.1 Majority vote (the gesture smoother)

`GestureSmoother(size=7, votes=5)` remembers the last 7 labels. It changes the reported gesture only when one label has at least 5 of the 7.

```
raw:     FIST FIST FIST UNKNOWN FIST FIST FIST      → stays FIST (glitch ignored)
raw:     ... POINT POINT POINT POINT POINT          → switches after 5 POINTs
```

This is a **mode filter** (the mode is the most common value). Properties:

- It tolerates up to 2 glitch frames out of 7.
- A real change needs at least 5 matching frames: about `5 / 30 s ≈ 167 ms` of delay at 30 FPS.

### 7.2 Exponential moving average (cursor smoothing)

For the cursor, the smoothed position moves a fraction `α` of the way to the new target each frame:

```
smooth_new = smooth_old + α × (target − smooth_old)         α = cursor_smoothing = 0.35
```

Expanding it shows each old measurement gets exponentially less weight: the latest frame has weight α, the previous α(1−α), the one before α(1−α)², and so on.

| Property | Formula | At α = 0.35, 30 FPS |
|---|---|---|
| Average lag | (1 − α) / α frames | 1.86 frames ≈ 62 ms |
| Fraction of a sudden jump reached after n frames | 1 − (1 − α)ⁿ | 3 frames: 72.5%, 5 frames: 88.4% |

- **Larger α** → follows quickly, jitters more.
- **Smaller α** → very smooth, but lags behind your hand.

### 7.3 A better filter for the future (One-Euro)

A fixed α is a compromise. The **One-Euro filter** (Casiez, Roussel and Vogel, 2012) changes the amount of smoothing with speed: strong smoothing when the hand is almost still (removes jitter), little smoothing when it moves fast (removes lag). It is the planned upgrade in the development plan.

### 7.4 Averaging in space

Using the palm centre (chapter 4.7) is a spatial average, another form of noise reduction.

---

## 8. Control logic: states, hysteresis, cooldowns

### 8.1 Finite-state machine

A **finite-state machine (FSM)** is a program that is always in exactly one *state* and moves to another state only on defined *events*. `ActionController` is one.

```
 DISARMED ──open palm 1 s──▶ ARMED ──point──▶ CURSOR ──pinch──▶ DRAG
    ▲                         │  ▲               │                │
    │                         │  └──not pointing─┘◀──un-pinch─────┘
    │                         └──peace──▶ SCROLL
    └── shaka 1 s, or hand away 3 s (from any armed state)
```

Why a state machine? Because the *same* gesture must mean different things depending on context. A pinch is ignored when disarmed, a click in cursor mode, and a drag while held.

### 8.2 Arming as a safety lock

While `DISARMED`, every gesture is ignored. The arming rule needs a pose held for 1 second, so ordinary hand movement almost never satisfies it. This is the main protection against accidental actions.

### 8.3 Hold time

Passing through a fist while changing hands must not pause the music. Actions require the pose to be held `hold_time = 0.5 s`. A one-shot action fires once and then needs you to leave the pose before it can fire again. Volume actions repeat every 0.3 s.

### 8.4 Hysteresis (the pinch)

Suppose a single threshold: press the mouse when `ratio < 0.3`. If your fingers hover near 0.3, the ratio wobbles across it and the button flickers up and down. **Hysteresis** uses two thresholds:

```
press when ratio < 0.25        release when ratio > 0.40
                 ◀─── dead band 0.25 to 0.40: state does NOT change ───▶
```

Inside the dead band the state stays whatever it was. This is also called a **Schmitt trigger**, and is common in electronics.

### 8.5 Click versus drag

When the mouse is released, the code measures how far the cursor moved since the press:

```
moved = √((x_release − x_press)² + (y_release − y_press)²)
moved > 15 px → "DRAG end"      otherwise → "CLICK"
```

A click is simply a press and release at about the same place.

### 8.6 Cooldowns and debouncing

**Debouncing** means ignoring repeated triggers that come from one real action (named after electrical switches that "bounce"). The global `_cooldown_until` timestamp is set after each swipe or one-shot action, and new actions are ignored until the clock passes it.

### 8.7 Safe cleanup

Because a held mouse button is dangerous (a stuck button can select or drag things unintentionally), the button is released when:

- the hand disappears for more than 0.3 s,
- the system disarms,
- the program exits (the `finally` block calls `controller.close()`).

---

## 9. From hand position to screen position

### 9.1 Why not map the whole camera image to the screen?

If the whole frame covered the screen, you would need to reach the very edge of the camera view to reach the screen edge, and your hand tracks poorly near the edges. So only a **region** of the frame is used: `region = (left, top, right, bottom) = (0.2, 0.15, 0.8, 0.75)`.

### 9.2 Linear interpolation and clamping

```
nx = (px − left) / (right − left)        clamp to [0, 1]
ny = (py − top)  / (bottom − top)        clamp to [0, 1]
screen_x = nx × screen_width             screen_y = ny × screen_height
```

**Worked example:** hand at `px = 0.5` → `nx = (0.5 − 0.2) / 0.6 = 0.5` → `screen_x = 0.5 × 1920 = 960` (the middle of a 1920-pixel screen). Hand at `px = 0.3` → `nx = 0.1667` → 320 px.

The region is 0.6 of the frame wide, so hand movement is amplified by `1 / 0.6 ≈ 1.67`. A smaller region means more amplification (less arm movement, more jitter).

### 9.3 Why the knuckle drives the cursor

When you pinch, the thumb and index **tips** travel several centimetres, but the index **knuckle** (landmark 5) barely moves. If the cursor followed the fingertip, clicking would shove the cursor off its target. The knuckle stays put.

### 9.4 Screen edges

The cursor is clamped to at least 5 pixels inside the screen. PyAutoGUI has a safety rule that raises an error if the mouse reaches a screen corner; clamping stops your hand from triggering it by accident.

### 9.5 Scrolling: accumulating fractions

Scrolling needs whole numbers, but one frame's hand movement is tiny:

```
accumulator += −dy × scroll_gain          (scroll_gain = 4000)
whole = int(accumulator)
send scroll(whole);  accumulator −= whole      (keep the leftover fraction)
```

**Worked example:** the hand rises 0.01 in one frame → `+40` units that frame, `1200` units per second at 30 FPS. On Windows, 120 units is one wheel notch, so that is about 10 notches per second. A dead zone (`scroll_deadzone = 0.002`) ignores jitter. Without the accumulator, small movements would round to zero and nothing would ever scroll.

---

## 10. How a program presses keys and moves the mouse

### 10.1 The operating system owns input

Applications never read the keyboard directly. The operating system receives hardware events, decides which window is active, and delivers them. To control another application, a program must **inject events into the operating system**, so that the OS treats them like real hardware input.

### 10.2 PyAutoGUI

PyAutoGUI is a Python library that wraps those operating-system functions. On Windows it calls functions in `user32.dll` through Python's `ctypes` module. You write `pyautogui.hotkey("ctrl", "win", "right")` and it:

1. Looks up each key name's **virtual-key code** (a number the OS uses to identify each key).
2. Sends **key down** events in order: Ctrl, Win, Right.
3. Sends **key up** events in **reverse** order: Right, Win, Ctrl. This matches how a person releases a shortcut, so the system recognises it.

Media keys also have virtual-key codes (for example volume up, volume down, mute, play/pause, next track), which is why `pyautogui.press("volumeup")` works.

### 10.3 The shortcuts used

| Gesture | Keys | Windows feature |
|---|---|---|
| Swipe left | `Ctrl + Win + Right` | Next virtual desktop |
| Swipe right | `Ctrl + Win + Left` | Previous virtual desktop |
| Swipe up | `Win + Tab` | Task view |
| Swipe down | `Win + D` | Show desktop |

### 10.4 Protections and limits

- **Fail-safe.** If enabled, PyAutoGUI raises an exception when the mouse reaches a screen corner, giving you an emergency stop.
- **Administrator windows.** Windows runs programs at different **integrity levels**. A normal program cannot send input to a window running as administrator (this is **UIPI**, User Interface Privilege Isolation). That is why the gestures may not work in Task Manager.
- **Wayland and macOS permissions.** On other systems, synthetic input may be blocked or need explicit permission.

### 10.5 Dry-run design

`InputBackend` is the only code that touches PyAutoGUI. With `dry_run=True` it prints instead of acting. Isolating the "dangerous" part behind one class is a standard design technique: it also lets tests substitute a fake backend that just records calls.

---

## 11. Python ideas used in this project

| Idea | Where | Why |
|---|---|---|
| `Enum` (`Gesture`) | `gestures.py` | A fixed list of names; typos become errors instead of silent bugs. `class Gesture(str, Enum)` also lets the values print as text. |
| `@dataclass` | `ActionConfig`, `Action`, `FingerStates` | Generates `__init__` etc. from a list of fields; ideal for settings. `frozen=True` makes instances read-only. |
| `field(default_factory=...)` | `ActionConfig` | Gives each instance its own dictionary instead of one shared mutable default. |
| `collections.deque` | `SwipeDetector`, `GestureSmoother` | Fast add at one end and remove at the other; `maxlen` drops old items automatically. |
| `collections.Counter` | `GestureSmoother` | `most_common(1)` finds the mode. |
| `is` vs `==` | `gesture is Gesture.FIST` | Enum members are single objects, so identity comparison is exact. |
| Lazy import | `InputBackend.__init__` | PyAutoGUI is imported only when needed, so dry-run works without it. |
| `try / finally` | `main()` | Cleanup (release mouse, close camera) runs even on error or Ctrl+C. |
| Passing time in (`now`) | `ActionController.update` | Makes the logic testable with fake clocks. |
| `time.monotonic()` | everywhere | Never goes backwards, unlike the wall clock. |
| Tuple unpacking | `frame, hand = proc.process(frame)` | A function returns several values at once. |

**Design principles to recognise:** separation of concerns (one job per file), single point of contact with the outside world (`InputBackend`), and keeping pure logic (`gestures.py`) free of hardware so it is easy to test.

---

## 12. Performance and latency

### 12.1 Frame budget

At 30 FPS you have about 33 ms per frame for everything. Almost all of that is the neural-network tracking. Our own logic (about 21 distance calculations and a few comparisons) takes a negligible fraction of a millisecond.

### 12.2 Where the delay comes from

| Stage | Delay |
|---|---|
| Camera exposure and transfer | one frame (about 33 ms) plus driver buffering |
| Tracking | hardware dependent |
| Gesture smoother | 5 frames to switch ≈ 167 ms |
| Hold time | 500 ms for media actions |
| Cursor smoothing | about 62 ms average lag (α = 0.35) |

The cursor deliberately uses the raw hand position instead of the smoothed gesture label, so it feels responsive even though gesture switching is slower. Total latency is the *sum* of the stages on the path an action takes.

### 12.3 Improving speed

- Lower the camera resolution (640×480 is plenty).
- Good lighting makes tracking faster and steadier.
- Tracking mode (not detection) is cheaper: don't lose the hand.
- Process the newest frame only (threaded capture is a planned improvement).

---

## 13. Exercises

Try them first, then compare with the answers.

1. A hand has wrist `(0.50, 0.80)` and middle knuckle `(0.50, 0.60)`. The thumb tip is at `(0.46, 0.38)` and the index tip at `(0.48, 0.36)`. Is it a pinch? (`PINCH_RATIO = 0.3`)
2. You want the cursor smoothing lag to be 3 frames. What `α` do you need?
3. Your hand drifts steadily 0.3 of the frame in 0.5 s. Does it trigger a swipe with the default settings?
4. The pinch ratio over six frames is `0.30, 0.24, 0.30, 0.35, 0.45, 0.30`. Using thresholds 0.25 (press) and 0.40 (release), what is the button state after each frame? Start released.
5. The region is `(0.2, 0.15, 0.8, 0.75)` and your screen is 1920 px wide. Your index knuckle is at `x = 0.3`. Which screen x is the cursor aimed at?
6. A finger's tip is 0.30 from the wrist and its PIP joint 0.28 from the wrist. With `EXTEND_MARGIN = 1.1`, is it up?
7. Why does the index knuckle drive the cursor instead of the fingertip?
8. Why can a single threshold for pinching cause the mouse button to flicker?
9. Explain why a swipe right is usually followed by a (wrong) swipe-left detection if there is no cooldown.
10. The smoother has the last 7 labels `FIST ×7` and then `POINT` frames arrive one at a time. After how many `POINT` frames does the reported gesture change?

### Answers

1. `hand_size = 0.20`. Tip distance = √(0.02² + 0.02²) = 0.0283. Ratio = 0.0283 / 0.20 = **0.141 < 0.3**, so yes it is touching (provided another finger is up, otherwise it is a fist).
2. Lag = (1 − α)/α = 3, so 1 − α = 3α, giving **α = 0.25**.
3. **No.** The window only holds the last 0.4 s, which is a movement of 0.24 (≥ 0.20 ✓). Speed = 0.24 / 0.4 = 0.6, which is below 0.8 ✗.
4. 0.30 → released. 0.24 → **pressed**. 0.30 → pressed (inside the dead band). 0.35 → pressed. 0.45 → **released**. 0.30 → released.
5. `nx = (0.3 − 0.2) / 0.6 = 0.1667`, so `0.1667 × 1920 = ` **320 px**.
6. Needed: 0.28 × 1.1 = 0.308. The tip distance 0.30 is smaller, so **no**, it counts as down. The margin makes a half-bent finger count as down.
7. When pinching, the fingertip moves a lot but the knuckle barely moves, so the cursor would jump off its target at the moment of the click.
8. Noise makes the ratio wobble across the threshold; each crossing toggles the button. Two thresholds (hysteresis) require a clear change before switching.
9. The hand moves back to where it started at a similar speed over a similar distance, so the same distance/speed/straightness rules match in the opposite direction. The cooldown blocks that return stroke.
10. After **5** `POINT` frames (the 5th makes POINT win 5 of the last 7).

---

## 14. Glossary

| Term | Meaning |
|---|---|
| Anchor | A predefined box shape/position a detector scores |
| BGR / RGB | Colour channel order (OpenCV / MediaPipe) |
| Centroid | The average position of several points |
| Classifier | Something that assigns a label to an input |
| Cooldown | Dead time after an action |
| CNN | Convolutional neural network, a model suited to images |
| Debounce | Ignoring repeated triggers caused by one real event |
| Dead band | The range between two hysteresis thresholds |
| EMA | Exponential moving average |
| Euclidean distance | Straight-line distance |
| FSM | Finite-state machine |
| FPS | Frames per second |
| Hysteresis | Using two different thresholds for switching on and off |
| IoU | Intersection over union, a box overlap measure |
| Landmark | One of the 21 tracked hand points |
| NMS | Non-maximum suppression, removing duplicate detections |
| Normalised coordinates | Positions expressed as a fraction (0 to 1) of image size |
| Regression | Predicting numbers rather than labels |
| Rotation invariance | Unchanged when the object is rotated |
| Scale invariance | Unchanged when the object is bigger or smaller |
| Sliding window | The most recent N samples (or seconds) of a signal |
| TFLite | TensorFlow Lite, a compact model format for devices |
| UIPI | Windows rule blocking input to higher-privilege windows |
| Virtual-key code | The number Windows uses to identify a key |

---

## 15. Further reading

- Zhang et al., *MediaPipe Hands: On-device Real-time Hand Tracking* (2020), the paper behind the tracker
- Google MediaPipe documentation for the Hand Landmarker task
- Casiez, Roussel and Vogel, *1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems* (CHI 2012)
- OpenCV documentation for `VideoCapture`, `cvtColor`, `flip`
- PyAutoGUI documentation (keyboard, mouse and fail-safe sections)
- Python documentation for `dataclasses`, `enum` and `collections`
- Introductory material on: Euclidean distance, exponential smoothing, Schmitt triggers, finite-state machines, non-maximum suppression
