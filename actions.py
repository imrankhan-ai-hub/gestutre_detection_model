"""
actions.py
==========
Decides what to DO for each gesture, then does it (keyboard / mouse).

Data flow, once per camera frame:

    hand landmarks ─┐
                    ├──▶ ActionController.update() ──▶ keyboard / mouse
    gesture name  ──┘                                   (through InputBackend)

What each gesture does (only while ARMED):

    OPEN PALM + fast move  → SWIPE: switch desktop / task view / show desktop
    POINT or L SHAPE       → CURSOR: your index knuckle moves the mouse
        + thumb touches index   → hold left mouse button (quick = CLICK, moving = DRAG)
    PEACE                  → SCROLL: move your hand up/down
    FIST (hold)            → play / pause
    ROCK (hold)            → mute
    THUMBS UP / DOWN (hold)→ volume up / down (repeats while held)
    THREE (hold)           → next track
    SHAKA (hold)           → disarm

ARMING: nothing happens until you hold an OPEN PALM for 1 second. This is the
safety switch that stops normal hand movement from triggering actions.
"""
import math
import time
from collections import deque
from dataclasses import dataclass, field

from gestures import Gesture, pinch_ratio

# MediaPipe landmark numbers used here
WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 0, 5, 9, 13, 17
PALM_POINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


# =============================================================================
# 1. SETTINGS (change numbers here to tune the feel, no other code needs editing)
# =============================================================================
@dataclass(frozen=True)
class Action:
    """One thing we can do: a label for the screen and the keys to press."""
    label: str
    keys: tuple
    repeat: bool = False        # True = keep firing while the gesture is held


@dataclass
class ActionConfig:
    dry_run: bool = False        # True = print what WOULD happen, press nothing

    # --- arming ---------------------------------------------------------------
    arm_hold: float = 1.0       # seconds of open palm needed to arm
    disarm_hold: float = 1.0    # seconds of SHAKA needed to disarm
    idle_disarm: float = 3.0    # hand out of view this long -> disarm automatically

    # --- swipe ----------------------------------------------------------------
    swipe_distance: float = 0.20    # hand must travel this fraction of the frame...
    swipe_window: float = 0.4       # ...within this many seconds
    swipe_min_speed: float = 0.8    # and at least this fast (frame-fractions/second)
    swipe_axis_ratio: float = 1.5   # how straight: main direction must beat the other by this
    swipe_cooldown: float = 1.0     # dead time after a swipe (also blocks the return stroke)
    palm_grace: float = 0.4         # an open palm seen this recently still counts mid-swipe

    # --- cursor / click / drag -------------------------------------------------
    # Only this box of the camera image is mapped to the whole screen:
    # (left, top, right, bottom) as fractions of the frame. A smaller box = you
    # need less arm movement to reach the screen edges.
    region: tuple = (0.2, 0.15, 0.8, 0.75)
    cursor_smoothing: float = 0.35  # 0..1. Lower = smoother but laggier
    pinch_down: float = 0.25        # thumb-index closer than this -> press mouse
    pinch_up: float = 0.40          # farther than this -> release (gap prevents flicker)
    drag_pixels: int = 15           # moved more than this while pressed = DRAG, else CLICK

    # --- scroll ---------------------------------------------------------------
    scroll_gain: float = 4000       # bigger = scrolls faster. (Windows: 120 units = 1 notch)
    scroll_deadzone: float = 0.002  # ignore tiny hand jitter

    # --- hold gestures ---------------------------------------------------------
    hold_time: float = 0.5          # hold the pose this long to trigger
    action_cooldown: float = 0.8    # dead time after a one-shot action
    repeat_interval: float = 0.3    # seconds between repeats (volume)

    # --- what each thing does (Windows key names; edit for Mac/Linux) ----------
    swipe_actions: dict = field(default_factory=lambda: {
        "left":  Action("next desktop",     ("ctrl", "win", "right")),
        "right": Action("previous desktop", ("ctrl", "win", "left")),
        "up":    Action("task view",        ("win", "tab")),
        "down":  Action("show desktop",     ("win", "d")),
    })
    hold_actions: dict = field(default_factory=lambda: {
        Gesture.FIST:        Action("play/pause",  ("playpause",)),
        Gesture.ROCK:        Action("mute",        ("volumemute",)),
        Gesture.THREE:       Action("next track",  ("nexttrack",)),
        Gesture.THUMBS_UP:   Action("volume up",   ("volumeup",),   repeat=True),
        Gesture.THUMBS_DOWN: Action("volume down", ("volumedown",), repeat=True),
    })


# =============================================================================
# 2. INPUT BACKEND: the only place that touches the real keyboard and mouse
# =============================================================================
class InputBackend:
    """Thin wrapper around PyAutoGUI. In dry-run mode it prints instead of acting."""

    EDGE = 5  # stay this many pixels from the screen edge (PyAutoGUI's fail-safe is a corner)

    def __init__(self, dry_run):
        self.dry_run = dry_run
        self.screen_size = (1920, 1080)
        self.position = (0, 0)
        self._pg = None
        try:
            import pyautogui
            pyautogui.PAUSE = 0
            pyautogui.FAILSAFE = True
            self.screen_size = tuple(pyautogui.size())
            self._pg = pyautogui
        except Exception as exc:
            if not dry_run:
                raise RuntimeError(f"pyautogui is required when dry_run=False: {exc}") from exc

    def move(self, x, y):
        w, h = self.screen_size
        x = int(min(max(x, self.EDGE), w - 1 - self.EDGE))
        y = int(min(max(y, self.EDGE), h - 1 - self.EDGE))
        self.position = (x, y)
        if not self.dry_run:
            self._pg.moveTo(x, y, _pause=False)

    def mouse_down(self):
        self._say("mouse DOWN")
        if not self.dry_run:
            self._pg.mouseDown(_pause=False)

    def mouse_up(self):
        self._say("mouse UP")
        if not self.dry_run:
            self._pg.mouseUp(_pause=False)

    def scroll(self, amount):
        if not self.dry_run:
            self._pg.scroll(amount, _pause=False)

    def hotkey(self, *keys):
        self._say("keys " + "+".join(keys))
        if not self.dry_run:
            self._pg.hotkey(*keys)

    def _say(self, text):
        if self.dry_run:
            print(f"[dry-run] {text}")


# =============================================================================
# 3. SWIPE DETECTOR: a MOTION gesture (needs several frames, not one)
# =============================================================================
class SwipeDetector:
    """Remembers where the palm was over the last fraction of a second.

    A swipe is: far enough, fast enough, and mostly in a straight line.

        positions over the last 0.4 s:   ●   ●   ●   ●   ●   ●  →
        distance travelled  = newest - oldest          must be >= swipe_distance
        speed               = distance / time          must be >= swipe_min_speed
        straightness        = main axis vs other axis  must differ by swipe_axis_ratio

    Coordinates are fractions of the image. y grows DOWNWARD, so a move with
    decreasing y is a swipe UP.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._samples = deque()          # (time, x, y)

    def reset(self):
        self._samples.clear()

    def update(self, now, x, y):
        """Add this frame's palm position. Returns 'left'/'right'/'up'/'down' or None."""
        c = self.cfg
        self._samples.append((now, x, y))
        while self._samples and now - self._samples[0][0] > c.swipe_window:
            self._samples.popleft()          # forget positions that are too old
        if len(self._samples) < 2:
            return None

        t0, x0, y0 = self._samples[0]
        dt = now - t0
        if dt <= 0:
            return None
        dx, dy = x - x0, y - y0
        ax, ay = abs(dx), abs(dy)

        direction = None
        if ax >= c.swipe_distance and ax >= c.swipe_axis_ratio * ay and ax / dt >= c.swipe_min_speed:
            direction = "right" if dx > 0 else "left"
        elif ay >= c.swipe_distance and ay >= c.swipe_axis_ratio * ax and ay / dt >= c.swipe_min_speed:
            direction = "down" if dy > 0 else "up"

        if direction:
            self._samples.clear()            # one movement = one swipe
        return direction


# =============================================================================
# 4. THE CONTROLLER: the state machine that ties gestures to actions
# =============================================================================
class ActionController:
    """Call update(hand, gesture) once per frame. It returns a list of text events.

    Public attributes you can show on screen:
        armed          True/False
        mode           "DISARMED", "ARMED", "CURSOR", "DRAG" or "SCROLL"
        arm_progress   0..1 while you hold an open palm to arm
        pointer        (x, y) of the cursor target as a fraction of the camera frame
        last_action    text of the most recent action
    """

    def __init__(self, config=None):
        self.cfg = config or ActionConfig()
        self.backend = InputBackend(self.cfg.dry_run)
        self._swipe = SwipeDetector(self.cfg)

        # public state for the on-screen display
        self.armed = False
        self.mode = "DISARMED"
        self.arm_progress = 0.0
        self.pointer = None
        self.last_action = ""

        # arming / presence
        self._arm_since = None
        self._disarm_since = None
        self._last_seen = None
        self._cooldown_until = 0.0

        # swipe
        self._last_open_palm = float("-inf")

        # hold gestures
        self._hold_gesture = Gesture.NONE
        self._hold_since = 0.0
        self._hold_last_fire = None       # None = not fired yet during this hold

        # cursor / click / drag
        self._cursor_active = False
        self._smooth = None               # smoothed (x, y) inside the region, 0..1
        self._pinch_down = False
        self._press_pos = (0, 0)

        # scroll
        self._prev_scroll_y = None
        self._scroll_acc = 0.0

    # ------------------------------------------------------------------ public

    def update(self, hand, gesture, now=None):
        """hand = 21 landmarks or None, gesture = a Gesture. Returns list of event strings."""
        now = time.monotonic() if now is None else now
        events = []

        if hand is None:
            self._on_hand_lost(now, events)
            return events

        self._last_seen = now
        if self.armed:
            self._handle_armed(hand, gesture, now, events)
        else:
            self._handle_disarmed(gesture, now, events)
        return events

    def close(self):
        """Always call before exiting so the mouse button is never left pressed."""
        self._release_mouse([], "shutdown")

    # ------------------------------------------------------------- arming logic

    def _handle_disarmed(self, gesture, now, events):
        self.mode = "DISARMED"
        if gesture is Gesture.OPEN_PALM:
            if self._arm_since is None:
                self._arm_since = now
            self.arm_progress = min(1.0, (now - self._arm_since) / self.cfg.arm_hold)
            if self.arm_progress >= 1.0:
                self._arm(now, events)
        else:
            self._arm_since = None
            self.arm_progress = 0.0

    def _arm(self, now, events):
        self.armed = True
        self.mode = "ARMED"
        self._arm_since = None
        self.arm_progress = 0.0
        self._cooldown_until = now + 1.0     # don't swipe straight after arming
        self._swipe.reset()
        self._say(events, "ARMED")

    def _disarm(self, events, reason):
        self._release_mouse(events, reason)
        self.armed = False
        self.mode = "DISARMED"
        self._cursor_active = False
        self._smooth = None
        self.pointer = None
        self._prev_scroll_y = None
        self._reset_hold()
        self._swipe.reset()
        self._disarm_since = None
        self._say(events, f"DISARMED ({reason})")

    # ------------------------------------------------------ the armed behaviour

    def _handle_armed(self, hand, gesture, now, events):
        # 1) SHAKA held = disarm
        if gesture is Gesture.SHAKA:
            if self._disarm_since is None:
                self._disarm_since = now
            if now - self._disarm_since >= self.cfg.disarm_hold:
                self._disarm(events, "shaka")
                return
        else:
            self._disarm_since = None

        if gesture is Gesture.OPEN_PALM:
            self._last_open_palm = now

        # 2) cursor mode (point -> move, pinch -> press)
        if self._cursor_wanted(gesture):
            self._do_cursor(hand, now, events)
            self._swipe.reset()
            self._prev_scroll_y = None
            self._reset_hold()
            return
        self._smooth = None
        self.pointer = None

        # 3) scroll mode
        if gesture is Gesture.PEACE:
            self._do_scroll(hand)
            self.mode = "SCROLL"
            self._swipe.reset()
            self._reset_hold()
            return
        self._prev_scroll_y = None
        self._scroll_acc = 0.0

        # 4) swipes and hold actions
        self.mode = "ARMED"
        self._do_swipe(hand, now, events)
        self._do_hold_actions(gesture, now, events)

    # --------------------------------------------------------------- swipe

    def _do_swipe(self, hand, now, events):
        # Only swipe with an open palm (seen recently: one blurry frame must not cancel it)
        if now - self._last_open_palm > self.cfg.palm_grace:
            self._swipe.reset()
            return
        x = sum(hand[i].x for i in PALM_POINTS) / len(PALM_POINTS)
        y = sum(hand[i].y for i in PALM_POINTS) / len(PALM_POINTS)

        direction = self._swipe.update(now, x, y)
        if direction and now >= self._cooldown_until:
            action = self.cfg.swipe_actions.get(direction)
            if action:
                self.backend.hotkey(*action.keys)
                self._say(events, f"swipe {direction} -> {action.label}")
            self._cooldown_until = now + self.cfg.swipe_cooldown
            self._reset_hold()

    # ------------------------------------------------------ hold gestures

    def _do_hold_actions(self, gesture, now, events):
        action = self.cfg.hold_actions.get(gesture)
        if action is None:
            self._reset_hold()
            return

        if gesture is not self._hold_gesture:          # a new pose just started
            self._hold_gesture = gesture
            self._hold_since = now
            self._hold_last_fire = None
            return
        if now - self._hold_since < self.cfg.hold_time:
            return                                      # not held long enough yet

        first = self._hold_last_fire is None
        again = (action.repeat and not first
                 and now - self._hold_last_fire >= self.cfg.repeat_interval)
        if (first and now >= self._cooldown_until) or again:
            self.backend.hotkey(*action.keys)
            self._hold_last_fire = now
            self._say(events, f"{gesture.value.lower()} -> {action.label}")
            if not action.repeat:
                self._cooldown_until = now + self.cfg.action_cooldown

    def _reset_hold(self):
        self._hold_gesture = Gesture.NONE
        self._hold_last_fire = None

    # ------------------------------------------------- cursor, click and drag

    def _cursor_wanted(self, gesture):
        """Cursor mode starts with POINT / L SHAPE and continues through the pinch."""
        if self._pinch_down:
            return True                    # never drop out of a drag halfway
        if gesture in (Gesture.POINT, Gesture.L_SHAPE):
            self._cursor_active = True
        elif gesture not in (Gesture.PINCH, Gesture.OK_SIGN):
            self._cursor_active = False
        return self._cursor_active

    def _do_cursor(self, hand, now, events):
        c = self.cfg
        # Use the index KNUCKLE, not the fingertip: the knuckle barely moves when
        # you pinch, so pinching to click does not shove the cursor off its target.
        px, py = hand[INDEX_MCP].x, hand[INDEX_MCP].y

        left, top, right, bottom = c.region
        nx = min(max((px - left) / (right - left), 0.0), 1.0)    # 0..1 inside the box
        ny = min(max((py - top) / (bottom - top), 0.0), 1.0)

        if self._smooth is None:
            self._smooth = (nx, ny)
        else:                              # exponential smoothing: move a fraction toward target
            sx, sy = self._smooth
            a = c.cursor_smoothing
            self._smooth = (sx + a * (nx - sx), sy + a * (ny - sy))

        sw, sh = self.backend.screen_size
        self.backend.move(self._smooth[0] * sw, self._smooth[1] * sh)
        self.pointer = (left + self._smooth[0] * (right - left),
                        top + self._smooth[1] * (bottom - top))     # for drawing on the frame

        # Pinch with hysteresis: press below pinch_down, release above pinch_up
        ratio = pinch_ratio(hand)
        if not self._pinch_down and ratio < c.pinch_down:
            self._pinch_down = True
            self._press_pos = self.backend.position
            self.backend.mouse_down()
        elif self._pinch_down and ratio > c.pinch_up:
            self._release_mouse(events, "pinch released")
        self.mode = "DRAG" if self._pinch_down else "CURSOR"

    def _release_mouse(self, events, reason):
        if not self._pinch_down:
            return
        self._pinch_down = False
        self.backend.mouse_up()
        x0, y0 = self._press_pos
        x1, y1 = self.backend.position
        moved = math.hypot(x1 - x0, y1 - y0)
        self._say(events, "DRAG end" if moved > self.cfg.drag_pixels else "CLICK")

    # --------------------------------------------------------------- scroll

    def _do_scroll(self, hand):
        c = self.cfg
        y = hand[MIDDLE_MCP].y
        if self._prev_scroll_y is not None:
            dy = y - self._prev_scroll_y
            if abs(dy) > c.scroll_deadzone:
                self._scroll_acc += -dy * c.scroll_gain      # hand up -> scroll up
            whole = int(self._scroll_acc)
            if whole:
                self.backend.scroll(whole)
                self._scroll_acc -= whole
        self._prev_scroll_y = y

    # ------------------------------------------------------ hand disappeared

    def _on_hand_lost(self, now, events):
        self._arm_since = None
        self.arm_progress = 0.0
        self._disarm_since = None
        self._prev_scroll_y = None
        self._reset_hold()
        # NOTE: swipe history is kept on purpose. Fast motion blurs the hand and the
        # tracker drops a frame or two; old samples expire by themselves.
        if self._last_seen is None:
            return
        gone = now - self._last_seen
        if gone > 0.3:
            self._release_mouse(events, "hand lost")      # never leave a button stuck
            self._cursor_active = False
            self._smooth = None
            self.pointer = None
        if self.armed and gone >= self.cfg.idle_disarm:
            self._disarm(events, "hand away")

    # ------------------------------------------------------------- utilities

    def _say(self, events, text):
        self.last_action = text
        events.append(text)

