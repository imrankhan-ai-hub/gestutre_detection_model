r"""
gestures.py
===========
Turns the 21 hand landmarks from MediaPipe into a GESTURE NAME.

This file knows nothing about the camera, MediaPipe or the mouse/keyboard.
It only does maths on points, so it is easy to test and easy to extend.
Later, a separate actions.py will decide what to DO for each gesture.

Landmark numbers (what MediaPipe gives us, 0..20):

            8   12  16  20          4 = thumb tip      8 = index tip
            |   |   |   |           12 = middle tip    16 = ring tip
            7   11  15  19          20 = pinky tip
        4   |   |   |   |           
        |   6   10  14  18          Each finger has: tip, DIP, PIP, MCP (the knuckle)
        3   |   |   |   |           Index:  MCP 5,  PIP 6,  DIP 7,  TIP 8
        |   5   9   13  17          Middle: MCP 9,  PIP 10, DIP 11, TIP 12
        2    \  |   |  /            Ring:   MCP 13, PIP 14, DIP 15, TIP 16
        |     \ |   | /             Pinky:  MCP 17, PIP 18, DIP 19, TIP 20
        1      \|   |/              Thumb:  CMC 1,  MCP 2,  IP 3,   TIP 4
         \      0 (wrist)           Wrist:  0

Every landmark has .x and .y between 0 and 1 (fraction of the image width/height).
IMPORTANT: y grows DOWNWARDS (top of the image is y = 0).
"""
import math
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum

# 1. THE LIST OF GESTURES

class Gesture(str, Enum):
    NONE = "NONE"              # no hand in view
    UNKNOWN = "UNKNOWN"        # a hand is visible, but the pose matches nothing below
    FIST = "FIST"
    OPEN_PALM = "OPEN PALM"
    POINT = "POINT"
    PEACE = "PEACE"
    THREE = "THREE"
    FOUR = "FOUR"
    ROCK = "ROCK"
    SHAKA = "SHAKA"
    L_SHAPE = "L SHAPE"
    THUMBS_UP = "THUMBS UP"
    THUMBS_DOWN = "THUMBS DOWN"
    PINCH = "PINCH"
    OK_SIGN = "OK SIGN"
    fuck = "FUCK you"  # for testing only, not a real gesture
    fingring = "FINGRING"  # for testing only, not a real gesture



# 2. BUILDING BLOCKS (small helper functions the gestures are made from)
# Landmark numbers we use, given names so the code reads like English
WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# How "clearly" a finger must be straight to count as up. Higher = stricter.
# 1.0 would mean "tip even slightly farther than the joint"; 1.1 adds a safety margin.
EXTEND_MARGIN = 1.1

# Pinch: thumb tip to index tip distance, measured in "hand sizes".
# 0.3 means the two tips are closer than 30% of (wrist -> middle knuckle).
PINCH_RATIO = 0.3

# Thumbs up/down: how far (in hand sizes) the thumb tip must be above/below the wrist.
THUMB_VERTICAL = 0.5


def dist(a, b):
    """Straight-line distance between two landmarks (in image units 0..1)."""
    return math.hypot(a.x - b.x, a.y - b.y)


def hand_size(hand):
    """Wrist to middle-finger knuckle. We measure everything relative to this,
    so gestures work the same whether your hand is near or far from the camera."""
    return dist(hand[WRIST], hand[MIDDLE_MCP])


@dataclass(frozen=True)
class FingerStates:
    """True = that finger is straight (extended). False = curled."""
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    @property
    def count(self):
        return sum((self.thumb, self.index, self.middle, self.ring, self.pinky))


def finger_states(hand):
    """Work out which of the five fingers are straight.

    Idea for the four normal fingers:
        a STRAIGHT finger  -> fingertip is FAR from the wrist
        a CURLED finger    -> fingertip folds back, so it is CLOSER to the wrist
                              than the finger's middle joint (PIP)

            straight            curled
              ● tip               ● PIP
              |                  / ● tip   <- tip is nearer the wrist than the PIP
              ● PIP             |
              |                 |
              ● wrist           ● wrist

    The thumb folds sideways across the palm, so the wrist test does not work.
    Instead we compare against the base of the pinky: a straight thumb sticks
    out FARTHER from the pinky base than its own middle joint does.
    """
    wrist = hand[WRIST]

    def up(pip, tip):
        return dist(hand[tip], wrist) > dist(hand[pip], wrist) * EXTEND_MARGIN

    pinky_base = hand[PINKY_MCP]
    thumb = dist(hand[THUMB_TIP], pinky_base) > dist(hand[THUMB_IP], pinky_base) * EXTEND_MARGIN

    return FingerStates(
        thumb=thumb,
        index=up(INDEX_PIP, INDEX_TIP),
        middle=up(MIDDLE_PIP, MIDDLE_TIP),
        ring=up(RING_PIP, RING_TIP),
        pinky=up(PINKY_PIP, PINKY_TIP),
    )


def pinch_ratio(hand):
    """Thumb tip to index tip distance divided by hand size. Small = touching."""
    size = hand_size(hand)
    return dist(hand[THUMB_TIP], hand[INDEX_TIP]) / size if size > 1e-6 else 99.0


def is_pinching(hand):
    """True when the thumb and index fingertips touch.

    Needs at least one other finger straight, because in a closed FIST the thumb
    also ends up near the index finger and we do not want that to count.
    Later, actions.py uses this for click (pinch starts) and drag (pinch held).
    """
    f = finger_states(hand)
    return pinch_ratio(hand) < PINCH_RATIO and (f.middle or f.ring or f.pinky)


def thumb_direction(hand):
    """'up', 'down' or 'side', judged by the thumb tip's height relative to the wrist.
    Assumes the hand is roughly upright in the image (y grows downwards)."""
    size = hand_size(hand)
    if size <= 1e-6:
        return "side"
    offset = (hand[WRIST].y - hand[THUMB_TIP].y) / size   # positive = thumb above wrist
    if offset > THUMB_VERTICAL:
        return "up"
    if offset < -THUMB_VERTICAL:
        return "down"
    return "side"


# 3. THE CLASSIFIER: finger states  ->  gesture name

def classify(hand):
    """Look at which fingers are up and decide the gesture.

    The order matters: more specific gestures are checked before general ones.
    Quick reference (T=thumb I=index M=middle R=ring P=pinky, 1=up 0=curled):

      Gesture      T I M R P   Extra condition
      -----------  ---------   ----------------------------------------------
      OPEN PALM    1 1 1 1 1
      FOUR         0 1 1 1 1
      THREE        - 1 1 1 0
      PEACE        - 1 1 0 0
      POINT        0 1 0 0 0
      L SHAPE      1 1 0 0 0
      ROCK         - 1 0 0 1   (the "rock on" horns sign)
      SHAKA        1 0 0 0 1   (the "call me" sign)
      THUMBS UP    1 0 0 0 0   thumb tip clearly ABOVE the wrist
      THUMBS DOWN  1 0 0 0 0   thumb tip clearly BELOW the wrist
      FIST         - 0 0 0 0   (thumb tucked, or sideways)
      PINCH        thumb tip touches index tip, other fingers not all up
      OK SIGN      thumb tip touches index tip, middle+ring+pinky all up
      ('-' means the thumb is ignored for that gesture)
      fuck         middle finger is up
      fingring     index finger and middle finger are up and close together
    """
    f = finger_states(hand)
    touching = pinch_ratio(hand) < PINCH_RATIO

    # --- Pinch family first: thumb and index touching -------------------------
    if touching and (f.middle or f.ring or f.pinky):
        return Gesture.OK_SIGN if (f.middle and f.ring and f.pinky) else Gesture.PINCH

    # --- All four fingers up --------------------------------------------------
    if f.index and f.middle and f.ring and f.pinky:
        return Gesture.OPEN_PALM if f.thumb else Gesture.FOUR

    # --- Three fingers: index, middle, ring ----------------------------------
    if f.index and f.middle and f.ring and not f.pinky:
        return Gesture.THREE

    # --- Two fingers: index + middle (peace) ---------------------------------
    # if f.index and f.middle and not f.ring and not f.pinky:
    #     return Gesture.PEACE

    # --- Index + pinky (rock) ------------------------------------------------
    if f.index and f.pinky and not f.middle and not f.ring:
        return Gesture.ROCK

    # --- Index only: pointing, or L shape if the thumb sticks out too ---------
    if f.index and not (f.middle or f.ring or f.pinky):
        return Gesture.L_SHAPE if f.thumb else Gesture.POINT

    # --- Thumb + pinky only (shaka) -------------------------------------------
    if f.thumb and f.pinky and not (f.index or f.middle or f.ring):
        return Gesture.SHAKA
    if f.middle and not f.index and not f.ring and not f.pinky:
        return Gesture.fuck
    # --- No finger up: fist, or thumbs up/down -------------------------------
    if not(f.index or f.middle or f.ring or f.pinky):
            if f.thumb:
                direction = thumb_direction(hand)
                if direction == "up":
                    return Gesture.THUMBS_UP
                if direction == "down":
                    return Gesture.THUMBS_DOWN
            return Gesture.FIST
    
    if f.index and f.middle and not f.ring and not f.pinky and dist(hand[INDEX_TIP], hand[MIDDLE_TIP]) < 0.05:
        return Gesture.fingring
    elif f.index and f.middle and not f.ring and not f.pinky:
        return Gesture.PEACE 
    else:
        return Gesture.UNKNOWN
    

    
    

    
    # --- No finger up: fist, or thumbs up/down -------------------------------
    # if not(f.index or f.middle or f.ring or f.pinky):
    #     if f.thumb:
    #         direction = thumb_direction(hand)
    #         if direction == "up":
    #             return Gesture.THUMBS_UP
    #         if direction == "down":
    #             return Gesture.THUMBS_DOWN
    #     return Gesture.FIST

    

# 4. SMOOTHING: stops the label flickering from frame to frame

class GestureSmoother:
    """Only changes the reported gesture when it has been stable for a few frames.

    Raw detection can flicker (e.g. POINT, UNKNOWN, POINT, POINT...) because one
    badly-tracked frame misreads a finger. We look at the last `size` frames and
    switch to a new gesture only when it wins at least `votes` of them.
    """

    def __init__(self, size=7, votes=5):
        self._recent = deque(maxlen=size)
        self._votes = votes
        self.stable = Gesture.NONE

    def update(self, gesture):
        self._recent.append(gesture)
        winner, count = Counter(self._recent).most_common(1)[0]
        if count >= self._votes:
            self.stable = winner
        return self.stable

    def reset(self):
        self._recent.clear()
        self.stable = Gesture.NONE