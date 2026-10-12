import cv2

from actions import ActionConfig, ActionController
from camera import webcam
from gestures import Gesture, GestureSmoother, classify
from processor import HandProcessor

DRY_RUN = False   # True = only print what would happen. Set False to really control the PC.

MODE_COLORS = {            # BGR colours for the mode label
    "DISARMED": (80, 80, 255),
    "ARMED": (80, 220, 100),
    "CURSOR": (255, 200, 0),
    "DRAG": (0, 160, 255),
    "SCROLL": (255, 120, 220),
}


def draw_status(frame, controller, gesture):
    """Show the mode, arming progress, cursor box + target and the last action."""
    h, w = frame.shape[:2]

    color = MODE_COLORS.get(controller.mode, (255, 255, 255))
    cv2.putText(frame, controller.mode, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(frame, f"pose: {gesture.value}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if not controller.armed:
        if controller.arm_progress > 0:                     # progress bar while arming
            cv2.rectangle(frame, (10, 85), (210, 100), (255, 255, 255), 1)
            cv2.rectangle(frame, (10, 85), (10 + int(200 * controller.arm_progress), 100), color, -1)
        cv2.putText(frame, "hold OPEN PALM 1 s to arm", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        left, top, right, bottom = controller.cfg.region    # the box mapped to your screen
        cv2.rectangle(frame, (int(left * w), int(top * h)), (int(right * w), int(bottom * h)),
                      (120, 120, 120), 1)
        if controller.pointer is not None:                  # where the cursor is aimed
            px, py = int(controller.pointer[0] * w), int(controller.pointer[1] * h)
            cv2.circle(frame, (px, py), 12, color, 2)
            cv2.circle(frame, (px, py), 3, color, -1)
        if controller.last_action:
            cv2.putText(frame, controller.last_action, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)


def main():
    cam = webcam()
    proc = HandProcessor()
    smoother = GestureSmoother()
    controller = ActionController(ActionConfig(dry_run=DRY_RUN))

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            frame, hand = proc.process(frame)            # 1. find + draw the hand
            if hand is None:                              # 2. what pose is it?
                smoother.reset()
                gesture = Gesture.NONE
            else:
                gesture = smoother.update(classify(hand))

            for event in controller.update(hand, gesture):   # 3. do something about it
                print(event)

            draw_status(frame, controller, gesture)
            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        controller.close()          # releases the mouse button if it was held
        proc.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()