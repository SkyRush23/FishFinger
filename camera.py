# pyrefly: ignore [missing-import]
import cv2
import logging
import time
from hand_tracker import HandTracker
from gestures import GestureRecognizer
from active_window import ActiveWindowDetector
from gemini_brain import GeminiBrain
from executor import CommandExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Camera:
    """
    Modular Camera handler for GestureOS.
    Manages webcam capture, hand tracking processing (MediaPipe Solutions),
    gesture recognition, active window detection, and live video display.
    """

    def __init__(self, camera_index=0, width=640, height=480, window_name="GestureOS - Camera Feed"):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.window_name = window_name
        self.cap = None
        self.hand_tracker = HandTracker()
        self.gesture_recognizer = GestureRecognizer()
        self.window_detector = ActiveWindowDetector()
        self.brain = GeminiBrain()
        self.executor = CommandExecutor()
        self.command_cooldown = 1.0
        self.last_command_time = 0.0
        self.command_cache = {}
        self.last_gesture = None
        self.last_app = None
        self.last_executed_key = None

    def start(self) -> bool:
        """Initialize and open the webcam."""
        logging.info(f"Opening camera index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            logging.error(f"Failed to open camera with index {self.camera_index}.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logging.info(f"Camera opened successfully ({self.width}x{self.height}).")
        return True

    def read(self):
        """
        Read a frame from the camera.
        Returns:
            (bool, numpy.ndarray): Success flag and the captured frame.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def show(self, frame, window_name=None):
        """Display a frame in an OpenCV window."""
        target_window = window_name if window_name else self.window_name
        if frame is not None:
            cv2.imshow(target_window, frame)

    def stop(self):
        """Release camera and tracker resources, and close display windows."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logging.info("Camera released.")
        if hasattr(self, "hand_tracker") and self.hand_tracker:
            self.hand_tracker.close()
        cv2.destroyAllWindows()
        logging.info("OpenCV windows closed.")

    def run(self):
        """
        Main camera loop. Captures frame, queries active application window,
        processes hand landmarks and gesture recognitions, renders overlays, and displays video feed.
        Exits when 'q' or 'Q' is pressed.
        """
        if not self.start():
            logging.error("Unable to start camera stream.")
            return

        logging.info("Starting live camera feed with MediaPipe Solutions. Press 'q' or 'Q' to exit.")

        try:
            while True:
                ret, frame = self.read()
                if not ret or frame is None:
                    logging.warning("Failed to grab frame. Exiting feed...")
                    break

                # Process frame using HandTracker (MediaPipe Solutions)
                annotated_frame, results = self.hand_tracker.process_frame(frame)
                display_frame = annotated_frame if annotated_frame is not None else frame

                h, w, _ = display_frame.shape

                # 1. Detect and display active application in top-left corner
                active_app = self.window_detector.get_active_app()
                app_text = f"App: {active_app}"
                cv2.putText(
                    display_frame, app_text, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA
                )
                cv2.putText(
                    display_frame, app_text, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA
                )

                if results and getattr(results, "multi_hand_landmarks", None):
                    # Analyze all gestures in the current frame
                    gesture_data = self.gesture_recognizer.recognize_all_gestures(results)

                    gesture_name = gesture_data.get("multi_hand_gesture")
                    if not gesture_name:
                        gesture_name = next(
                            (
                                hand_info["gesture"]
                                for hand_info in gesture_data.get("hand_gestures", [])
                                if hand_info["gesture"].upper() not in {
                                    "NO_GESTURE", "UNKNOWN", "UNKNOWN GESTURE"
                                }
                            ),
                            None,
                        )

                    if gesture_name:
                        now = time.monotonic()
                        key = (active_app, gesture_name)
                        key_changed = key != (self.last_app, self.last_gesture)
                        cooldown_expired = now - self.last_command_time >= self.command_cooldown

                        if key in self.command_cache:
                            command = self.command_cache[key]
                            cache_status = "HIT"
                        else:
                            command = self.brain.get_command(active_app, gesture_name)
                            self.command_cache[key] = command
                            cache_status = "MISS"

                        if key_changed:
                            logging.info(
                                "%s -> %s -> Cache %s -> %s",
                                gesture_name, active_app, cache_status, command
                            )

                        if (
                            command != "NO_ACTION"
                            and cooldown_expired
                            and key != self.last_executed_key
                        ):
                            self.executor.execute(command)
                            self.last_command_time = time.monotonic()
                            self.last_executed_key = key

                        self.last_gesture = gesture_name
                        self.last_app = active_app
                    else:
                        self.last_gesture = None
                        self.last_app = None
                        self.last_executed_key = None

                    # 2. Render multi-hand gesture banner if active (e.g. "Both Hands Open")
                    multi_gesture = gesture_data.get("multi_hand_gesture")
                    if multi_gesture:
                        banner_text = f"*** {multi_gesture} ***"
                        cv2.putText(
                            display_frame, banner_text, (w // 2 - 170, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA
                        )
                        cv2.putText(
                            display_frame, banner_text, (w // 2 - 170, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA
                        )

                    # 3. Render per-hand gesture name and finger count overlay above each hand
                    for hand_info in gesture_data.get("hand_gestures", []):
                        i = hand_info["hand_index"]
                        hand_landmarks = results.multi_hand_landmarks[i]
                        label = hand_info["label"]
                        gesture = hand_info["gesture"]
                        count = hand_info["finger_count"]

                        # Find top-left bounding position of the hand using hand_landmarks.landmark
                        min_x = min(lm.x for lm in hand_landmarks.landmark)
                        min_y = min(lm.y for lm in hand_landmarks.landmark)

                        text_x = max(10, int(min_x * w))
                        text_y = max(70, int(min_y * h) - 15)

                        label_text = f"{label}: {gesture} ({count})" if label else f"{gesture} ({count})"

                        # Draw background outline for contrast
                        cv2.putText(
                            display_frame, label_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA
                        )
                        cv2.putText(
                            display_frame, label_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA
                        )

                else:
                    self.last_gesture = None
                    self.last_app = None
                    self.last_executed_key = None

                self.show(display_frame)

                # Wait 1ms for key press; exit if 'q' or 'Q' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    logging.info("'q' pressed. Exiting camera stream...")
                    break
        finally:
            self.stop()


if __name__ == "__main__":
    cam = Camera()
    cam.run()
