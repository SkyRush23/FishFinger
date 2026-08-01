# pyrefly: ignore [missing-import]
import cv2
import logging
import mediapipe as mp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class HandTracker:
    """
    Modular Hand Tracker using legacy MediaPipe Solutions API (mp.solutions.hands).
    Detects hand landmarks and renders connections on OpenCV BGR camera frames.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe Solutions Hands detector.

        Args:
            static_image_mode (bool): Whether to treat input images as static or stream.
            max_num_hands (int): Maximum number of hands to detect (default: 2).
            min_detection_confidence (float): Minimum confidence for hand detection.
            min_tracking_confidence (float): Minimum confidence for hand tracking.
        """
        self.max_num_hands = max_num_hands

        # MediaPipe Solutions components
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize MediaPipe Hands solution
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logging.info(f"HandTracker (MediaPipe Solutions) initialized (max_num_hands={max_num_hands}).")

    def process_frame(self, frame):
        """
        Process a BGR OpenCV frame, detect hands using mp.solutions.hands, and draw landmarks.

        Args:
            frame (numpy.ndarray): BGR image frame from camera.

        Returns:
            tuple: (annotated_frame, results)
                - annotated_frame: Copy of frame with drawn hand landmarks & connections.
                - results: MediaPipe Solutions SolutionOutputs object.
        """
        if frame is None or self.hands is None:
            return None, None

        annotated_frame = frame.copy()

        # Convert OpenCV BGR to RGB for MediaPipe Solutions
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame to detect hand landmarks
        results = self.hands.process(rgb_frame)

        # Draw hand landmarks and connections if detected
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

        return annotated_frame, results

    def draw_landmarks(self, frame, results):
        """
        Draw landmarks on a copy of the frame using MediaPipe drawing utilities.

        Args:
            frame (numpy.ndarray): Input BGR OpenCV frame.
            results (SolutionOutputs): MediaPipe detection output.

        Returns:
            numpy.ndarray: Annotated BGR OpenCV frame.
        """
        annotated_frame = frame.copy()
        if not results or not getattr(results, "multi_hand_landmarks", None):
            return annotated_frame

        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        return annotated_frame

    def close(self):
        """Release MediaPipe Hands resources."""
        if hasattr(self, "hands") and self.hands:
            self.hands.close()
            self.hands = None
            logging.info("HandTracker (MediaPipe Solutions) closed.")


if __name__ == "__main__":
    from camera import Camera

    cam = Camera()
    tracker = HandTracker()

    if cam.start():
        logging.info("Starting HandTracker Solutions API test stream. Press 'q' or 'Q' to exit.")
        try:
            while True:
                ret, frame = cam.read()
                if not ret or frame is None:
                    break

                annotated_frame, _ = tracker.process_frame(frame)
                cam.show(annotated_frame, window_name="GestureOS - Legacy Solutions Test")

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    break
        finally:
            cam.stop()
            tracker.close()
