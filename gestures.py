import logging
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GestureRecognizer:
    """
    Gesture Recognizer module for GestureOS.
    Analyzes MediaPipe Solutions NormalizedLandmarkList objects to count fingers and recognize key hand gestures.
    """

    # Landmark indices for finger tips and reference joints (PIP / MCP)
    FINGER_TIPS = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    FINGER_PIPS = [6, 10, 14, 18]  # Index, Middle, Ring, Pinky PIP joints

    WRIST = 0
    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2

    INDEX_TIP = 8
    INDEX_MCP = 5
    PINKY_MCP = 17

    def __init__(self):
        logging.info("GestureRecognizer initialized.")

    def count_fingers(self, hand_landmarks, label: str = None) -> int:
        """
        Count the number of extended fingers (0-5) on a single hand.

        Args:
            hand_landmarks: MediaPipe Solutions NormalizedLandmarkList object for a single hand.
            label (str, optional): Hand label ("Left" or "Right") if available.

        Returns:
            int: Number of extended fingers (0 to 5).
        """
        if not hand_landmarks or not hasattr(hand_landmarks, "landmark") or len(hand_landmarks.landmark) < 21:
            return 0

        finger_states = self.get_finger_states(hand_landmarks, label=label)
        return sum(finger_states)

    def get_finger_states(self, hand_landmarks, label: str = None) -> list:
        """
        Determine the extended/folded state for each of the 5 fingers.

        Args:
            hand_landmarks: MediaPipe Solutions NormalizedLandmarkList object.
            label (str, optional): Hand label ("Left" or "Right").

        Returns:
            list[bool]: [Thumb, Index, Middle, Ring, Pinky] state (True = extended, False = folded).
        """
        if not hand_landmarks or not hasattr(hand_landmarks, "landmark") or len(hand_landmarks.landmark) < 21:
            return [False, False, False, False, False]

        landmarks = hand_landmarks.landmark
        states = []

        # --- 1. Thumb State ---
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        pinky_mcp = landmarks[self.PINKY_MCP]

        dist_tip_pinky = math.hypot(thumb_tip.x - pinky_mcp.x, thumb_tip.y - pinky_mcp.y)
        dist_ip_pinky = math.hypot(thumb_ip.x - pinky_mcp.x, thumb_ip.y - pinky_mcp.y)

        # Distance ratio (1.15x) for robust thumb extension detection
        thumb_is_extended = dist_tip_pinky > (dist_ip_pinky * 1.15)

        # Secondary check using label if provided
        if label:
            if label == "Right":
                thumb_is_extended = thumb_is_extended or (thumb_tip.x < (thumb_ip.x - 0.03))
            elif label == "Left":
                thumb_is_extended = thumb_is_extended or (thumb_tip.x > (thumb_ip.x + 0.03))

        states.append(thumb_is_extended)

        # --- 2. Four Fingers State (Index, Middle, Ring, Pinky) ---
        for tip_idx, pip_idx in zip(self.FINGER_TIPS, self.FINGER_PIPS):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]

            # In image space, y increases downwards; tip.y < pip.y means finger is extended upwards
            is_extended = tip.y < pip.y
            states.append(is_extended)

        return states

    def recognize_hand_gesture(self, hand_landmarks, label: str = None) -> str:
        """
        Recognize specific one-hand gestures from MediaPipe Solutions NormalizedLandmarkList.

        Recognized gestures:
        - "Open Palm": All 5 fingers extended.
        - "Closed Fist": All 5 fingers folded or ambiguous thumb state.
        - "Thumb Up": Thumb extended pointing UP; all 4 fingers folded.
        - "Thumb Down": Thumb extended pointing DOWN; all 4 fingers folded.
        - "Index Up": Index finger pointing UP; middle/ring/pinky folded.
        - "Index Down": Index finger pointing DOWN; middle/ring/pinky folded.
        - "Unknown Gesture": Any other combination.

        Args:
            hand_landmarks: MediaPipe Solutions NormalizedLandmarkList object.
            label (str, optional): Hand label ("Left" or "Right").

        Returns:
            str: Gesture name string.
        """
        if not hand_landmarks or not hasattr(hand_landmarks, "landmark") or len(hand_landmarks.landmark) < 21:
            return "Unknown Gesture"

        finger_states = self.get_finger_states(hand_landmarks, label=label)
        thumb, index, middle, ring, pinky = finger_states

        # 1. Open Palm (All 5 fingers extended)
        if all(finger_states):
            return "Open Palm"

        four_fingers_folded = not (index or middle or ring or pinky)
        landmarks = hand_landmarks.landmark

        if four_fingers_folded:
            thumb_tip_y = landmarks[self.THUMB_TIP].y
            thumb_ip_y = landmarks[self.THUMB_IP].y
            thumb_mcp_y = landmarks[self.THUMB_MCP].y
            index_mcp_y = landmarks[self.INDEX_MCP].y

            THUMB_VERTICAL_THRESHOLD = 0.07

            is_thumb_pointing_up = (
                thumb_tip_y < (thumb_mcp_y - THUMB_VERTICAL_THRESHOLD)
                and thumb_tip_y < thumb_ip_y
            )

            is_thumb_pointing_down = (
                thumb_tip_y > (thumb_mcp_y + THUMB_VERTICAL_THRESHOLD)
                and thumb_tip_y > thumb_ip_y
            )

            # Strict Thumb Up
            if is_thumb_pointing_up and thumb:
                return "Thumb Up"

            # Strict Thumb Down
            if is_thumb_pointing_down:
                return "Thumb Down"

            index_tip_y = landmarks[self.INDEX_TIP].y
            index_pip_y = landmarks[self.FINGER_PIPS[0]].y

            if index_tip_y > index_pip_y and index_tip_y > (index_mcp_y + 0.1):
                return "Index Down"

            return "Closed Fist"

        # 3. Single Finger Up gesture (Index Up: Index extended upwards; middle/ring/pinky folded)
        other_three_folded = not (middle or ring or pinky)
        if other_three_folded and index:
            index_tip_y = landmarks[self.INDEX_TIP].y
            index_mcp_y = landmarks[self.INDEX_MCP].y

            if index_tip_y < index_mcp_y:
                return "Index Up"

        return "Unknown Gesture"

    def count_fingers_all(self, results) -> list:
        """
        Count extended fingers for all detected hands in a MediaPipe Solutions SolutionOutputs result.

        Args:
            results (SolutionOutputs): Result object from mp.solutions.hands.Process.

        Returns:
            list[int]: List of finger counts for each detected hand.
        """
        counts = []
        if not results or not getattr(results, "multi_hand_landmarks", None):
            return counts

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = None
            if (
                getattr(results, "multi_handedness", None)
                and i < len(results.multi_handedness)
                and results.multi_handedness[i].classification
            ):
                label = results.multi_handedness[i].classification[0].label

            count = self.count_fingers(hand_landmarks, label=label)
            counts.append(count)

        return counts

    def recognize_all_gestures(self, results) -> dict:
        """
        Analyze all detected hands in a frame and recognize one-hand and two-hand gestures.

        Args:
            results (SolutionOutputs): Output from mp.solutions.hands.process.

        Returns:
            dict: {
                "multi_hand_gesture": str or None,  # e.g., "Both Hands Open"
                "hand_gestures": [
                    {"hand_index": 0, "label": str, "gesture": str, "finger_count": int},
                    ...
                ]
            }
        """
        results_data = {
            "multi_hand_gesture": None,
            "hand_gestures": []
        }

        if not results or not getattr(results, "multi_hand_landmarks", None):
            return results_data

        detected_gestures = []

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = None
            if (
                getattr(results, "multi_handedness", None)
                and i < len(results.multi_handedness)
                and results.multi_handedness[i].classification
            ):
                label = results.multi_handedness[i].classification[0].label

            gesture = self.recognize_hand_gesture(hand_landmarks, label=label)
            count = self.count_fingers(hand_landmarks, label=label)

            results_data["hand_gestures"].append({
                "hand_index": i,
                "label": label,
                "gesture": gesture,
                "finger_count": count
            })
            detected_gestures.append(gesture)

        # Check for two-hand gesture: "Both Hands Open"
        if len(detected_gestures) == 2 and all(g == "Open Palm" for g in detected_gestures):
            results_data["multi_hand_gesture"] = "Both Hands Open"

        return results_data


if __name__ == "__main__":
    recognizer = GestureRecognizer()

    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class MockNormalizedLandmarkList:
        def __init__(self, landmark_list):
            self.landmark = landmark_list

    def create_mock_hand(finger_tips_and_pips, thumb_pos=(0.2, 0.3), thumb_ip=(0.3, 0.35), pinky_mcp=(0.7, 0.6)):
        landmarks = [MockLandmark(0.5, 0.5) for _ in range(21)]
        landmarks[4] = MockLandmark(*thumb_pos)   # Thumb tip
        landmarks[3] = MockLandmark(*thumb_ip)    # Thumb IP
        landmarks[2] = MockLandmark(0.3, 0.4)     # Thumb MCP
        landmarks[17] = MockLandmark(*pinky_mcp)  # Pinky MCP
        landmarks[5] = MockLandmark(0.4, 0.4)     # Index MCP

        for idx, (tip_y, pip_y) in zip([8, 12, 16, 20], finger_tips_and_pips):
            landmarks[idx] = MockLandmark(0.5, tip_y)
            landmarks[idx - 2] = MockLandmark(0.5, pip_y)

        return MockNormalizedLandmarkList(landmarks)

    class MockClassification:
        def __init__(self, label):
            self.label = label

    class MockHandedness:
        def __init__(self, label):
            self.classification = [MockClassification(label)]

    class MockSolutionOutputs:
        def __init__(self, multi_hand_landmarks, multi_handedness=None):
            self.multi_hand_landmarks = multi_hand_landmarks
            self.multi_handedness = multi_handedness or []

    # Test 1: Open Palm
    open_hand = create_mock_hand([(0.2, 0.4), (0.2, 0.4), (0.2, 0.4), (0.2, 0.4)])
    print(f"Open Palm Test: {recognizer.recognize_hand_gesture(open_hand)} (Expected: Open Palm)")

    # Test 2: Closed Fist
    fist_hand = create_mock_hand([(0.45, 0.4), (0.45, 0.4), (0.45, 0.4), (0.45, 0.4)], thumb_pos=(0.3, 0.4), thumb_ip=(0.3, 0.4))
    print(f"Closed Fist Test: {recognizer.recognize_hand_gesture(fist_hand)} (Expected: Closed Fist)")

    # Test 3: Strict Thumb Up
    thumb_up = create_mock_hand([(0.45, 0.4), (0.45, 0.4), (0.45, 0.4), (0.45, 0.4)], thumb_pos=(0.1, 0.1), thumb_ip=(0.2, 0.25))
    print(f"Thumb Up Test: {recognizer.recognize_hand_gesture(thumb_up)} (Expected: Thumb Up)")

    # Test 4: Strict Thumb Down
    thumb_down = create_mock_hand([(0.45, 0.4), (0.45, 0.4), (0.45, 0.4), (0.45, 0.4)], thumb_pos=(0.3, 0.8), thumb_ip=(0.3, 0.6))
    print(f"Thumb Down Test: {recognizer.recognize_hand_gesture(thumb_down)} (Expected: Thumb Down)")

    # Test 5: Ambiguous Thumb
    ambiguous_thumb = create_mock_hand([(0.45, 0.4), (0.45, 0.4), (0.45, 0.4), (0.45, 0.4)], thumb_pos=(0.3, 0.36), thumb_ip=(0.3, 0.38))
    print(f"Ambiguous Thumb Test: {recognizer.recognize_hand_gesture(ambiguous_thumb)} (Expected: Closed Fist)")

    # Test 6: Index Up
    index_up = create_mock_hand([(0.2, 0.4), (0.45, 0.4), (0.45, 0.4), (0.45, 0.4)], thumb_pos=(0.3, 0.4), thumb_ip=(0.3, 0.4))
    print(f"Index Up Test: {recognizer.recognize_hand_gesture(index_up)} (Expected: Index Up)")

    # Test 7: Index Down
    index_down = create_mock_hand([(0.8, 0.4), (0.45, 0.4), (0.45, 0.4), (0.45, 0.4)], thumb_pos=(0.3, 0.4), thumb_ip=(0.3, 0.4))
    print(f"Index Down Test: {recognizer.recognize_hand_gesture(index_down)} (Expected: Index Down)")

    # Test 8: Both Hands Open
    two_hands_open_result = MockSolutionOutputs([open_hand, open_hand], [MockHandedness("Right"), MockHandedness("Left")])
    gestures_res = recognizer.recognize_all_gestures(two_hands_open_result)
    print(f"Both Hands Open Test: {gestures_res['multi_hand_gesture']} (Expected: Both Hands Open)")
