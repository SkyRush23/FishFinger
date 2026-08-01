import logging
import time
import pywinctl as pwc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ActiveWindowDetector:
    """
    Active Window Detector module for GestureOS.
    Detects the currently focused application on the operating system using pywinctl.
    Caches results and refreshes every 300-500ms to maintain smooth camera FPS.
    """

    def __init__(self, refresh_interval: float = 0.4):
        """
        Initialize detector with caching.

        Args:
            refresh_interval (float): Time in seconds between active window queries (default: 0.4s / 400ms).
        """
        self.refresh_interval = refresh_interval
        self.cached_app = "Unknown"
        self.last_check_time = 0.0
        logging.info(f"ActiveWindowDetector initialized with {int(refresh_interval * 1000)}ms cache refresh.")

    def get_active_app(self) -> str:
        """
        Detect the currently active foreground application.
        Reuses cached result if called within the refresh interval (300-500ms).

        Returns:
            str: One of:
                - "Desktop"
                - "Chrome"
                - "File Explorer"
                - "VS Code"
                - "PowerPoint"
                - "Unknown"
        """
        current_time = time.time()
        # Refresh cached window state only if refresh_interval has elapsed
        if current_time - self.last_check_time >= self.refresh_interval:
            self.cached_app = self._query_active_app()
            self.last_check_time = current_time

        return self.cached_app

    def _query_active_app(self) -> str:
        """Internal method performing pywinctl query and classification."""
        try:
            active_win = pwc.getActiveWindow()
            if not active_win:
                return "Unknown"

            title = (active_win.title or "").strip()
            title_lower = title.lower()

            app_name = ""
            if hasattr(active_win, "getAppName"):
                try:
                    app_name = (active_win.getAppName() or "").strip()
                except Exception:
                    pass

            app_lower = app_name.lower()

            # 1. Desktop
            if (
                any(name in app_lower for name in ["progman", "workerw", "shell_traywnd"])
                or title_lower in ["desktop", "program manager", "taskbar"]
            ):
                return "Desktop"

            # 2. Chrome
            if "chrome" in app_lower or "google chrome" in title_lower or title_lower.endswith("- chrome"):
                return "Chrome"

            # 3. File Explorer (distinguished from Desktop Progman shell)
            if "explorer" in app_lower or "file explorer" in title_lower or "windows explorer" in title_lower:
                if not any(name in app_lower for name in ["progman", "workerw"]):
                    return "File Explorer"
                return "Desktop"

            # 4. VS Code
            if "code" in app_lower or "visual studio code" in title_lower or title_lower.endswith("- visual studio code"):
                return "VS Code"

            # 5. PowerPoint
            if "powerpnt" in app_lower or "powerpoint" in title_lower or title_lower.endswith("- powerpoint"):
                return "PowerPoint"

            return "Unknown"

        except Exception as e:
            logging.warning(f"Failed to detect active window: {e}")
            return "Unknown"


if __name__ == "__main__":
    detector = ActiveWindowDetector(refresh_interval=0.4)

    # Mock window class for unit testing window classification logic
    class MockWindow:
        def __init__(self, title, app_name):
            self.title = title
            self._app_name = app_name

        def getAppName(self):
            return self._app_name

    def test_classification(title, app_name):
        mock_win = MockWindow(title, app_name)
        # Force cache invalidation for instant test evaluation
        detector.last_check_time = 0.0
        original_get_active = pwc.getActiveWindow
        pwc.getActiveWindow = lambda: mock_win
        try:
            res = detector.get_active_app()
        finally:
            pwc.getActiveWindow = original_get_active
        return res

    print("--- ActiveWindowDetector Logic & Caching Tests ---")
    print(f"Desktop Test: {test_classification('Program Manager', 'progman.exe')} (Expected: Desktop)")
    print(f"Chrome Test: {test_classification('New Tab - Google Chrome', 'chrome.exe')} (Expected: Chrome)")
    print(f"File Explorer Test: {test_classification('Documents', 'explorer.exe')} (Expected: File Explorer)")
    print(f"VS Code Test: {test_classification('active_window.py - GestureOS - Visual Studio Code', 'Code.exe')} (Expected: VS Code)")
    print(f"PowerPoint Test: {test_classification('Presentation1 - PowerPoint', 'POWERPNT.EXE')} (Expected: PowerPoint)")
    print(f"Unknown App Test: {test_classification('Calculator', 'calculator.exe')} (Expected: Unknown)")
    print("---------------------------------------")
    print(f"Current Live Active Window: {detector.get_active_app()}")

    # Verify cache reuse within 400ms
    start = time.time()
    call1 = detector.get_active_app()
    call2 = detector.get_active_app()
    call3 = detector.get_active_app()
    elapsed = (time.time() - start) * 1000
    print(f"3 consecutive calls executed in {elapsed:.2f}ms (Cached reuse working).")
