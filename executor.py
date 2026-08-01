import keyboard
import pyautogui


class CommandExecutor:
    def __init__(self):
        self.commands = {
            "OPEN_NEW_TAB": lambda: keyboard.press_and_release("ctrl+t"),
            "NEW_FOLDER": lambda: keyboard.press_and_release("ctrl+shift+n"),
            "RUN_CODE": lambda: keyboard.press_and_release("ctrl+f5"),
            "PLAY_PAUSE_MEDIA": lambda: keyboard.press_and_release("play/pause media"),
            "VOLUME_UP": lambda: pyautogui.press("volumeup"),
            "VOLUME_DOWN": lambda: pyautogui.press("volumedown"),
            "TOGGLE_DESKTOP": lambda: keyboard.press_and_release("windows+d"),
            "BRIGHTNESS_UP": lambda: print("Brightness Up"),
            "BRIGHTNESS_DOWN": lambda: print("Brightness Down"),
            "NO_ACTION": lambda: None,
        }

    def execute(self, command: str):
        action = self.commands.get(command)

        if action:
            try:
                action()
            except Exception as e:
                print(f"Error executing {command}: {e}")
        else:
            print(f"Unknown command: {command}")
