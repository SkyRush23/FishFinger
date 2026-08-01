import logging
from google import genai
from google.genai import types

try:
    from config import GEMINI_API_KEY, GEMINI_MODEL
except ImportError:
    GEMINI_API_KEY = "ENTER YOUR API CODE HERE"
    GEMINI_MODEL = "ENTER YOUR MODEL HERE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Strict system prompt: instructs Gemini to return only a command identifier
SYSTEM_PROMPT = """\
You are a gesture-to-command translator for a hands-free OS controller called GestureOS.

You will receive exactly two pieces of information:
- The currently active application (e.g. Chrome, File Explorer, VS Code, PowerPoint, Desktop)
- A detected hand gesture (e.g. Open Palm, Closed Fist, Thumb Up, Both Hands Open)

Your job is to return EXACTLY ONE command identifier string.

Rules:
- Return ONLY the command identifier. Nothing else.
- No explanations. No punctuation. No markdown. No reasoning.
- Use UPPERCASE_WITH_UNDERSCORES format.
- If the gesture has no meaningful command for that app, return: NO_ACTION
-Do not explain.
-Do not use markdown.
-Do not add punctuation.
-Do not output anything except the identifier.

Command examples (not exhaustive):
  Chrome + Both Hands Open        -> OPEN_NEW_TAB
  File Explorer + Both Hands Open -> NEW_FOLDER
  VS Code + Both Hands Open       -> RUN_CODE
  Desktop + Open Palm             -> PLAY_PAUSE_MEDIA
  Chrome + Open Palm              -> PLAY_PAUSE_MEDIA
  Chrome + Thumb Up               -> SCROLL_UP
  Chrome + Thumb Down             -> SCROLL_DOWN
  Desktop + Thumb Up              -> VOLUME_UP
  Desktop + Thumb Down            -> VOLUME_DOWN
  VS Code + Index Up              -> SCROLL_UP
  VS Code + Index Down            -> SCROLL_DOWN
  Chrome + Closed Fist            -> TOGGLE_DESKTOP
  Desktop + Closed Fist           -> TOGGLE_DESKTOP
"""


class GeminiBrain:
    """
    GeminiBrain module for GestureOS.
    Maps detected gestures + active applications to command identifiers via the Gemini API.
    """

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the Gemini client.

        Args:
            api_key (str, optional): Gemini API key. Defaults to GEMINI_API_KEY from config.py.
            model (str, optional): Gemini model name. Defaults to GEMINI_MODEL from config.py.
        """
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or GEMINI_MODEL

        if not self.api_key or self.api_key == "your-gemini-api-key-here":
            logging.warning("GeminiBrain: No valid API key found in config.py. get_command() will return NO_ACTION.")

        self.client = genai.Client(api_key=self.api_key)
        self.fallback_commands = {
            ("Chrome", "Both Hands Open"): "OPEN_NEW_TAB",
            ("Google Chrome", "Both Hands Open"): "OPEN_NEW_TAB",

            ("File Explorer", "Both Hands Open"): "NEW_FOLDER",

            ("VS Code", "Both Hands Open"): "RUN_CODE",
            ("Visual Studio Code", "Both Hands Open"): "RUN_CODE",

            ("Desktop", "Open Palm"): "PLAY_PAUSE_MEDIA",
            ("Chrome", "Open Palm"): "PLAY_PAUSE_MEDIA",
            ("Google Chrome", "Open Palm"): "PLAY_PAUSE_MEDIA",

            ("Chrome", "Thumb Up"): "SCROLL_UP",
            ("Google Chrome", "Thumb Up"): "SCROLL_UP",

            ("Chrome", "Thumb Down"): "SCROLL_DOWN",
            ("Google Chrome", "Thumb Down"): "SCROLL_DOWN",

            ("Desktop", "Thumb Up"): "VOLUME_UP",
            ("Desktop", "Thumb Down"): "VOLUME_DOWN",

            ("VS Code", "Index Up"): "SCROLL_UP",
            ("Visual Studio Code", "Index Up"): "SCROLL_UP",

            ("VS Code", "Index Down"): "SCROLL_DOWN",
            ("Visual Studio Code", "Index Down"): "SCROLL_DOWN",

            ("Chrome", "Closed Fist"): "TOGGLE_DESKTOP",
            ("Google Chrome", "Closed Fist"): "TOGGLE_DESKTOP",
            ("Desktop", "Closed Fist"): "TOGGLE_DESKTOP",
        }
        logging.info(f"GeminiBrain initialized (model={self.model}).")

    def get_command(self, active_app: str, gesture: str) -> str:
        """
        Ask Gemini to map the given application and gesture to a command identifier.

        Args:
            active_app (str): The currently active application (e.g. "Chrome", "VS Code").
            gesture (str): The detected hand gesture (e.g. "Open Palm", "Thumb Up").

        Returns:
            str: A command identifier string in UPPERCASE_WITH_UNDERSCORES format.
                 Returns "NO_ACTION" on API failure or empty response.
        """
        if not self.api_key or self.api_key == "your-gemini-api-key-here":
            logging.warning("GeminiBrain: No API key. Returning NO_ACTION.")
            return "NO_ACTION"

        user_message = f"Active App: {active_app}\nGesture: {gesture}"

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=200,
                    temperature=0.0,
                    # Disable internal reasoning so output tokens are not consumed
                    # by the thinking step (required for thinking models like gemini-3.5-flash)
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
            )

            raw = response.text.strip() if response.text else ""

            # Sanitize: strip whitespace, newlines, backticks, and ensure uppercase
            command = raw.strip().replace("\n", "").replace("`", "").upper()

            if not command:
                logging.warning(f"GeminiBrain: Empty response for ({active_app}, {gesture}).")
                return "NO_ACTION"

            logging.info(f"GeminiBrain: ({active_app}, {gesture}) -> {command}")
            return command

        except Exception as e:
            logging.error(f"GeminiBrain: API call failed: {e}")

            fallback = self.fallback_commands.get(
                (active_app, gesture),
                "NO_ACTION"
            )

            logging.info(
                f"GeminiBrain: Using fallback -> ({active_app}, {gesture}) = {fallback}"
            )

            return fallback


if __name__ == "__main__":
    brain = GeminiBrain()

    test_cases = [
        ("Chrome", "Both Hands Open"),
        ("File Explorer", "Both Hands Open"),
        ("VS Code", "Both Hands Open"),
        ("Desktop", "Open Palm"),
        ("Chrome", "Thumb Up"),
        ("Chrome", "Thumb Down"),
        ("VS Code", "Index Up"),
        ("PowerPoint", "Open Palm"),
        ("Chrome", "Closed Fist"),
        ("Chrome", "Open Palm"),

    ]

    print("=== GeminiBrain Command Tests ===")
    for app, gesture in test_cases:
        command = brain.get_command(app, gesture)
        print(f"  {app:15s} + {gesture:20s} -> {command}")
    print("================================")
