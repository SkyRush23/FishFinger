import logging
from camera import Camera

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    """
    GestureOS Entry Point.
    Initializes and launches the camera feed.
    """
    logging.info("Starting GestureOS...")
    cam = Camera()
    cam.run()


if __name__ == "__main__":
    main()
