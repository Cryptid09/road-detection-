import time

class FPSCounter:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

    def update(self):
        """Update the FPS counter."""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time

    def get_fps(self):
        """Get the current FPS."""
        return self.fps

    def reset(self):
        """Reset the FPS counter."""
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0