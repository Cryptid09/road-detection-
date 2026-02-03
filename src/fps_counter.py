"""
FPS Counter Module
Measures and displays frames per second
"""

import time
from collections import deque
from typing import Optional


class FPSCounter:
    """
    FPS counter using moving average
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = None
        self.current_fps = 0.0
    
    def update(self) -> float:
        """
        Update FPS counter with current frame
        
        Returns:
            Current FPS (averaged over window)
        """
        current_time = time.time()
        
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
            
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        self.last_time = current_time
        return self.current_fps
    
    def get_fps(self) -> float:
        """
        Get current FPS without updating
        
        Returns:
            Current FPS
        """
        return self.current_fps
    
    def reset(self):
        """Reset FPS counter"""
        self.frame_times.clear()
        self.last_time = None
        self.current_fps = 0.0

