import numpy as np
import math

class QualityMonitor:
    def __init__(self, window_size=50, precision_cutoff_degrees=1.0, screen_dpi=110, user_distance_mm=600):
        self.window_size = window_size
        self.precision_cutoff_degrees = precision_cutoff_degrees
        self.screen_dpi = screen_dpi
        self.user_distance_mm = user_distance_mm
        self.x_center_coords = []
        self.y_center_coords = []

    def pixels_to_degrees(self, pixels, ppi=96.0, distance_cm=60.0):
        # pixels in one centimeter calculation
        pixels_per_cm = ppi / 2.54

        # Convert the pixel distance to physical size in centimeters
        size_cm = pixels / pixels_per_cm

        # Visual angle calculation
        angle_rad = 2 * np.arctan(size_cm / (2 * distance_cm))
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def calculate_rms_s2s_precision(self, x_coords, y_coords):
        if len(x_coords) < 2:
            return 0.0
        
        x = np.array(x_coords)
        y = np.array(y_coords)

        # Calculate dfferences between consecutive points
        dx = np.diff(x)
        dy = np.diff(y)

        # Calculate squared distances between consecutive samples
        distances = np.sqrt(dx**2 + dy**2)

        # Root Mean Square calculation
        rms_s2s_px = np.sqrt(np.mean(distances**2))
        return rms_s2s_px