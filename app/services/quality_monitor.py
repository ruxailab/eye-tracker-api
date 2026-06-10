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

    def process_prediction(self, prediction, should_print=True):

        if not prediction or len(prediction) == 0:
            return {"status": "no_prediction"}
        
        face = prediction[0]
        annotations = face.get("annotations", {})
        right_iris = annotations.get("rightEyeIris", [])

        if not right_iris:
            return {"status": "no_prediction"}
        
        right_iris_center = right_iris[0]
        x_center = right_iris_center[0]
        y_center = right_iris_center[1]

        self.x_center_coords.append(x_center)
        self.y_center_coords.append(y_center)

        # Keep only the last N points
        if len(self.x_center_coords) > self.window_size:
            self.x_center_coords.pop(0)
            self.y_center_coords.pop(0)

        # RMS-S2S metric calculation
        precision_px = self.calculate_rms_s2s_precision(self.x_center_coords, self.y_center_coords)
        precision_deg = self.pixels_to_degrees(precision_px)

        status = "good" if precision_deg < self.precision_cutoff_degrees else "poor"
        points_collected = len(self.x_center_coords)

        if should_print:
            print(f"Coordinates: X={x_center:.2f}, Y={y_center:.2f} | Precision: {precision_deg:.2f}° (RMS-S2S) | Status: {status}")
            
        return {"status": status, "precision_degrees": float(precision_deg), "precision_pixels": float(precision_px), "points_collected": points_collected}

# Initialize with standard 0.5 degrees cutoff
quality_monitor_instance = QualityMonitor(window_size=50, precision_cutoff_degrees=0.5)
