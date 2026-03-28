import math

def classify_gaze_movements(gaze_data, velocity_threshold=0.5):
    """
    Classifies an array of gaze points into fixations or saccades.
    Expected format: [{'x': float, 'y': float, 'timestamp': float}, ...]
    """
    if not gaze_data or len(gaze_data) < 2:
        return gaze_data

    # Default the first point to a fixation
    gaze_data[0]['movement_type'] = 'fixation'

    for i in range(1, len(gaze_data)):
        prev = gaze_data[i-1]
        curr = gaze_data[i]

        # Calculate Euclidean distance
        distance = math.sqrt((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)
        
        # Calculate time delta
        time_delta = curr['timestamp'] - prev['timestamp']

        if time_delta > 0:
            velocity = distance / time_delta
            # Classify based on the threshold
            curr['movement_type'] = 'saccade' if velocity > velocity_threshold else 'fixation'
        else:
            curr['movement_type'] = 'fixation' 

    return gaze_data