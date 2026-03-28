from gaze_processing import classify_gaze_movements

mock_data = [
    {'x': 100, 'y': 200, 'timestamp': 1000},
    {'x': 102, 'y': 201, 'timestamp': 1016}, # Slow move (Fixation)
    {'x': 500, 'y': 600, 'timestamp': 1032}, # Fast jump (Saccade)
    {'x': 501, 'y': 602, 'timestamp': 1048}  # Stopped again (Fixation)
]

result = classify_gaze_movements(mock_data, velocity_threshold=5.0)
for point in result:
    print(f"Point ({point['x']}, {point['y']}) -> {point['movement_type']}")