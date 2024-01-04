from glob import glob
import os
import json
from math import atan2, asin, degrees

gesture_data = {}

def readJson(path):
    json_files = glob(path)

    for json_file in json_files:
        # Extract the gesture name from the file name
        gesture_name = os.path.basename(json_file).split('-')[0]
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        frames_data = []
        
        # Loop through each frame's data
        for frame in data:
            frame_data = frame.get('FrameData', [{}])[0]  
            hand_orientation = frame_data.get('HandOrientation', None)
            hand_position = frame_data.get('HandPosition', None)
            hand_rotation = frame_data.get('HandRotation', None)
            finger_positions = frame_data.get('FingerPositions', None)
            combined_data = [hand_orientation, hand_position, hand_rotation, finger_positions]
            
            frames_data.append(combined_data)
        
        gesture_data[gesture_name] = frames_data
        
    return gesture_data