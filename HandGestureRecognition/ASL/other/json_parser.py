import pandas as pd
import numpy as np
import feature_extraction

# Function to convert quaternion to Euler angles (roll, pitch, yaw)
def quaternion_to_euler(quaternion):
    w, x, y, z = quaternion
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(2 * (w * y - z * x))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    # Convert angles from radians to degrees
    roll, pitch, yaw = np.degrees([roll, pitch, yaw])
    return [roll, pitch, yaw]

# Initialize an empty dictionary to hold the ASL gesture data using Pandas
asl_gesture_data_pd = {}

# Define the list of file names
file_names = ['0-Annotated.json', '1-Annotated.json', '2-Annotated.json', '3-Annotated.json', '4-Annotated.json']

# Loop through all the uploaded JSON files
for file_name in file_names:
    file_path = f'/Users/mohddanishkhan/Downloads/VolunterWork/json/{file_name}'
    # Read JSON file into a DataFrame
    df = pd.read_json(file_path)
    
    # Extract and transform the relevant columns
    df['Gesture'] = df['Gesture'].apply(lambda x: x if pd.notna(x) else "Unknown")
    df['FrameData'] = df['FrameData'].apply(lambda x: x[0] if pd.notna(x) and len(x) > 0 else {})
    df['HandPosition'] = df['FrameData'].apply(lambda x: x.get("HandPosition", []))
    df['HandRotation'] = df['FrameData'].apply(lambda x: x.get("HandRotation", []))
    df['HandOrientation'] = df['FrameData'].apply(lambda x: x.get("HandOrientation", ""))
    df['FingerPositions'] = df['FrameData'].apply(lambda x: x.get("FingerPositions", []))
    
    # Drop the original FrameData column
    df.drop(columns=['FrameData', 'ScreenshotPath'], inplace=True)
    
    # Append the DataFrame to the dictionary
    for gesture in df['Gesture'].unique():
        if gesture not in asl_gesture_data_pd:
            asl_gesture_data_pd[gesture] = pd.DataFrame()
        asl_gesture_data_pd[gesture] = pd.concat([asl_gesture_data_pd[gesture], df[df['Gesture'] == gesture]], ignore_index=True)
        
# Replace the 'HandRotation' column with Euler angles
for gesture, df in asl_gesture_data_pd.items():
    df['HandRotation'] = df['HandRotation'].apply(quaternion_to_euler)


# Initialize an empty dictionary to hold the structured ASL gesture data
structured_asl_data = {}

# Loop through each gesture label and its corresponding DataFrame
for gesture, df in asl_gesture_data_pd.items():
    # Convert the DataFrame to a list of dictionaries
    list_of_dicts = df.to_dict('records')
    
    # Store the list of dictionaries in the new dictionary, using the gesture label as the key
    structured_asl_data[gesture] = list_of_dicts

# Extract features
feature_vectors_all_gestures = feature_extraction.extract_features(structured_asl_data)

print(feature_vectors_all_gestures["One"][6])