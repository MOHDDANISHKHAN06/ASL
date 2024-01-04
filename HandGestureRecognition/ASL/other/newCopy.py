from glob import glob
import os
import json
import pandas as pd
from math import atan2, asin, degrees, sqrt, acos
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Function to pad sequences to the same length
def pad_sequences(sequences):
    # Find the length of the longest sequence
    max_len = max(len(seq) for seq in sequences)
    
    # Pad each sequence with zeros to match the length of the longest sequence
    padded_sequences = np.zeros((len(sequences), max_len))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    
    return padded_sequences

# Function to calculate angle between two vectors with safety checks, considering hand as origin
def angle_between_fingers(finger1, finger2, hand_position):
    vector1 = [b - a for a, b in zip(hand_position, finger1)]
    vector2 = [b - a for a, b in zip(hand_position, finger2)]
    
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = sqrt(sum(a * a for a in vector1))
    magnitude2 = sqrt(sum(b * b for b in vector2))
    
    # Safety check to ensure the value is within the domain for acos
    value = dot_product / (magnitude1 * magnitude2)
    value = max(min(value, 1.0), -1.0)
    
    angle = acos(value)
    return degrees(angle)

# Function to flatten the features into a single list
def flatten_features(frame):
    flat_frame = []
    for feature in frame:
        if isinstance(feature, list):
            flat_frame.extend(feature)
        else:
            flat_frame.append(feature)
    return flat_frame

# Function to calculate Euclidean distance between two 3D points
def euclidean_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


# Function to convert quaternion to Euler angles
def quaternion_to_euler(w, x, y, z):
    roll = atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = asin(2 * (w * y - z * x))
    yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    # Convert to degrees
    roll = degrees(roll)
    pitch = degrees(pitch)
    yaw = degrees(yaw)
    
    return [roll, pitch, yaw]

# Initialize an empty dictionary to store the extracted data
gesture_data = {}

# List all the uploaded JSON files
json_files = glob('/Users/mohddanishkhan/Downloads/VolunterWork/json/*-Annotated.json')

# Loop through each file and extract the relevant data
for json_file in json_files:
    # Extract the gesture name from the file name
    gesture_name = os.path.basename(json_file).split('-')[0]
    
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Initialize a list to store the frame data for this gesture
    frames_data = []
    
    # Loop through each frame's data
    for frame in data:
        frame_data = frame.get('FrameData', [{}])[0]  # Taking the first item as it seems to contain the relevant data
        hand_orientation = frame_data.get('HandOrientation', None)
        hand_position = frame_data.get('HandPosition', None)
        hand_rotation = frame_data.get('HandRotation', None)
        finger_positions = frame_data.get('FingerPositions', None)
        
        # Combine the extracted data into a single list for this frame
        combined_data = [hand_orientation, hand_position, hand_rotation, finger_positions]
        
        # Add this frame's combined data to the list of frames for this gesture
        frames_data.append(combined_data)
    
    # Add this gesture's frames data to the main dictionary
    gesture_data[gesture_name] = frames_data

# Show a sample from the extracted data to verify
# print({key: len(value) for key, value in gesture_data.items()}, gesture_data['0'][:1])  # Show the number of frames for each gesture and first frame of '0'

# Convert quaternions to Euler angles for each frame in each gesture
for gesture, frames in gesture_data.items():
    for i, frame in enumerate(frames):
        hand_orientation, hand_position, hand_rotation, finger_positions = frame
        w, x, y, z = hand_rotation
        euler_angles = quaternion_to_euler(w, x, y, z)
        frames[i][2] = euler_angles  # Replace quaternions with Euler angles

# Initialize a dictionary to store the corrected features
feature_data = {}

# Loop through each gesture and frame to extract corrected features for angles between adjacent fingers
for gesture, frames in gesture_data.items():
    feature_frames = []
    for frame in frames:
        hand_orientation, hand_position, hand_rotation, finger_positions = frame
        
        # Existing features
        #feature1 : Hand orientation
        feature1 = hand_orientation  # 
        #feature2 : Hand rotation (Euler angles)
        feature2 = hand_rotation  # 
        #feature3 : Distances from hand to fingertips
        feature3 = [euclidean_distance(hand_position, fingertip) for fingertip in finger_positions]  
        #Feature 4: Distance between each fingertip and all other fingertips
        feature4 = []
        for i in range(len(finger_positions)):
            for j in range(i+1, len(finger_positions)):
                feature4.append(euclidean_distance(finger_positions[i], finger_positions[j]))        
        
        #Feature 5: Angles between adjacent fingers considering hand as origin
        feature5 = []
        for i in range(len(finger_positions) - 1):
            feature5.append(angle_between_fingers(finger_positions[i], finger_positions[i + 1], hand_position))
        
        # Combine all features for this frame
        features = [feature1, feature2, feature3, feature4, feature5]
        feature_frames.append(features)
    
    # Store the corrected features for this gesture
    feature_data[gesture] = feature_frames

# Show a sample from the extracted features to verify
# print(feature_data['0'][:1][0][4])


# Initialize a dictionary to store the feature combinations
feature_combinations = {}

# Define the combinations
combinations = {
    'Combination 1': [0, 1, 2],  # Features 1, 2, 3
    'Combination 2': [2, 3, 4],  # Features 3, 4, 5
    'Combination 3': [2, 1, 3],  # Features 3, 2, 4
    'Combination 4': [0, 1, 2, 3, 4]  # All Features
}

# Create the feature combinations for each gesture and each frame
for gesture, frames in feature_data.items():
    feature_combinations[gesture] = {}
    for comb_name, comb_indices in combinations.items():
        feature_combinations[gesture][comb_name] = []
        for frame in frames:
            combined_features = [frame[i] for i in comb_indices]
            feature_combinations[gesture][comb_name].append(combined_features)

# Show a sample from the created feature combinations to verify
# print({key: feature_combinations['1'][key][:1] for key in feature_combinations['1'].keys()})

# Initialize lists to store the separated and normalized features and labels
separated_data_non_normalized = []
separated_data_normalized = []
labels = []

# Separate and normalize the features
for gesture, combinations in feature_combinations.items():
    for combination_name, frames in combinations.items():
        for frame in frames:
            # Separate 'Hand Orientation' from numerical features
            hand_orientation = frame[0]
            numerical_features = flatten_features(frame[1:])
            
            # Normalize the numerical features
            numerical_features_normalized = (numerical_features - np.mean(numerical_features)) / np.std(numerical_features) if np.std(numerical_features) != 0 else np.zeros_like(numerical_features)
            
            # Re-combine the 'Hand Orientation' with the non-normalized and normalized numerical features
            separated_data_non_normalized.append([hand_orientation] + numerical_features)
            separated_data_normalized.append([hand_orientation] + numerical_features_normalized.tolist())
            
            # Store the label
            labels.append(gesture)

# Convert to NumPy arrays for easier handling
X_non_normalized = np.array(separated_data_non_normalized, dtype=object)
X_normalized = np.array(separated_data_normalized, dtype=object)
y = np.array(labels)

# Split the data into training and test sets
X_train_non_normalized, X_test_non_normalized, y_train_non_normalized, y_test_non_normalized = train_test_split(X_non_normalized, y, test_size=0.2, random_state=42)
X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Check the shapes of the data sets
X_train_non_normalized.shape, X_test_non_normalized.shape, X_train_normalized.shape, X_test_normalized.shape

# Pad the non-normalized and normalized numerical features
X_train_non_normalized_numerical_padded = pad_sequences([x[1:] for x in X_train_non_normalized])
X_test_non_normalized_numerical_padded = pad_sequences([x[1:] for x in X_test_non_normalized])
X_train_normalized_numerical_padded = pad_sequences([x[1:] for x in X_train_normalized])
X_test_normalized_numerical_padded = pad_sequences([x[1:] for x in X_test_normalized])

# Check the shapes of the padded data sets
# print(X_train_non_normalized_numerical_padded.shaspe, X_test_non_normalized_numerical_padded.shape, X_train_normalized_numerical_padded.shape, X_test_normalized_numerical_padded.shape)

# Redefine the models
models = {
    'k-NN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Initialize a dictionary to store the performance metrics for each model and each feature set (non-normalized and normalized)
performance_metrics = {}

# Train and evaluate each model on both non-normalized and normalized data
for model_name, model in models.items():
    # Metrics for non-normalized data
    model.fit(X_train_non_normalized_numerical_padded, y_train_non_normalized)
    y_pred_non_normalized = model.predict(X_test_non_normalized_numerical_padded)
    accuracy_non_normalized = accuracy_score(y_test_non_normalized, y_pred_non_normalized)
    precision_non_normalized = precision_score(y_test_non_normalized, y_pred_non_normalized, average='weighted')
    recall_non_normalized = recall_score(y_test_non_normalized, y_pred_non_normalized, average='weighted')
    f1_non_normalized = f1_score(y_test_non_normalized, y_pred_non_normalized, average='weighted')
    
    # Metrics for normalized data
    model.fit(X_train_normalized_numerical_padded, y_train_normalized)
    y_pred_normalized = model.predict(X_test_normalized_numerical_padded)
    accuracy_normalized = accuracy_score(y_test_normalized, y_pred_normalized)
    precision_normalized = precision_score(y_test_normalized, y_pred_normalized, average='weighted')
    recall_normalized = recall_score(y_test_normalized, y_pred_normalized, average='weighted')
    f1_normalized = f1_score(y_test_normalized, y_pred_normalized, average='weighted')
    
    # Store the metrics
    performance_metrics[model_name] = {
        'Non-Normalized': {
            'Accuracy': accuracy_non_normalized,
            'Precision': precision_non_normalized,
            'Recall': recall_non_normalized,
            'F1 Score': f1_non_normalized
        },
        'Normalized': {
            'Accuracy': accuracy_normalized,
            'Precision': precision_normalized,
            'Recall': recall_normalized,
            'F1 Score': f1_normalized
        }
    }

# Convert the performance metrics dictionary to a Pandas DataFrame for better visualization
performance_df = pd.DataFrame.from_dict({(i, j): performance_metrics[i][j] 
                                         for i in performance_metrics.keys() 
                                         for j in performance_metrics[i].keys()}, 
                                       orient='index')

# Convert the metrics to percentages for easier interpretation
performance_df *= 100
performance_df = performance_df.round(2)

# Show the DataFrame
print(performance_df)
