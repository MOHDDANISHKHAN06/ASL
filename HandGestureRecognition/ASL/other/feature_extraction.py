import numpy as np

# Helper function to calculate Euclidean distance between two points in 3D space
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

# Group D: Distance between palm center and each fingertip
def calculate_group_D(palm_position, finger_positions):
    return [euclidean_distance(palm_position, finger_position) for finger_position in finger_positions]

# Group A: Angle between two adjacent fingertips
def calculate_group_A(finger_positions):
    angles = []
    for i in range(len(finger_positions) - 1):
        vector1 = np.array(finger_positions[i]) - np.array(finger_positions[i+1])
        vector2 = np.array(finger_positions[i+1]) - np.array(finger_positions[i])
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        angles.append(np.degrees(angle))
    return angles

# Group L: Distance between one fingertip and the consecutive fingertip
def calculate_group_L(finger_positions):
    distances = []
    for i in range(len(finger_positions)):
        for j in range(i+1, len(finger_positions)):
            distances.append(euclidean_distance(finger_positions[i], finger_positions[j]))
    return distances

# Main function to apply feature extraction to a given dataset
def extract_features(dataset):
    feature_vectors_all_gestures = {}
    
    for gesture, data in dataset.items():
        feature_vectors = []
        
        for entry in data:
            palm_position = entry['HandPosition']
            finger_positions = entry['FingerPositions']

            group_D = calculate_group_D(palm_position, finger_positions)
            group_A = calculate_group_A(finger_positions)
            group_L = calculate_group_L(finger_positions)
            
            feature_vector = {
                'DistPalm&Fingers': group_D,
                'AngleFingers': group_A,
                'DistBtwFingers': group_L
            }
            
            feature_vectors.append(feature_vector)
        
        feature_vectors_all_gestures[gesture] = feature_vectors
    
    return feature_vectors_all_gestures
