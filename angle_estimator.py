import cv2
import mediapipe as mp
import numpy as np
import time
import json
import sys

def angle_calc(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def find_distance(a, b, h):
    a, b, h = np.array(a), np.array(b), np.array(h)
    return np.arctan2(a[1] - b[1], a[0] - b[0]) - np.arctan2(a[1] - h[1], a[0] - h[0])

def is_stationary(prev_landmarks, curr_landmarks, threshold=0.02):
    if prev_landmarks is None or curr_landmarks is None:
        return False
    
    total_diff = 0
    for i in range(len(prev_landmarks)):
        p_x, p_y = prev_landmarks[i]
        c_x, c_y = curr_landmarks[i]
        diff = np.sqrt((p_x - c_x)**2 + (p_y - c_y)**2)
        total_diff += diff
    
    avg_diff = total_diff / len(prev_landmarks)
    return avg_diff < threshold

def check_angles(current_angles, target_angles):
    feedback = []
    for joint, angle in current_angles.items():
        if joint in target_angles:
            target = target_angles[joint]
            if angle < target['min']:
                feedback.append(f"{joint} angle too small")
            elif angle > target['max']:
                feedback.append(f"{joint} angle too large")
    return feedback

def estimate_distance(landmarks):
    # Simple distance estimation based on shoulder width
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_width = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + 
                           (left_shoulder.y - right_shoulder.y)**2)
    # Approximate distance in meters (this is a rough estimation)
    return 0.5 / shoulder_width

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

# Get exercise data from command line arguments
exercise_data = json.loads(sys.argv[1])
target_angles = exercise_data['targetAngles']
distance_threshold = exercise_data['distanceThreshold']

cap = cv2.VideoCapture(0)

# Variables for stationary detection
prev_landmarks = None
stationary_start_time = None
stationary_duration = 3.0  # seconds
detailed_view = False
last_angles = {}

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            
            # Extract key points for all relevant joints
            # Left side
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Right side
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate angles
            angles = {
                "Left Elbow": round(angle_calc(left_shoulder, left_elbow, left_wrist), 2),
                "Right Elbow": round(angle_calc(right_shoulder, right_elbow, right_wrist), 2),
                "Left Shoulder": round(angle_calc(left_hip, left_shoulder, left_elbow), 2),
                "Right Shoulder": round(angle_calc(right_hip, right_shoulder, right_elbow), 2),
                "Left Hip": round(angle_calc(left_shoulder, left_hip, left_knee), 2),
                "Right Hip": round(angle_calc(right_shoulder, right_hip, right_knee), 2),
                "Left Knee": round(angle_calc(left_hip, left_knee, left_ankle), 2),
                "Right Knee": round(angle_calc(right_hip, right_knee, right_ankle), 2),
                "Trunk": round(angle_calc(
                    [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2],
                    [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2],
                    [(left_knee[0] + right_knee[0])/2, (left_knee[1] + right_knee[1])/2]), 2)
            }
            
            # Estimate distance
            distance = estimate_distance(landmarks)
            
            # Check angles and distance
            angle_feedback = check_angles(angles, target_angles)
            distance_feedback = []
            if distance < distance_threshold['min']:
                distance_feedback.append("Too close to camera")
            elif distance > distance_threshold['max']:
                distance_feedback.append("Too far from camera")
            
            # Prepare data to send to frontend
            data = {
                'angles': angles,
                'distance': distance,
                'feedback': angle_feedback + distance_feedback,
                'isCorrectPose': len(angle_feedback) == 0 and len(distance_feedback) == 0
            }
            
            # Send data to frontend
            print(json.dumps(data))
            
            # Display feedback on video
            y_offset = 30
            for feedback in data['feedback']:
                cv2.putText(image, feedback, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 30
            
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            cv2.imshow("Pose Estimation", image)
            
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
                
        except Exception as e:
            print(json.dumps({'error': str(e)}))
            continue

cap.release()
cv2.destroyAllWindows()
