import cv2
import mediapipe as mp
import numpy as np
import time

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

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

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
            
            # Additional points
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            
            # Calculate angles for different joints
            angles = {
                # Arms
                "Left Elbow": round(angle_calc(left_shoulder, left_elbow, left_wrist), 2),
                "Right Elbow": round(angle_calc(right_shoulder, right_elbow, right_wrist), 2),
                
                # Shoulders
                "Left Shoulder": round(angle_calc(left_hip, left_shoulder, left_elbow), 2),
                "Right Shoulder": round(angle_calc(right_hip, right_shoulder, right_elbow), 2),
                
                # Hips
                "Left Hip": round(angle_calc(left_shoulder, left_hip, left_knee), 2),
                "Right Hip": round(angle_calc(right_shoulder, right_hip, right_knee), 2),
                
                # Knees
                "Left Knee": round(angle_calc(left_hip, left_knee, left_ankle), 2),
                "Right Knee": round(angle_calc(right_hip, right_knee, right_ankle), 2),
                
                # Upper Body (between shoulders)
                "Shoulder Line": round(angle_calc(left_shoulder, 
                                                [(left_shoulder[0] + right_shoulder[0])/2, 
                                                 (left_shoulder[1] + right_shoulder[1])/2], 
                                                right_shoulder), 2),
                
                # Trunk angle
                "Trunk": round(angle_calc(
                    [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2],
                    [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2],
                    [(left_knee[0] + right_knee[0])/2, (left_knee[1] + right_knee[1])/2]), 2)
            }
            
            # Store the current angles
            last_angles = angles
            
            # Create list of key points for stationary detection
            current_landmarks = [left_shoulder, left_elbow, left_wrist, right_shoulder, right_elbow, 
                                right_wrist, left_hip, right_hip, left_knee, right_knee, 
                                left_ankle, right_ankle, nose]
            
            # Check if the person is stationary
            if prev_landmarks is not None and is_stationary(prev_landmarks, current_landmarks):
                if stationary_start_time is None:
                    stationary_start_time = time.time()
                elif time.time() - stationary_start_time >= stationary_duration:
                    detailed_view = True
            else:
                stationary_start_time = None
                detailed_view = False
            
            # Update previous landmarks
            prev_landmarks = current_landmarks
            
            # Display basic angle information
            y_offset = 30
            if detailed_view:
                # Display detailed angles in a panel
                overlay = image.copy()
                cv2.rectangle(overlay, (10, 10), (300, 320), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
                
                cv2.putText(image, "STATIC POSE DETECTED", (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 30
                
                for name, angle in angles.items():
                    cv2.putText(image, f"{name}: {angle} deg", (20, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 25
            else:
                # Display minimal info when moving
                cv2.putText(image, f"L Elbow: {angles['Left Elbow']}", 
                           tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(image, f"R Elbow: {angles['Right Elbow']}", 
                           tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(image, f"Trunk: {angles['Trunk']}", 
                           tuple(np.multiply([(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2], 
                                           [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display stationary progress when holding still
            if stationary_start_time is not None and not detailed_view:
                elapsed = time.time() - stationary_start_time
                progress = min(elapsed / stationary_duration, 1.0)
                bar_width = int(200 * progress)
                cv2.rectangle(image, (220, 20), (420, 40), (0, 0, 0), -1)
                cv2.rectangle(image, (220, 20), (220 + bar_width, 40), (0, 255, 0), -1)
                cv2.putText(image, "Hold still...", (220, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        except Exception as e:
            pass

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        cv2.imshow("Pose Estimation", image)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
