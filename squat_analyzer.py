import cv2
import mediapipe as mp
import numpy as np
import time

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

def get_feedback(angle, hip, knee, ankle):
    """Returns specific feedback based on angles and positions."""
    hip_x, knee_x = hip[0], knee[0]
    hip_y, knee_y = hip[1], knee[1]
    knee_angle = angle
    hip_knee_x_dist = knee_x - hip_x

    if knee_angle > 140:
        return "Hips too high – Go lower"
    elif 120 < knee_angle <= 140:
        return "Shallow squat – Bend knees more"
    elif knee_angle < 120:
        if hip_knee_x_dist > 0.15:
            return "Hips too far back"
        elif hip_knee_x_dist < -0.05:
            return "Hips tucked in – Lean slightly back"
        else:
            return "Good Squat"
    else:
        return "Adjust form"

def draw_feedback(frame, rep_count, stage, feedback, inactivity_time):
    cv2.putText(frame, f'Reps: {rep_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f'Stage: {stage}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    
    color = (0, 255, 0) if feedback == "Good Squat" else (0, 0, 255)
    cv2.putText(frame, f'Feedback: {feedback}', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    if inactivity_time > 5:
        cv2.putText(frame, f'Inactive for {int(inactivity_time)}s', (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# -----------------------------
# Main Squat Trainer
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Use webcam by default, or specify a video file path if available
use_webcam = False  # Set to False and specify video_path below if you have a video file
video_path = "squat_3.mp4"  # Optional: path to video file

if use_webcam:
    cap = cv2.VideoCapture(0)  # Use webcam
    print("Using webcam for pose analysis")
    print("Press 'q' to quit the analysis")
else:
    cap = cv2.VideoCapture(video_path)  # Use video file
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        print("Switching to webcam...")
        cap = cv2.VideoCapture(0)
        use_webcam = True
    else:
        print(f"Successfully opened video: {video_path}")
        print("Press 'q' to quit the video analysis")

# Create a named window with proper flags
cv2.namedWindow('AI Squat Trainer', cv2.WINDOW_NORMAL)

rep_count = 0
stage = None
feedback = "Fix Form"
last_rep_time = time.time()
error_count = 0  # Track consecutive errors

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if pose landmarks were detected
        if results.pose_landmarks is None:
            error_count += 1
            if error_count <= 3:  # Only print first few errors
                print(f"No pose detected in frame (error {error_count})")
            elif error_count == 4:
                print("... (suppressing further pose detection errors)")
            continue
        else:
            error_count = 0  # Reset error count when pose is detected

        try:
            landmarks = results.pose_landmarks.landmark
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            angle = calculate_angle(hip, knee, ankle)
            current_time = time.time()
            inactivity_time = current_time - last_rep_time

            feedback = get_feedback(angle, hip, knee, ankle)

            if angle > 160:
                stage = "Up"
            if angle < 100 and stage == "Up":
                stage = "Down"
                rep_count += 1
                last_rep_time = time.time()

            draw_feedback(image, rep_count, stage, feedback, inactivity_time)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except Exception as e:
            print(f"Error processing landmarks: {e}")

        # Resize frame to fit window properly
        if not use_webcam:
            # Get current frame dimensions
            height, width = image.shape[:2]
            
            # Calculate window size (max 80% of screen size)
            screen_width = 1920  # Approximate screen width
            screen_height = 1080  # Approximate screen height
            
            max_width = int(screen_width * 0.8)
            max_height = int(screen_height * 0.8)
            
            # Calculate scaling factor
            scale_x = max_width / width
            scale_y = max_height / height
            scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize the frame
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imshow('AI Squat Trainer', image)

        # Resize window to fit video properly
        if not use_webcam:
            # Get video dimensions
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate window size (max 80% of screen size)
            screen_width = 1920  # Approximate screen width
            screen_height = 1080  # Approximate screen height
            
            max_width = int(screen_width * 0.8)
            max_height = int(screen_height * 0.8)
            
            # Calculate scaling factor
            scale_x = max_width / video_width
            scale_y = max_height / video_height
            scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down
            
            # Calculate new dimensions
            new_width = int(video_width * scale)
            new_height = int(video_height * scale)
            
            # Resize the window
            cv2.resizeWindow('AI Squat Trainer', new_width, new_height)

        # Add a small delay to make video playback more viewable
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Analysis complete. Total reps detected: {rep_count}") 