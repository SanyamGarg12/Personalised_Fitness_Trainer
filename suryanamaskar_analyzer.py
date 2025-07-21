import cv2
import mediapipe as mp
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os

class SuryanamaskarAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Load Suryanamaskar poses data
        with open('suryanamaskar_poses.json', 'r') as f:
            self.suryanamaskar_data = json.load(f)
        
        self.setup_gui()
    
    def angle_calc(self, a, b, c):
        """Calculate angle between three points in 3D"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        # Vectors BA and BC
        ba = a - b
        bc = c - b
        # Normalize vectors
        ba_norm = ba / np.linalg.norm(ba)
        bc_norm = bc / np.linalg.norm(bc)
        # Dot product and angle
        cosine_angle = np.dot(ba_norm, bc_norm)
        # Clamp value to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def estimate_distance(self, landmarks):
        """Estimate distance from camera based on shoulder width"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        shoulder_width = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + 
                               (left_shoulder.y - right_shoulder.y)**2)
        return 0.5 / shoulder_width
    
    def check_angles(self, current_angles, target_angles):
        """Check if current angles match target angles"""
        feedback = []
        for joint, angle in current_angles.items():
            if joint in target_angles:
                target = target_angles[joint]
                if angle < target['min']:
                    feedback.append(f"{joint} angle too small ({angle}° < {target['min']}°)")
                elif angle > target['max']:
                    feedback.append(f"{joint} angle too large ({angle}° > {target['max']}°)")
        return feedback
    
    def remove_background(self, image):
        """Remove background using MediaPipe Selfie Segmentation with improved mask post-processing"""
        import mediapipe as mp
        import cv2
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
            results = segmenter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            mask = results.segmentation_mask
            # Improved mask: higher threshold, morphological closing, largest contour
            mask_bin = (mask > 0.3).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask_closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
            # Keep only largest contour
            contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                mask_person = np.zeros_like(mask_closed)
                cv2.drawContours(mask_person, [largest], -1, 255, thickness=cv2.FILLED)
            else:
                mask_person = mask_closed
            # Apply mask
            condition = mask_person > 0
            bg_image = np.ones(image.shape, dtype=np.uint8) * 255  # white background
            output_image = np.where(condition[..., None], image, bg_image)
            return output_image, mask_person

    def crop_and_resize_person(self, image, mask, target_size=700):
        """Crop the image to the bounding box of the person and resize to target_size keeping aspect ratio."""
        # Find bounding box from mask
        mask_bin = (mask > 0.1).astype(np.uint8)
        coords = cv2.findNonZero(mask_bin)
        x, y, w, h = cv2.boundingRect(coords)
        cropped = image[y:y+h, x:x+w]
        # Resize
        scale = target_size / max(h, w)
        resized = cv2.resize(cropped, (int(w*scale), int(h*scale)))
        # Place on white background
        out_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
        y_offset = (target_size - resized.shape[0]) // 2
        x_offset = (target_size - resized.shape[1]) // 2
        out_img[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
        return out_img

    def analyze_pose(self, image_path, pose_name):
        """Analyze pose from image and compare with target angles"""
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}

        # Remove background (improved)
        processed_image, mask = self.remove_background(image)
        # Crop and resize (now with larger target size)
        processed_image = self.crop_and_resize_person(processed_image, mask, target_size=700)
        # Display preprocessed image in GUI
        self.display_preprocessed_image(processed_image)

        # Convert to RGB
        image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # Initialize pose detection
        with self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        ) as pose:
            results = pose.process(image_rgb)
            
            if not results.pose_landmarks:
                return {"error": "No pose detected in the image"}
            
            landmarks = results.pose_landmarks.landmark
            
            # Extract key points (now with z coordinate)
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].z]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].z]
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
            left_ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].z]
            right_ear = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].z]
            nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[self.mp_pose.PoseLandmark.NOSE.value].y,
                    landmarks[self.mp_pose.PoseLandmark.NOSE.value].z]

            # Define vertical vector for shoulder reference (upward from shoulder)
            vertical_up = [0, -1, 0]  # In normalized image coordinates, y decreases upward

            # Calculate angles
            angles = {
                # Elbow: angle between upper arm and forearm
                "Left Elbow": round(self.angle_calc(left_shoulder, left_elbow, left_wrist), 2),
                "Right Elbow": round(self.angle_calc(right_shoulder, right_elbow, right_wrist), 2),
                # Shoulder: angle between upper arm and vertical axis (upward from shoulder)
                "Left Shoulder": round(self.angle_calc(left_elbow, left_shoulder, [left_shoulder[0], left_shoulder[1]-1, left_shoulder[2]]), 2),
                "Right Shoulder": round(self.angle_calc(right_elbow, right_shoulder, [right_shoulder[0], right_shoulder[1]-1, right_shoulder[2]]), 2),
                # Hip: angle between torso and upper leg
                "Left Hip": round(self.angle_calc(left_shoulder, left_hip, left_knee), 2),
                "Right Hip": round(self.angle_calc(right_shoulder, right_hip, right_knee), 2),
                # Knee: angle between upper leg and lower leg
                "Left Knee": round(self.angle_calc(left_hip, left_knee, left_ankle), 2),
                "Right Knee": round(self.angle_calc(right_hip, right_knee, right_ankle), 2),
                # Neck: angle between torso (shoulder-hip midpoint) and head (nose)
                "Neck": round(self.angle_calc(
                    [(left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2, (left_shoulder[2]+right_shoulder[2])/2],
                    [(left_ear[0]+right_ear[0])/2, (left_ear[1]+right_ear[1])/2, (left_ear[2]+right_ear[2])/2],
                    nose), 2),
                # Ankle: angle between foot and lower leg (vertical shin)
                "Left Ankle": round(self.angle_calc(left_knee, left_ankle, [left_ankle[0], left_ankle[1]-1, left_ankle[2]]), 2),
                "Right Ankle": round(self.angle_calc(right_knee, right_ankle, [right_ankle[0], right_ankle[1]-1, right_ankle[2]]), 2),
                # Trunk: angle between shoulders-hips-knees midpoints (as before)
                "Trunk": round(self.angle_calc(
                    [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2, (left_shoulder[2] + right_shoulder[2])/2],
                    [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2, (left_hip[2] + right_hip[2])/2],
                    [(left_knee[0] + right_knee[0])/2, (left_knee[1] + right_knee[1])/2, (left_knee[2] + right_knee[2])/2]), 2)
            }
            
            # Get target angles for the selected pose
            if pose_name not in self.suryanamaskar_data['suryanamaskar_poses']:
                return {"error": f"Pose '{pose_name}' not found in database"}
            
            pose_data = self.suryanamaskar_data['suryanamaskar_poses'][pose_name]
            target_angles = pose_data['targetAngles']
            
            # Check angles
            feedback = self.check_angles(angles, target_angles)
            
            # Estimate distance
            distance = self.estimate_distance(landmarks)
            distance_feedback = []
            if distance < pose_data['distanceThreshold']['min']:
                distance_feedback.append("Too close to camera")
            elif distance > pose_data['distanceThreshold']['max']:
                distance_feedback.append("Too far from camera")
            
            # Create result
            result = {
                'pose_name': pose_data['name'],
                'description': pose_data['description'],
                'current_angles': angles,
                'target_angles': target_angles,
                'feedback': feedback + distance_feedback,
                'distance': distance,
                'is_correct_pose': len(feedback) == 0 and len(distance_feedback) == 0,
                'image_with_landmarks': self.draw_landmarks_on_image(image, results)
            }
            
            return result
    
    def draw_landmarks_on_image(self, image, results):
        """Draw pose landmarks on image"""
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(
            annotated_image, 
            results.pose_landmarks, 
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        return annotated_image
    
    def setup_gui(self):
        """Setup the GUI"""
        self.root = tk.Tk()
        self.root.title("Suryanamaskar Pose Analyzer")
        self.root.geometry("800x600")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Suryanamaskar Pose Analyzer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Pose selection
        ttk.Label(main_frame, text="Select Suryanamaskar Pose:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.pose_var = tk.StringVar()
        pose_names = list(self.suryanamaskar_data['suryanamaskar_poses'].keys())
        pose_display_names = [self.suryanamaskar_data['suryanamaskar_poses'][name]['name'] for name in pose_names]
        
        self.pose_combo = ttk.Combobox(main_frame, textvariable=self.pose_var, 
                                      values=pose_display_names, state="readonly", width=40)
        self.pose_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        self.pose_combo.bind('<<ComboboxSelected>>', self.on_pose_selected)
        
        # Upload button
        self.upload_btn = ttk.Button(main_frame, text="Upload Exercise Image", 
                                    command=self.upload_image)
        self.upload_btn.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Add two image display areas: one for preprocessed, one for pose result
        self.pre_image_label = ttk.Label(main_frame, text="Preprocessed Image will appear here")
        self.pre_image_label.grid(row=4, column=0, columnspan=2, pady=5)
        self.image_label = ttk.Label(main_frame, text="Pose Result Image will appear here")
        self.image_label.grid(row=5, column=0, columnspan=2, pady=5)

        # Pose info label (restored for on_pose_selected)
        self.pose_info_label = ttk.Label(main_frame, text="", font=("Segoe UI", 11, "italic"), padding=(0,0,0,10))
        self.pose_info_label.grid(row=5, column=0, columnspan=2, sticky=tk.W)

        # Results frame: angles and feedback side by side
        self.results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        self.results_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        self.results_frame.grid_rowconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(1, weight=1)

        # Angles (left column)
        self.results_canvas = tk.Canvas(self.results_frame, borderwidth=0, height=220)
        self.angles_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_canvas.yview)
        self.scrollable_results = ttk.Frame(self.results_canvas)
        self.scrollable_results.bind(
            "<Configure>", lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all")))
        self.results_canvas.create_window((0, 0), window=self.scrollable_results, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.angles_scrollbar.set)
        self.results_canvas.grid(row=0, column=0, sticky="nsew")
        self.angles_scrollbar.grid(row=0, column=2, sticky="ns")
        self.angles_frame = self.scrollable_results

        # Feedback (right column)
        self.feedback_canvas = tk.Canvas(self.results_frame, borderwidth=0, height=220, bg='#fff8f0', highlightthickness=0)
        self.feedback_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.feedback_canvas.yview)
        self.feedback_frame = ttk.Frame(self.feedback_canvas, style="Feedback.TFrame")
        self.feedback_frame.bind(
            "<Configure>", lambda e: self.feedback_canvas.configure(scrollregion=self.feedback_canvas.bbox("all")))
        self.feedback_canvas.create_window((0, 0), window=self.feedback_frame, anchor="nw")
        self.feedback_canvas.configure(yscrollcommand=self.feedback_scrollbar.set)
        self.feedback_canvas.grid(row=0, column=1, sticky="nsew", padx=(10,0))
        self.feedback_scrollbar.grid(row=0, column=3, sticky="ns")

        # Section headers
        self.angles_header = ttk.Label(self.scrollable_results, text="Joint Angles", font=("Segoe UI", 12, "bold"), padding=(0,5,0,5))
        self.angles_header.grid(row=0, column=0, columnspan=3, sticky=tk.W)
        self.feedback_header = ttk.Label(self.feedback_frame, text="Issues to Fix", font=("Segoe UI", 12, "bold"), foreground="#b22222", padding=(0,5,0,5))
        self.feedback_header.grid(row=0, column=0, sticky=tk.W)

        # Result label below both columns
        self.result_label = ttk.Label(self.results_frame, text="", font=("Segoe UI", 14, "bold"), padding=(0,10,0,10))
        self.result_label.grid(row=1, column=0, columnspan=4, pady=10)

        # Style improvements
        style = ttk.Style()
        style.configure("TFrame", background="#f7f7fa")
        style.configure("Feedback.TFrame", background="#fff8f0")
        style.configure("TLabel", font=("Segoe UI", 11), background="#f7f7fa")
        style.configure("Feedback.TLabel", font=("Segoe UI", 11), background="#fff8f0", foreground="#b22222")
        style.configure("Result.TLabel", font=("Segoe UI", 14, "bold"))
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def setup_results_display(self):
        """Setup the results display area"""
        # Pose info
        self.pose_info_label = ttk.Label(self.results_frame, text="")
        self.pose_info_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Angles comparison
        self.angles_frame = ttk.Frame(self.results_frame)
        self.angles_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Feedback
        self.feedback_label = ttk.Label(self.results_frame, text="", foreground="red")
        self.feedback_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Overall result
        self.result_label = ttk.Label(self.results_frame, text="", font=("Arial", 12, "bold"))
        self.result_label.grid(row=3, column=0, columnspan=2, pady=10)
    
    def on_pose_selected(self, event):
        """Handle pose selection"""
        selected_index = self.pose_combo.current()
        if selected_index >= 0:
            pose_names = list(self.suryanamaskar_data['suryanamaskar_poses'].keys())
            selected_pose = pose_names[selected_index]
            pose_data = self.suryanamaskar_data['suryanamaskar_poses'][selected_pose]
            
            info_text = f"Selected: {pose_data['name']}\nDescription: {pose_data['description']}"
            self.pose_info_label.config(text=info_text)
    
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            title="Select Exercise Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if not file_path:
            return
        
        # Check if pose is selected
        if not self.pose_var.get():
            messagebox.showerror("Error", "Please select a Suryanamaskar pose first!")
            return
        
        # Get selected pose
        selected_index = self.pose_combo.current()
        pose_names = list(self.suryanamaskar_data['suryanamaskar_poses'].keys())
        selected_pose = pose_names[selected_index]
        
        # Analyze pose
        try:
            result = self.analyze_pose(file_path, selected_pose)
            
            if 'error' in result:
                messagebox.showerror("Error", result['error'])
                return
       
            # Display results
            self.display_results(result)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def display_image(self, image, label_text=None):
        """Display the image with optional label"""
        # Resize image to fit in GUI
        height, width = image.shape[:2]
        max_size = 300
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        # Convert to PIL format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        # Update label
        self.image_label.config(image=photo, text=label_text or "")
        self.image_label.image = photo  # Keep a reference
    
    def display_preprocessed_image(self, image):
        """Display the preprocessed image in the pre_image_label area"""
        height, width = image.shape[:2]
        max_size = 300
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        self.pre_image_label.config(image=photo, text="Preprocessed Image")
        self.pre_image_label.image = photo

    def display_results(self, result):
        """Display analysis results"""
        # Clear previous results
        for widget in self.angles_frame.winfo_children():
            if widget != self.angles_header:
                widget.destroy()
        for widget in self.feedback_frame.winfo_children():
            if widget != self.feedback_header:
                widget.destroy()

        # Display angles comparison
        row = 1  # Start after header
        for joint, current_angle in result['current_angles'].items():
            if joint in result['target_angles']:
                target = result['target_angles'][joint]
                target_range = f"{target['min']}° - {target['max']}°"
                # Determine color based on whether angle is in range
                color = "green" if target['min'] <= current_angle <= target['max'] else "red"
                ttk.Label(self.angles_frame, text=f"{joint}:").grid(row=row, column=0, sticky=tk.W)
                ttk.Label(self.angles_frame, text=f"{current_angle}°", foreground=color).grid(row=row, column=1, padx=10)
                ttk.Label(self.angles_frame, text=f"(Target: {target_range})").grid(row=row, column=2, sticky=tk.W)
                row += 1
            else:
                # Show extra angles (like Neck, Ankle, Trunk) if present
                ttk.Label(self.angles_frame, text=f"{joint}:").grid(row=row, column=0, sticky=tk.W)
                ttk.Label(self.angles_frame, text=f"{current_angle}°").grid(row=row, column=1, padx=10)
                row += 1

        # Display feedback in scrollable feedback_frame
        if result['feedback']:
            for i, issue in enumerate(result['feedback'], start=1):
                ttk.Label(self.feedback_frame, text=f"• {issue}", style="Feedback.TLabel").grid(row=i, column=0, sticky=tk.W, pady=1)
        else:
            ttk.Label(self.feedback_frame, text="All angles are correct!", foreground="green", style="Feedback.TLabel").grid(row=1, column=0, sticky=tk.W, pady=1)

        # Display overall result
        if result['is_correct_pose']:
            self.result_label.config(text="✅ Perfect Pose!", foreground="green", style="Result.TLabel")
        else:
            self.result_label.config(text="❌ Pose needs adjustment", foreground="red", style="Result.TLabel")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function"""
    try:
        app = SuryanamaskarAnalyzer()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main() 