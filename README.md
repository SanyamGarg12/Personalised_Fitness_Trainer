# Suryanamaskar Pose Analyzer

A Python application that analyzes Suryanamaskar (Sun Salutation) poses from uploaded images and provides feedback on joint angles that need to be corrected.

## Features

- **Image Upload**: Upload exercise images in various formats (JPG, PNG, BMP, GIF)
- **Pose Selection**: Choose from 9 different Suryanamaskar poses
- **Angle Analysis**: Compare current joint angles with target angles for each pose
- **Visual Feedback**: Display pose landmarks on the uploaded image
- **Detailed Feedback**: Show specific angles that need adjustment
- **Distance Estimation**: Check if the person is at the correct distance from camera

## Supported Suryanamaskar Poses

1. **Prayatna (Prayer Pose)** - Standing with hands in prayer position
2. **Hasta Uttanasana (Raised Arms Pose)** - Arms raised above head, slight backbend
3. **Padahastasana (Standing Forward Bend)** - Forward bend with hands touching feet
4. **Ashwa Sanchalanasana (Low Lunge)** - One leg forward in lunge position
5. **Phalakasana (Plank Pose)** - Body straight like a plank
6. **Ashtanga Namaskara (Eight-Limbed Salutation)** - Eight points of body touching ground
7. **Bhujangasana (Cobra Pose)** - Cobra pose with backbend
8. **Adho Mukha Svanasana (Downward Dog)** - Inverted V-shape pose
9. **Uttanasana (Standing Forward Bend)** - Forward bend with hands on ground

## Installation

1. **Clone or download the project files**
2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**:
   ```bash
   python suryanamaskar_analyzer.py
   ```

2. **Select a Suryanamaskar pose** from the dropdown menu

3. **Upload an exercise image** by clicking the "Upload Exercise Image" button

4. **View the analysis results**:
   - The image with pose landmarks will be displayed
   - Current angles will be compared with target angles
   - Feedback will show which angles need adjustment
   - Overall result will indicate if the pose is correct

## How It Works

1. **Pose Detection**: Uses MediaPipe Pose to detect 33 body landmarks
2. **Angle Calculation**: Calculates joint angles between three points (e.g., shoulder-elbow-wrist)
3. **Comparison**: Compares current angles with predefined target ranges for each pose
4. **Feedback**: Provides specific feedback on angles that are too small or too large
5. **Distance Check**: Estimates distance from camera and provides feedback if too close or too far

## File Structure

- `suryanamaskar_analyzer.py` - Main application file
- `suryanamaskar_poses.json` - Database of pose definitions and target angles
- `angle_estimator.py` - Original angle estimation code (for reference)
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Technical Details

### Joint Angles Measured
- Left/Right Elbow angles
- Left/Right Shoulder angles  
- Left/Right Hip angles
- Left/Right Knee angles
- Trunk angle (overall body alignment)

### Target Angle Ranges
Each pose has specific target angle ranges for optimal form:
- **Min/Max values** ensure proper alignment
- **Color coding** (green/red) indicates correct/incorrect angles
- **Detailed feedback** shows exact angle values and target ranges

### Image Requirements
- Clear, full-body images work best
- Good lighting improves pose detection
- Person should be clearly visible in the frame
- Avoid cluttered backgrounds

## Troubleshooting

- **No pose detected**: Ensure the person is clearly visible and well-lit
- **Incorrect angles**: Check that the image shows the full body and the pose is clearly visible
- **Application errors**: Make sure all dependencies are installed correctly

## Dependencies

- **OpenCV**: Image processing and computer vision
- **MediaPipe**: Pose detection and landmark extraction
- **NumPy**: Mathematical calculations
- **PIL (Pillow)**: Image handling for GUI
- **Tkinter**: GUI framework (included with Python)

## Contributing

Feel free to add more poses or improve the angle calculations by modifying the `suryanamaskar_poses.json` file. 