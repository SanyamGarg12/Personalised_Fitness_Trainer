"""
Exercise Index for Suryanamaskar Pose Analyzer

This module provides utilities for managing and accessing Suryanamaskar pose data.
"""

import json
import os
from typing import Dict, List, Optional

class ExerciseIndex:
    """Manages the index of Suryanamaskar exercises and their target angles."""
    
    def __init__(self, data_file: str = 'suryanamaskar_poses.json'):
        """Initialize the exercise index with data from JSON file."""
        self.data_file = data_file
        self.load_data()
    
    def load_data(self) -> None:
        """Load exercise data from JSON file."""
        try:
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {self.data_file} not found. Creating empty data structure.")
            self.data = {"suryanamaskar_poses": {}}
        except json.JSONDecodeError as e:
            print(f"Error reading {self.data_file}: {e}")
            self.data = {"suryanamaskar_poses": {}}
    
    def save_data(self) -> None:
        """Save exercise data to JSON file."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def get_all_poses(self) -> Dict[str, Dict]:
        """Get all available poses."""
        return self.data.get('suryanamaskar_poses', {})
    
    def get_pose_names(self) -> List[str]:
        """Get list of all pose names."""
        poses = self.get_all_poses()
        return [pose_data['name'] for pose_data in poses.values()]
    
    def get_pose_keys(self) -> List[str]:
        """Get list of all pose keys (internal identifiers)."""
        return list(self.get_all_poses().keys())
    
    def get_pose_data(self, pose_key: str) -> Optional[Dict]:
        """Get data for a specific pose by key."""
        poses = self.get_all_poses()
        return poses.get(pose_key)
    
    def get_pose_by_name(self, pose_name: str) -> Optional[Dict]:
        """Get pose data by display name."""
        poses = self.get_all_poses()
        for key, pose_data in poses.items():
            if pose_data['name'] == pose_name:
                return {'key': key, **pose_data}
        return None
    
    def add_pose(self, pose_key: str, pose_data: Dict) -> bool:
        """Add a new pose to the index."""
        try:
            self.data['suryanamaskar_poses'][pose_key] = pose_data
            self.save_data()
            return True
        except Exception as e:
            print(f"Error adding pose: {e}")
            return False
    
    def update_pose(self, pose_key: str, pose_data: Dict) -> bool:
        """Update an existing pose."""
        if pose_key in self.data['suryanamaskar_poses']:
            return self.add_pose(pose_key, pose_data)
        return False
    
    def delete_pose(self, pose_key: str) -> bool:
        """Delete a pose from the index."""
        try:
            if pose_key in self.data['suryanamaskar_poses']:
                del self.data['suryanamaskar_poses'][pose_key]
                self.save_data()
                return True
            return False
        except Exception as e:
            print(f"Error deleting pose: {e}")
            return False
    
    def get_pose_summary(self) -> List[Dict]:
        """Get a summary of all poses with key information."""
        summary = []
        poses = self.get_all_poses()
        
        for key, pose_data in poses.items():
            summary.append({
                'key': key,
                'name': pose_data['name'],
                'description': pose_data['description'],
                'joints_measured': len(pose_data.get('targetAngles', {})),
                'distance_range': pose_data.get('distanceThreshold', {})
            })
        
        return summary
    
    def validate_pose_data(self, pose_data: Dict) -> List[str]:
        """Validate pose data structure and return list of errors."""
        errors = []
        
        required_fields = ['name', 'description', 'targetAngles', 'distanceThreshold']
        for field in required_fields:
            if field not in pose_data:
                errors.append(f"Missing required field: {field}")
        
        if 'targetAngles' in pose_data:
            if not isinstance(pose_data['targetAngles'], dict):
                errors.append("targetAngles must be a dictionary")
            else:
                for joint, angles in pose_data['targetAngles'].items():
                    if not isinstance(angles, dict) or 'min' not in angles or 'max' not in angles:
                        errors.append(f"Invalid angle data for {joint}")
                    elif angles['min'] >= angles['max']:
                        errors.append(f"Invalid angle range for {joint}: min >= max")
        
        if 'distanceThreshold' in pose_data:
            if not isinstance(pose_data['distanceThreshold'], dict):
                errors.append("distanceThreshold must be a dictionary")
            elif 'min' not in pose_data['distanceThreshold'] or 'max' not in pose_data['distanceThreshold']:
                errors.append("distanceThreshold must have 'min' and 'max' values")
            elif pose_data['distanceThreshold']['min'] >= pose_data['distanceThreshold']['max']:
                errors.append("Invalid distance range: min >= max")
        
        return errors
    
    def export_poses_to_csv(self, filename: str = 'poses_summary.csv') -> bool:
        """Export pose summary to CSV file."""
        try:
            import csv
            summary = self.get_pose_summary()
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['key', 'name', 'description', 'joints_measured', 'distance_min', 'distance_max']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for pose in summary:
                    writer.writerow({
                        'key': pose['key'],
                        'name': pose['name'],
                        'description': pose['description'],
                        'joints_measured': pose['joints_measured'],
                        'distance_min': pose['distance_range'].get('min', ''),
                        'distance_max': pose['distance_range'].get('max', '')
                    })
            
            print(f"Pose summary exported to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False

def create_default_poses():
    """Create default Suryanamaskar poses if they don't exist."""
    default_poses = {
        "prayatna": {
            "name": "Prayatna (Prayer Pose)",
            "description": "Standing with hands in prayer position",
            "targetAngles": {
                "Left Shoulder": {"min": 80, "max": 100},
                "Right Shoulder": {"min": 80, "max": 100},
                "Left Elbow": {"min": 80, "max": 100},
                "Right Elbow": {"min": 80, "max": 100},
                "Left Hip": {"min": 170, "max": 180},
                "Right Hip": {"min": 170, "max": 180},
                "Left Knee": {"min": 170, "max": 180},
                "Right Knee": {"min": 170, "max": 180}
            },
            "distanceThreshold": {"min": 1.5, "max": 3.0}
        }
    }
    
    index = ExerciseIndex()
    for key, pose_data in default_poses.items():
        if key not in index.get_all_poses():
            index.add_pose(key, pose_data)
    
    return index

if __name__ == "__main__":
    # Example usage
    index = ExerciseIndex()
    
    print("Available Suryanamaskar Poses:")
    print("=" * 40)
    
    summary = index.get_pose_summary()
    for i, pose in enumerate(summary, 1):
        print(f"{i}. {pose['name']}")
        print(f"   Key: {pose['key']}")
        print(f"   Description: {pose['description']}")
        print(f"   Joints measured: {pose['joints_measured']}")
        print(f"   Distance range: {pose['distance_range']['min']}-{pose['distance_range']['max']}m")
        print()
    
    # Export to CSV
    index.export_poses_to_csv() 