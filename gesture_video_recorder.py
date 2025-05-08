#!/usr/bin/env python3
"""
Gesture-Controlled Video Recorder
- Uses OpenCV + MediaPipe for hand tracking
- State machine for recording control
- Left hand fingers = # of videos to record
- Right hand fingers = control (1=record, 2=prepare)
- Saves videos with timestamps
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
import mediapipe as mp
from typing import Dict, List, Optional

class GestureVideoRecorder:
    def __init__(
        self,
        output_dir: str = "recordings",
        camera_index: int = 0,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        video_duration: float = 5.0,  # Default duration per video (seconds)
        focus_delay: float = 1.5,  # Delay before recording starts
        frame_size: tuple = (640, 480),
        fps: int = 30,
    ):
        """
        Initialize gesture-controlled video recorder.
        
        Args:
            output_dir (str): Directory to save videos.
            camera_index (int): Webcam index (usually 0).
            min_detection_confidence (float): Hand detection threshold.
            min_tracking_confidence (float): Hand tracking threshold.
            video_duration (float): Max recording duration per video (seconds).
            focus_delay (float): Delay before recording after preparation.
            frame_size (tuple): Video resolution (width, height).
            fps (int): Frames per second for recording.
        """
        self.output_dir = output_dir
        self.camera_index = camera_index
        self.video_duration = video_duration
        self.focus_delay = focus_delay
        self.frame_size = frame_size
        self.fps = fps
        
        # MediaPipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Video recording variables
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording_start_time: float = 0
        self.video_count: int = 0
        self.current_video_frames: List[np.ndarray] = []
        
        # Camera state machine
        self.state: str = "IDLE"  # States: IDLE, PREPARING, RECORDING
        self.last_gesture_time: float = 0
        
        # Performance tracking
        self.frame_count: int = 0
        self.start_time: float = time.time()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera!")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def __del__(self):
        """Release resources."""
        if self.cap.isOpened():
            self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
    
    def _get_timestamp(self) -> str:
        """Generate a timestamp for filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _count_fingers(self, landmarks, handedness: str) -> int:
        """Count extended fingers (0-5)."""
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP,
        ]
        finger_mcp = [
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP,
        ]
        
        thumb_tip = self.mp_hands.HandLandmark.THUMB_TIP
        thumb_ip = self.mp_hands.HandLandmark.THUMB_IP
        
        count = 0
        
        # Thumb detection (right/left hand logic)
        if handedness == "Right":
            if landmarks[thumb_tip].x > landmarks[thumb_ip].x:
                count += 1
        else:
            if landmarks[thumb_tip].x < landmarks[thumb_ip].x:
                count += 1
        
        # Other fingers
        for tip, mcp in zip(finger_tips, finger_mcp):
            if landmarks[tip].y < landmarks[mcp].y:
                count += 1
        
        return count
    
    def _process_gestures(self, frame: np.ndarray, results) -> Dict[str, int]:
        """Detect hands and count fingers."""
        gesture_info = {"Left": 0, "Right": 0}
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_label = handedness.classification[0].label
                finger_count = self._count_fingers(hand_landmarks.landmark, hand_label)
                gesture_info[hand_label] = finger_count
                
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Display finger count
                cv2.putText(
                    frame,
                    f"{hand_label}: {finger_count}",
                    (10, 30 if hand_label == "Left" else 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
        
        return gesture_info
    
    def _update_state(self, gesture_info: Dict[str, int]) -> None:
        """State machine for recording control."""
        left_fingers = gesture_info["Left"]
        right_fingers = gesture_info["Right"]
        
        # Reset if no hands detected for 2 seconds
        if left_fingers == 0 and right_fingers == 0:
            if time.time() - self.last_gesture_time > 2.0:
                self.state = "IDLE"
            return
        
        self.last_gesture_time = time.time()
        
        # State transitions
        if self.state == "IDLE" and right_fingers == 2:
            self.state = "PREPARING"
            self.prepare_start_time = time.time()
        elif self.state == "PREPARING" and (time.time() - self.prepare_start_time) > self.focus_delay:
            self.state = "READY"
        elif self.state == "READY" and right_fingers == 1:
            self.state = "RECORDING"
            self.recording_start_time = time.time()
            self.video_count = left_fingers if left_fingers > 0 else 1
            self._start_recording()
        elif self.state == "RECORDING":
            elapsed = time.time() - self.recording_start_time
            if elapsed >= self.video_duration or right_fingers == 0:
                self._stop_recording()
                self.video_count -= 1
                if self.video_count > 0:
                    self.state = "READY"
                else:
                    self.state = "IDLE"
    
    def _start_recording(self) -> None:
        """Initialize video writer."""
        timestamp = self._get_timestamp()
        filename = os.path.join(self.output_dir, f"video_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            filename, fourcc, self.fps, self.frame_size
        )
        print(f"Recording started: {filename}")
    
    def _stop_recording(self) -> None:
        """Stop recording and save video."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print("Recording stopped.")
    
    def _display_status(self, frame: np.ndarray) -> None:
        """Show system status on screen."""
        fps = self.frame_count / (time.time() - self.start_time)
        
        # Display state
        cv2.putText(
            frame,
            f"State: {self.state}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        
        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        
        # Visual feedback for PREPARING state
        if self.state == "PREPARING":
            progress = (time.time() - self.prepare_start_time) / self.focus_delay
            cv2.rectangle(
                frame,
                (0, 0),
                (int(frame.shape[1] * progress), 20),
                (0, 255, 0),
                -1,
            )
        
        # Visual feedback for RECORDING state
        elif self.state == "RECORDING":
            elapsed = time.time() - self.recording_start_time
            remaining = max(0, self.video_duration - elapsed)
            cv2.putText(
                frame,
                f"Recording... {remaining:.1f}s left",
                (frame.shape[1] // 4, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
    
    def run(self) -> None:
        """Main application loop."""
        print("Gesture-Controlled Video Recorder | Press 'Q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame!")
                break
            
            # Mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hands
            results = self.hands.process(rgb_frame)
            
            # Process gestures
            gesture_info = self._process_gestures(frame, results)
            
            # Update state machine
            self._update_state(gesture_info)
            
            # Write frame if recording
            if self.state == "RECORDING" and self.video_writer is not None:
                self.video_writer.write(frame)
            
            # Display status
            self._display_status(frame)
            
            # Show frame
            cv2.imshow("Gesture Video Recorder", frame)
            
            # Exit on 'Q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Increment frame counter
            self.frame_count += 1
        
        print("Shutting down...")

def main():
    """Entry point."""
    try:
        recorder = GestureVideoRecorder(
            output_dir="gesture_recordings",
            camera_index=0,
            video_duration=5.0,
            focus_delay=1.5,
            frame_size=(640, 480),
            fps=30,
        )
        recorder.run()
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    return 0

if __name__ == "__main__":
    main()
