import cv2
import mediapipe as mp
import pygame
import math
import numpy as np

# Initialize MediaPipe pose module
mp_pose = mp.solutions.pose

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame for sound playback
pygame.mixer.init()

# Load the sound file
sound_file = 'count.mp3'
sound = pygame.mixer.Sound(sound_file)

# Load the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Set up MediaPipe pose
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    hands_up_count = 0   # Initialize hands-up count
    hands_up = False  # Initialize hands-up state

    shoulder_width = 0  # Initialize shoulder width (in pixels)
    distance_text = "Distance: N/A"  # Default distance text

    # Initialize smoothing window size and buffer
    smoothing_window_size = 5
    left_wrist_buffer = []
    right_wrist_buffer = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe pose
        results = pose.process(image_rgb)

        # Check if pose landmarks are detected
        if results.pose_landmarks:
            # Get the coordinates of the landmarks
            landmarks = results.pose_landmarks.landmark

            # Get the y-coordinate of the left and right wrists
            left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y

            # Add y-coordinates to the buffer
            left_wrist_buffer.append(left_wrist_y)
            right_wrist_buffer.append(right_wrist_y)

            # Maintain buffer size
            if len(left_wrist_buffer) > smoothing_window_size:
                left_wrist_buffer.pop(0)
                right_wrist_buffer.pop(0)

            # Calculate the smoothed vertical distances
            left_wrist_smoothed = np.mean(left_wrist_buffer)
            right_wrist_smoothed = np.mean(right_wrist_buffer)

            # Calculate the vertical distance from the wrists to the top of the image
            image_height, image_width, _ = image.shape
            left_wrist_to_top_distance = int(left_wrist_y * image_height)
            right_wrist_to_top_distance = int(right_wrist_y * image_height)

            # Set the threshold for the "hands-up" posture
            hands_up_threshold = 0.1

            # Check if both wrists are above the threshold (hands up)
            if left_wrist_to_top_distance < hands_up_threshold * image_height and right_wrist_to_top_distance < hands_up_threshold * image_height:
                if not hands_up:
                    hands_up = True
                    hands_up_count += 1
                    sound.play()
            else:
                hands_up = False

            # Calculate the distance from the camera to the user
            if shoulder_width == 0:  # Measure shoulder width once (assuming the first frame has a proper pose detection)
                shoulder_width = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width)

            # Calculate the distance based on shoulder width and focal length (you can adjust the focal length value based on your camera)
            focal_length = 20  # Adjust this value based on your camera
            distance_to_user = focal_length * shoulder_width / (2 * abs(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width - landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width))

            distance_text = f"Distance: {distance_to_user:.2f} meters"

            # Add visual feedback (green checkmark) if the distance is between 7 and 9.5 meters
            if 9 <= distance_to_user <= 13:
                save_distance_text = "Save Distance"
                save_distance_position = (
                    image.shape[1] - 30 * len(save_distance_text) - 50, 100)  # Adjust position for the right corner
                cv2.putText(image, save_distance_text, save_distance_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),
                            3)

            # Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.putText(image, f'Hands Up Count: {hands_up_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the image
        cv2.imshow('Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press Esc to exit
            break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
