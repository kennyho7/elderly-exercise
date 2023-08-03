import cv2
import mediapipe as mp
import math
import time


# Load the video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the path to a video file
cap.set(3, 1280)
cap.set(4, 720)

# Initialize OpenPose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Set up a pose detection model
with mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.5) as pose:
    is_calibrated = False
    neck_stretch_count = 0
    cooldown_time = 5  # Cooldown period in seconds
    last_stretch_time = 0

    while True:
        # Capture a video frame from the webcam
        ret, image = cap.read()

        if not ret:
            break

        # Convert the image to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform pose estimation on the image
        results = pose.process(image_rgb)

        # Convert the image back to BGR format for display
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Calculate the neck stretch angle
        neck_stretch = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            image_height, image_width, _ = image_bgr.shape  # Get the height and width of the frame

            # Get the pixel coordinates of the detected landmarks
            nose_x = int(landmarks[mp_pose.PoseLandmark.NOSE].x * image_width)
            nose_y = int(landmarks[mp_pose.PoseLandmark.NOSE].y * image_height)
            left_shoulder_x = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width)
            left_shoulder_y = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height)
            right_shoulder_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width)
            right_shoulder_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height)

            neck_angle = math.degrees(math.atan2(nose_x - right_shoulder_x, nose_y - right_shoulder_y))

            if not is_calibrated:
                initial_neck_angle = neck_angle
                is_calibrated = True

            # Calculate the absolute neck stretch angle relative to the initial_neck_angle
            neck_stretch = abs(neck_angle - initial_neck_angle)

            # Draw the neck angle on the image
            cv2.putText(image_bgr, f'Neck Stretch: {neck_stretch:.2f} degrees', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show direction based on the neck stretch angle
            direction = None
            if 12 <= neck_stretch <= 15:
                direction = 'Up'
            elif 2 <= neck_stretch <= 4:
                direction = 'Down'
            elif 0 <= neck_stretch <= 1:
                direction = 'Left'
            elif 30 <= neck_stretch <= 33:
                direction = 'Right'

            # Draw the direction on the image
            if direction:
                cv2.putText(image_bgr, f'Direction: {direction}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                            2)

        # Draw the pose landmarks on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the image in a window
        cv2.imshow('Neck Stretch Exercise', image_bgr)

        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
