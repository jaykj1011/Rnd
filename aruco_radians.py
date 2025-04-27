import cv2
print("cv")
import numpy as np
from picamera2 import Picamera2, Preview
print("np")
import time

picam2 = Picamera2() 
camera_config = picam2.create_preview_configuration() 
picam2.configure(camera_config) 
picam2.start_preview(Preview.NULL) 
picam2.start()

# Define camera matrix (intrinsic parameters)
# You can calibrate your camera to get these values (fx, fy, cx, cy)
# This is an example, you should replace with actual values from your camera calibration
camera_matrix = np.array([[9810.144958754117,    0.0,                  260.7917835500185],
                          [0.0,                 6342.787960213781,     289.77752006459065],
                          [0.0,                    0.0,                   1.0]], dtype="float64")

# # Example of more typical camera matrix values (replace with your actual calibration)
# camera_matrix = np.array([[800, 0, 320],
#                          [0, 800, 240],
#                          [0, 0, 1]], dtype="float64")

# Define distortion coefficients (use camera calibration to get these)
# dist_coeffs = np.array([0, 0, 0, 0], dtype="float32")  # assuming no distortion for simplicity
dist_coeffs = np.array([-0.1, 0.01, 0, 0], dtype="float32")  # Example values

# Load the ArUco dictionary and parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)  # Change this as needed
parameters = cv2.aruco.DetectorParameters_create()

# Camera setup
#cap = cv2.VideoCapture(0)  # or use `cap = cv2.VideoCapture("video.mp4")` for a video

while True:
    frame = picam2.capture_array()
    ret = True
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if len(corners) > 0:
        # If markers are detected, draw them
        #cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Estimate pose for each detected marker
        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.07, camera_matrix, dist_coeffs)

            # rvec: rotation vector (3x1), tvec: translation vector (3x1)
            # Convert rotation vector to a rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # Get the position (x, y, z) in meters and orientation (in radians)
            x, y, z = tvec[0][0]
          # Apply different scaling for X/Y vs Z if needed
            x_corrected = x * 100
            y_corrected = y * 100
            z_corrected = z / 10  
       
            roll, pitch, yaw = rvec[0][0]  # These are in radians

            # Display the coordinates and orientation
            print(f"Marker ID: {ids[i]}", end = '')
            print(f" Position (X, Y, Z in cm): ({x_corrected:.2f}, {y_corrected:.2f}, {z_corrected:.2f})",end = '')
            # print(f"Orientation (roll, pitch, yaw in radians): ({roll:.2f}, {pitch:.2f}, {yaw:.2f})",end='\n')
            # Draw the axis
            #cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
    
    # Show the image
    #cv2.imshow('ArUco Marker Detection', frame)

    # Break the loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
