import cv2
import numpy as np
import time
from picamera2 import Picamera2

# ArUco dictionary that includes AprilTag patterns
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()

# Camera calibration parameters
TAG_SIZE = 0.144  # meters
CAMERA_FX = 600
CAMERA_FY = 600
CAMERA_CX = 320
CAMERA_CY = 240

# Camera matrix
camera_matrix = np.array([
    [CAMERA_FX, 0, CAMERA_CX],
    [0, CAMERA_FY, CAMERA_CY],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros(5, dtype=np.float32)  # Assume no distortion

# Initialize camera with a smaller resolution to reduce load
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)}))
picam2.start()
time.sleep(2)  # Give camera time to initialize

print("Camera initialized. Press Ctrl+C to quit.")

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()
        
        # Convert to grayscale for tag detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect AprilTags
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
        # Process detected tags
        if ids is not None:
            for i in range(len(ids)):
                # Estimate pose
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], TAG_SIZE, camera_matrix, dist_coeffs
                )
                
                # Calculate angles and distance
                x_rad = np.arctan2(tvec[0][0][0], tvec[0][0][2])
                y_rad = np.arctan2(tvec[0][0][1], tvec[0][0][2])
                dist = np.linalg.norm(tvec)
                
                # Print info
                print(f"[INFO] Tag {ids[i][0]} @ x={x_rad[0]:.2f}rad, y={y_rad[0]:.2f}rad, d={dist[0]:.2f}m")
                
        # Small delay to prevent CPU overload
        time.sleep(0.1)
            
except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    # Clean up
    picam2.stop()
    print("Test completed.")
