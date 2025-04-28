import cv2
import threading
print("cv")
import numpy as np
from picamera2 import Picamera2, Preview
print("np")
import time
import sys
if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    import collections
    setattr(collections, "MutableMapping", collections.abc.MutableMapping)
from apscheduler.schedulers.background import BackgroundScheduler
from dronekit import connect, VehicleMode
from pymavlink import mavutil

roll = 0
pitch = 0
z = 0
vehicle = None
compass_enabled = 0
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

# Define distortion coefficients (use camera calibration to get these)
# dist_coeffs = np.array([0, 0, 0, 0], dtype="float32")  # assuming no distortion for simplicity
dist_coeffs = np.array([-0.1, 0.01, 0, 0], dtype="float32")  # Example values

# Load the ArUco dictionary and parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)  # Change this as needed
parameters = cv2.aruco.DetectorParameters_create()

#######################################
# Functions for MAVLink
#######################################

# Define function to send landing_target mavlink message for mavlink based precision landing
# http://mavlink.org/messages/common#LANDING_TARGET
def send_land_target_message():
    global current_time, H_camera_tag, is_landing_tag_detected

    if is_landing_tag_detected == True:

        x_offset_rad = m.atan(x / z)
        y_offset_rad = m.atan(y / z)
        distance = np.sqrt(x * x + y * y + z * z)

        msg = vehicle.message_factory.landing_target_encode(
            current_time,                       # time target data was processed, as close to sensor capture as possible
            0,                                  # target num, not used
            mavutil.mavlink.MAV_FRAME_BODY_NED, # frame, not used
            x_offset_rad,                       # X-axis angular offset, in radians
            y_offset_rad,                       # Y-axis angular offset, in radians
            z,                           # distance, in meters
            0,                                  # Target x-axis size, in radians
            0,                                  # Target y-axis size, in radians
            0,                                  # x	float	X Position of the landing target on MAV_FRAME
            0,                                  # y	float	Y Position of the landing target on MAV_FRAME
            0,                                  # z	float	Z Position of the landing target on MAV_FRAME
            (1,0,0,0),      # q	float[4]	Quaternion of landing target orientation (w, x, y, z order, zero-rotation is 1, 0, 0, 0)
            2,              # type of landing target: 2 = Fiducial marker
            1,              # position_valid boolean
        )
        vehicle.send_mavlink(msg)
        vehicle.flush()
# Connect to FCU through serial port
def vehicle_connect():
    global vehicle

    try:
        vehicle = connect("/dev/serial0", wait_ready = True, baud = 115200, source_system = 1)
    except KeyboardInterrupt:    
        pipe.stop()
        print("INFO: Exiting")
        sys.exit()
    except:
        print('Connection error! Retrying...')

    if vehicle == None:
        return False
    else:
        return True


# Set up a mutex to share data between threads 
frame_mutex = threading.Lock()

print("INFO: Connecting to vehicle.")
while (not vehicle_connect()):
    pass
print("INFO: Vehicle connected.")

data = None
current_confidence = None
H_aeroRef_aeroBody = None
H_camera_tag = None
is_landing_tag_detected = False # This flag returns true only if the tag with landing id is currently detected
heading_north_yaw = None

# Send MAVlink messages in the background
sched = BackgroundScheduler()
sched.add_job(send_land_target_message, 'interval', seconds = 1/30)


sched.start()

if compass_enabled == 1:
    # Wait a short while for yaw to be correctly initiated
    time.sleep(1)

print("INFO: Starting main loop...")

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
            x = x / 2
            y = y / 2
            z = z / 10 
            roll, pitch, yaw = rvec[0][0]  # These are in radians

            # Display the coordinates and orientation
            print(f"Marker ID: {ids[i]}", end = '')
            print(f" Position (X, Y, Z in m): ({x:.2f}, {y:.2f}, {z:.2f})",end = '')
            print(f"Orientation (roll, pitch, yaw in radians): ({roll:.2f}, {pitch:.2f}, {yaw:.2f})",end='\n')
            # Draw the axis
            #cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
    
    # Show the image
    #cv2.imshow('ArUco Marker Detection', frame)

    # Break the loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
