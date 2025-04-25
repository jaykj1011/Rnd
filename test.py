from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()
picam2.start_preview(Preview.NULL)
picam2.start_and_capture_file("sa.jpg")
