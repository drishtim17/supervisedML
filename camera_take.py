#!/usr/bin/python3
#camera_take
import cv2

#to start camera
cam=cv2.VideoCapture(0)

#will open the camera
if cam.isOpened():
	print("Camera is ready to use")

#frame is current camera data
#status is after frame take camera status
status,frame=cam.read()

#will show the frame name "framme1"
cv2.imshow("framme1",frame)

#hold the camera
cv2.waitKey(0)

#ek bar image le kar it will close
cam.release()
