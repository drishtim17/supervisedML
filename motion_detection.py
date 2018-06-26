#!/usr/bin/pyhton3

import cv2

#this is a function which will check the image difference(3 frames)
def img_diff(x,y,z):
	img1=cv2.absdiff(x,y)#absoulte difference
	img2=cv2.absdiff(y,z)
	com_diff=cv2.bitwise_and(img1,img2)
	return com_diff
#camera for video
cap=cv2.VideoCapture(0)

#read takes aa list [status,frame] frames are used
frame1=cap.read()[1]
frame2=cap.read()[1]
frame3=cap.read()[1]

#changed colour image to gray scale
gray1=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
gray3=cv2.cvtColor(frame3,cv2.COLOR_BGR2GRAY)

#camera is open
while cap.isOpened():
	#function calling
	imgDiff=img_diff(gray1,gray2,gray3)
	cv2.imshow("Difference",imgDiff)
	
	status,frame=cap.read()
	gray1=gray2
	gray2=gray3
	gray3=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	#printing data of motion detection
	print(gray3)

	#hold the camera screen and to close that window press 'q'
	if cv2.waitKey(1) & 0XFF == ord('q'):
		break

cv2.destroyAllWindows()
cap.release()
