#!/usr/bin/python
import  cv2

# now starting cam
cap=cv2.VideoCapture(0)

while   cap.isOpened() :
	print  ("camera is working")
	#frame is current camera data
	#status is after frame take camera status
	status,frame=cap.read()

	onlyred=cv2.inRange(frame,(0,0,0),(40,40,255))

	#to show the image window	
	#cv2.imshow("camera1",frame)
	cv2.imshow("camera2",onlyred)
	

	#hold the camera window and window will close by pressing 'q'
	if   cv2.waitKey(1)  &  0xFF  == ord('q')  :
		break  

#it will close alll opened window
cv2.destroyAllWindows()
cap.release()




