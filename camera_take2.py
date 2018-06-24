#!/usr/bin/python

import  cv2
import random
# now starting cam

cap=cv2.VideoCapture(0)

while   cap.isOpened() :
	print  ("camera is working")


	#frame is current camera data
	#status is after frame take camera status
	status,frame=cap.read()
	
	#frame which will capture black n white
	bwimg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		
	#to show the image window	
	cv2.imshow("camera1",frame)
	cv2.imshow("camera2",bwimg)
	
	#to generate random no.
	x=random.random()
	#from random folat no is generated so we converted it ito string and take 4 nos.
	y=str(x)[5:8]
	#tosave the image('save file name',frame)
	cv2.imwrite('adhoc'+y+'.jpg',frame)

	#hold the camera window and window will close by pressing 'q'
	if   cv2.waitKey(1)  &  0xFF  == ord('q')  :
		break  

#it will close alll opened window
cv2.destroyAllWindows()
cap.release()




