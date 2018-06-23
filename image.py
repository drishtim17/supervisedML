#!/usr/bin/python3
import cv2
import numpy

#image read hogi
img1=cv2.imread('index.jpeg')
img2=cv2.imread('images.jpeg')

#printing the shape of the images(rows,col,color(3))
#print(img1.shape)
a=img1.shape
b=img2.shape

for i in range(a[0]):
	for j in range(a[1]):
		#image into full white
		img1[i][j]=[255,255,255]

for i in range(b[0]):
	for j in range(b[1]):
		#image into full black
		img2[i][j]=[0,0,0]
#it will display the image on screen 
cv2.imshow("Dog1",img1)
cv2.imshow("Dog2",img2)

#to save the image
cv2.imwrite("White.jpeg",img1)
cv2.imwrite("Black.jpeg",img2)

#It will hold the image on the screen
cv2.waitKey(0)

#close the image
cv2.destroyAllWindows()


