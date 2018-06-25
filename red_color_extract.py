#!/usr/bin/python3
import cv2
import time

#image read hogi
img1=cv2.imread('redhat.jpg')

#printing the shape of the images(rows,col,color(3))
print(img1.shape)
a=img1.shape

time.sleep(5)
print(img1)

#extracting only red colour
#(jis par range aplly krni hai,(starting range of colour),(ending range))
red=cv2.inRange(img1,(0,0,0),(255,40,40))
cv2.imshow("original",img1)
cv2.imshow("only red",red)

#It will hold the image on the screen
cv2.waitKey(0)

#close the image
cv2.destroyAllWindows()


