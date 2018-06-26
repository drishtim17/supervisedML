#!/usr/bin/python3
#image_action
import cv2
import numpy

#image reading
img1=cv2.imread('images.jpeg')

x='''
Press 1:To draw a Line in a image
Press 2:To draw a Rectangle in a image
Press 3:To draw a Circle in a image
Press 4:To write some text in the image
'''
print (x)
choice=input(" Enter your choice: ")

if choice=='1':
	print("DRAW LINE")
	#inputing the cordinatees
	cor=[[]for i in range(4)]
	for i in range(4):
    		cor[i]=input("Enter Cordinate of line :")
	
	#inputing the color cordinatees
	col=[[]for i in range(3)]
	print("In RGB form")
	for i in range(3):
 		col[i]=input("Color:")
	
	##inputing the thickness of line
	thick=input("Enter the Thickness of the line :")
	
	#type casting converting string to int
	for i in range(4):	
		cor[i]=int(cor[i])
	for i in range(3):
		col[i]=int(col[i])
	thick=int(thick)
	
	#to draw line(image,(x1,y1),(x2,y2),(r,g,b),thickness)
	cv2.line(img1,(cor[0],cor[1]),(cor[2],cor[3]),(col[0],col[1],col[2]),thick)
	
	#it will display the image on screen 
	cv2.imshow("Dog1",img1)

	#It will hold the image on the screen
	cv2.waitKey(0)

	#close the image
	cv2.destroyAllWindows()

if choice=='2':
	print("DRAW RECTANGLE")
	#inputing the cordinatees

	cor=[[]for i in range(4)]
	print("Enter four Cordinates(Diagonal Coordinates)")
	for i in range(4):
    		cor[i]=input("Enter Cordinate of line :")
	
	#inputing the color cordinatees
	col=[[]for i in range(3)]
	print("In RGB form")
	for i in range(3):
 		col[i]=input("Color:")
	
	##inputing the thickness of line
	thick=input("Enter the Thickness of the line :")
	
	#type casting converting string to int
	for i in range(4):	
		cor[i]=int(cor[i])
	for i in range(3):
		col[i]=int(col[i])
	thick=int(thick)
	
	#to draw rectangle(image,(x1,y1),(x2,y2),(r,g,b),thickness)
	cv2.rectangle(img1,(cor[0],cor[1]),(cor[2],cor[3]),(col[0],col[1],col[2]),thick)
	
	#it will display the image on screen 
	cv2.imshow("Dog1",img1)

	#It will hold the image on the screen
	cv2.waitKey(0)

	#close the image
	cv2.destroyAllWindows()

if choice=='3':
	print("DRAW CIRCLE")
	#inputing the centre
	centre=[[]for i in range(2)]
	for i in range(2):
    		centre[i]=input("Enter Centre of circle :")

	#inputing the radius of circle
	radius=input("Enter the radius of the circle :")
	
	#inputing the color cordinatees
	col=[[]for i in range(3)]
	print("In RGB form")
	for i in range(3):
 		col[i]=input("Color:")
	
	#inputing the thickness of line
	thick=input("Enter the Thickness of the line :")
	
	#type casting converting string to int

	for i in range(2):	
		centre[i]=centre(cor[i])

	radius=int(radius)

	for i in range(3):
		col[i]=int(col[i])

	thick=int(thick)
	
	#to draw circle(image,(c1,c2),radius,(r,g,b),thickness)
	cv2.circle(img1,(centre[0],centre[1]),radius,(col[0],col[1],col[2]),thick)
	
	#it will display the image on screen 
	cv2.imshow("Dog1",img1)

	#It will hold the image on the screen
	cv2.waitKey(0)

	#close the image
	cv2.destroyAllWindows()


if choice=='4':
	print("WRITING TEXT")

	#inputing the text
	text=input("Enter the text :")

	#inputing the cordinates
	cor=[[]for i in range(2)]
	for i in range(2):
    		cor[i]=input("Enter Cordinators where you want to write text :")
	
	#specifing font
	font=cv2.FONT_HERSHEY_SIMPLEX

	#inputing the font size
	font_size=input("Enter the Font size :")
	
	#inputing the color cordinatees
	col=[[]for i in range(3)]
	print("In RGB form")
	for i in range(3):
 		col[i]=input("Color:")
	
	#type casting converting string to int
	for i in range(2):	
		cor[i]=int(cor[i])
	for i in range(3):
		col[i]=int(col[i])


	font_size=int(font_size)
	
	#to write text(image,"text",(x1,y1),font,font_size,(r,g,b),line_AA)
	cv2.putText(img1,text,(cor[0],cor[1]),font,font_size,(col[0],col[1],col[2]),cv2.LINE_AA)
	
	#it will display the image on screen 
	cv2.imshow("Dog1",img1)

	#It will hold the image on the screen
	cv2.waitKey(0)

	#close the image
	cv2.destroyAllWindows()
else:
	print("Wrong Choice")
	


