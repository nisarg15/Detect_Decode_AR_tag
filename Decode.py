# Importing the Libraries
import cv2
import copy 

image = cv2.imread("C:/Users/nisar/Desktop/Projects/nisarg15_Project1/ref_marker.png")
image_info = copy.deepcopy(image)

#Creating the grid
GRID_SIZE = 25
height, width, channel = image.shape
for x in range(0, height -1, GRID_SIZE):
     cv2.line(image, (x, 0), (x, width), (255, 0, 0), 1)
for x in range(0, height -1, GRID_SIZE):  
    cv2.line(image,(200,x),(0,x),(0,0,255), thickness=1)



img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, image = cv2.threshold(image, 127, 255, 0)
# Checking the orintation of the AR TAG
# Top Left
if 255 in image[51:75, 51:75]:
 print("The position of the TAG is 'top_left'")
 orintation = "TL"
 
# Top Right
elif 255 in image[126:150, 51:75]:
    print("The position of the TAG is 'top_right")
    orintation = "TR"

# Bottom right
elif 255 in image[51:75,126:150 ]:
    print("The position of the TAG is 'bottom_right")
    orintation = "BR"
    
# Bottom Left
else:
    print("The position of the TAG is 'bottom_left")
    orintation = "BL"

# Printing the orinatation of the AR TAG 
print(orintation)

# Checking the value of the AR TAG
Tag_code = ''

# Top Left
if 255 in image[76:100,76:100] :
    Tag_code = Tag_code + '1'
else:
    Tag_code = Tag_code + '0'

# Top Right
if 255 in image[76:100,101:125] :
    Tag_code = Tag_code + '1'
else:
    Tag_code = Tag_code + '0'
    
# Bottom right
if 255 in image[101:125,76:100] :
    Tag_code = Tag_code + '1'
else:
    Tag_code = Tag_code + '0'

# Bottom Left
if 255 in  image[101:125,101:125] :
    Tag_code = Tag_code + '1'
else:
    Tag_code = Tag_code + '0'

# Printing the value of the TAG    
print(Tag_code)


image_info = cv2.putText(image_info, 'Tag_name =', (0,15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
image_info = cv2.putText(image_info, Tag_code, (115,15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
image_info = cv2.putText(image_info, 'Orientation =', (0,35), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
image_info = cv2.putText(image_info, orintation , (115,35), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

# Displaying the Image with grid
cv2.imshow('Image', image)

# Displaying the Image with orianantion and the value of the AR TAG
cv2.imshow('Image_Information', image_info)
cv2.waitKey(0)
cv2.destroyAllWindows()




