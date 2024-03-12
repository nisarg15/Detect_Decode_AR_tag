# Importing the Libraries
import cv2
import copy
import numpy as np

# Defining the variables
x = []
y = []

# Function to resize the image
def resize(frames,scale):
    
    widht = int(frames.shape[1] * scale)
    height = int(frames.shape[0] * scale)
    dim = (widht,height)
    return cv2.resize(frames,dim)

Video = cv2.VideoCapture("C:/Users/nisar/Desktop/Projects/nisarg15_Project1/1tagvideo.mp4")

# Getting a single frame to perform operation (frame npo. 6)
Video.set(1, 6)
is_success, frame = Video.read()
frame_ = resize(frame, 0.5)

# Converting that frame to grayscale to perform FFT
frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret, frame_ = cv2.threshold(frame_, 127, 255, 0)

# FFT is a NP array with 2 Dimensions (Rows and columns)
# First channel consists of real part and second channel is imaginary part of FFT
# Important to note that FFT is an algorithm to find DFT
FFT = cv2.dft(np.float32(frame_), flags=cv2.DFT_COMPLEX_OUTPUT)

# Shifting array so that the center represents the zero coordinate
FFT_shift_ = np.fft.fftshift(FFT)
mag_spec_ = 20 * np.log(cv2.magnitude(FFT_shift_[:, :, 0], FFT_shift_[:, :, 1]))

# Creating mask
# The circular mask with center at (0, 0)
# The radius determine the cutoff frequency

# Defining no. od rows and columns
r, c = frame_.shape

# Finding center of frame
CenR, CenC = int(r/2), int(c/2)

# Note that the mask is a 2D array with 2 channels
mask = np.ones((r, c, 2), np.uint8)

# The radii is the parameter to change cutoff
radii = 100

# This function will generate a 2D grid of defined rows and columns
X, Y = np.ogrid[:r, :c]

# Creating a masking circle by assigning values 0 that fall in the range defined
masking_portion = (X - CenR) ** 2 + (Y - CenC) ** 2 <= radii*radii
mask[masking_portion] = 0

# apply mask to the FFT
FFT_shift = FFT_shift_ * mask

mag_spec = 20 * np.log(cv2.magnitude(FFT_shift[:, :, 0], FFT_shift[:, :, 1]))

# Applying inverse FFT
FFT_is_shift = np.fft.ifftshift(FFT_shift)

# Getting the modified frame back
frame_back_ = cv2.idft(FFT_is_shift)
frame_back = cv2.magnitude(frame_back_[:, :, 0], frame_back_[:, :, 1])
frame_back = np.array(frame_back)
frame_back = cv2.GaussianBlur(frame_back, (9,9), 0, borderType=cv2.BORDER_CONSTANT)
frame_back = cv2.normalize(frame_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
r, frame_back = cv2.threshold(frame_back, 127, 255, 0)
frame_back = resize(frame_back, 0.5)

# Applying the Blob dectator and the mask to rmeove the noise from the image
params = cv2.SimpleBlobDetector_Params()

# Define thresholds
#Can define thresholdStep. See documentation. 
params.minThreshold = 0
params.maxThreshold = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 10000
params.maxArea = 1000000000

# Filter by Color (black=0)
params.filterByColor = True  
params.blobColor = 0

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5
params.maxCircularity = 1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
params.maxConvexity = 1000

# Filter by InertiaRatio
params.filterByInertia = True
params.minInertiaRatio = 0
params.maxInertiaRatio = 1

# Distance Between Blobs
params.minDistBetweenBlobs = 200

# Setup the detector with parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(frame_back)




# Draw blobs
img_with_blobs = cv2.drawKeypoints(frame_back, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
frame_back2 = copy.deepcopy(frame_back)
img_with_blobs2 = cv2.drawKeypoints(frame_back2, keypoints, np.array([]), (0,1,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_with_blobs2 = cv2.cvtColor(img_with_blobs2, cv2.COLOR_BGR2GRAY)
ret, img_with_blobs2 = cv2.threshold(img_with_blobs2, 127, 255, 0)
img_with_blobs = cv2.cvtColor(img_with_blobs, cv2.COLOR_BGR2HSV)
    
#Extarxting the red channel to form a circular mask

l_1 = np.array([0, 100, 20])
u_1 = np.array([10, 255, 255])
    
 
l_2 = np.array([160,100,20])
u_2 = np.array([179,255,255])
    
lower_mask = cv2.inRange(img_with_blobs, l_1, u_1)
upper_mask = cv2.inRange(img_with_blobs, l_2, u_2)
full_mask = lower_mask + upper_mask;

# Creating the mask image
coordinates = []
coordinates = np.flip(np.argwhere(full_mask == 255))
full_mask = cv2.fillPoly(full_mask, [coordinates], [1,0,0])

# Removing the noise by using the mask from the image
masked_img = full_mask*img_with_blobs2

# Harris corner detection
frame_back = np.float32(masked_img)
dst = cv2.cornerHarris(frame_back, 5, 3, 0.04)
ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(frame_back, np.float32(centroids), (5, 5), (-1, -1), criteria)

# Extraxting the max and min values of the X & Y coordinates
for i in range(len(corners)): 
       value = (corners[i])
       a = value[0]
       b = value[1]
       x.append(a)
       y.append(b)


x_max = np.max(x)
x_max = x_max.astype(int)
x_min = np.min(x)
x_min = x_min.astype(int)
y_max = np.max(y)
y_max = y_max.astype(int)
y_min = np.min(y)
y_min = y_min.astype(int)

corners = [[x_min,y_min],[x_max,y_max]]
frame_back = frame_back[y_min:y_max , x_min:x_max]

# Cropping the image to remove the paper around the AR TAG
dim = np.shape(frame_back)
frame_back = frame_back[int(dim[1]/2)-75:int(dim[1]/2)+75, int(dim[0]/2)-75:int(dim[0]/2)+65]

# Harris corner detection
frame_back = np.float32(frame_back)
dst = cv2.cornerHarris(frame_back, 5, 3, 0.04)
ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(frame_back, np.float32(centroids), (5, 5), (-1, -1), criteria)

# Extraxting the max and min values of the X & Y coordinates
for i in range(len(corners)): 
       value = (corners[i])
       a = value[0]
       b = value[1]
       x.append(a)
       y.append(b)
       corner = [x,y]
       
       
    
x_max = np.max(x)
x_max = x_max.astype(int)
x_min = np.min(x)
x_min = x_min.astype(int)
y_max = np.max(y)
y_max = y_max.astype(int)
y_min = np.min(y)
y_min = y_min.astype(int)

# Cropping the image to the size of the AR TAG
frame_back = frame_back[y_min:y_max , x_min:x_max]

# Display of the AR TAG
cv2.imshow('image',frame_back)
cv2.waitKey(0)
cv2.destroyAllWindows()
