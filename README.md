  
## Requirements
       ***Important****************************************************************
       *Change the path of the video in cv2.VideoCapture() function in python file*
       *Change the path of the refrence image and the testudo image in the Part1b.py, Part2a.py respectively
       *The video files are not genetrting but u can see the output in the seperate window frame by frame in a form of a video
       		
       ****************************************************************************
       
### To run this code following libraries are required
* OpenCV,  

* NumPy, 

* copy

### Installation (For ubuntu 18.04) ###
* OpenCV
	````
	sudo apt install python3-opencv
	````

* NumPy
	````
	pip install numpy
	````
	
### Running code in ubuntu
After changing the path of the video source file and installing dependencies
Make sure that current working derectory is same as the directory of program
You can change the working derectory by using **cd** command
* Run the following command which will give the result
````
python Detect.py
````
* Run the following command which will decode the reference April Tag
  The code will display the final output that is the image of the AR Tag on a sepreate window
````
python Decode.py
````  
It is important to note that all python files are in different directory
we have to change to the correct directory again.



### Troubleshooting ###
	Most of the cases the issue will be incorrect file path.
	Double check the path by opening the properies of the video and the image of the refrence Tag and the testudo image
	and copying path directly from there.

	For issues that you may encounter create an issue on GitHub.
  
### Maintainers ###
	Nisarg Upadhyay (nisargupadhyay1@gmail.com)
