###########################
Dronedeploy-coding-challenge
###########################


Challenge-1 Problem statement:

This zip file https://www.dropbox.com/s/g67dmolq79ko4jk/Camera%20Localization.zip?dl=0 contains a number of images taken from different positions and orientations with an iPhone 6. Each image is the view of a pattern on a flat surface. The original pattern that was photographed is 8.8cm x 8.8cm and is included in the zip file. Write a Python program that will visualize (i.e. generate a graphic) where the camera was when each image was taken and how it was posed, relative to the algorithm.

You can assume that the pattern is at 0,0,0 in some global coordinate system and are thus looking for the x, y, z and yaw, pitch, roll of the camera that took each image. Please submit a link to a Github repository contain the code for your solution. Readability and comments are taken into account too. You may use 3rd party libraries like OpenCV and Numpy.


Steps to solve the challenge:

- Installed Python,OpenCV, matplotlib, scipy and Numpy libraries.

1. Go to terminal
2. Navigate to drone_deploy folder
3. run using the command "python drone_deploy.py" ( I have specified 'IMG_6726.JPG' as the test-image)

- The output pictures drawn using the Camera can be found in the 'captured_output_images' folder.
- Sample Input: IMG_6726.JPG (From testimages folder)
- Sample Output: capture_6726.png (From captured_output_images folder)
