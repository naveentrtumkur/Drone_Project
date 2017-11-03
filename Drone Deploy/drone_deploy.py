############## Import the required built in libary ###############
import cv2
import numpy as np
from scipy import linalg
from pylab import *


#sets parameters of camera
def camera_calibration(sizes):
    row, colum = sizes
    x = 2555 * colum/2448
    y = 2586 * row/3264
    camera_parameters = np.diag([x, y, 1])
    camera_parameters[0,2] = 0.5*colum
    camera_parameters[1,2] = 0.5*row
    return camera_parameters


#this function helps us to draw a cube
def cube_points(c, wid): 
    cube_parameters = []
    
    cube_parameters.append([c[0] - wid, c[1] - wid, c[2] - wid])
    cube_parameters.append([c[0] - wid, c[1] + wid, c[2] - wid])
    cube_parameters.append([c[0] + wid, c[1] + wid, c[2] - wid])
    cube_parameters.append([c[0] + wid, c[1] - wid, c[2] - wid])
    
    cube_parameters.append([c[0] - wid, c[1] - wid, c[2] - wid])
    
    cube_parameters.append([c[0] - wid, c[1] - wid, c[2] + wid])
    cube_parameters.append([c[0] - wid, c[1] + wid, c[2] + wid])
    cube_parameters.append([c[0] + wid, c[1] + wid, c[2] + wid])
    cube_parameters.append([c[0] + wid, c[1] - wid, c[2] + wid])
    
    cube_parameters.append([c[0] - wid, c[1] - wid, c[2] + wid])
    
    cube_parameters.append([c[0] - wid, c[1] - wid, c[2] + wid])
    cube_parameters.append([c[0] - wid, c[1] + wid, c[2] + wid])
    cube_parameters.append([c[0] - wid, c[1] + wid, c[2] - wid])
    cube_parameters.append([c[0] + wid, c[1] + wid, c[2] - wid])
    cube_parameters.append([c[0] + wid, c[1] + wid, c[2] + wid])
    cube_parameters.append([c[0] + wid, c[1] - wid, c[2] + wid])
    cube_parameters.append([c[0] + wid, c[1] - wid, c[2] - wid])
    return np.array(cube_parameters).T


#converts cube's points to plot
def homogeneous_coordinates(points):      
    return np.vstack((points, np.ones((1, points.shape[1]))))

class Camera(object):

    def __init__(self,P):
	
        self.P = P
        self.K = None
        self.R = None
        self.t = None
        self.c = None

#calculating the camera's position as a matrix
    def project(self,X):
        xco = np.dot(self.P,X)

        for i in range(3):
            xco[i] = np.divide(xco[i],xco[2])
        return xco



######## TO read the two images #########

ref_image = cv2.imread('testimages/pattern_copy.jpg',0); # actual QR code given
main_image = cv2.imread('testimages/IMG_6726.jpg',0); # QR code photo taken on floor


################ Logic to find the match between two images ###################

#sift object

sift = cv2.xfeatures2d.SIFT_create()

#reference image key points detecting
keyPoint1, descriptor1 = sift.detectAndCompute(ref_image, None)

#main image keypoint detecting
keyPoint2, descriptor2 = sift.detectAndCompute(main_image, None)

#Descriptors matching of main image with reference image(Brute force approach)
bf_obj = cv2.BFMatcher()

#Descriptors are matched
match = bf_obj.knnMatch(descriptor1, descriptor2, k=2)


#calculation of close match from the 2 images.
closeMatch = []

for a,b in match:
    if a.distance < 0.7 * b.distance:
        closeMatch.append(a)

#check whether close Match of atleast 10 was achieved
if len(closeMatch) > 10:

#Then find the positions
    source_points = np.float32([keyPoint1[a.queryIdx].pt for a in closeMatch]).reshape(-1, 1, 2)
    destination_points = np.float32([keyPoint2[a.trainIdx].pt for a in closeMatch]).reshape(-1, 1, 2)

    
    #return mask which specifies the inside and outside points of the images.
    M, mask = cv2.findHomography(source_points, destination_points, method=cv2.RANSAC, ransacReprojThreshold=5.0) 

    # creates a thin silver line around the QR Code pattern
    height, width = ref_image.shape
    corners = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)

    #perspective transform will  help  find the pattern present in image 1 that's also present in other image. 
    transformCorners = cv2.perspectiveTransform(corners, M)

    #polylines will help in circling the pattern of image  
    main_image = cv2.polylines(main_image, [np.int32(transformCorners)], True, (255, 255, 255), 2, cv2.LINE_AA) 

#Else If matches is less than 10
else:
    print("Insufficient matches found to calculate the QR pattern")



############Calculate Pose of the  Camera and drawing Cube##############

#Values of camera is set
#2448*3264(dimension of main_image)
camera_values = camera_calibration((2448, 3264))

#Length, breadth & height of the cube will be set 
box = cube_points([0, 0, 0.1], 0.1) 

#camera's matrix is calculated by help of mask ( closeMatch(M) from the homography and position of pattern image)
camera_pos1 = Camera(np.hstack((camera_values, np.dot(camera_values, np.array([[0], [0], [-1]])))))
camera_pos2 = Camera(np.dot(M, camera_pos1.P))
A = np.dot(linalg.inv(camera_values), camera_pos2.P[:, :3])
A = np.array([A[:, 0], A[:, 1], np.cross(A[:, 0], A[:, 1])]).T
camera_pos2.P[:, :3] = np.dot(camera_values, A)

#how the camera is kept while taking the picture is found by using the camera matrix.
#In other words camera's co-ordinates are calculated..
camera_cube = camera_pos2.project(homogeneous_coordinates(box))  


###########Plot the matches along with cube#####################

pattern = np.array(ref_image)
carpet_image = np.array(main_image)

imshow(carpet_image)

plot(camera_cube[0,:],camera_cube[1,:],linewidth=6)

axis('off')

show()

####### End of the Program #########
