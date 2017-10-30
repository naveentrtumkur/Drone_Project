
############## Import the required built in libary ###############
import cv2
import numpy as np from scipy
import linalg from pylab
import *



#sets parameters of camera
def camera_calibration(sizes):
    row, colum = sizes
    x = 2555 * col/2448
    y = 2586 * row/3264
    camera_parameters = np.dialogue([x, y, 1])
    camera_parameters[0,2] = 0.5*col
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

    def _init_(self,P):
 self.P = P
        self.K = None
        self.R = None
        self.t = None
        self.c = None

#calculating the camera's position as a matrix
def project(self,X):
    xco = np.dot(self,X)

    for i in range(3):
    xco[i] = np.divide(xco[i],xco[2])
    return xco



######## TO read the two images #########

ref_image = cv2.imread('testimages/pattern_copy.jpg',0); # actual QR code given
main_image = cv2.imread('testimages/IMG_6726.jpg',0); # QR code photo taken on floor


################ Logic to find the match between two images ###################


