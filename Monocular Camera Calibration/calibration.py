# -*- coding: utf-8 -*-
"""

Created on Mon May 21 15:26:48 2018
@author: Amos
reference: Camera Calibration 
          ("http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials
          /py_calib3d/py_calibration/py_calibration.html#calibration")

"""
import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('../../left/*.jpg') #read a series of images

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert the image to gray

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2=cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) #refine the corner locations
        imgpoints.append(corners2)

        # Draw and display the corners
        
        cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

#calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#save parameters needed in undistortion
np.savez("parameters.npz",mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs)
np.savez("points.npz",objpoints=objpoints,imgpoints=imgpoints)

print('intrinsic matrix=\n', mtx)
print('distortion coefficients=\n', dist)
print('rotation vector for each image=', *rvecs, sep = "\n")
print('translation vector for each image=', *tvecs, sep= "\n")


