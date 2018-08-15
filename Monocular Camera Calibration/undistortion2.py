# -*- coding: utf-8 -*-
"""

Created on Fri May 25 20:48:25 2018
@author: Amos
reference: Camera Calibration 
          ("http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials
          /py_calib3d/py_calibration/py_calibration.html#calibration")
          
"""
import numpy as np
import cv2
import os

#basic setting
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objpoints = []
imgpoints = []
        
# Take one image as an example to undistort

path = input('Please input the path of the image needed to be calibrated:\n')

if os.path.exists(path) == True:
    img = cv2.imread(path)
    print('Image has been read successfully.\n')
else:
    print('Error: image not found.\n')
    exit()

#get the parameters
parameters = np.load("parameters.npz")
mtx = parameters["mtx"]
dist = parameters["dist"]
rvecs = parameters["rvecs"]
tvecs = parameters["tvecs"]

h,  w = img.shape[:2] #get the size of images
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
# write the image
new_path='calibresult2/new_'+path[11:]
cv2.imwrite(new_path,dst)
if cv2.imwrite(new_path,dst) == True:
    print('New calibrated image has been saved to the folder calibresult2.\n')

#re-projection error
mean_error = 0
points = np.load("points.npz")
objpoints = points["objpoints"]
imgpoints = points["imgpoints"]
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist) #transform the object point to image point
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print ('total error: ', mean_error/len(objpoints))
