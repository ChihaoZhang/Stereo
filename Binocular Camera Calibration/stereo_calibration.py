# -*- coding: utf-8 -*-
"""

Created on Tue Jul 31 16:05:21 2018
@author: Amos
"""

import numpy as np
import cv2
import glob

# monocular camera calibration

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints1 = []
imgpoints2 = []

# left camera calibration
images = glob.glob('../left/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret == True:
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints1.append(corners2)
ret, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints1, gray.shape[::-1],None,None)

# right camera calibration
images = glob.glob('../right/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret == True:
        corners2=cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints2.append(corners2)
ret, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints2, gray.shape[::-1],None,None)

# binocular camera calibration
ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, mtx_l, dist_l, mtx_r, dist_r, gray.shape[::-1])

np.savez("parameters for calibration.npz",ret=ret,mtx_l=mtx_l,mtx_r=mtx_r,dist_l=dist_l,dist_r=dist_r,R=R,T=T)
np.savez("points.npz",objpoints=objpoints,imgpoints1=imgpoints1,imgpoints2=imgpoints2)

print('intrinsic matrix of left camera=\n', mtx_l)
print('intrinsic matrix of right camera=\n', mtx_r)
print('distortion coefficients of left camera=\n', dist_l)
print('distortion coefficients of right camera=\n', dist_r)
print('Transformation from left camera to right:\n')
print('R=\n', R)
print('\n')
print('T=\n', T)
print('\n')
print('Reprojection Error=\n', ret)

# stereo rectification
R1, R2, P1, P2, Q, ROI1, ROI2= cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, gray.shape[::-1], R, T, flags=0, alpha=-1)

# undistort rectifying mapping
map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, gray.shape[::-1], cv2.CV_16SC2)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, gray.shape[::-1], cv2.CV_16SC2)

# undistort the original image, take img#3 as an example
left3 = cv2.imread('../left/left03.jpg')
dst_l = cv2.remap(left3, map1_l, map2_l, cv2.INTER_LINEAR)
cv2.imwrite('rectifyresult/left03(rectified).jpg', dst_l)
if cv2.imwrite('rectifyresult/left03(rectified).jpg', dst_l)==True:
    print('rectification of left camera has been done successfully.\n')
right3 = cv2.imread('../right/right03.jpg')
dst_r = cv2.remap(right3, map1_r, map2_r, cv2.INTER_LINEAR)
cv2.imwrite('rectifyresult/right03(rectified).jpg', dst_r)
if cv2.imwrite('rectifyresult/right03(rectified).jpg', dst_r) == True:
    print('rectification of right camera has been done successfully.\n')

np.savez("parameters for rectification.npz", R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, ROI1=ROI1, ROI2=ROI2)
