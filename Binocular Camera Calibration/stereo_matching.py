# -*- coding: utf-8 -*-
"""

Created on Mon Aug 13 12:39:11 2018
@author: Amos
"""

import numpy as np
import cv2


if __name__ == '__main__':
    print('loading images...')
    imgL = cv2.imread('rectifyresult/left03(rectified).jpg')  
    imgR = cv2.imread('rectifyresult/right03(rectified).jpg')

    # SGBM Parameters -----------------
    window_size = 3 
    left_matcher = cv2.StereoSGBM_create(
            minDisparity = 0,
            numDisparities = 16, 
            blockSize = 3,
            P1 = 8*1*window_size** 2,    
            P2 = 32*1*window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32,
            preFilterCap = 63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    
    # use WLS_Filter to do filtering 
    # FILTER Parameters
    lmbda = 8000
    sigma = 1.0
    visual_multiplier = 1.0
 
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # compute the disparities and convert the resulting images to int16 format
    print('computing disparity...')
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)

    # normalize the depth map and show it
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    cv2.imshow("left", imgL)
    cv2.imshow('Disparity Map', filteredImg)
    cv2.waitKey()
    cv2.destroyAllWindows()