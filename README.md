# Stereo
Monocular camera calibration, binocular camera claibration and stereo matching

## Monocular camera calibration

This part is just the same as [camera-calibration-and-image-undistortion](https://github.com/ChihaoZhang/camera-calibration-and-image-undistortion), please visit it for more detail.

## Binocular camera calibration

### stereo_calibration.py
- computes the camera matrices, distortion efficients of left and right cameras.
- obtains the transformation from left to right camera, the essential matrix, the fundamental matrix.
- take an image as an example to to recitification

### stereo_matching.py
- uses SGBM method to compute the disparity map for the rectified image

## Note
If you have any question or advice, contact with me

email: zhangchihao@zju.edu.cn

## Revised
- 31st, Oct. 2018: path error problem raised by @wyh6789
