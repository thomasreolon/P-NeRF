import cv2
import os, numpy as np

'''
Script to extract capera poses for each pair of cameras
Create a folder images/ and upload the images on it
N_MATCHING_POINT -> number of points to be matched for each pair of images
focal_length -> focal_length of the camera
This script assumes that all the cameras have the same parameters (the cameras are equal)
'''

N_MATCHING_POINT = 50

focal_length = 153.6

# Parameters estimated using Colmap software
intrinsic_param = np.matrix([[focal_length, 0, 64], [0, focal_length, 64], [0, 0, 1]])

images = os.listdir("images/")

imgs = []

for image in images:
    img = cv2.imread("images/" + image)
    imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# create SIFT object
sift = cv2.xfeatures2d.SIFT_create()

# detect SIFT features
keypoints = []
descriptors = []

for img in imgs:
    kp, des = sift.detectAndCompute(img,None)
    keypoints.append(kp)
    descriptors.append(des)

# create feature matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# match descriptors
matches = {}

for i, des1 in enumerate(descriptors):
    for i2 in range(0, len(descriptors)):
        matches[(i,i2)] = bf.match(des1,descriptors[i2])

relative_poses = {}

for key in matches:
    a, b = key[0], key[1]
    match = matches[key]
    matches[key] = sorted(match, key = lambda x:x.distance)
    matched_img = cv2.drawMatches(imgs[a], keypoints[a], imgs[b], keypoints[b], matches[key][:N_MATCHING_POINT], imgs[b], flags=2)

    pts1 = []
    pts2 = []

    for m in matches[key][:N_MATCHING_POINT]:
        pts1.append(keypoints[a][m.queryIdx].pt)
        pts2.append(keypoints[b][m.trainIdx].pt)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    # compute fundamental matrix
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=intrinsic_param, distCoeffs=None)
    pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=intrinsic_param, distCoeffs=None)

    # compute essential matrix
    E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.),
                                       method=cv2.RANSAC, prob=0.999, threshold=3.0 / focal_length)

    points, R_est, t_est, mask_pose = cv2.recoverPose(E, pts_l_norm, pts_r_norm)

    cam_pose = -np.transpose(R_est)@t_est

    print((a,b)) # referred images

    R_t = np.concatenate((R_est, t_est), axis = 1)
    R_t = np.concatenate((R_t, np.matrix([0,0,0,1])), axis = 0)

    print(R_t) # print pose matrix of pair (a,b)

    # show the image
    cv2.imshow('image', matched_img)
    # save the image
    cv2.imwrite("matched_images.jpg", matched_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
