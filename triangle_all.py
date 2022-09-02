# -*- coding: utf-8 -*-
import os
import random

import cv2 as cv
import numpy as np

# SETUP #
imageSize = 256 # nr of blocks per row / column
nrOfImagesPerClass = 25000 # number of images to be generated
sizeMin = int(16  * imageSize / 256)
sizeMax = int(35  * imageSize / 256)
MIN_ANGLE = 25
MAX_ANGLE = 20
ANGLE_EPS = 8
MIN_ANGLE_DIFF = 25
MAX_ANGLE_DIFF = 35
MIN_OFFSET = 28 * imageSize / 256
MAX_OFFSET = 56 * imageSize / 256
path = os.path.join(f'generated_images{imageSize}x{imageSize}', 'kanizsa_triangle_all') # place where generated images are stored
##


def distOk2pts(p1, p2, radius):
    vec = p2 - p1
    dist = np.linalg.norm(vec)

    return (dist > radius * np.sqrt(8) + 2)
  

def distOk(p1, p2, p3, radius):
    return distOk2pts(p1, p2, radius) and distOk2pts(p2, p3, radius) and distOk2pts(p1, p3, radius)


def rotateCorners(corners, offset, angle):    
    c = np.cos(np.deg2rad(angle))
    s = np.sin(np.deg2rad(angle))
    rotMat = np.matrix([[c, s], [-s, c]])
    
    for idx in range(len(corners)):
        corners[idx] = (offset + np.dot(rotMat, corners[idx])).astype(int) 


for sub in ["yes", "no"]:
    currp = os.path.join(path, sub)    
    os.path.exists(currp) or os.makedirs(currp)


for i in range(2):
    imgNr = 0    
    
    while imgNr < nrOfImagesPerClass:
        img = np.zeros((imageSize, imageSize, 3), np.uint8)
        size = int(random.uniform(0, 1) * (sizeMax - sizeMin) + sizeMin)
        
        while True:
            posRand1 = (np.array([random.uniform(0, 1) for _ in range(2)]) * (imageSize - 2.5 * size) + 1.25 * np.array([size, size])).astype(int)
            posRand2 = (np.array([random.uniform(0, 1) for _ in range(2)]) * (imageSize - 2.5 * size) + 1.25 * np.array([size, size])).astype(int)
            posRand3 = (np.array([random.uniform(0, 1) for _ in range(2)]) * (imageSize - 2.5 * size) + 1.25 * np.array([size, size])).astype(int)
                        
            if (distOk(posRand1, posRand2, posRand3, size)):
                a = np.linalg.norm(posRand1 - posRand2)
                b = np.linalg.norm(posRand2 - posRand3)
                c = np.linalg.norm(posRand3 - posRand1)
                
                gamma = np.arccos(((c ** 2) - (a ** 2) - (b ** 2)) / (-2 * a * b))
                beta = np.arcsin(np.sin(gamma) * b / c)
                alpha = np.arcsin(np.sin(gamma) * a / c)
                
                alpha = np.rad2deg(alpha)
                beta = np.rad2deg(beta)
                gamma = np.rad2deg(gamma)
                
                vec = posRand2 - posRand1
                lenLine = np.linalg.norm(vec)
                angle = np.arctan2(vec[1], vec[0])
                
                if ((abs(alpha) > MIN_ANGLE) and (abs(beta) > MIN_ANGLE) and (abs(gamma) > MIN_ANGLE)):
                    break
                
        img = cv.circle(img, tuple(posRand1), size, (255, 255, 255), -1)
        img = cv.circle(img, tuple(posRand2), size, (255, 255, 255), -1)
        img = cv.circle(img, tuple(posRand3), size, (255, 255, 255), -1)
        
        triangle = np.array([posRand1, posRand2, posRand3], np.int32)
        img = cv.fillConvexPoly(img, triangle, (0, 0, 0)) 
           
        l1 = posRand1[1] - size - 1 
        l2 = posRand1[1] + size + 1
        l3 = posRand1[0] - size - 1
        l4 = posRand1[0] + size + 1
        
        modType = random.uniform(0, 1)
        
        randOffset = random.uniform(0, 1) * (MAX_OFFSET - MIN_OFFSET) + MIN_OFFSET
        roi = np.copy(img[l1 : l2, l3 : l4])
        randSign = -1 if random.uniform(0, 1) < 0.5 else 1
        translVector = (randSign * (posRand2 - posRand3)).astype(float)
        translVector /= np.linalg.norm(translVector)
        translVector *= randOffset
        translVector = translVector.astype(int)
        
        try:            
            dummy = img[l1 + translVector[1] : l2 + translVector[1], l3 + translVector[0] : l4 + translVector[0]] # hack to force the same prerequisites for both sets
        except:
            continue
                
        if (i == 1):   
            if modType < (1/3): # offset        
                img[l1 : l2, l3 : l4] = np.zeros_like(img[l1 : l2, l3 : l4])
            
                try:
                    if (not np.equal(img[l1 + translVector[1] : l2 + translVector[1], l3 + translVector[0] : l4 + translVector[0]], np.zeros_like(roi)).all()):
                        continue;  
                        
                    img[l1 + translVector[1] : l2 + translVector[1], l3 + translVector[0] : l4 + translVector[0]] = roi
                except:
                    continue;
            elif modType < (2/3): # angle            
                randAngleDiff = random.uniform(0, 1) * (MAX_ANGLE_DIFF - MIN_ANGLE_DIFF) + MIN_ANGLE_DIFF
                angle = np.deg2rad(randAngleDiff / 2)                 
                (h, w) = roi.shape[:2]
                center = (w / 2, h / 2)
                rotMat = cv.getRotationMatrix2D(center, randAngleDiff, 1)
                rotMat2 = cv.getRotationMatrix2D(center, -randAngleDiff, 1)
                img[l1 : l2 , l3 : l4] &= cv.warpAffine(roi, rotMat, (h, w)) & cv.warpAffine(roi, rotMat2, (h, w))
            else: # rotation
                randAngle = random.uniform(0, 1) * (MAX_ANGLE - ANGLE_EPS) + ANGLE_EPS
                (h, w) = roi.shape[:2]
                center = (w / 2, h / 2)
                randSign = -1 if random.uniform(0, 1) < 0.5 else 1
                rotMat = cv.getRotationMatrix2D(center, randSign * randAngle, 1)
                img[l1 : l2 , l3 : l4] = cv.warpAffine(roi, rotMat, (h, w))
            
        imgFileName = os.path.join(path, ("no" if (i == 1) else "yes"), f"{imgNr}.png")
        print(imgFileName)
        cv.imwrite(imgFileName, img, [cv.IMWRITE_PNG_COMPRESSION, 0])
        imgNr += 1
