# -*- coding: utf-8 -*-
import os
import random

import cv2 as cv
import numpy as np

# SETUP #
imageSize = 256 # nr of blocks per row / column
nrOfTrainImagesPerClass = 15000 # number of images to be generated
nrOfValImagesPerClass = 5000 # number of images to be generated
nrOfTestImagesPerClass = 5000 # number of images to be generated
sizeMin = int(16  * imageSize / 256)
sizeMax = int(35  * imageSize / 256)
MAX_ANGLE = 20
ANGLE_EPS = 8
MIN_ANGLE = 25
path = os.path.join(f'generated_images{imageSize}x{imageSize}', 'kanizsa_triangle_rot') # place where generated images are stored
##

test_path = os.path.join(path, "test")
train_path = os.path.join(path, "train")
val_path = os.path.join(path, "val")

nrOfImagesPerClass = nrOfTrainImagesPerClass + nrOfValImagesPerClass + nrOfTestImagesPerClass


def currPhase(currNr):
    if (currNr < nrOfTrainImagesPerClass):
        res = "train"
    elif (currNr < nrOfTrainImagesPerClass + nrOfValImagesPerClass):
        res = "val"
    else:
        res = "test"
    
    return res
    

def distOk2pts(p1, p2, radius):
    vec = p2 - p1
    dist = np.linalg.norm(vec)

    return (dist > radius * np.sqrt(8) + 2)


def distOk(p1, p2, p3, radius):
    return distOk2pts(p1, p2, radius) and distOk2pts(p2, p3, radius) and distOk2pts(p1, p3, radius)


def randRotatePos(img, posRand):    
    randAngle = random.uniform(0, 1) * (MAX_ANGLE - ANGLE_EPS) + ANGLE_EPS
    roi = img[posRand[1] - size - 1 : posRand[1] + size + 1, posRand[0] - size - 1 : posRand[0] + size + 1]                    
    (h, w) = roi.shape[:2]
    center = (w / 2, h / 2)
    randSign = -1 if random.uniform(0, 1) < 0.5 else 1
    rotMat = cv.getRotationMatrix2D(center, randSign * randAngle, 1)
    img[posRand[1] - size - 1 : posRand[1] + size + 1, posRand[0] - size - 1 : posRand[0] + size + 1] = cv.warpAffine(roi, rotMat, (h, w))


for p in [test_path, train_path, val_path]:
    for sub in ["yes", "no"]:
        currp = os.path.join(p, sub)    
        os.path.exists(currp) or os.makedirs(currp)
        

for i in range(2): 
    for imgNr in range(nrOfImagesPerClass):
        img = np.zeros((imageSize, imageSize, 3), np.uint8)
        size = int(random.uniform(0, 1) * (sizeMax - sizeMin) + sizeMin)
        
        while True:
            posRand1 = (np.array([random.uniform(0, 1) for _ in range(2)]) * (imageSize - 2.5 * size) + 1.25 * np.array([size, size])).astype(int)
            posRand2 = (np.array([random.uniform(0, 1) for _ in range(2)]) * (imageSize - 2.5 * size) + 1.25 * np.array([size, size])).astype(int)
            posRand3 = (np.array([random.uniform(0, 1) for _ in range(2)]) * (imageSize - 2.5 * size) + 1.25 * np.array([size, size])).astype(int)
            posRand1 = [int(posRand1[0]), int(posRand1[1])]
            vec = posRand2 - posRand1
            lenLine = np.linalg.norm(vec)
            angle = np.arctan2(vec[1], vec[0])
            
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
        
        if (i == 1):
            randRotatePos(img, posRand1)
            
            #if (random.uniform(0, 1) < 0.5):
            #    randRotatePos(img, posRand2)
            
            #if (random.uniform(0, 1) < 0.5):
            #    randRotatePos(img, posRand3)
            
        imgFileName = os.path.join(path, currPhase(imgNr), ("no" if (i == 1) else "yes"), f"{imgNr}.png")
        print(imgFileName)
        cv.imwrite(imgFileName, img, [cv.IMWRITE_PNG_COMPRESSION, 0])