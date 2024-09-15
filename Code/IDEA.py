from PIL import ImageChops
import numpy as np
import cv2
from PIL import Image

def IDEA(original,distored,th=100):

    original = original.resize(original.size)
    distored = distored.resize(original.size)

    diff_od = ImageChops.difference(original, distored)

    diff_od = np.array(diff_od)
    diff_od = cv2.cvtColor(diff_od, cv2.COLOR_RGB2BGR)

    th = 100
    diff_od = np.where(diff_od <=th ,0,diff_od)
    diff_od = np.where(diff_od>th,255,diff_od)
    diff_od = np.where(diff_od==255,1,diff_od)
    score_od = np.sum(diff_od)/(original.size[0]*original.size[1])

    return score_od

def P2P(original,distored):
    
    h, w, c = original.shape
    gray_origin = cv2.resize(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), (w,h))
    gray_distored = cv2.resize(cv2.cvtColor(distored, cv2.COLOR_BGR2GRAY), (w,h))

    corners_origin = cv2.goodFeaturesToTrack(gray_origin, maxCorners=300, qualityLevel=0.3, minDistance=7)
    corners_distored, dist_status, dist_errors = cv2.calcOpticalFlowPyrLK(gray_origin, gray_distored, corners_origin, None)


    distances_distored = []
    for i in range(len(corners_origin)):
        x1, y1 = corners_origin[i].ravel()
        x2, y2 = corners_distored[i].ravel()
        distance_distored = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
        distances_distored.append(distance_distored)

    # 이동 벡터의 크기 평균 계산
    if len(distances_distored) != 0:
        Distored_mean_distance = sum(distances_distored) / len(distances_distored)
    
    return Distored_mean_distance

if __name__ == "__main__":
    original = Image.open('origin/000000_0_0_0_0_0.jpg')
    distored = Image.open('Distortion/2.Shoulder_50_down/000000_0_-50_0_0_0.jpg')
    IDEA_score = IDEA(original,distored)
    print('IDEA Score : %s'%IDEA_score)

    origin = cv2.imread('origin/000000_0_0_0_0_0.jpg')
    distored = cv2.imread('Distortion/2.Shoulder_50_down/000000_0_-50_0_0_0.jpg')
    P2P_score = P2P(origin,distored)
    print('P2P Score : %s'%P2P_score)
