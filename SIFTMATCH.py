import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
https://www.jianshu.com/p/9b0b3b4a19de
'''

def processData(imgPth1,imgPth2):
    img1 = cv2.imread(imgPth1)
    gray1 = cv2.cvtColor(imgPth1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
    img2 = cv2.imread(imgPth2)
    gray2 = cv2.cvtColor(imgPth2, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
    hmerge = np.hstack((gray1, gray2))  # 水平拼接
    cv2.imshow("gray", hmerge)  # 拼接显示为gray
    cv2.waitKey(0)


def detectFPoint(imgPth1,imgPth2):
    sift = cv2.SIFT_create()

if __name__ == '__main__':
    imgname1 = "./imgPackage/Img1.jpg"
    imgname2 = "./imgPackage/Img2.jpg"
    img1 = cv2.imread(imgname1, cv2.IMREAD_GRAYSCALE)  # 查询图片
    img2 = cv2.imread(imgname2, cv2.IMREAD_GRAYSCALE)  # 训练(目标)图片

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 使用SIFT找到关键点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 使用FLANN matcher进行匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    #可以尝试 调整不同的关键点匹配算法--
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    matchesMask = [[0,0] for i in range(len(matches))]

    # 应用比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good_matches.append([m])

    # 绘制匹配结果
    img_matches = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)
    kp_matched_kpts = np.float32([keypoints1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sensed_matched_kpts = np.float32([keypoints2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 使用两张图对应上的关键点，得到变换矩阵
    H, status = cv2.findHomography(sensed_matched_kpts, kp_matched_kpts, cv2.RANSAC, 5.0)
    warped_image = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

    dim = (800,600)
    resized_image = cv2.resize(warped_image, dim, interpolation=cv2.INTER_AREA)
    # 显示图片
    cv2.imshow('Matches', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
