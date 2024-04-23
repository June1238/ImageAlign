# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 读取原始图片
    face = cv2.imread('imgPackage/DownLoadBD.jpg')
    # 转换为灰度图
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # 实例化sift对象
    sift = cv2.SIFT_create()
    '''sift特征检测器来查找图像中的关键点（也称为特征点）和计算它们的描述子
            每个关键点对象包含了有关关键点的信息，例如位置、尺度、方向等'''
    kps = sift.detect(gray)
    # 在原始图像上绘制出关键点
    face_sift = cv2.drawKeypoints(face, kps, face.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('face_sift', face_sift)
    cv2.waitKey(0)
    # 使用sift.compute（）计算关键点点的描述符，以便后期的特征配对
    kps, des = sift.compute(face, kps)
    # 返回kps和des的维度信息
    print(np.array(kps).shape, des.shape)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
