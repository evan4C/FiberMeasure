import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

"""
在实际计算中，一般情况下会根据输入的theta与lambd的不同，得到一系列的Gabor的滤波器组合，
然后把它们的结果相加输出，得到最终的输出结果，在纹理提取，图像分割、纹理分类中特别有用，
Gabor滤波器的任意组合提供了非常强大的图像分类能力，被认为是最接近于现代深度学习方式进行图像分类的算法之一。
Gabor滤波器应用也非常广泛，几乎从图像处理、分割、分类、对象匹配、人脸识别、文字OCR等领域都有应用。
"""

def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray


def build_filters():
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 4):  # gabor方向，0°，45°，90°，135°，共四个
        kernel = cv2.getGaborKernel((11, 11), sigma=1.5, theta=theta, lambd=3, gamma=1.2, psi=0, ktype=cv2.CV_32F)

        # kernel normalization
        kernel /= 1.5 * kernel.sum()
        filters.append(kernel)
    return filters


def process(img, filters):
    accu = np.zeros_like(img)
    for kern in filters:
        f_img = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.add(accu, f_img, out=accu)
    accu = accu / accu.max() * 255
    return accu


# 加载图片并转换为0-1之间数值float32格式
img = cv2.imread('test.jpg').astype(np.float32)
gray = BGR2GRAY(img).astype(np.float32)

filters = build_filters()
out = process(gray, filters)
cv2.imshow("result", out)
cv2.waitKey(0)
