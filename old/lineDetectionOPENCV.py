import numpy as np
import math
import cv2

pic = cv2.imread("test2.jpg")
w = pic.shape[0]
h = pic.shape[1]

gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(gray)
pic2 = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
hsv = cv2.cvtColor(pic2, cv2.COLOR_BGR2HSV)
ROI = hsv[0:int(h / 2), 0:w]

low_white = np.array([0, 0, 170])
high_white = np.array([255, 255, 255])

mask = cv2.inRange(ROI, low_white, high_white)
# Length for lines to find
minLineLength = 150
maxLineGap = 40
lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 10, minLineLength, maxLineGap)
print(lines)

for i in range(len(lines)):
    x1, y1, x2, y2 = lines[i][0]
    cv2.line(pic, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('original', pic)
cv2.imshow('dead', mask)
cv2.waitKey(0)
