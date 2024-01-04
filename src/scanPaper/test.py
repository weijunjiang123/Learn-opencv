import cv2
from transform import four_point_transform
import numpy as np
import imutils

# 读取图片
image = cv2.imread('./images/paper0.jpg')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 0, 50)

print("STEP 1: 边缘检测")
# cv2.imshow("Image", image)
# cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# 从大到小排序
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
for c in cnts:
    # 计算轮廓近似
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # 近似轮廓有四个顶点
    if len(approx) == 4:
        screenCnt = approx
        break
print("STEP 2: 获取轮廓")
# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# 二值化
ref = cv2.threshold(warped, 80, 255, cv2.THRESH_BINARY)[1]
# 展示结果
print("STEP 3: 获取文本")
cv2.imshow("Scanned", imutils.resize(ref, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()