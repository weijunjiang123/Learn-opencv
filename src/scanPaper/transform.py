import numpy as np
import cv2


def order_points(pts):
    """
    对原始点进行排序
    :param pts: 原始点
    :return: 排序后的点
    """

    # 一共4个坐标点
    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下点的总和，差为右上左下的总和
    rect = np.zeros((4, 2), dtype="float32")
    # 左上角点的x+y最小，右下角点的x+y最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 右上角点的x-y最小，左下角点的x-y最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 返回排序好的四个坐标点
    return rect


def four_point_transform(image, pts) -> np.ndarray:
    """
    透视变换
    :param image: 原图像
    :param pts: 原图像中待变换区域的四个顶点
    :return: 变换后的图像
    """

    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # 计算输入的maxWidth和maxHeight值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 变换后对应坐标位置
    dat = np.float32([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]])
    # 计算变换矩阵
    m = cv2.getPerspectiveTransform(rect, dat)
    # 进行透视变换
    warped = cv2.warpPerspective(image, m, (maxWidth, maxHeight))
    return warped

# test
# if __name__ == '__main__':
#     image = cv2.imread('./images/paper.jpg')
#     # cv2.imshow('image', image)
#     pts = np.array([(73, 239), (356, 117), (475, 265), (187, 443)], dtype="float32")
#     warped = four_point_transform(image, pts)
#     cv2.imshow('warped', warped)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()




