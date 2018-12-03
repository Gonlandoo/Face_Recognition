import cv2 as cv
import numpy as np
from PIL import Image


# import matplotlib.pyplot as plt
def HistogramEqualization(Img):
    '''
    读取单张图片，均衡化
    :param Img: 需要处理的图片
    :return: 均衡化后的图像数组
    '''
    # 获取直方图
    # img = cv.imread("./Yale/1/s1.bmp")
    # lut=np.zeros(256,dtype=img.dtype)
    # color = ('r', 'g', 'b')
    # for i, col in enumerate(color):
    #     hist = cv.calcHist([img], [i], None, [256], [0, 256])
    #     plt.plot(hist, color=col)
    #     plt.xlim([0, 256])
    # plt.show()
    # hist=cv.calcHist([img],[0],None,[256],[0.0,255.0])

    # img = cv.imread(Img)
    img = cv.imread(Img, cv.IMREAD_GRAYSCALE)  # 读取彩色图像为灰度图




    # # print('原始图像数组',img)
    # # lut = np.zeros(256, dtype=img.dtype)  # 创建空的查找表
    # hist, bins = np.histogram(img.flatten(), 256, [0, 256])  # 获取直方图
    # cdf = hist.cumsum()  # 计算累积直方图
    # cdf_m = np.ma.masked_equal(cdf, 0)  # 除去直方图中的0值
    # cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # 将掩模处理掉的元素补为0
    #
    # # 直方图均衡化
    # result = cdf[img]

    result = cv.equalizeHist(img)



    # print('均衡化后图像数组', result)
    # cv.imshow("NumPyLUT", result)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # print('均衡化后的图像矩阵维度',result.shape)

    return result
# if __name__ == '__main__':
#    HistogramEqualization("./Yale/1/s1.bmp")
