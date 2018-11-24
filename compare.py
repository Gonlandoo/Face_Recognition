import numpy as np
import createImageSet
import ImageSet
import PCA


def compare(dataMat, testImg, label):
    '''
    :param dataMat:样本矩阵
    :param testImg: 测试图像矩阵,最原始形式（未均衡化处理）
    :param label: 标签矩阵
    :return: 结果文件名
    '''
    testImg = ImageSet.HistogramEqualization(testImg)
    testImg = np.reshape(testImg, (-1, 1))
    lowMat, redVects = PCA.pca(dataMat, 0.99)
    print(' 变换矩阵维度', redVects.shape)
    print('测试图像维度', testImg.shape)
    # redVects=np.reshape(redVects, (-1, 1))
    testImg = redVects.T * testImg  #########
    print('样本变化后的维度', testImg.shape)
    disList = []
    testVec = np.reshape(testImg, (1, -1))
    print('testVec', testVec.shape)

    for sample in lowMat.T:
        disList.append(np.linalg.norm(testVec - sample))
    print('disList', disList)
    sortIndex = np.argsort(disList)
    return label[sortIndex[0]]


def predict(dirName, testFileName):
    '''
    :param dirName: 包含训练数据集的文件夹路径
    :param testFileName: 测试图像文件名
    :return: 预测结果
    '''
    # testImg=cv2.imread(testFileName)
    # cv2.imshow("testImg",testImg)
    dataMat, label = createImageSet.createImageMat(dirName, 15, 11, 165, 100 * 100)
    print('加载数据集矩阵', dataMat)
    print('加载图片标签', label)
    ans = compare(dataMat, testFileName, label)
    return ans
# if __name__ == '__main__':
#      result=predict('Yale','./Yale/2/s1.bmp')
#      print('result',result)

# def judge(judgeImg,FaceVector,meanImg,diffMat):
#     diff=judgeImg.T-meanImg
#     weiVec=FaceVector.T*diff
#     res=0
#     resVal=float("inf")
#     for i in range(1,165):
#         TrainVec=FaceVector.T*diffMat[:,i]
#         if (array(weiVec-TrainVec)**2).sum()<resVal：
