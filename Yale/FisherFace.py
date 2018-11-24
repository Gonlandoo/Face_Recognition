# encoding=utf-8
import numpy as np
import cv2
import os


class FisherFace(object):
    def __init__(self, threshold, k, dsize):
        self.threshold = threshold  # 阈值，暂未使用
        self.k = k  # 指定投影w的个数
        self.dsize = dsize  # 统一尺寸大小

    def loadImg(self, fileName, dsize):
        '''
        载入图像，灰度化处理，统一尺寸，直方图均衡化
        :param fileName: 图像文件名
        :param dsize: 统一尺寸大小。元组形式
        :return: 图像矩阵
        '''
        img = cv2.imread(fileName)
        retImg = cv2.resize(img, dsize)
        retImg = cv2.cvtColor(retImg, cv2.COLOR_RGB2GRAY)
        retImg = cv2.equalizeHist(retImg)
        # cv2.imshow('img',retImg)
        # cv2.waitKey()
        return retImg

    def createImgMat(self, dirName):
        '''
        生成图像样本矩阵，组织形式为行为属性，列为样本
        :param dirName: 包含训练数据集的图像文件夹路径
        :return: 包含样本矩阵的列表，标签列表
        '''
        dataMat = np.zeros((15, 1))
        label = []
        dataList = []
        for parent, dirnames, filenames in os.walk(dirName):
            # print parent
            # print dirnames
            # print filenames
            # index = 0
            for dirname in dirnames:
                for subParent, subDirName, subFilenames in os.walk(parent + '/' + dirname):
                    for index, filename in enumerate(subFilenames):
                        img = self.loadImg(subParent + '/' + 's' + filename, self.dsize)
                        tempImg = np.reshape(img, (-1, 1))
                        if index == 0:
                            dataMat = tempImg
                        else:
                            dataMat = np.column_stack((dataMat, tempImg))
                dataList.append(dataMat)
                label.append(subParent)
        return dataList, label

    def LDA(self, dataList, k):
        '''
        多分类问题的线性判别分析算法
        :param dataList: 样本矩阵列表
        :param k: 投影向量k的个数
        :return: 变换后的矩阵列表和变换矩阵
        '''
        n = dataList[0].shape[0]
        W = np.zeros((n, self.k))
        Sw = np.zeros((n, n))
        Sb = np.zeros((n, n))
        u = np.zeros((n, 1))
        N = 0
        meanList = []
        sampleNum = []

        for dataMat in dataList:
            meanMat = np.mat(np.mean(dataMat, 1)).T
            meanList.append(meanMat)
            sampleNum.append(dataMat.shape[1])

            dataMat = dataMat - meanMat
            sw = dataMat * dataMat.T
            Sw += sw
        print('Sw的维度', Sw.shape)

        for index, meanMat in enumerate(meanList):
            m = sampleNum[index]
            u += m * meanMat
            N += m
        u = u / N
        print('u的维度', u.shape)

        for index, meanMat in enumerate(meanList):
            m = sampleNum[index]
            sb = m * (meanMat - u) * (meanMat - u).T
            Sb += sb
        print('Sb的维度', Sb.shape)

        eigVals, eigVects = np.linalg.eig(np.mat(np.linalg.inv(Sw) * Sb))
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[::-1]
        eigValInd = eigValInd[:k]  # 取出指定个数的前k大的特征值
        print('选取的特征值', eigValInd.shape)
        eigVects = eigVects / np.linalg.norm(eigVects, axis=0)  # 归一化特征向量
        redEigVects = eigVects[:, eigValInd]
        print('变换矩阵维度', redEigVects.shape)

        transMatList = []
        for dataMat in dataList:
            transMatList.append(redEigVects.T * dataMat)
        return transMatList, redEigVects

    def compare(self, dataList, testImg, label):
        '''
        比较函数，这里只是用了最简单的欧氏距离比较，还可以使用KNN等方法，如需修改修改此处即可
        :param dataList: 样本矩阵列表
        :param testImg: 测试图像矩阵，最原始形式
        :param label: 标签矩阵
        :return: 与测试图片最相近的图像文件夹，也就是类别
        '''
        #testImg = cv2.resize(testImg, self.dsize)
        #testImg = cv2.cvtColor(testImg, cv2.COLOR_RGB2GRAY)
        testImg = cv2.imread(testImg, cv2.IMREAD_GRAYSCALE)  # 读取彩色图像为灰度图
        testImg = np.reshape(testImg, (-1, 1))
        transMatList, redVects = fisherface.LDA(dataList, self.k)
        testImg = redVects.T * testImg
        #print('检测样本变换后的维度', testImg.shape)
        disList = []
        testVec = np.reshape(testImg, (1, -1))
        sumVec = np.mat(np.zeros((self.dsize[0] * self.dsize[1], 1)))
        for transMat in transMatList:
            for sample in transMat.T:
                disList.append(np.linalg.norm(testVec - sample))
        #print(disList)
        sortIndex = np.argsort(disList)
        return label[sortIndex[0] / 14]

    def predict(self, dirName, testFileName):
        '''
        预测函数
        :param dirName: 包含训练数据集的文件夹路径
        :param testFileName: 测试图像文件名
        :return: 预测结果
        '''
        testImg = cv2.imread(testFileName)
        dataMat, label = self.createImgMat(dirName)
        print('加载图片标签', label)
        ans = self.compare(dataMat, testImg, label)
        return ans


if __name__ == "__main__":
    fisherface = FisherFace(15, 11, (100, 100))
    ans = fisherface.predict('./Yale', './Yale/1/s1.bmp')
    print(ans)