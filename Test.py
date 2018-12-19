import ImageSet
import createImageSet
import numpy as np
import LDA
import KNN


def compare(disc_set, Train_LDA, W, label, testNum, classNum,k):
    '''
    计算一个测试集的正确率
    :param disc_set: PCA投影空间
    :param Train_LDA: 最终训练集
    :param W: LDA投影空间
    :param label: 训练集标签矩阵
    :param testNum: 每类中用来测试的图像数量
    :param testInClass: 当前样本类的第几个图像
    :param k: 前k个距离中最小元素所属分类
    :return: 当前样本类的正确率
    '''
    # testImg=ImageSet.HistogramEqualization(testImg)
    # testImg=np.reshape(testImg,(-1,1))

    # disc_set,disc_value=LDA.pca(dataMat,50)
    # redVects,lowMat=LDA.lda(dataMat,label,50,15,11,165)#LDA投影空间，最终的训练集
    # print(' 变换矩阵维度',redVects.shape)
    # print('测试图像维度',testImg.shape)
    # redVects=np.reshape(redVects, (-1, 1))
    #print('redV', W.shape)
    testImgSet=createImageSet.createTestMat('Yale',classNum,testNum,100*100)
    #testImgSet = createImageSet.createTestMat('Yale', testInClass, testNum, testInClass, 100 * 100)
    testImgSet = disc_set.T.dot(testImgSet)
    testImgSet = W.T*(testImgSet)
    #print('testSet',testImgSet.shape)
    resVal = 0
    for res in range(testNum):
        # TrainVec = FaceVector.T * diffTrain[:, i]
        #print(Train_LDA.shape)
        #print('res',res+1)
        testRes=int(KNN.classify0(testImgSet.T[res], Train_LDA.T, label, k))
        if (testRes == classNum):
            resVal = resVal + 1
    correctCount = resVal / testNum  # 正确率
    print("correctCount", correctCount)
    return correctCount

 
def getResult(dataMat, label, PCA_dim, testNum, classNum, classInNum ):
    '''
    加载每类测试集，计算总正确率
    :param dataMat: 训练集
    :param label: 训练集标签矩阵
    :param testNum: 每类测试集的测试个数
    :param classNum: 共几类
    :param classInNum:每类有几个图
    :return:
    '''
    Count = 0
    # disc_set, disc_value ,meanFace= LDA.pca(dataMat, PCA_dim)
    disc_set,x,y=LDA.pca(dataMat,PCA_dim)
    Total=classNum*classInNum
    redVects, Train_LDA = LDA.lda(dataMat, label,PCA_dim, classNum, classInNum, Total)  # LDA投影空间，最终的训练集
    for classnum in range(1, classNum + 1):
        print('第',classnum,'类')
        Count += compare(disc_set, Train_LDA, redVects, label, testNum,classnum, 7)
    print('Final correctCount:', Count/classNum)


if __name__ == '__main__':
    PCA_dim=40 #####从22开始识别率提升为1.0
    testNum=10
    classNum=17
    classInNum=11
    Train_Total=classNum*classInNum
    dataMat, label = createImageSet.createImageMat('Yale', classNum, classInNum, Train_Total, 100 * 100)
    getResult(dataMat, label, PCA_dim, testNum , classNum ,classInNum)
    # disc_set, disc_value = LDA.pca(dataMat, 50)
    # redVects, Train_LDA = LDA.lda(dataMat, 50, 15, 11, 165)  # LDA投影空间，最终的训练集
    # testImgSet = './Yale/1/s1.bmp'
    # # testImgSet = createImageSet.createTestMat('Yale', testInClass, testNum, testInClass, 100 * 100)
    # testImgSet = ImageSet.HistogramEqualization(testImgSet)
    # testImgSet = np.reshape(testImgSet, (-1, 1))
    # testImgSet = disc_set.T.dot(testImgSet)
    # testImgSet = redVects.T.dot(testImgSet)
    # disList = []
    # testVec = np.reshape(testImgSet, (1, -1))
    # print('testVec', testVec.shape)
    #
    # for sample in Train_LDA.T:
    #     disList.append(np.linalg.norm(testVec - sample))
    # #print('disList', disList)
    # sortIndex = np.argsort(disList)
    # print(label[sortIndex[0]])