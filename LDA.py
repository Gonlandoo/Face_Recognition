import numpy as np
import numpy.linalg as lp
import math


def pca(dataMat, dimNum):
    '''
    对多类数据集ImageSize*ImageNum,降维至(k+1)*ImageNum
    :param dataMat: 数据集
    :param dimNum: 降维后的维数
    :return: 特征值，特征向量,均值矩阵
    '''
    ImgNum = dataMat.shape[1]
    ImgInfo = dataMat.shape[0]
    # 去中心化
    meanMat = np.mean(dataMat, axis=1)  # 对行求均值
    diffMat = dataMat - meanMat  # 偏差矩阵(去中心化后的特征矩阵)
    # 计算协方差矩阵
    eigVals, eigVects = np.linalg.eig(diffMat.T * diffMat / (ImgNum - 1))
    eigSortIndex = np.argsort(-eigVals)  # 按行降序排列(一行为一个特征),返回一个序列
    V = eigVects[:, eigSortIndex[0:dimNum]]  # 取特征向量前dimNum维
    # 取前dimNum大特征值
    for i in range(dimNum):
        S = eigVals[eigSortIndex[:i]]

    # 通过奇异值分解（小样本问题，因为样本维数远大于样本数），得到协方差矩阵特征向量
    disc_value = S
    disc_value = np.reshape(disc_value, (1, -1))
    # print('discvalue',disc_value.shape)
    disc_set = np.zeros((ImgInfo, dimNum))

    Train_SET = diffMat / math.sqrt(ImgNum - 1)
    # 得到协方差矩阵特征向量
    for j in range(dimNum - 1):
        disc_value = disc_value.real
        temp = Train_SET * V[:, j]
        temp = np.reshape(temp, (10000))
        disc_set[:, j] = (1 / math.sqrt(disc_value[:, j])) * temp
    # print('PCA变换特征向量',disc_set.shape)
    return disc_set, disc_value


# 奇异矩阵求逆
def inv(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")

    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]


def lda(dataMat, label, dimNum, classNum, classInNum, ImgNum):
    disc_set, disc_value = pca(dataMat, dimNum)
    # print('ds',disc_set.shape)
    # print('dataMat',dataMat.shape)
    Train_PCA = disc_set.T * (dataMat)  # 训练样本PCA投影集
    # print('Train_PCA',Train_PCA.shape)
    Mean_all = np.mean(Train_PCA, axis=1)  # 所有训练样本的均值
    Mean_classMat = np.zeros((dimNum, classNum))  # 每类样本均值矩阵
    classMat = np.zeros((dimNum, classInNum))  # 每类样本矩阵
    # index=0
    # Swi = np.zeros((dimNum, ImgNum))  # 类内散度矩阵
    # for classnum in range(classNum):
    #     for classinNum in range(classInNum):
    #         temp=Train_PCA[:,index]
    #
    #         temp = np.reshape(temp, (50))
    #         classMat[:,classinNum]=temp
    #         index+=1
    #     Mean_classMat[:,classnum]=np.mean(classMat,axis=1)
    #     temp=Mean_classMat[:,classnum]
    #     temp=np.reshape(temp,(50,1))
    #     Swi=(classMat-temp).dot((classMat-temp).T)

    index = 0
    temp = np.zeros((dimNum, classInNum + 1))
    for i in range(1, classNum + 1):
        for j in range(1, classInNum + 1):
            temp[:, j] = Train_PCA[:, index].flatten()
            index = index + 1
        Mean_classMat[:, i - 1] = np.mean(temp, 1)
    Sw = np.zeros((dimNum, ImgNum))  # 类内散度矩阵
    Sw = np.mat(Sw)
    for i in range(0, ImgNum):
        mess = Mean_classMat[:, int(label[i]) - 1]
        mess = np.reshape(mess, (-1, 1))
        mess1 = Train_PCA[:, i] - mess
        Sw[:, i] = mess1.flatten().reshape((-1, 1))
    print('Swi', Sw.shape)
    Swi = Sw * (Sw.T)

    # Sb=np.zeros((dimNum,classNum))#类间散度矩阵
    Classdiff = Mean_classMat - Mean_all  # 每类均值-所有样本均值
    Sb = math.sqrt(classInNum) * Classdiff
    Sbi = Sb * (Sb.T)

    # print('Sb',Sb.shape)
    # eigVals, eigVects = np.linalg.eig(inv(Swi) * Sbi)
    eigVals, eigVects = np.linalg.eig(Swi * Sbi)
    # print('eigVects',eigVects.shape)
    eigSortIndex = np.argsort(-eigVals)  # 按行降序排列(一行为一个特征),返回一个序列
    LDA_dimNum = classNum - 1
    W = eigVects[:, eigSortIndex[0:LDA_dimNum]]  # 取特征向量前 维

    Train_LDA = W.T * Train_PCA  # 训练集在LDA空间投影
    # print('W',W)
    return W, Train_LDA
