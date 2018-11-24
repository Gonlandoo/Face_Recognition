import numpy as np
# def pca(dataMat,dimNum):
    # '''
    # PCA函数，用于数据降维
    # :parameter dataMat:样本矩阵
    # :parameter dimNum:降维后的目标维度dimNum
    # :return:降维后的样本矩阵和变换矩阵
    # '''
    #
    # #均值化矩阵
    # meanMat=np.mat(np.mean(dataMat,1)).T #axis=1，压缩列，对列求均值
    # print('平均值矩阵维度',meanMat.shape)
    # diffMat=dataMat.T-meanMat
    # #求协方差矩阵，但不直接求，因为样本维度远大于样本数目
    # covMat=(diffMat.T*diffMat)/float(diffMat.shape[1]) #归一化
    # #基本方法求协方差矩阵：
    # #covMat2=np.cov(dataMat,bias=True)
    # #print('协方差矩阵维度',covMat2.shape)
    #
    # #计算特征值，特征向量
    # eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    # print('特征向量维度',eigVects.shape)
    #
    # print('特征值',eigVals)
    # eigVects=diffMat*eigVects
    #
    # eigValInd=np.argsort(eigVals)#对特征值进行排序
    # eigValInd=eigValInd[::-1]
    # eigValInd=eigValInd[:dimNum]#取出指定个数的前n大的特征值
    # #np.seterr(divide='ignore', invalid='ignore')
    # eigVects=eigVects/np.linalg.norm(eigVects,axis=0)#归一化特征向量##########
    # #eigVects = np.squeeze(eigVects)
    # redEigVects=eigVects[:,eigValInd]
    # print('选取的特征向量',redEigVects.shape)
    # print('均值矩阵维度',diffMat.shape)
    #
    # # redEigVects=np.reshape(redEigVects,(1,-1))
    # # print(redEigVects.shape)
    #
    # lowMat=redEigVects.T*diffMat
    # return lowMat,redEigVects
def pca(dataMat,selecthr):
    meanMat=np.mean(dataMat,axis=1)#对行求均值，计算平均图像
    print('样本矩阵',dataMat)
    print('平均图像',meanMat)
    diffMat=dataMat-meanMat#偏差矩阵
    print('偏差矩阵维度',diffMat.shape)
    print('偏差矩阵',diffMat)

    print('计算特征值 特征向量')
    eigVals,eigVects=np.linalg.eig(diffMat.T*diffMat)
    print('特征值',eigVals.shape)
    print('特征向量维度',eigVects.shape)
    eigSortIndex=np.argsort(-eigVals)#按行降序排列(一行为一个特征),返回一个序列

    np.seterr(divide='ignore', invalid='ignore')
    for i in range(dataMat.shape[1]):
        if (eigVals[eigSortIndex[:i]]/eigVals.sum()).sum()>=selecthr:
            eigSortIndex=eigSortIndex[:i]
            break
    #print('排序后的特征值',eigVects[:,eigSortIndex])
    covVects=diffMat*eigVects[:,eigSortIndex]#协方差矩阵的特征向量
    print('协方差矩阵特征向量维度',covVects.shape)
    # print('变换矩阵维度',diffMat.shape)
    lowMat=covVects.T*diffMat
    # print('降维后的样本矩阵维度',lowMat.shape)
    return lowMat,covVects





