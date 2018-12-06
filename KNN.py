import numpy as np
import operator


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 得到行数
    diffMat = np.array(np.tile(inX, (dataSetSize, 1)) - dataSet)  # 计算输入向量inX与训练样本的差
    sqDiffMat = diffMat ** 2  # 计算差值的平方
    sqDistances = sqDiffMat.sum(axis=1)  # 距离平方和
    distances = sqDistances ** 0.5  # 开方得到距离
    sortedDistIndicies = distances.argsort()  # 距离进行排序,得到排序的下标
    classCount = {}
    for i in range(k):  # 确定前k个距离中最小元素所属分类
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 对出现的label进行计数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 按照计数值进行降序排序
    # operator.itemgetter(1)确定一个函数取出classCount中的第一个域的值，即将value取出
    #print('测试图像属于class',sortedClassCount[0][0])
    return sortedClassCount[0][0]  # 返回最大的计数值的分类
