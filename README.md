# Face_Recognition
PCA+LDA
catchPic.py:调取摄像头，派取11张照片，并写入Yale库中，作为样本

程序入口：faceTest
程序运行结果：在控制台输出人名，或者检测失败

LDA.py:两个函数，pca作初次降维，lda在pca降维的基础上进行分类降维

PCA.py：本项目未使用，用作理解原理

KNN.py:本项目未使用，但可以与PCA方法联用

compare:本项目未使用，可以用来检测PCA+KNN的正确率

createImageSet:两个方法，分别对样本集与测试集进行处理

ImageSet:读取照片为矩阵形式
