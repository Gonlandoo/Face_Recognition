import ImageSet
import cv2
import numpy as np


# def createImageSet(folder,ClassNum,InClassNum,ImageSize):
# Facemat=mat(zeros((InClassNum,ImageSize)))
# k=0
# for i in 1,ClassNum+1:
#     for j in 1,InClassNum+1:
#         try:
#             img=cv2.imread('s'+j+'.bmp',0)
#             img=ImageSet.HistogramEqualization()
#         except:
#             print('load %s s%s failed'%i %j)
#         Facemat[k,:]=mat(img).flatten()
#         k+=1
# return Facemat

# def createImageMat(dirName,ImageNum,ImageSize):
#     '''
#     :param dirName: 训练数据集的图像文件夹路径
#     :param ImageNum: 总图像数
#     :param ImageSize: 图像大小
#     :return: 图像样本矩阵,
#     '''
#
#     dataMat=np.zeros((ImageNum,ImageSize))#生成用0填充的数据集（数组）
#     label=[]
#     for parent,dirnames,filename in os.walk(dirName):
#         index=0
#         for dirname in dirnames:
#             for subParent,subDirName,subFileNames in os.walk(parent+'/'+dirname):
#                 for filename in subFileNames:
#                     img=ImageSet.HistogramEqualization('./'+parent+'/'+subParent+'/'+filename+'.bmp')
#                     tempImg=np.reshape(img,(-1,1))#转化为一列
#                     if index==0:
#                         dataMat=tempImg
#                     else:
#                         dataMat=np.column_stack((dataMat,tempImg))#列合并
#                         label.append(subParent+'/'+filename)
#                     index+=1
#     return dataMat,label

def createImageMat(dirName, classNum, classInNum, ImageNum, ImageSize):
    '''
    创建训练集矩阵
    :param dirName:包含数据集的文件夹名称
    :param classNum: 类数目
    :param classInNum: 每类包含的图像数目
    :param ImageNum: 图像总数
    :param ImageSize: 每张图像的大小
    :return: 图像矩阵，#每张图像对应的标签
    '''
    dataMat = np.zeros((ImageNum, ImageSize))  # 生成用0填充的数据集(数组),维度为图像数目*图像大小
    # label = np.zeros((1,ImageNum))
    label = []
    index = 0
    for classnum in range(1, classNum + 1):
        for classinnum in range(1, classInNum + 1):
            img = ImageSet.HistogramEqualization(
                './' + str(dirName) + '/' + str(classnum) + '/' + 's' + str(classinnum) + '.bmp')
            # print('均衡化后的图像矩阵',img)
            tempImg = np.reshape(img, (-1, 1))  # 每张图转化为一列
            if index == 0:
                dataMat = tempImg
                label.append(str(classnum))
                # label[:,index]=classnum
                index += 1
            else:
                dataMat = np.column_stack((dataMat, tempImg))
                dataMat = np.mat(dataMat)  # 列合并，然后转为矩阵
                label.append(str(classnum))
                # label[:, index] = classnum
                index += 1
            # print('index',index)
    return dataMat, label


def createTestMat(dirName, classNum, testNum, ImageSize):
    '''
    创建测试集
    :param dirName: 文件夹名称
    :param classNum: 哪一类
    :param testNum: 每类中用来测试的图像数量
    :param ImageSize: 图像大小
    :return: 第classNum类的测试集
    '''
    testMat = np.zeros((testNum, ImageSize))
    index = 0
    for i in range(1, testNum + 1):
        # dic = './' + str(dirName) + '/' + str(classNum) + '/' + 's' + str(i) + '.bmp'
        img = ImageSet.HistogramEqualization(
            './' + str(dirName) + '/' + str(classNum) + '/' + 's' + str(i) + '.bmp')
        # cv2.imshow(str(classNum) + '/' + 's', img)
        # name=str(classNum) + '/' + 's'+ str(i)
        # print('name',name)
        # cv2.waitKey(0)
        tempImg = np.reshape(img, (-1, 1))  # 每张图转化为一列
        if index == 0:
            testMat = tempImg
            index += 1
        else:
            testMat = np.column_stack((testMat, tempImg))
            testMat = np.mat(testMat)  # 列合并，然后转为矩阵
            index += 1
    return testMat
