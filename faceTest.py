# coding:utf8
import threading

import numpy as np
import os

import cv2

import ImageSet
import LDA
import createImageSet

camera = cv2.VideoCapture(0)


def detect(dataMat, label):
    # 创建人脸检测的对象
    face_cascade = cv2.CascadeClassifier("./venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

    k = 0

    # disc_set, disc_value = LDA.pca(dataMat, 50)
    # redVects, Train_LDA = LDA.lda(dataMat, label, 50, 16, 11, 11 * 16)  # LDA投影空间，最终的训练集

    while True:

        # 读取当前帧
        ret, frame = camera.read()

        # 转为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸 返回列表 每个元素都是(x, y, w, h)表示矩形的左上角和宽高
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # 画出人脸的矩形
        for (x, y, w, h) in faces:
            roi_gray = gray[y: y + h, x: x + w]
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            img1 = frame[y: y + h, x: x + w]
            cv2.imwrite('./pic1/s0.bmp', roi_gray)
            pic = cv2.imread('./pic1/s0.bmp')
            pic = cv2.resize(pic, (100, 100), interpolation=cv2.INTER_CUBIC)
            os.remove('./pic1/s0.bmp')
            cv2.imwrite('./pic/s0.bmp', pic)
            # print("hello world")
        if k == 0:
            t = threading.Thread(target=testPic, args=(dataMat, label))
            # t = threading.Thread(target=testPic, args=(dataMat, label, disc_set, disc_value, redVects, Train_LDA))
            t.start()
        k += 1

        # disc_set, disc_value = LDA.pca(dataMat, 50)
        # redVects, Train_LDA = LDA.lda(dataMat, label, 50, 16, 11, 11 * 16)  # LDA投影空间，最终的训练集
        # testImgSet = './pic/s0.bmp'
        # # testImgSet = createImageSet.createTestMat('Yale', testInClass, testNum, testInClass, 100 * 100)
        # testImgSet = ImageSet.HistogramEqualization(testImgSet)
        # testImgSet = np.reshape(testImgSet, (-1, 1))
        # testImgSet = disc_set.T.dot(testImgSet)
        # testImgSet = redVects.T.dot(testImgSet)
        # disList = []
        # testVec = np.reshape(testImgSet, (1, -1))
        # for sample in Train_LDA.T:
        #     disList.append(np.linalg.norm(testVec - sample))
        # # print('disList', disList)
        # sortIndex = np.argsort(disList)
        # print(label[sortIndex[0]])
        # if 16 == int(label[sortIndex[0]]):
        #     isRight = isRight + 1
        # os.remove('./pic/s0.bmp')
        # j = j + 1
        # if j == 5:
        #     if isRight >= 4:
        #         print("测试成功")
        #         break
        #     else:
        #         isRight = 0
        #         testTimes += 1
        #         if testTimes >= 5:
        #             print("测试失败")
        #             break
        #         j = 0
        # testPic(camera,dataMat, isRight, j, label, testTimes)
        cv2.imshow("camera", frame)
        if cv2.waitKey(5) & 0xff == ord("q"):
            break
        if not camera.isOpened():
            break

    if os.path.isfile('./pic/s0.bmp'):
        os.remove('./pic/s0.bmp')
    camera.release()
    cv2.destroyAllWindows()


def testPic(dataMat, label):
    # def testPic(dataMat, label, disc_set, disc_value, redVects, Train_LDA):
    print("thread")
    j = 0
    isRight = 0
    isRight2 = 0
    testTimes = 0
    while True:
        testImgSet = './pic/s0.bmp'
        if not os.path.isfile(testImgSet):
            continue

        disc_set, disc_value ,meanFace= LDA.pca(dataMat, 40)
        redVects, Train_LDA = LDA.lda(dataMat, label, 40, 17, 11, 11 * 17)  # LDA投影空间，最终的训练集

        # testImgSet = createImageSet.createTestMat('Yale', testInClass, testNum, testInClass, 100 * 100)
        testImgSet = ImageSet.HistogramEqualization(testImgSet)
        # print("shape", testImgSet.shape)
        testImgSet = np.reshape(testImgSet, (-1, 1))
        testImgSet = disc_set.T.dot(testImgSet)
        testImgSet = redVects.T.dot(testImgSet)
        disList = []
        testVec = np.reshape(testImgSet, (1, -1))
        for sample in Train_LDA.T:
            disList.append(np.linalg.norm(testVec - sample))
        # print('disList', disList)
        sortIndex = np.argsort(disList)
        print(label[sortIndex[0]])
        if 16 == int(label[sortIndex[0]]):
            isRight = isRight + 1
        if 17 == int(label[sortIndex[0]]):
            isRight2 = isRight2 + 1
        os.remove('./pic/s0.bmp')
        j = j + 1
        # j = j + 1
        # 在脸上检测眼睛   (40, 40)是设置最小尺寸，再小的部分会不检测
        # eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
        # 把眼睛画出来
        # for(ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        if j == 5:
            if isRight >= 4 or isRight2 >= 4:
                if isRight >= 4:
                    print("欢迎你，史长顺！")
                    # break
                    camera.release()
                    cv2.destroyAllWindows()
                    break
                if isRight2 >= 4:
                    print("欢迎你，饶丝雨！")
                    # break
                    camera.release()
                    cv2.destroyAllWindows()
                    break
            else:
                if isRight < 4:
                    isRight = 0
                    testTimes += 1
                    print("测试失败")
                    if testTimes >= 5:
                        # break
                        camera.release()
                    j = 0
                if isRight2 < 4:
                    isRight2 = 0
                    testTimes += 1
                    print("测试失败2")
                    if testTimes >= 5:
                        # break
                        camera.release()
                    j = 0


if __name__ == '__main__':
    dataMat, label = createImageSet.createImageMat('Yale', 17, 11, 11 * 17, 100 * 100)
    detect(dataMat, label)