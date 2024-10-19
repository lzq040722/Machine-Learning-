#code by Desperate
#time : 2024/10/16

import numpy as np
import KNN as KNN
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

#将32 * 32 的二进制图像转换成1*1024 的向量
def img2vector(filename):
    returnVec = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()

        for j in range(32):
            returnVec[0,32*i+j] = int(lineStr[j])
    return returnVec

def handwritingClassTest():
    #测试集的Labels
    hwlabels = []
    filename = "D:/ky/thorough-pytorch-main/Machine-Learning-in-Action-master/Ch02-KNN/trainingDigits"
    trainingFileList = listdir(filename)
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        filenameStr = trainingFileList[i]
        #获得分类的数字
        #根据文件的名字来获取分类的数字，例如文件名是：0_1.txt，那么分类的数字就是0
        classNumStr = int(filenameStr.split('_')[0])
        hwlabels.append(classNumStr)
        trainingMat[i,:] = img2vector(filename + '/' + filenameStr)
    filename = "D:/ky/thorough-pytorch-main/Machine-Learning-in-Action-master/Ch02-KNN/testDigits"
    testFileList = listdir(filename)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        #把测试集的数据向量化
        vectorUnderTest = img2vector(filename  + "/" + fileNameStr)
        #利用KNN算法获得预测结果
        classifyResult = KNN.classify(vectorUnderTest,trainingMat,hwlabels,3)
        #print("the classifier came back with: %d , the real answer is: %d"% (classifyResult, classNumber))
        if(classifyResult != classNumber) : errorCount += 1.0
    print("总共错了%d个数据\t错误率为%f%%" % (errorCount,float(errorCount/mTest)*100))

if __name__ == '__main__':
    handwritingClassTest()


   
        

