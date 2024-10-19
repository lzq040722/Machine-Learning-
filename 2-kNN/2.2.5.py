#coded by Desperate
#time : 2024/10/16

import numpy as np
import KNN as KNN
def classifyPerson():
    resultList = ['讨厌','有些喜欢','喜欢']
    #输入用户的三维特征
    percentTats = float(input("玩游戏所耗时间百分比："))
    ffMiles = float(input("每年获得飞行常客里程数："))
    iceCream = float(input("每周消耗冰激凌公升数："))

    filename='D:/ky/thorough-pytorch-main/Machine-Learning-in-Action-master/Ch02-KNN/datingTestSet.txt'
    datingDataMat,datingLabels = KNN.file2matrix(filename)
    normMat,ranges,minVals = KNN.autoNorm(datingDataMat)
    #输入数据的归一化
    inArr = np.array([percentTats,ffMiles,iceCream])
    norminArr = (inArr - minVals)/ranges

    classifyResult = KNN.classify(norminArr,normMat,datingLabels,3)

    print("你可能%s这个人" % (resultList[classifyResult]))

if __name__ == '__main__':
    classifyPerson()