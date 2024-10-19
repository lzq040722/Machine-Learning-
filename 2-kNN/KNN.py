import numpy as np
import operator

def createDataset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] #shape[0]返回矩阵第一个维度的长度
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet #tile函数将inX重复dataSetSize次，形成一个dataSetSize*1的矩阵
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)#按行求和
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() #按照一定的顺序进行排序后返回相应的索引
    classCount={} 
    #设置一个记录类别数的字典
    for i in range(k):
        #属于哪个标签
        voteILabel = labels[sortedDistIndicies[i]]
        #统计属于哪个标签的数量，用到了字典中的get方法，如果没有这个标签，则返回0
        classCount[voteILabel] = classCount.get(voteILabel,0)+1
    #operator.itemgetter(1) 对字典按照值进行排序，reverse = True是降序排序
    #operator.itemgetter(0) 对字典按照键进行排序
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)

    #返回数量最多的类别
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() #去掉回车符
        lineFormLine = line.split('\t')
        #将每一行的特征数据存档到returnMat中
        returnMat[index,:] = lineFormLine[0:3]
        #存放类标签，根据喜欢的程度：1表示不喜欢，2表示魅力一般，3表示极具魅力
        if lineFormLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif lineFormLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif lineFormLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index+=1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0) #0表示从每一列中选取最小值最大值
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet / np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.10 #以百分之十的数据作为测试集
    filename='D:/ky/thorough-pytorch-main/Machine-Learning-in-Action-master/Ch02-KNN/datingTestSet.txt'
    datingDataMat,datingLabels = file2matrix(filename)
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print(f"the classifier came back with:{classifierResult},the real answer is : {datingLabels[i]}")
        if (classifierResult != datingLabels[i]) :
            errorCount  += 1.0
    print(f"the total erroe rate is :{errorCount / float(numTestVecs)}")

def classifyPerson():
    resultList = ['讨厌','有些喜欢','喜欢']
    #输入用户的三维特征
    percentTats = float(input("玩游戏所耗时间百分比："))
    ffMiles = float(input("每年获得飞行常客里程数："))
    iceCream = float(input("每周消耗冰激凌公升数："))

    filename='D:/ky/thorough-pytorch-main/Machine-Learning-in-Action-master/Ch02-KNN/datingTestSet.txt'
    datingDataMat,datingLabels = file2matrix(filename)
    normMat,ranges,minVals = autoNorm(datingDataMat)
    #输入数据的归一化
    inArr = np.array([percentTats,ffMiles,iceCream])
    norminArr = (inArr - minVals)/ranges

    classifyResult = classify(norminArr,normMat,datingLabels,3)

    print("你可能%s这个人" % (resultList[classifyResult]))


if __name__ == '__main__':
    group,labels = createDataset()