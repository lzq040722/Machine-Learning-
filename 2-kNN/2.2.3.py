import numpy as np
import operator

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

#进行数据的归一化处理
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0) #0表示从每一列中选取最小值最大值
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet / np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

if __name__ == '__main__':
    filename='D:/ky/thorough-pytorch-main/Machine-Learning-in-Action-master/Ch02-KNN/datingTestSet.txt'
    normDataSet,ranges,minVals = autoNorm(file2matrix(filename)[0])
    print(np.shape(minVals))
    print(normDataSet[:10])

