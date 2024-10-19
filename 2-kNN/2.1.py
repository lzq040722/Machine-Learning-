#Code by Desperate
#time : 2024/10/14

import numpy as np
import operator

def classsify(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] #shape[0]返回矩阵第一个维度的长度
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet #tile函数将inX重复dataSetSize次，形成一个dataSetSize*1的矩阵
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)#按列求和
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




    