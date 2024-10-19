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

if __name__ =='__main__':
    #记得复制过来的相对路径中的‘\’要改成   '/'
    filename='D:/ky/thorough-pytorch-main/Machine-Learning-in-Action-master/Ch02-KNN/datingTestSet.txt'
    DatingDataMat,DatingLabels = file2matrix(filename)
    print(DatingDataMat[:10])
    print(DatingLabels[:10])
