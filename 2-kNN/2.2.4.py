import KNN

filename='D:/ky/thorough-pytorch-main/Machine-Learning-in-Action-master/Ch02-KNN/datingTestSet.txt'

def datingClassTest():
    hoRatio = 0.10 #以百分之十的数据作为测试集
    datingDataMat,datingLabels = KNN.file2matrix(filename)
    normMat,ranges,minVals = KNN.autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = KNN.classify(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print(f"the classifier came back with:{classifierResult},the real answer is : {datingLabels[i]}",3)
        if (classifierResult != datingLabels[i]) :
            errorCount  += 1.0
    print(f"the total erroe rate is :{errorCount / float(numTestVecs)}")

if __name__=='__main__':
    datingClassTest()