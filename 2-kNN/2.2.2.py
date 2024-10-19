#coded by desperate
#time 2024/10/16

import KNN
import matplotlib.pyplot as plt

filename='D:/ky/thorough-pytorch-main/Machine-Learning-in-Action-master/Ch02-KNN/datingTestSet.txt'
DatingDataMat,DatingLabels=KNN.file2matrix(filename)

def showdatas(DatingDataMat,DatingLabels):
    
    '''fig = plt.figure(figsize = (8,5))
    ax = fig.add_subplot(111)
    ax.scatter(DatingDataMat[:,1],DatingDataMat[:,2])
    plt.show()'''

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)


    colorLabels={1:'r',2:'g',3:'b'}
    for i in range(len(DatingLabels)):
        ax.scatter(DatingDataMat[i,1],DatingDataMat[i,2],c = colorLabels[int(DatingLabels[i])], alpha = 0.68,label=str(i))
        #ax.annotate()
    plt.legend()
    plt.show()

showdatas(DatingDataMat,DatingLabels)
