# -*-coding:utf-8-*-
import numpy as np
from stumpClassify import stumpClassify

class adaboost:
    def __init__(self, weakc):
        # weakc is a weakclassify which includes a train() function return
        # the classify, the error and the classify result
        self.weakc = weakc


    def train(self, X, y, numIt =10):
        # M为弱分类器最大个数
        X = np.mat(X)
        y = np.mat(y)
        m = X.shape[0]
        self.G = {}
        weights = np.mat(np.ones((m, 1))) / m
        aggClassEst = np.mat(np.zeros((m, 1)))
        for i in range(numIt):
            self.G[i], error, classEst = self.weakc(X, y).train(weights)
            # print "weight_sample:", weights.T
            alpha = float(0.5 * np.log((1-error)/max(error,1e-16)))
            # print alpha
            self.G[i]['alpha'] = alpha
            # print "classEst: ", classEst
            weights = np.multiply(weights, np.exp(np.multiply(-alpha * classEst, y)))
            weights = weights / weights.sum()

            aggClassEst += alpha * classEst
            # print "aggClassEst: ", aggClassEst
            aggError = np.multiply((np.sign(aggClassEst) != y), np.ones((m, 1)))
            errorRate = aggError.sum() / m
            # print "total error: ", errorRate, "\n"
            if errorRate == 0.0:
                break
        return self.G

    def predict(self, X):
        X = np.mat(X)
        m = X.shape[0]
        predictClass = np.zeros((m, 1))
        for i in range(len(self.G)):

            predictClass += self.G[i]['alpha'] * \
                            self.weakc(X).predict(self.G[i]['dim'],\
                                                  self.G[i]['thresh'],\
                                                  self.G[i]['ineq'])
            # print predictClass
        return np.sign(predictClass)

def loadData(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat -1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append([float(curLine[-1])])
    return dataMat,labelMat


if __name__ == "__main__":
    data, labels = loadData("horseColicTraining2.txt")
    nn = adaboost(stumpClassify)
    nn.train(data, labels)
    testdata, testlabel = loadData("horseColicTest2.txt")
    result = nn.predict(testdata)
    testlabel = np.mat(testlabel)
    m = testlabel.shape[0]
    error = np.mat(np.ones((m, 1)))
    error[result == testlabel] = 0
    errorrate = float(error.sum())/m
    print errorrate



