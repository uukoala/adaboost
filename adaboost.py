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
        # for i in range(numIt):
        #     self.G.setdefault(i)
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
            # print self.G[i]['alpha']
            # print self.G[i]['dim']
            # print self.G[i]['thresh']
            # print self.G[i]['ineq']
            a = self.weakc(X).predict(self.G[i]['dim'],\
                                                  self.G[i]['thresh'],\
                                                  self.G[i]['ineq'])
            print a
            predictClass += self.G[i]['alpha'] * \
                            self.weakc(X).predict(self.G[i]['dim'],\
                                                  self.G[i]['thresh'],\
                                                  self.G[i]['ineq'])
            print predictClass
        return np.sign(predictClass)



if __name__ == "__main__":
    data = [[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]]
    labels = [[1.0], [1.0], [-1.0], [-1.0], [1.0]]
    nn = adaboost(stumpClassify)
    nn.train(data, labels)
    result = nn.predict([0,0])
    print result



