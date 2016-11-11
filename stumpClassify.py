# -*-coding:utf-8-*-
import numpy as np

class stumpClassify:
    #create a decision stump as weak classifier
    def __init__(self, X, y=None):
        # X are the inputs
        # y are the labels
        self.X = np.mat(X)

        if y is not None:
            self.y = np.mat(y)

    def predict(self, dim, threshVal, threshIneq):
        # output the classification
        retArray = np.mat(np.ones((self.X.shape[0], 1)))
        if threshIneq == 'lt':
            retArray[self.X[:, dim] <= threshVal] = -1.0
        else:
            retArray[self.X[:, dim] > threshVal] = -1.0
        return retArray

    def train(self, weights):
        m, n = np.shape(self.X)
        numSteps = 10.0
        self.bestStump = {}
        bestClasEst = np.mat(np.zeros((m, 1)))
        minError = float("inf")
        for i in range(n):
            rangeMin = self.X[:, i].min()
            rangeMax = self.X[:, i].max()
            stepSize = (rangeMax - rangeMin) / numSteps
            for j in range(-1, int(numSteps) + 1):
                for inequal in ['lt', 'gt']:
                    threshVal = (rangeMin + float(j) * stepSize)
                    predictVal = self.predict(i, threshVal, inequal)
                    errArr = np.mat(np.ones((m, 1)))
                    errArr[predictVal == self.y] = 0
                    weightError = np.dot(weights.T, errArr)
                    # print "split: din %d, thresh %.2f, thresh ineqal:\
                    #       %s, the weights error is %.3f" %(i, threshVal, inequal, weightError)
                    if weightError < minError:
                        bestClasEst = predictVal
                        minError = weightError
                        self.bestStump['dim'] = i
                        self.bestStump['thresh'] = threshVal
                        self.bestStump['ineq'] = inequal
        return self.bestStump, minError, bestClasEst

if __name__ == "__main__":
    data = [[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]]
    labels = [[1.0], [1.0], [-1.0], [-1.0], [1.0]]
    weights = np.mat([[0.2],[0.2],[0.2],[0.2],[0.2]])
    nn = stumpClassify(data, labels)
    bestStump, minError, bestClasEst = nn.train(weights)











