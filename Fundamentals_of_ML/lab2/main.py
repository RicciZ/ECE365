import numpy as np
import scipy.spatial.distance as dist
from scipy import stats
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time

class Question1(object):
    def bayesClassifier(self,data,pi,means,cov):
        C_inv = np.linalg.inv(cov)
        delta = np.log(pi) + np.dot(means,np.dot(C_inv,data.T)).T - 1/2*np.sum(np.dot(means,C_inv)*means, axis=1)  # 3xn
        labels = np.argmax(delta,axis=1)
        return labels

    def classifierError(self,truelabels,estimatedlabels):
        error = 1 - np.mean(truelabels == estimatedlabels)
        return error


class Question2(object):
    def trainLDA(self,trainfeat,trainlabel):
        nlabels = int(trainlabel.max())+1 # Assuming all labels up to nlabels exist.
        pi = np.zeros(nlabels)            # Store your prior in here
        means = np.zeros((nlabels,trainfeat.shape[1]))            # Store the class means in here
        cov = np.zeros((trainfeat.shape[1],trainfeat.shape[1]))   # Store the covariance matrix in here
        # Put your code below
        pi = np.array([np.mean(trainlabel==i) for i in range(nlabels)])
        means = np.array([np.mean(trainfeat[trainlabel==i],axis=0) for i in range(nlabels)])
        for i in range(nlabels):
            cov += np.dot((trainfeat[trainlabel==i]-means[i]).T,(trainfeat[trainlabel==i]-means[i]))
        cov /= len(trainfeat) - nlabels
        # Don't change the output!
        return (pi,means,cov)

    def estTrainingLabelsAndError(self,trainingdata,traininglabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        pi, means, cov = self.trainLDA(trainingdata,traininglabels)
        esttrlabels = q1.bayesClassifier(trainingdata,pi,means,cov)
        trerror = q1.classifierError(traininglabels,esttrlabels)
        # Don't change the output!
        return (esttrlabels, trerror)

    def estValidationLabelsAndError(self,trainingdata,traininglabels,valdata,vallabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        pi, means, cov = self.trainLDA(trainingdata,traininglabels)
        estvallabels = q1.bayesClassifier(valdata,pi,means,cov)
        valerror = q1.classifierError(vallabels,estvallabels)
        # Don't change the output!
        return (estvallabels, valerror)


class Question3(object):
    def kNN(self,trainfeat,trainlabel,testfeat, k):
        distance = dist.cdist(testfeat,trainfeat)
        index = np.argpartition(distance,k-1)[:,:k]
        knn_labels = np.array([np.take_along_axis(trainlabel, index[:,i], axis=-1) for i in range(k)]).T
        labels = stats.mode(knn_labels.T)[0][0]
        return labels

    def kNN_errors(self,trainingdata, traininglabels, valdata, vallabels):
        q1 = Question1()
        trainingError = np.zeros(4)
        validationError = np.zeros(4)
        k_array = [1,3,4,5]

        for i in range(len(k_array)):
            # Please store the two error arrays in increasing order with k
            # This function should call your previous self.kNN() function.
            # Put your code below
            labels = self.kNN(trainingdata,traininglabels,trainingdata,k_array[i])
            trainingError[i] = q1.classifierError(traininglabels,labels)
            labels = self.kNN(trainingdata,traininglabels,valdata,k_array[i])
            validationError[i] = q1.classifierError(vallabels,labels)
        # Don't change the output!
        return (trainingError, validationError)

class Question4(object):
    def sklearn_kNN(self,traindata,trainlabels,valdata,vallabels):
        classifier = neighbors.KNeighborsClassifier(1,algorithm='brute')
        start = time.time()
        classifier.fit(traindata,trainlabels)
        fit = time.time()
        res = classifier.predict(valdata)
        pred = time.time()
        valerror = 1 - np.mean(res == vallabels)
        fitTime = fit - start
        predTime = pred - fit
        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

    def sklearn_LDA(self,traindata,trainlabels,valdata,vallabels):
        classifier = LinearDiscriminantAnalysis()
        start = time.time()
        classifier.fit(traindata,trainlabels)
        fit = time.time()
        res = classifier.predict(valdata)
        pred = time.time()
        valerror = 1 - np.mean(res == vallabels)
        fitTime = fit - start
        predTime = pred - fit
        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

###
