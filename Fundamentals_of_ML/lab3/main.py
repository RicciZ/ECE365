import numpy as np
import time
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

error = lambda y, yhat: np.mean(y!=yhat)

class Question1(object):
    # The sequence in this problem is different from the one you saw in the jupyter notebook. This makes it easier to grade. Apologies for any inconvenience.
    def BernoulliNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        # Put your code below
        classifier = BernoulliNB()
        start = time.time()
        classifier.fit(traindata,trainlabels)
        fit = time.time()
        valres = classifier.predict(valdata)
        valpred = time.time()
        trainres = classifier.predict(traindata)
        trainingError = 1 - np.mean(trainres == trainlabels)
        validationError = 1 - np.mean(valres == vallabels)
        fittingTime = fit - start
        valPredictingTime = valpred - fit
        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def MultinomialNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        # Put your code below
        classifier = MultinomialNB()
        start = time.time()
        classifier.fit(traindata,trainlabels)
        fit = time.time()
        valres = classifier.predict(valdata)
        valpred = time.time()
        trainres = classifier.predict(traindata)
        trainingError = 1 - np.mean(trainres == trainlabels)
        validationError = 1 - np.mean(valres == vallabels)
        fittingTime = fit - start
        valPredictingTime = valpred - fit
        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)


    def LinearSVC_classifier(self, traindata, trainlabels, valdata, vallabels):
        # Put your code below
        classifier = LinearSVC()
        start = time.time()
        classifier.fit(traindata,trainlabels)
        fit = time.time()
        valres = classifier.predict(valdata)
        valpred = time.time()
        trainres = classifier.predict(traindata)
        trainingError = 1 - np.mean(trainres == trainlabels)
        validationError = 1 - np.mean(valres == vallabels)
        fittingTime = fit - start
        valPredictingTime = valpred - fit
        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LogisticRegression_classifier(self, traindata, trainlabels, valdata, vallabels):
        # Put your code below
        classifier = LogisticRegression()
        start = time.time()
        classifier.fit(traindata,trainlabels)
        fit = time.time()
        valres = classifier.predict(valdata)
        valpred = time.time()
        trainres = classifier.predict(traindata)
        trainingError = 1 - np.mean(trainres == trainlabels)
        validationError = 1 - np.mean(valres == vallabels)
        fittingTime = fit - start
        valPredictingTime = valpred - fit
        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def NN_classifier(self, traindata, trainlabels, valdata, vallabels):
        # Put your code below
        classifier = KNeighborsClassifier(1)
        start = time.time()
        classifier.fit(traindata,trainlabels)
        fit = time.time()
        valres = classifier.predict(valdata)
        valpred = time.time()
        trainres = classifier.predict(traindata)
        trainingError = 1 - np.mean(trainres == trainlabels)
        validationError = 1 - np.mean(valres == vallabels)
        fittingTime = fit - start
        valPredictingTime = valpred - fit
        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def confMatrix(self,truelabels,estimatedlabels):
        cm = np.zeros((2,2))
        # Put your code below
        true = truelabels == 1
        false = truelabels == -1
        pos = estimatedlabels == 1
        neg = estimatedlabels == -1
        cm[0,0] = np.sum(true & pos)
        cm[0,1] = np.sum(false & pos)
        cm[1,0] = np.sum(true & neg)
        cm[1,1] = np.sum(false & neg)
        return cm

    def classify(self, traindata, trainlabels, testdata, testlabels):
        # Put your code below
        classifier = LogisticRegression()
        classifier.fit(traindata,trainlabels)
        est_labels = classifier.predict(testdata)
        testError = 1 - np.mean(est_labels == testlabels)
        confusionMatrix = self.confMatrix(testlabels, est_labels)
        # Do not change this sequence!
        return (classifier, testError, confusionMatrix)

class Question2(object):
    def crossValidationkNN(self, trainData, trainLabels, k):
        # Put your code below
        err = np.zeros(k+1)
        n = trainData.shape[0]
        for i in range(1,k+1):
            valerr = 0
            for j in range(5):
                traindata = np.concatenate((trainData[:n*j//5,:],trainData[n*(j+1)//5:,:]))
                valdata = trainData[n*j//5:n*(j+1)//5,:]
                trainlabels = np.concatenate((trainLabels[:n*j//5],trainLabels[n*(j+1)//5:]))
                vallabels = trainLabels[n*j//5:n*(j+1)//5]
                classifier = KNeighborsClassifier(i)
                classifier.fit(traindata,trainlabels)
                valres = classifier.predict(valdata)
                valerr += 1 - np.mean(valres == vallabels)
            valerr /= 5
            err[i] = valerr
        return err

    def minimizer_K(self, trainData, trainLabels, k):
        err = self.crossValidationkNN(trainData, trainLabels, k)
        # Put your code below
        k_min = np.argmin(err[1:]) + 1
        err_min = err[k_min]
        # Do not change this sequence!
        return (err, k_min, err_min)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        # Put your code below
        classifier = KNeighborsClassifier(14)
        classifier.fit(traindata,trainlabels)
        testres = classifier.predict(testdata)
        testError = 1 - np.mean(testres == testlabels)
        # Do not change this sequence!
        return (classifier, testError)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class Question3(object):
    def LinearSVC_crossValidation(self, data_train, label_train):
        # Put your code below
        C = np.array([2**i for i in range(-5,16)])
        err = np.zeros(len(C))
        for i in range(len(C)):
            classifier = LinearSVC(C=C[i])
            error = 1 - cross_val_score(classifier,data_train,label_train,cv=10)
            err[i] = np.mean(error)
        idx = np.argmin(err)
        C_min = C[idx]
        min_err = err[idx]
        # Do not change this sequence!
        return (C_min, min_err)

    def SVC_crossValidation(self, data_train, label_train):
        # Put your code below
        C = np.array([2**i for i in range(-5,16)])
        gamma = np.array([2**i for i in range(-15,4)])
        param = {'C':C,'gamma':gamma}
        classifier = SVC()
        grid_search = GridSearchCV(classifier,param,cv=10)
        res = grid_search.fit(data_train,label_train)
        min_err = 1 - res.best_score_
        C_min = res.best_params_['C']
        gamma_min = res.best_params_['gamma']
        # Do not change this sequence!
        return (C_min, gamma_min, min_err)

    def LogisticRegression_crossValidation(self, data_train, label_train):
        # Put your code below
        C = np.array([2**i for i in range(-14,15)])
        param = {'C':C}
        classifier = LogisticRegression()
        grid_search = GridSearchCV(classifier,param,cv=10)
        res = grid_search.fit(data_train,label_train)
        min_err = 1 - res.best_score_
        C_min = res.best_params_['C']
        # Do not change this sequence!
        return (C_min, min_err)

    def classify(self, data_train, label_train, data_test, label_test):
        # Put your code below
        classifier = SVC(C=8,gamma=0.125)
        classifier.fit(data_train,label_train)
        est_labels = classifier.predict(data_test)
        testError = 1 - np.mean(est_labels == label_test)
        # Do not change this sequence!
        return (classifier, testError)
