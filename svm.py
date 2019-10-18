from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier



Cs = [1, 10, 100, 1000]
gammas = [0.0001, 0.001, 0.01, 0.1]
degrees = [2, 3, 4, 5]
classifier_types = ['ovr', 'ovo']
class_weights = ['unbalanced', 'balanced']
kernels = ['rbf', 'linear', 'sigmoid', 'poly']

grid_search_params = {'estimator__C': Cs,
                      'estimator__gamma': gammas,
                      'estimator__degree': degrees}


def readData(file, norm=True):
    '''
    reads the content of the file and returns X(features) and Y(labels)
    :param file: name of the file
    :return: features, labels
    '''
    pd.set_option('display.max_columns', 15)

    dataset = pd.read_csv(file, header=None)
    dataset = dataset.sample(frac=1)

    X = dataset.iloc[:, 1:10].values
    Y = dataset.iloc[:, 10].values

    if norm:

        scalar = MinMaxScaler()
        X = scalar.fit_transform(X)

    return X.reshape((-1,9)), Y



def SVMGridSearch(X_train, Y_train, kernel, nfolds, classifier_type, class_weight):
    '''
    Performs grid search of SVM
    :param X_train: features of training data
    :param Y_train: labels of training data
    :param kernel:
    :param nfolds: n of n-fold cross validation
    :param classifier_type: type of classifier, either 'ovo' or 'ovr'
    :param class_weight: type of class weight, either 'balanced' or 'unbalanced'
    :return: best model obtained after grid search
    '''
    # if kernel=='poly':
    #     grid_search_params = {'estimator__C': Cs,
    #                           'estimator__gamma': gammas,
    #                           "estimator__degree": degrees}
    # if kernel == 'linear':
    #     grid_search_params = {'estimator__C': Cs}
    # else:
    #     grid_search_params = {'estimator__C': Cs,
    #                           'estimator__gamma': gammas}


    if class_weight=='balanced':
        # print(classifier_type, class_weight)
        # model_to_set = svm.SVC(classifier_type=classifier_type, class_weight=class_weight)
        if classifier_type=='ovo':
            model_to_set = OneVsOneClassifier(svm.SVC(kernel=kernel, class_weight=class_weight))

        elif classifier_type=='ovr':
            model_to_set = OneVsRestClassifier(svm.SVC(kernel=kernel, class_weight=class_weight))

    elif class_weight=='unbalanced':
        # print(classifier_type, class_weight)
        # model_to_set = svm.SVC(classifier_type=classifier_type)

        if classifier_type == 'ovo':
            model_to_set = OneVsOneClassifier(svm.SVC(kernel=kernel))

        elif classifier_type == 'ovr':
            model_to_set = OneVsRestClassifier(svm.SVC(kernel=kernel))


    best_model = GridSearchCV(model_to_set, grid_search_params, cv=nfolds, iid=False)
    best_model.fit(X_train, Y_train)

    # print(best_model.cv_results_)

    return best_model


def plotAccuracyVsGamma(results, kernel):
    '''
    plots validation accuracy vs gamma for different values of C
    :param results: results from grid_search
    :param kernel: name of the kernel for which the plot is being generated
    :return: none
    '''
    means = results["mean_test_score"]
    if len(grid_search_params)==3:
        means = np.concatenate((means[0:4], means[16:20], means[32:36], means[48:52])).reshape(len(Cs), len(gammas))
    else:
        means = np.array(means).reshape(len(Cs), len(gammas))

    plt.subplot(2,2, kernels.index(kernel)+1)

    for ind, i in enumerate(Cs):
        plt.plot(gammas, means[ind], label='C: ' + str(i))
    plt.legend()
    plt.xlabel('Gamma (log scale)')
    plt.ylabel('Mean accuracy')
    plt.xscale('log')
    plt.tight_layout()
    plt.title('Kernel: {}'.format(kernel))



def SVMwithCrossValidation(X, Y, kernel, cv, classifier_type, class_weight, plot=False):
    '''
    implementation of SVM with cross validation
    :param X: Features
    :param Y: Labels
    :param kernel: name of the kernel to be used
    :param cv: n-fold cross-validation
    :param classifier_type: either 'ovo' or 'ovr'. Ovo represents OneVsOne classifier, Ovr represents OneVsRest
    :param class_weight: either 'balanced' or none
    :param plot: if plot is set to true, a graph of accuracy vs gamma is plotted at the end of the program
    :return: average_train_accuracy, average_test_accuracy, average_fit_time
    '''
    accuracies_train = []
    accuracies_test = []
    fit_time_list = []
    num_samples = len(X)

    length_test = int((1 / cv) * num_samples)
    u = 0
    v = length_test
    print("\n========================================================================================================")
    print("========================================================================================================")
    print("""Training begins with Parameters: 
          \tkernel: {0} 
          \tclass_weight: {1} 
          \tclassifier type: {2}""".format(kernel, class_weight, classifier_type))

    for i in range(cv):
        print("\n******Iteration {0} of {1}-fold cross-validation******".format(i + 1, cv))

        X_test = X[u:v]
        X_train = np.concatenate((X[0:u], X[v:num_samples]), axis=0)
        Y_test = Y[u:v]
        Y_train = np.concatenate((Y[0:u], Y[v:num_samples]), axis=0)
        u = u + length_test
        v = v + length_test

        best_model = SVMGridSearch(X_train, Y_train, kernel=kernel, nfolds=5,
                                   classifier_type=classifier_type,
                                   class_weight=class_weight)
        best_params = best_model.best_params_
        best_train_accuracy = best_model.best_score_
        fit_time = best_model.refit_time_

        test_accuracy = accuracy_score(Y_test, best_model.predict(X_test))
        print("=>Best params: ", best_params)
        print("=>Training accuracy: ", best_train_accuracy)
        print("=>Test accuracy: ", test_accuracy)
        print("=>Fit Time: ", fit_time)

        accuracies_train.append(best_train_accuracy)
        accuracies_test.append(test_accuracy)
        fit_time_list.append(fit_time)

        if plot and i==0 and classifier_type=='ovo' and class_weight=='unbalanced':
            plotAccuracyVsGamma(best_model.cv_results_, kernel)


        # if plot and i == 0 and classifier_type == 'ovo' and class_weight == 'unbalanced' and kernel=='poly':
        #     plotAccuracyVsDegree(best_model.cv_results_, kernel)

    average_train_accuracy = sum(accuracies_train) / len(accuracies_train)
    average_test_accuracy = sum(accuracies_test) / len(accuracies_test)
    average_fit_time = sum(fit_time_list)/len(fit_time_list)

    return average_train_accuracy, average_test_accuracy, average_fit_time



def main():
    '''
    Main function. Program begins here
    :return: 
    '''
    print("Loading Dataset...")
    X, Y = readData("data/glass.data", norm=True)


    for classifier_type in classifier_types:
        for class_weight in class_weights:
            for kernel in kernels:
                average_train_accuracy, average_test_accuracy, average_training_time = \
                    SVMwithCrossValidation(X, Y,
                                           kernel=kernel, cv=5,
                                           classifier_type = classifier_type,
                                           class_weight = class_weight,
                                           plot=True)

                print("\nSummary: ")
                print("\tAverage train Accuracy: ", average_train_accuracy)
                print("\tAverage test Accuracy: ", average_test_accuracy)
                print("\tAverage training time: ", average_training_time)
                print("Completed!")

    plt.show()



if __name__ == '__main__':

    main()
    # test()