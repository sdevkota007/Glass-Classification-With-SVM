import matplotlib.pyplot as plt
import numpy as np
from svm import kernels, class_weights, classifier_types


def plotTrainingTime():
    training_time_dict = {}
    for classifier_type in classifier_types:
        for class_weight in class_weights:
            training_time_dict['{0}-{1}'.format(classifier_type, class_weight)]= None

    print(training_time_dict)

    with open(result_file, 'r') as file:
        lines = file.readlines()
        training_time_list = []
        for line in lines:
            if "Average training time:" in line:
                time = line.split(":  ")[1]
                training_time_list.append(float(time) * 1000)
        training_time_array = np.array(training_time_list)
        training_time_array = training_time_array.reshape((-1, 4))
        print(kernels)
        print(training_time_array)

        i = 0
        for key in training_time_dict:
            training_time_dict[key] = training_time_array[i]
            i += 1

        print(training_time_dict)

    labels = kernels
    index = np.arange(len(labels))
    plt.subplot(111)
    w = 0.2
    xpos = index - (3 * w / 2)
    for key in training_time_dict:
        print(xpos)
        plt.bar(xpos, training_time_dict[key], width=w, align='center')
        xpos = xpos + w

    plt.legend(training_time_dict.keys(), loc=2)
    # plt.autoscale(tight=True)
    plt.xlabel('Kernels')
    plt.ylabel('Average training Time (milli-sec)')
    plt.xticks(index, labels, rotation=30)

    plt.title('Training Time')
    plt.show()




def generateTable():
    with open(result_file, 'r') as file:
        lines = file.readlines()
        test_accuracy_list = []
        for line in lines:
            #if ""
            if "=>Test accuracy:" in line:
                accuracy = line.split(":  ")[1]
                test_accuracy_list.append(round(float(accuracy), 4))

            if "	Average test Accuracy:" in line:
                accuracy = line.split(":  ")[1]
                test_accuracy_list.append(round(float(accuracy), 4))

        test_accuracy_array = np.array(test_accuracy_list)
        test_accuracy_array = np.transpose(test_accuracy_array.reshape((-1, 6)))
        print(test_accuracy_array, test_accuracy_array.shape)
        # print(test_accuracy_array, len(test_accuracy_array))

    i=0
    offset=4
    labels = kernels
    titles = [""]
    for a in range(4):

        data = test_accuracy_array[:, i:i+offset]
        print(data)



        n_folds = np.array([1,2,3,4,5, "Average"]).reshape(6,1)
        clust_data = np.c_[n_folds, data]

        collabel = ['n-fold']
        collabel = collabel + labels

        plt.subplot(2, 2, a + 1)
        print(clust_data)
        plt.axis('tight')
        plt.axis('off')
        the_table = plt.table(cellText=clust_data, colLabels=collabel, loc='center', fontsize=100)

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1.1, 1.1)

        i=i+offset

    plt.show()


def generateBestParamTable():
    with open(result_file, 'r') as file:
        lines = file.readlines()
        test_accuracy_list = []
        for line in lines:
            #if ""
            if "=>Test accuracy:" in line:
                accuracy = line.split(":  ")[1]
                test_accuracy_list.append(round(float(accuracy), 4))

            if "	Average test Accuracy:" in line:
                accuracy = line.split(":  ")[1]
                test_accuracy_list.append(round(float(accuracy), 4))

        test_accuracy_array = np.array(test_accuracy_list)
        test_accuracy_array = np.transpose(test_accuracy_array.reshape((-1, 6)))
        print(test_accuracy_array, test_accuracy_array.shape)
        # print(test_accuracy_array, len(test_accuracy_array))

    i=0
    offset=4
    labels = kernels
    titles = [""]
    for a in range(4):

        data = test_accuracy_array[:, i:i+offset]
        print(data)



        n_folds = np.array([1,2,3,4,5, "Average"]).reshape(6,1)
        clust_data = np.c_[n_folds, data]

        collabel = ['n-fold']
        collabel = collabel + labels

        plt.subplot(2, 2, a + 1)
        print(clust_data)
        plt.axis('tight')
        plt.axis('off')
        the_table = plt.table(cellText=clust_data, colLabels=collabel, loc='center', fontsize=100)

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1.1, 1.1)

        i=i+offset



    # clust_data = test_accuracy_array[5:10]
    # collabel = kernels
    # plt.subplot()
    # plt.axis('tight')
    # plt.axis('off')
    # plt.table(cellText=clust_data, colLabels=collabel, loc='center')



    plt.show()

result_file = "results/run5.txt"
plotTrainingTime()
generateTable()
#generateBestParamTable()