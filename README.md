# Glass-Classification-With-SVM
This repository contains implementation of SVM for classification of glass. The dataset can be found here
https://archive.ics.uci.edu/ml/datasets/glass+identification\
To achieve the best classification accuracy, four different types of kernels were tested: RBF, linear, polynomial and sigmoid.
In addition of accuracy, the training time of two different classifier type: OneVsOne and OneVsAll classifiers is also compared. \
Here, \
ovo-balanced means OneVsOne classifier with balanced class weight \
ovr-balanced means OneVsRest classifier with balanced class weight \
ovo-balanced means OneVsOne classifier with unbalanced class weight \
ovR-balanced means OneVsRest classifier with unbalanced class weight 

#Results:

###Training Time:  
![html dark](https://github.com/sdevkota007/KNN-And-NaiveBayes-Classifier/blob/master/screenshots/training-time.png) \

###Accuracy table for 5 fold cross validation:
![html dark](https://github.com/sdevkota007/KNN-And-NaiveBayes-Classifier/blob/master/screenshots/accuracy1.png) \
![html dark](https://github.com/sdevkota007/KNN-And-NaiveBayes-Classifier/blob/master/screenshots/accuracy2.png) \
![html dark](https://github.com/sdevkota007/KNN-And-NaiveBayes-Classifier/blob/master/screenshots/accuracy3.png) \
![html dark](https://github.com/sdevkota007/KNN-And-NaiveBayes-Classifier/blob/master/screenshots/accuracy4.png) \