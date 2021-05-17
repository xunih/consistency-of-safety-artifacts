from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

import write_data

ground_truth_file = 'classification.csv'


def classifier(arr, file_name):
    # Open the file where ground truth data is in
    data = pd.read_csv(ground_truth_file)
    feature = arr # Features that are fed to the classifier
    ground_truth = data['Classification'] # Get ground truth data from a specific column
    # Cross validation
    clf_one = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=27)
    score_SVC = cross_val_score(clf_one, feature, ground_truth, cv=cv)
    mean_SVC = score_SVC.mean()
    std_SVC = score_SVC.std()
    print("%0.2f accuracy with a standard deviation of %0.2f" % (score_SVC.mean(), score_SVC.std()))
    # print('the accuracy of a linear kernel support vector machine on the dataset')
    # print(scores)
    clf_two = make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    score_KNN = cross_val_score(clf_two, feature, ground_truth, cv=cv)
    mean_KNN = score_KNN.mean()
    std_KNN = score_KNN.std()
    print("%0.2f accuracy with a standard deviation of %0.2f" % (score_KNN.mean(), score_KNN.std()))

    # Split the data to implement 10-fold x-validation
    kf = KFold(n_splits=10)
    for train, test in kf.split(feature):
        # print('k fold')
        # print("%s %s" % (train, test))
        feature = np.array(feature)
        # print('feature is ')
        # print(feature)
        ground_truth = np.array(ground_truth)
        # print('Ground truth is')
        # print(ground_truth)
        feature_train, feature_test, ground_truth_train, ground_truth_test = feature[train], feature[test], ground_truth[train], ground_truth[test]

        SVC_model = svm.SVC()
        KNN_model = KNeighborsClassifier(n_neighbors=5)

        SVC_model.fit(feature_train, ground_truth_train)
        KNN_model.fit(feature_train, ground_truth_train)

        SVC_prediction = SVC_model.predict(feature_test)
        KNN_prediction = KNN_model.predict(feature_test)

        # Check accuracy of SVC/KNN classifier
        SVC_acc = accuracy_score(SVC_prediction, ground_truth_test)
        KNN_acc = accuracy_score(KNN_prediction, ground_truth_test)

        # Get confusion matrix of SVC/KNN classifier
        SVC_confusion_matrix = confusion_matrix(SVC_prediction, ground_truth_test)
        KNN_confusion_matrix = confusion_matrix(KNN_prediction, ground_truth_test)

        # Get classification report of SVC/KNN classifier, where can check precision, recall
        SVC_report = classification_report(SVC_prediction, ground_truth_test)
        KNN_report = classification_report(KNN_prediction, ground_truth_test)

        print('accuracy SVC')
        print(accuracy_score(SVC_prediction, ground_truth_test))
        print('accuracy KNN')
        print(accuracy_score(KNN_prediction, ground_truth_test))
        print(confusion_matrix(SVC_prediction, ground_truth_test))
        print(classification_report(KNN_prediction, ground_truth_test))

        write_data.write_data(score_SVC, mean_SVC,
                              std_SVC, SVC_acc, score_KNN,
                              mean_KNN,
                              std_KNN, SVC_acc, SVC_confusion_matrix, SVC_report, KNN_acc, KNN_confusion_matrix,
                              KNN_report,
                              file_name)
