from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn import model_selection as ms
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
    data = pd.read_csv(ground_truth_file)
    X = arr
    y = data['Classification']
    clf_one = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=27)
    score_SVC = cross_val_score(clf_one, X, y, cv=cv)
    mean_SVC = score_SVC.mean()
    std_SVC = score_SVC.std()
    print("%0.2f accuracy with a standard deviation of %0.2f" % (score_SVC.mean(), score_SVC.std()))
    # print('the accuracy of a linear kernel support vector machine on the dataset')
    # print(scores)
    clf_two = make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    score_KNN = cross_val_score(clf_two, X, y, cv=cv)
    mean_KNN = score_KNN.mean()
    std_KNN = score_KNN.std()
    print("%0.2f accuracy with a standard deviation of %0.2f" % (score_KNN.mean(), score_KNN.std()))

    kf = KFold(n_splits=10)
    for train, test in kf.split(X):
        # print('k fold')
        # print("%s %s" % (train, test))
        X = np.array(X)
        # print('X is ')
        # print(X)
        y = np.array(y)
        # print('Y is')
        # print(y)
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        SVC_model = svm.SVC()
        KNN_model = KNeighborsClassifier(n_neighbors=5)

        SVC_model.fit(X_train, y_train)
        KNN_model.fit(X_train, y_train)

        SVC_prediction = SVC_model.predict(X_test)
        KNN_prediction = KNN_model.predict(X_test)
        SVC_acc = accuracy_score(SVC_prediction, y_test)
        KNN_acc = accuracy_score(KNN_prediction, y_test)
        SVC_confusion_matrix = confusion_matrix(SVC_prediction, y_test)
        KNN_confusion_matrix = confusion_matrix(KNN_prediction, y_test)
        SVC_report = classification_report(SVC_prediction, y_test)
        KNN_report = classification_report(KNN_prediction, y_test)
        print('accuracy SVC')
        print(accuracy_score(SVC_prediction, y_test))
        print('accuracy KNN')
        print(accuracy_score(KNN_prediction, y_test))
        print(confusion_matrix(SVC_prediction, y_test))
        print(classification_report(KNN_prediction, y_test))

        write_data.write_data(score_SVC, mean_SVC,
                              std_SVC, SVC_acc, score_KNN,
                              mean_KNN,
                              std_KNN, SVC_acc, SVC_confusion_matrix, SVC_report, KNN_acc, KNN_confusion_matrix,
                              KNN_report,
                              file_name)
