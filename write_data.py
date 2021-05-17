import csv
import os.path


def write_data(score_SVC, mean_SVC,
               std_SVC, SVC_acc, score_KNN,
               mean_KNN,
               std_KNN, acc1, mat1, re1, acc2, mat2, re2, file_name):
    # Write classifier evaluation result into a csv file
    with open(file_name, mode='a', newline='') as csv_file:
        field = ['Score_SVC', 'Score_KNN', 'Mean_SVC', 'Mean_KNN', 'Std_SVC', 'Std_KNN', 'Accuracy_SVC_non',
                 'Confusion_matrix_SVC_non', 'Report_SVC_non', 'Accuracy_KNN_non',
                 'Confusion_matrix_KNN_non', 'Report_KNN_non']
        writer = csv.DictWriter(csv_file, fieldnames=field, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fileEmpty = os.stat(file_name).st_size == 0
        if fileEmpty:
            writer.writeheader()
        writer.writerow(
            {'Score_SVC': score_SVC, 'Score_KNN': score_KNN, 'Mean_SVC': mean_SVC, 'Mean_KNN': mean_KNN,
             'Std_SVC': std_SVC, 'Std_KNN': std_KNN,
             'Accuracy_SVC_non': acc1, 'Confusion_matrix_SVC_non': mat1, 'Report_SVC_non': re1,
             'Accuracy_KNN_non': acc2, 'Confusion_matrix_KNN_non': mat2, 'Report_KNN_non': re2})
