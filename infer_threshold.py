import numpy
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
import csv
import pandas as pd

data_file = 'data.csv'
ground_truth_file = 'classification.csv'


def main():
    ground_truth_one, ground_truth_two, similarity_score_uni, similarity_score_lev, similarity_score_jac, similarity_score_pair, similarity_score_cos = get_data()
    infer_threshold(ground_truth_one, similarity_score_uni)

    infer_threshold(ground_truth_two, similarity_score_uni)

    infer_threshold(ground_truth_one, similarity_score_lev)

    infer_threshold(ground_truth_two, similarity_score_lev)

    infer_threshold(ground_truth_one, similarity_score_jac)

    infer_threshold(ground_truth_two, similarity_score_jac)

    infer_threshold(ground_truth_one, similarity_score_pair)

    infer_threshold(ground_truth_two, similarity_score_pair)

    infer_threshold(ground_truth_one, similarity_score_cos)

    infer_threshold(ground_truth_two, similarity_score_cos)


def get_data():
    ground_truth_list_one = []
    ground_truth_list_two = []
    similarity_score_uni = []
    similarity_score_lev = []
    similarity_score_jac = []
    similarity_score_pair = []
    similarity_score_cos = []  # small model
    file = pd.read_csv(data_file)

    similairty_universal = file['Similarity_universal_normalized']
    similarity_lev = file['Similarity_Levenshtein']
    similarity_jac = file['Similarity_Jaccard']
    similarity_pair = file['Similarity_Pairwise_normalized']
    similarity_cos = file['Similarity_cosine_large_model_nonnormalized']
    file_class = pd.read_csv(ground_truth_file)
    column_classification = file_class['Classification']
    for row in similairty_universal:
        similarity_score_uni.append(row)
    for row in similarity_lev:
        similarity_score_lev.append(row)

    for row in similarity_jac:
        similarity_score_jac.append(row)

    for row in similarity_pair:
        similarity_score_pair.append(row)
    for row in similarity_cos:
        similarity_score_cos.append(row)

    for row in column_classification:
        if 'consistent' in str(row):
            ground_truth_one = 1
            ground_truth_two = 1
        if 'suspected' in str(row):
            ground_truth_one = 1
            ground_truth_two = 0
        if 'irrelevant' in str(row):
            ground_truth_one = 0
            ground_truth_two = 0
        ground_truth_list_one.append(ground_truth_one)
        ground_truth_list_two.append(ground_truth_two)

    return ground_truth_list_one, ground_truth_list_two, \
           similarity_score_uni, similarity_score_lev, \
           similarity_score_jac, similarity_score_pair, similarity_score_cos


def infer_threshold(ground_truth, similarity_score):
    # predict class values
    precisions, recalls, thresholds = precision_recall_curve(ground_truth, similarity_score)

    # Calculate the f-score
    fscores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the optimal threshold
    index = numpy.argmax(fscores)

    results = '\n'.join([
        f"Index:     {index} / {len(thresholds)}",
        f"Threshold: {thresholds[index]:4.2f}",
        f"F-score:   {fscores[index]:4.2f}",
        f"Recall:    {recalls[index]:4.2f}",
        f"Precision: {precisions[index]:4.2f}",
    ])

    index_text = ''.join(f'{index} / {len(thresholds)}')
    print(index_text)
    print(results)


if __name__ == '__main__':
    main()
