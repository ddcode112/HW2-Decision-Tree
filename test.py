import sys
import numpy as np

if __name__ == '__main__':
    trainFile = sys.argv[1]
    inspect = sys.argv[2]

    print("The train file is: %s" % trainFile)
    print("The inspect file is: %s" % inspect)

train_data = np.genfromtxt(trainFile, delimiter='\t', dtype=None, encoding=None)
title = train_data[0, ]
train_data = train_data[1:, ]


def matrix_info(data):
    rows = data.shape[0]
    i, counts = np.unique(data[:, -1], return_counts=True)
    return rows, i, counts


def majority_vote(data):
    if not data:
        return None
    rows, i, counts = matrix_info(data)
    max_count = np.amax(counts)
    majority_label = ''

    if np.count_nonzero(counts == max_count) > 1:
        for index, v in np.ndenumerate(counts):
            if v == max_count:
                majority_label = max(majority_label, i[index])
    else:
        majority_label = i[np.argmax(counts)]

    print("Majority label: " + str(majority_label))
    return majority_label


def entropy_cal(rows, label_counts):
    e = label_counts / rows
    e = np.sum(-np.log2(e) * e)
    return e


def mutual_info_cal(entropy_y, subset_by_feature):
    total_rows = subset_by_feature.shape[0]
    a = np.unique(subset_by_feature[:, 0])
    con_entropy = 0
    for feature_value in range(len(a)):
        a_set = subset_by_feature[subset_by_feature[:, 0] == a[feature_value]]
        a_rows, a_i, a_counts = matrix_info(a_set)
        a_entropy = entropy_cal(a_rows, a_counts)
        con_entropy += (a_rows / total_rows) * a_entropy

    return entropy_y - con_entropy


def split_criterion(title, data):
    rows, i, counts = matrix_info(data)
    entropy_y = entropy_cal(rows, counts)
    mi = []
    for d in range(len(data[0]) - 1):
        mi.append(mutual_info_cal(entropy_y, data[:, [d, -1]]))

    mi = np.array(mi)
    max_id = np.argmax(mi)
    if mi[max_id] > 1:
        return title[max_id]
    else:
        return None


print(split_criterion(title, train_data))
