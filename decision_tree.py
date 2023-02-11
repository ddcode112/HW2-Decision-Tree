import sys
import numpy as np


class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    """

    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.depth = None
        self.value = {}


if __name__ == '__main__':
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    max_depth = int(sys.argv[3])
    trainOutFile = sys.argv[4]
    testOutFile = sys.argv[5]
    metrics = sys.argv[6]

    print("The train file is: %s" % trainFile)
    print("The test file is: %s" % testFile)
    print("The maximum depth is: %s" % max_depth)
    print("The train output file is: %s" % trainOutFile)
    print("The test output file is: %s" % testOutFile)
    print("The metrics file is: %s" % metrics)


def matrix_info(data):
    rows = data.shape[0]
    i, counts = np.unique(data[:, -1], return_counts=True)
    return rows, i, counts


def majority_vote(data):
    if data is None:
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
        con_entropy += ((a_rows / total_rows) * a_entropy)
    return entropy_y - con_entropy


def split_criterion(title, data):
    rows, i, counts = matrix_info(data)
    entropy_y = entropy_cal(rows, counts)
    mi = []
    for d in range(data.shape[1] - 1):
        mi.append(mutual_info_cal(entropy_y, data[:, [d, -1]]))

    mi = np.array(mi)
    max_id = np.argmax(mi)
    if mi[max_id] > 0:
        return title[max_id]
    else:
        return None


def train(title, data, depth, master_majority_label):
    root = train_recurse(title, data, depth, 0, master_majority_label)
    return root


def train_recurse(title, data, depth, cur_d, master_majority_label):
    n = Node()
    n.depth = cur_d
    check_feature = False
    for x in range(data.shape[1]-1):
        if np.all(data[:, x] == data[0, x]):
            check_feature = True
        else:
            check_feature = False
            break

    if data.shape[1] <= 1 or np.all(data[:, -1] == data[0, -1]) or check_feature or (cur_d >= depth):
        if majority_vote(data):
            n.vote = majority_vote(data)
        else:
            n.vote = master_majority_label
    else:
        split_feature = split_criterion(title, data)
        if split_feature:
            n.attr = split_feature
            feature_id = np.where(title == split_feature)[0][0]
            i = np.unique(data[:, feature_id])
            i = np.sort(i)
            new_title = np.delete(title, feature_id)
            if new_title.shape[0] >= 1:
                n.value[i[0]] = data[data[:, feature_id] == i[0]]
                n.value[i[1]] = data[data[:, feature_id] == i[1]]
                data_1 = np.delete(n.value[i[0]], feature_id, axis=1)
                data_2 = np.delete(n.value[i[1]], feature_id, axis=1)
                n.left = train_recurse(new_title, data_1, depth, n.depth + 1, master_majority_label)
                n.right = train_recurse(new_title, data_2, depth, n.depth + 1, master_majority_label)

    return n


def print_tree(data, root):
    if data is not None:
        rows, i, count = matrix_info(data)
        i = np.sort(i)
        print(f'[{count[0]} {i[0]}/{count[1]} {i[1]}]')

    def recurse_tree(root, i):
        if root:
            if root.attr:
                x = 0
                for k, v in root.value.items():
                    a, b, c = matrix_info(v)
                    depth_note = '|'*(root.depth+1)
                    if b.shape[0] < 2:
                        if b[0] == i[0]:
                            print(f'{depth_note}{root.attr} = {k}: [{c[0]} {b[0]}/0 {i[1]}]')
                        else:
                            print(f'{depth_note}{root.attr} = {k}: [0 {i[0]}/{c[0]} {b[0]}]')
                    else:
                        print(f'{depth_note}{root.attr} = {k}: [{c[0]} {b[0]}/{c[1]} {b[1]}]')
                    if x == 0:
                        x += 1
                        recurse_tree(root.left, i)
                    else:
                        recurse_tree(root.right, i)

    recurse_tree(root, i)


def predict(title, data, root):
    predict_result = []

    def predict_by_row(r, root):
        if not root or not r:
            return None
        if root.vote:
            return root.vote
        k = root.value.keys()
        k = list(k)
        k.sort()
        if r[root.attr] == k[0]:
            result = predict_by_row(r, root.left)
        else:
            result = predict_by_row(r, root.right)
        return result

    for i in range(data.shape[0]):
        r = {}
        for t in range(title.shape[0]):
            r[title[t]] = data[i, t]
        predict_result.append(predict_by_row(r, root))

    return predict_result


def error_cal(data, result):
    error = 0
    rows = len(result)
    for i in range(rows):
        if data[i] != result[i]:
            error += 1
    return error / rows


def write_result_to_file(result, outfile):
    with open(outfile, 'w') as f:
        for line in result:
            f.write(line + '\n')


train_data = np.genfromtxt(trainFile, delimiter='\t', dtype=None, encoding=None)
title = train_data[0, ]
train_data = train_data[1:, ]

master_majority_label = majority_vote(train_data)
root = train(title, train_data, max_depth, master_majority_label)
print_tree(train_data, root)

test_data = np.genfromtxt(testFile, delimiter='\t', dtype=None, encoding=None)
title_t = test_data[0, ]
test_data = test_data[1:, ]
predict_result_train = predict(title, train_data, root)
predict_result_test = predict(title_t, test_data, root)
write_result_to_file(predict_result_train, trainOutFile)
write_result_to_file(predict_result_test, testOutFile)
train_error = error_cal(train_data[:, -1], predict_result_train)
test_error = error_cal(test_data[:, -1], predict_result_test)


with open(metrics, 'w') as f_metrics:
    f_metrics.write('error(train): ' + str(train_error) + '\n')
    f_metrics.write(('error(test): ' + str(test_error)))
