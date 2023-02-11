import sys
import numpy as np

if __name__ == '__main__':
    trainFile = sys.argv[1]
    inspect = sys.argv[2]

    print("The train file is: %s" % trainFile)
    print("The inspect file is: %s" % inspect)

train_data = np.genfromtxt(trainFile, delimiter='\t', dtype=None, encoding=None)
train_data = train_data[1:, ]


def stats(data):
    rows = data.shape[0]
    i, counts = np.unique(data[:, -1], return_counts=True)
    maxcount = np.amax(counts)
    majority_label = ''

    if np.count_nonzero(counts == maxcount) > 1:
        for index, v in np.ndenumerate(counts):
            if v == maxcount:
                majority_label = max(majority_label, i[index])
    else:
        majority_label = i[np.argmax(counts)]

    return rows, counts, majority_label


def countEntropy(rows, label_counts):
    e = label_counts / rows
    e = np.sum(-np.log2(e)*e)
    return e


def countError(data, rows, majority_label):
    error = 0
    for i in range(rows):
        if data[i, -1] != majority_label:
            error += 1
    return error/rows


rows, label_counts, majority_label = stats(train_data)
entropy = countEntropy(rows, label_counts)
error_rate = countError(train_data, rows, majority_label)


with open(inspect, 'w') as f_inspect:
    f_inspect.write('entropy: '+str(entropy)+'\n')
    f_inspect.write('error: '+str(error_rate))
