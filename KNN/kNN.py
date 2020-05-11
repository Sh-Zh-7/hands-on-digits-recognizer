from os import listdir
import numpy as np
import operator

# Convert matrix to vector
def Img2Vector(filename):
    # Return matrix
    ret_val = np.zeros((1, 1024))
    # Read file's content
    file = open(filename)
    content = file.readlines()
    for i in range(32):
        line = content[i]
        for j in range(32):
            ret_val[0, 32 * i + j] = int(line[j])
    return ret_val

# Normalize our data set
def AutoNorm(data_set):
    # Get all feature's limits
    min_val = data_set.min(0)
    max_val = data_set.max(0)
    val_range = max_val - min_val
    # Normalize our data set
    m = data_set.shape[0]
    normal_set = data_set - np.tile(min_val, (m, 1))
    normal_set = normal_set / np.tile(val_range, (m, 1))
    return normal_set, min_val, val_range

# Build our classifier
def ClassifyTrain(in_vector, data_set, labels, k):
    m = data_set.shape[0]
    # Calculate distance and sort
    diff_mat = data_set - np.tile(in_vector, (m, 1))
    square_diff_mat = diff_mat ** 2
    # Remember! axis is one!!!
    sum_square_diff_mat = np.sum(square_diff_mat, axis=1)
    indexes = np.argsort(sum_square_diff_mat)
    # Select K distance
    class_count = {}
    for i in range(k):
        index = indexes[i]
        label = labels[index]
        class_count[label] = class_count.get(label, 0) + 1
    result = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # Get the most similar label
    return result[0][0] 

# Get our data set
def GetDataSetByDir(dirname):
    training_set_dir = listdir(dirname)
    m = len(training_set_dir)
    # Prepare our data set and targets
    labels = []
    data_set = np.zeros((m, 1024))
    # Get our data set and targets
    for i in range(m):
        filename = training_set_dir[i]
        label = int(filename.split('_')[0])
        labels.append(label)
        data_set[i, :] = Img2Vector(dirname + filename)
    return data_set, labels


# Create Classify by using kNN
def ClassifyTest():
    # Select our directory
    test_dir_name = "./DataSet/TestDigits/"
    train_dir_name = "./DataSet/TrainingDigits/"
    data_set, labels = GetDataSetByDir(train_dir_name)
    test_set, test_labels = GetDataSetByDir(test_dir_name)
    # Normalize
    # data_set, min_val, val_range = AutoNorm(data_set)
    # Test and calculate error rate
    error_count = 0
    test_num = len(listdir(test_dir_name))
    for i in range(test_num):
        result = ClassifyTrain(test_set[i, :], data_set, labels, 3)
        print("The real answer is %d, and the classifier came back with %d" %
              (test_labels[i], result))
        if test_labels[i] != result:
            error_count += 1
    print("The error rate is %f" % (error_count / test_num))

if __name__ == "__main__":
    ClassifyTest()


