from svmlib import *
from os import listdir

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

# Get our data set and labels(real number)
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

# Get -1 and 1 class set
def GetClassSet(labels):
    ret_set = []
    for number in range(10):
        class_set = [1 if label == number else -1 for label in labels]
        ret_set.append(class_set)
    return ret_set

# Make multi-class classifier by using one-vs-all method
def OVR(data_set, labels, k_tup):
    bs = []
    alphas = []
    # Train 0~9 classifier
    class_sets = []
    for number in range(10):
        class_set = [1 if label == number else -1 for label in labels]      # Create binary labels
        # Get b and alpha in optimization
        b, alpha = SMO(data_set, class_set, 200, 0.0001, 10000, k_tup)
        print("Current number:%d" % number)
        bs.append(b)
        alphas.append(alpha)
        class_sets.append(class_set)
    return bs, alphas, class_sets

def GetSupportVectors(data_set, labels, alphas):
    sv_indexes = []
    svs_all_num = []
    labels_svs_all_num = []
    for alpha in alphas:
        sv_index = np.nonzero(alpha.A > 0)[0]
        svs = data_set[sv_index]
        labels_svs = labels[sv_index]
        # Add them to lists
        sv_indexes.append(sv_index)
        svs_all_num.append(svs)
        labels_svs_all_num.append(labels_svs)
    return sv_indexes, svs_all_num, labels_svs_all_num

# Store svm parameters
def StoreParams(alphas, bs):
    import pickle
    with open("./SvmParams/alphas.txt", "wb") as f1, open("./SvmParams/bs.txt", "wb") as f2:
        pickle.dump(alphas, f1)
        pickle.dump(bs, f2)

# Load parameters from existing files
def LoadParams():
    import pickle
    with open("./SvmParams/alphas.txt", "rb") as f1, open("./SvmParams/bs.txt", "rb") as f2:
        alphas = pickle.load(f1)
        bs = pickle.load(f2)
    return alphas, bs


def TestRBF(k_tup=("rbf", 10)):
    data_set, labels = GetDataSetByDir("./DataSet/TrainingDigits/")
    # Get b and alphas
    # bs, alphas, class_sets = OVR(data_set, labels, k_tup)
    # StoreParams(alphas, bs)
    # Convert data set to matrix and get their attribute
    alphas, bs = LoadParams()
    class_sets = GetClassSet(labels)
    data_set = np.mat(data_set)
    labels = np.mat(labels).transpose()
    m, n = np.shape(data_set)
    # Get support vectors
    sv_indexes, svs_all_num, labels_svs_all_num = GetSupportVectors(data_set, labels, alphas)
    error_count = 0
    for i in range(m):
        result = -1
        for num in range(10):
            kernel_eval = KernelTrans(svs_all_num[num], data_set[i, :], k_tup)
            predict = kernel_eval.T * np.multiply(labels_svs_all_num[num], alphas[num][sv_indexes[num]]) + bs[num]
            if np.sign(predict) == np.sign(class_sets[num][i]):
                result = num
                break
        print("The real answer is %d, and the classifier came back with %d" %
              (labels[i], result))
        if result != labels[i]:
            error_count += 1
    print("The training set error is %f" % (error_count / m))
    # Get test set error
    data_set, labels = GetDataSetByDir("./DataSet/TestDigits/")
    class_sets = GetClassSet(labels)
    error_count = 0
    data_set = np.mat(data_set)
    labels = np.mat(labels).transpose()
    m, n = np.shape(data_set)
    for i in range(m):
        result = -1
        for num in range(10):
            kernel_eval = KernelTrans(svs_all_num[num], data_set[i, :], k_tup)
            predict = kernel_eval.T * np.multiply(labels_svs_all_num[num], alphas[num][sv_indexes[num]]) + bs[num]
            if np.sign(predict) == np.sign(class_sets[num][i]):
                result = num
                break
        print("The real answer is %d, and the classifier came back with %d" %
              (labels[i], result))
        if result != labels[i]:
            error_count += 1
    print("The test set error is %f" % (error_count / m))

if __name__ == "__main__":
    TestRBF()
