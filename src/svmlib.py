import numpy as np
import random


# Kernel transporting
def KernelTrans(X, A, k_tup):
    m, n = np.shape(X)
    k = np.mat(np.zeros((m, 1)))
    if k_tup[0] == "lin":
        # Linear Kernel
        k = X * A.T
    elif k_tup[0] == "rbf":
        # Gauss Kernel
        for j in range(m):
            delta_row = X[j, :] - A
            k[j] = delta_row * delta_row.T
        k = np.exp(k / (-1 * k_tup[1] ** 2))
    return k


# A helper class
# Store necessary data of svm
class HelperStruct:
    def __init__(self, data_set, labels, C, toler, k_tup):
        # Store parameters in class's attributes
        self.X = data_set
        self.Y = labels
        self.C = C
        self.toler = toler
        # Get number of data
        self.m = np.shape(self.X)[0]
        # Get all necessary parameters
        self.alpha = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.error_cache = np.mat(np.zeros((self.m, 2)))  # Array of error rate
        self.k = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:, i] = KernelTrans(self.X, self.X[i, :], k_tup)


# Calculate error
# hs means a HelperStruct object
def GetEk(hs, k):
    fx = float(np.multiply(hs.alpha, hs.Y).T * hs.k[:, k] + hs.b)
    e = fx - float(hs.Y[k])
    return e


# Randomly select index j
def SelectRandJ(i, m):
    j = i
    while i == j:
        j = int(random.uniform(0, m))
    return j


# Clip alpha
def ClipAlpha(a, high, low):
    if a > high:
        a = high
    if a < low:
        a = low
    return a


# Randomly select index j
def SelectJ(i, hs, e_i):
    max_k = -1
    max_e = 0
    e_j = 0
    hs.error_cache[i] = [1, e_i]
    valid_error_cache_list = np.nonzero(hs.error_cache[:, 0].A)[0]
    if len(valid_error_cache_list) > 1:
        for k in valid_error_cache_list:
            if k == i:
                continue
            e_k = GetEk(hs, k)
            delta_e = abs(e_i - e_k)
            if delta_e > max_e:
                max_k = k
                max_e = delta_e
                e_j = e_k
        return max_k, e_j
    else:
        j = SelectRandJ(i, hs.m)
        e_j = GetEk(hs, j)
    return j, e_j


# Update error cache
def UpdateEk(hs, k):
    e_k = GetEk(hs, k)
    hs.error_cache[k] = [1, e_k]


# Do what SMO algorithm inner loop do
def InnerLoop(i, hs):
    e_i = GetEk(hs, i)
    if (hs.Y[i] * e_i < -hs.toler) and (hs.alpha[i] < hs.C) or \
            (hs.Y[i] * e_i > hs.toler) and (hs.alpha[i] > 0):
        j, e_j = SelectJ(i, hs, e_i)
        alpha_i_old = hs.alpha[i].copy()
        alpha_j_old = hs.alpha[j].copy()
        # Calculate lower and upper bound of a
        if hs.Y[i] != hs.Y[j]:
            low = max(0, hs.alpha[j] - hs.alpha[i])
            high = min(hs.C, hs.C + hs.alpha[j] - hs.alpha[i])
        else:
            low = max(0, hs.alpha[j] + hs.alpha[i] - hs.C)
            high = min(hs.C, hs.alpha[j] + hs.alpha[i])
        if low == high:
            print("L == H")
            return 0
        # Calculate eta
        eta = 2.0 * hs.k[i, j] - hs.k[i, i] - hs.k[j, j]
        if eta >= 0:
            print("eta >= 0")
            return 0
        # Calculate ai and aj
        hs.alpha[j] -= hs.Y[j] * (e_i - e_j) / eta
        hs.alpha[j] = ClipAlpha(hs.alpha[j], high, low)
        UpdateEk(hs, j)  # Update error cache
        if abs(hs.alpha[j] - alpha_j_old) < 0.00001:
            print("j not moving enough")
            return 0
        hs.alpha[i] += hs.Y[j] * hs.Y[i] * (alpha_j_old - hs.alpha[j])
        UpdateEk(hs, i)  # Update error cache
        # Calculate b
        b1 = hs.b - e_i - hs.Y[i] * (hs.alpha[i] - alpha_i_old) * hs.k[i, i] - \
             hs.Y[j] * (hs.alpha[j] - alpha_j_old) * hs.k[i, j]
        b2 = hs.b - e_j - hs.Y[i] * (hs.alpha[i] - alpha_i_old) * hs.k[i, j] - \
             hs.Y[j] * (hs.alpha[j] - alpha_j_old) * hs.k[j, j]
        if 0 < hs.alpha[i] < hs.C:
            hs.b = b1
        elif 0 < hs.alpha[j] < hs.C:
            hs.b = b2
        else:
            hs.b = (b1 + b2) / 2
        return 1
    else:
        return 0


# The real SMO algorithm
def SMO(data_set, labels, C, toler, max_iter, k_tup):
    hs = HelperStruct(np.mat(data_set), np.mat(labels).transpose(), C, toler, k_tup)
    iter = 0
    entire_set = True
    alpha_pairs_change = 0
    while (iter < max_iter) and (alpha_pairs_change > 0 or entire_set):
        alpha_pairs_change = 0
        if entire_set:
            for i in range(hs.m):
                alpha_pairs_change += InnerLoop(i, hs)
                print("Fullset, iter: %d i %d, pairs changed %d" %
                      (iter, i, alpha_pairs_change))
                iter += 1
        else:
            non_bound_i = np.nonzero((hs.alpha.A > 0) * (hs.alpha.A < C))[0]
            for i in non_bound_i:
                alpha_pairs_change += InnerLoop(i, hs)
                print("non-bound, iter: %d i:%d, pairs changed %d" %
                      (iter, i, alpha_pairs_change))
            iter += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_change == 0:
            entire_set = True
        print("iteration number: %d" % iter)
    return hs.b, hs.alpha


# Calculate w
def GetW(alpha, data, labels):
    X = np.mat(data)
    labels = np.mat(labels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alpha[i] * labels[i], X[i, :].T)
    return w
