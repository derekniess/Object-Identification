import numpy as np
import math
import time

"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):
    
    #IMPLEMENT HERE
    print("weight size = ", w1.shape)
    print("bias size = ", b1.shape)
    print("weight size = ", w2.shape)
    print("bias size = ", b2.shape)
    print("weight size = ", w3.shape)
    print("bias size = ", b3.shape)
    losses = [0 for i in range(epoch)]
    for j in range(epoch):
        X = x_train
        Y = y_train
        # losses = 0
        if shuffle:
            c = list(zip(x_train, y_train))
            np.random.shuffle(c)
            X = list(zip(*c))[0]
            Y = list(zip(*c))[1]
        for i in range(int(len(X) / 200.0)):
            X_T = X[0+i*200:200+i*200][:]
            Y_T = Y[0+i*200:200+i*200][:]
            # print("size of X_T: ", len(X_T))
            Z1, acache1 = affine_forward(X_T, w1, b1)
            A1, rcache1 = relu_forward(Z1)
            Z2, acache2 = affine_forward(A1, w2, b2)
            A2, rcache2 = relu_forward(Z2)
            Z3, acache3 = affine_forward(A2, w3, b3)
            A3, rcache3 = relu_forward(Z3)
            F, acache4  = affine_forward(A3, w4, b4)
           # print ("The size of F is : ",len(F))
            loss, dF = cross_entropy(F, Y_T)
            dA3, dW4, dB4 = affine_backward(dF, acache4)
            dZ3 = relu_backward(dA3, rcache3)
            dA2, dW3, dB3 = affine_backward(dZ3, acache3)
            dZ2 = relu_backward(dA2, rcache2)
            dA1, dW2, dB2 = affine_backward(dZ2, acache2)
            dZ1 = relu_backward(dA1, rcache1)
            dX,  dW1, dB1 = affine_backward(dZ1, acache1)
            w1 = w1 - 0.1 * dW1    
            w2 = w2 - 0.1 * dW2
            w3 = w3 - 0.1 * dW3
            w4 = w4 - 0.1 * dW4
            b1 = b1 - 0.1 * dB1
            b2 = b2 - 0.1 * dB2
            b3 = b3 - 0.1 * dB3
            b4 = b4 - 0.1 * dB4   
            losses[j] += loss

    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):

    avg_class_rate = 0.0
    class_rate_per_class = [0.0] * num_classes
    test = np.zeros(len(y_test))
    actual = np.zeros(len(y_test))
    Z1, acache1 = affine_forward(x_test, w1, b1)
    A1, rcache1 = relu_forward(Z1)
    Z2, acache2 = affine_forward(A1, w2, b2)
    A2, rcache2 = relu_forward(Z2)
    Z3, acache3 = affine_forward(A2, w3, b3)
    A3, rcache3 = relu_forward(Z3)
    F, acache4  = affine_forward(A3, w4, b4)
    for i in range(len(F)):
        val = -math.inf
        label = -1
        for j in range(len(F[i])):
            if F[i][j] > val:
                val = F[i][j]
                label = j
        test[i] = label
        actual[i]=y_test[i]

    for i in range(len(test)):
        if test[i] == actual[i]:
            avg_class_rate = avg_class_rate + 1
            class_rate_per_class[int(test[i])] = class_rate_per_class[int(test[i])] + 1

    for i in range(num_classes):
        actual[i] = np.count_nonzero(y_test == i)
        class_rate_per_class[i] = class_rate_per_class[i] / actual[i]

    avg_class_rate = avg_class_rate / len(x_test)
    return avg_class_rate, class_rate_per_class, test

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn():
    pass

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    
    # print("A ( ",len(A), ", ", len(A[0]), ")")
    # print("W ( ",len(W), ", ", len(W[0]), ")")

    # t = time.time()
    Z = np.dot(A,W)
    # print ("opt 1. ", time.time() - t)
    
    # t = time.time()
    # Z_t = [[0 for x in range(len(W[0]))] for y in range(len(A))]
    # for out_y in range(len(A)):
    #     for out_x in range(len(W[0])):
    #         sum = 0
    #         for i in range(len(A[0])):
    #             sum += A[out_y][i] * W[i][out_x]
    #         Z_t[out_y][out_x] = sum
    # print ("orig 1. ", time.time() - t,"\n")


    # t = time.time()
    Z[...] += b
    # print ("opt 2. ", time.time() - t)
    
    # t = time.time()
    # for i in Z_t:
    #     i = i + b
    # print ("orig 2. ", time.time() - t, "\n")

    cache = (A, W, b)
    
    return Z, cache

def affine_backward(dZ, cache):
    
    # print("dZ ( ",len(dZ), ", ", len(dZ[0]), ")")

    # t = time.time()
    dB = np.sum(dZ, axis=0)
    # print ("opt 3. ", time.time() - t)

    # t = time.time()
    # dB = np.copy(cache[2])
    # index = 0
    # for i in dZ.T:
    #     val = 0
    #     for j in i:
    #         val = val + j
    #     dB[index] = val
    #     index = index + 1
    # print ("orig 3. ", time.time() - t)

    dA = np.dot(dZ, cache[1].T)    
    x = np.array(cache[0])
    x = x.T
    dW = np.dot(x, dZ)

    return dA, dW, dB

def relu_forward(Z):
    
    # print("Z ( ",len(Z), ", ", len(Z[0]), ")")

    # t = time.time()
    A = Z.clip(min = 0)
    # print ("opt 4. ", time.time() - t)

    # t = time.time()
    # A = np.copy(Z)
    # for i in A:
    #     for j in i:
    #         j = max(0, j)
    # print ("orig 4. ", time.time() - t)
    
    cache = np.copy(A)

    return A, cache

def relu_backward(dA, cache):
    dZ = np.copy(dA)
    for row in range(cache.shape[0]):
        for col in range(cache.shape[1]):
            if cache[row,col] == 0:
                dZ[row,col] = 0
    return dZ

def cross_entropy(F, y):
    y = np.array(y)

    # t = time.time()
    exp_F = np.exp(F)
    sums = np.sum(exp_F, axis=1)
    loss = 0
    for i in range(len(F)):
        label = int(y[i])
        loss += F[i][label] - np.log(sums[i])
    # print ("opt 5. ", time.time() - t)

    # print("exp_F ( ",len(exp_F), ", ", len(exp_F[0]), ")")
    # print("F ( ",len(F), ", ", len(F[0]), ")")

    # t = time.time()
    # loss = 0
    # for i in range(len(F)):
    #     label = y[i]
    #     label = int(label)
    #     loss = loss + F[i][label]
    #     sum = 0
    #     for j in F[i]:
    #         sum = sum + np.exp(j)
    #     loss = loss - np.log(sum)
    # print ("orig 5. ", time.time() - t)

    loss = -loss / len(y)

    dF = np.copy(F)
    for i in range(len(F)):
        exp = 0
        for j in F[i]:
            exp = exp + np.exp(j)
        for j in range(len(F[i])):
            func = 0
            if j == y[i]:
                func = 1
            fval = func - (np.exp(F[i][j])/exp)
            fval = -fval / len(y)
            dF[i][j] = fval
    
    return loss, dF
