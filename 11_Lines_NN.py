import numpy as np

# sigmoid function
def sigmoid(x, deriv = False):
    if(deriv == True):
        return x*(1 - x)
    return 1 / (1 + np.exp(-x))

# input data

X = np.array([  [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1] ])

# output data
y = np.array([[0, 0, 1, 1]]).T

# Seed for deterministic behaviour
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

print(syn0)

for iter in range(10000):
    
    # forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))

    # calculate naive error
    l1_error = y - l1
    
    # print error to visualize training progress
    if(iter % 1000 == 0):
        print(l1_error)
    
    # Naive learning approach
    l1_delta = l1_error * sigmoid(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

result1 = l1

y = np.array([[0, 1, 1, 0]]).T

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in range(60000):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    # how much did we miss the target value?
    l2_error = y - l2

    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*sigmoid(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * sigmoid(l1, deriv = True)

    # update weigths
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
print("Output of first network after Training")
print(result1)

print("Output of second network after Training")
print(l2)

