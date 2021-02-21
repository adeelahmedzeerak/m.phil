import numpy as np


def tanh(a):
    return 2*sigmoid(2*a)

def tanh_derivative(a):
   return (1-(tanh(a)**2))  

def relu(a):
    if a > 0:
        return a
    else:
        return 0
  
def sigmoid(a):
    return 1/(1 + np.exp(-a))

def relu_derivative(a): 
    if a > 0:
        return 1
    else:
        return 0

def sigmoid_derivative(a):
    return np.exp(-a) / ((1 + np.exp(-a)) ** 2)

def lrelu(a, k):
    if a > 0:
        return a
    else:
        return k*a

def lrelu_derivative(a, k):  
    if a > 0:
        return 1
    else:
        return k

