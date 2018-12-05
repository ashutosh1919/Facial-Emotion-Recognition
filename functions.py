import pandas as pd
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A,cache

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A,cache

def softmax(Z):
    t = np.exp(Z)
    s = np.sum(t)
    A = t/s
    cache = Z
    return A,s,cache

def sigmoid_backward(dA,Z):
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def relu_backward(dA,Z):
    dZ = np.array(dA,copy=True)
    dZ[Z<=0] = 0
    return dZ

def softmax_backward(y,y_hat):
    dZ = -y + y_hat(np.sum(y))
    return dZ

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    # Initializing W's and b's
    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
    return parameters

def linear_forward(A,W,b):
    Z = W.dot(A)+b
    cache = (A,W,b)
    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
    if activation=="sigmoid":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
    elif activation=="relu":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)
    elif activation=="softmax":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,s,activation_cache = softmax(Z) 
        #parameters['S'] = s
    cache = (linear_cache,activation_cache)
    return A,cache

def L_model_forward(X,parameters):
    caches = []
    A = X
    L =len(parameters)//2
    for l in range(1,L):
        A_prev = A
        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]
        A,cache = linear_activation_forward(A_prev,W,b,"relu")
        caches.append(cache)
    W,b = parameters['W'+str(L)],parameters['b'+str(L)]
    AL,cache = linear_activation_forward(A,W,b,'softmax')
    caches.append(cache)
    return AL,caches

def compute_cost(AL,Y):
    m = Y.shape[1]
    cost = (1.0/m)*np.sum(-np.sum(Y*np.log(AL)))
    return cost

def linear_backward(dZ,cache):
    A_prev,W,b = cache
    m = A_prev.shape[1]
    dW = (1.0/m)*dZ.dot(A_prev.T)
    db = (1.0/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = W.T.dot(dZ)
    return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
    linear_cache,activation_cache = cache
    if activation=="relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    elif activation=="sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,dB = linear_backward(dZ,linear_cache)
    return dA_prev,dW,db

def L_model_backward(AL,Y,caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # last layer back propagation
    dZL = AL - Y
    current_cache = caches[L-1]
    grads["dA"+str(L)],grads['dW'+str(L)],grads['db'+str(L)] = linear_backward(dZL,current_cache[0])
    
    # Backpropagation for other layers.
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA'+str(l+1)],grads['dW'+str(l+1)],grads['db'+str(l+1)] = linear_activation_backward(grads['dA'+str(l+2)],current_cache,"relu")
    return grads

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters)//2
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]
    return parameters

def L_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0,num_iterations):
        AL,caches = L_model_forward(X,parameters)
        cost = compute_cost(AL,Y)
        grads = L_model_backward(AL,Y,caches)
        parameters = update_parameters(parameters,grads,learning_rate)
        if print_cost and i%100==0:
            print("Cost after iteration %i : %f"%(i,cost))
            costs.append(cost)
    #Plotting cost function.
    plt.plot(np.squeeze(costs))
    plt.ylabel('Costs')
    plt.xlabel('Iterations (per tens)')
    plt.title("Learning Rate = "+str(learning_rate))
    plt.show()
    return parameters