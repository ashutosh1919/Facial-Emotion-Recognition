import pandas as pd
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from functions import L_layer_model

data = pd.read_csv('data/legend.csv')
data.head()

list = []
for i in range(0,50):
    A = cv2.imread('images/'+data.iloc[i,1])
    B = cv2.resize(A, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
    list.append(B)
list = np.array(list)

print(data['emotion'].str.lower().unique())

data['emotion'] = data['emotion'].str.lower()
print(data['emotion'].str.lower().unique())
type_y = data['emotion'].str.lower().unique()
type(type_y)

train_X_orig = list
y = []
for i in range(0,50):
    out = data.iloc[i,2]
    A = np.zeros((type_y.size))
    for j in range(0,type_y.size):
        if type_y[j]==out:
            A[j] = 1
            break
    y.append(A)
train_y = np.array(y)
train_y_orig = train_y.T

shape_X = train_X_orig.shape
shape_y = train_y_orig.shape
print("Shape of X : ",shape_X)
print("Shape of y : ",shape_y)

plt.imshow(train_X_orig[1],interpolation='nearest')
plt.show()

train_X_flatten = train_X_orig.reshape(train_X_orig.shape[0],-1).T
train_X = train_X_flatten/255
train_y = train_y_orig
print("train_X shape : ",train_X.shape)
print("train_y shape : ",train_y.shape)

layers_dims = [train_X.shape[0],20,15,11,8]

parameters = L_layer_model(train_X,train_y,layers_dims,num_iterations=2500,print_cost=True)