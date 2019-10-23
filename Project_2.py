#!/usr/bin/env python
# coding: utf-8

# ## Step 1. Making Preparations

# ### Importing libraries

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# ### Initializing Random Number

seed = 7
numpy.random.seed(seed)


# ### Loading Data

# load dataset
dataframe = pandas.read_csv('C:/Users/FC/Documents/Deep_Learning_Project_Two/iris.csv', header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]


# ### Label Encoding

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# ## Step 3: Define the Neural Network Baseline Model

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
 
# ## Step 4. Evaluate The Model with k-Fold Cross Validation

kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## Step 5. Tuning Layers and Number of Neurons in The Model

# ### Step 5.2. Evaluate a Larger Network

# define Larger model
def Larger_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=Larger_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## Step 6. Really Scaling up: developing a model that overfits

# define overfit model
def overfit_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=overfit_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("overfit: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## Step 7. Tuning the Model

# define baseline model
def tuned_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



estimator = KerasClassifier(build_fn=tuned_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Tuned_model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## Step 8. Rewriting the code using the Keras Functional API

import keras
from keras import layers
def functional_model():
    # create model, write code below
    inputs = keras.Input(shape=(4,))
    x = layers.Dense(8,activation='relu')(inputs)
    outputs = layers.Dense(3,activation='softmax')(x)
    model = keras.Model(inputs,outputs)
    # Compile model
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
estimator = KerasClassifier(build_fn=functional_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("funtional_model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


#  ## Step 9. Rewriting the code by doing Model Subclassing

import keras
from keras import layers
class Mymodel(keras.Model):
    def __init__(self):
        super(Mymodel,self).__init__()
        self.dense1 = Dense(8, activation = 'relu')
        self.dense2 = Dense(3, activation = 'softmax')
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
def subclass_model():
    inputs = keras.Input(shape=(4,))
    model = Mymodel()
    outputs = model.call(inputs)
    model = keras.Model(inputs, outputs)
    # Compile model, write code below
    model.compile(optimizer='adam',loss='categorical_crossentropy' ,metrics=['accuracy'])
    return model
estimator = KerasClassifier(build_fn= subclass_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("funtional_model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

estimator = KerasClassifier(build_fn= subclass_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("funtional_model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## Step 10. Rewriting the code without using scikit-learn

data = pandas.read_csv('C:/Users/FC/Documents/Deep_Learning_Project_Two/iris.csv', header=None)

data[4][data[4]=='Iris-setosa']=0
data[4][data[4]=='Iris-versicolor']=1
data[4][data[4]=='Iris-virginica']=2


data_values = data.values
numpy.random.seed(7)
numpy.random.shuffle(data_values)


train_data = data_values[:120,0:4].astype(float)
train_labels =data_values[:120,4]
test_data= data_values[120:,:4].astype(float)
test_labels= data_values[120:,4]


from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


from keras import models
from keras import layers
def keras_model():
    # create model, write code below
    model = models.Sequential()
    model.add(layers.Dense(8, activation='relu', input_shape=(4,)))
    model.add(layers.Dense(3,activation='softmax'))
    # Compile model, write code below
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


import numpy as np
k = 4
num_val_samples = len(train_data) // k
num_epochs = 200
all_scores = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_labels[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_labels[:i * num_val_samples],train_labels[(i + 1) * num_val_samples:]],axis=0)
    model = keras_model()
    model.fit(partial_train_data, partial_train_targets,epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)


test_loss, test_acc = model.evaluate(test_data, test_labels)
print('test_acc:', test_acc)