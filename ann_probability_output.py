# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:20:48 2021

@author: Grant Isaacs
"""

#IMPORT LIBRARIES------------------------------------------------------------->
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score




#PREPROCESS DATA-------------------------------------------------------------->
#load data
dataset = pd.read_csv("_")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#check for missing values
print(sum(np.equal(X, None)))

#encode categorical variables
lencoder = LabelEncoder()
X[: , 2] = lencoder.fit_transform(X[:, 2])

ctransform = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ctransform.fit_transform(X))

#split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=361981)

#scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




#STRUCTURE THE ANN------------------------------------------------------------>
#initialize the neural network
neural_net = tf.keras.models.Sequential()

#create the input layer and first hidden layer to form a shallow learning model.
"""Layer quantity is determined by experimentation and expertise."""
neural_net.add(tf.keras.layers.Dense(units=6, activation='relu'))

#create the second hidden layer to form a deep learning model.
neural_net.add(tf.keras.layers.Dense(units=6, activation='relu'))

#add the output layer
"""Output units equals the output dimensions minus 1.
This model generates a probability between 0 and 1 (Sigmoid)"""
neural_net.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))




#TRAIN THE ANN---------------------------------------------------------------->
#compile the neural network
"""In this example the adam optimizer is used for stochastic gradient desecent.
   The output is binary so binary cross entropy is selected for the loss function."""
neural_net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train the neural network
"""Batch and Epoch arguments were chose based on previous training data.
Modify accordingly."""
neural_net.fit(X_train, y_train, batch_size=32, epochs=100)




#GENERATE PREDICTIONS--------------------------------------------------------->
#predict test set
y_pred = neural_net.predict(X_test)
y_pred = (y_pred > 0.5)
y_test = y_test.reshape(len(y_test), 1)
y_pred = y_pred.reshape(len(y_pred), 1)
print(np.concatenate((y_pred, y_test), 1))

#build confusion matrix
cmatrix = confusion_matrix(y_test, y_pred)
print(cmatrix)
print(accuracy_score(y_test, y_pred))

#individual prediction
"""Apply the transform method to scale the variables to the same distribution as the training data."""
pred = neural_net.predict(scaler.transform([[1.0, 0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
print("Predicted Probabilty the Customer will leave: {}".format(pred))

pred = neural_net.predict(scaler.transform([[1.0, 0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5
print("Binary statement will the customer leave: {}".format(pred))