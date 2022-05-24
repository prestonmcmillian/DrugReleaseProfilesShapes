# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tensorflow.kersa.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation

# import training dataset from excel file as a pandas dataframe
dataset = pd.read_excel(r"Path where the Excel file is stored\File name.xlsx")
np.random.shuffle(dataset)

# Split into targets(X) and features (Y) for classification
X_Classifier = dataset.drop['Shape']
Y_Classifier = dataset['Shape']
# SVM classifier training
X_train_classification, X_test_classification,  Y_train_classification, Y_test_classification = train_test_split(X_Classifier, Y_Classifier, test_size=0.2, random_state=25)
clf = svm.SVC()
clf.fit(X_train_classification, Y_train_classification)
y_pred = clf.predict(X_test_classification)

# Split into targets(X) and features (Y) for regression
X_Regressor = dataset.drop['Release']
Y_Regressor = dataset['Release']
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_Regressor, Y_Regressor, test_size=0.2, random_state=25)

# Regression
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1

model = Sequential([

    # reshape 28 row * 28 column data to 28*28 rows
    Flatten(input_shape=(28, 28)),

    # dense layer 1
    Dense(256, activation='sigmoid'),

    # dense layer 2
    Dense(128, activation='sigmoid'),

    # output layer
    Dense(10, activation='sigmoid'),
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train_regression, y_test_regression, epochs=10,
          batch_size=100,
          validation_split=0.2)
results = model.evaluate(X_test_regression,  y_test_regression, verbose = 0)

