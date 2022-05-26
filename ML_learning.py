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
from sklearn.decomposition import PCA

# import training dataset from excel file as a pandas dataframe
dataset = pd.read_excel(r"Path where the Excel file is stored\File name.xlsx")
np.random.shuffle(dataset)

# Split into targets(X) and features (Y) for classification of shape
X_Classifier_shape = dataset.drop['Shape']
Y_Classifier_shape = dataset['Shape']
# SVM classifier training
X_train_classification, X_test_classification,  Y_train_classification, Y_test_classification = train_test_split(X_Classifier, Y_Classifier, test_size=0.2, random_state=25)
clf = svm.SVC()
clf.fit(X_train_classification, Y_train_classification)
y_pred = clf.predict(X_test_classification)

# Split into targets(X) and features (Y) for regression
X_Regressor = dataset.drop['Release', 'Sample ID']
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
results_regression = model.fit(X_train_regression, y_train_regression, epochs=10,
          batch_size=100, validation_data=(X_test_regression, y_test_regression))
results = model.evaluate(X_test_regression,  y_test_regression, verbose = 0)

loss_train = results_regression.history['train_loss']

loss_val = results_regression.history['val_loss']

epochs = range(1,100)

plt.plot(epochs, loss_train, 'g', label='Training loss')

plt.plot(epochs, loss_val, 'b', label='validation loss')

plt.title('Training and Validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()

pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_train_classification)
clfreduced = svm.SVC()
clfreduced = clfreduced.fit(X_reduced, Y_train_classification)

z = lambda x,y: (-clfreduced.intercept_[0]-clfreduced.coef_[0][0]*x -clfreduced.coef_[0][1]*y) / clfreduced.coef_[0][2]
tmp = np.linspace(-5,5,30)
x,y = np.meshgrid(tmp,tmp)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot3D(X_reduced[Y_train_classification==0,0], X_reduced[Y_train_classification==0,1], X_reduced[Y_train_classification==0,2],'ob')
ax.plot3D(X_reduced[Y_train_classification==1,0], X_reduced[Y_train_classification==1,1], X_reduced[Y_train_classification==1,2],'sr')
ax.plot_surface(x, y, z(x,y))
ax.view_init(30, 60)
plt.show()

