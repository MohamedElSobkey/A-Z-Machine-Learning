# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = ColumnTransformer([("Geography", OneHotEncoder(), [4])], remainder = "passthrough")
X = onehotencoder.fit_transform(X)

X = X[:, 1:] #Remove the first column of the dummy variable (Redundancy).

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # It is a module that is used to initialize ANN.
from keras.layers import Dense # It is a module that is used to build the layers in
                               #our neural networks. 
                               
                               
# Initialising the ANN
classifier = Sequential() # At first, create an object of sequential class
                          # This object will be nothing but the model itself.
                          # So, model object is our neural network and we willnot 
                          # need o define any arguements because we will define each layer.

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, activation='relu', kernel_initializer='glorot_uniform',input_dim=11))
               
# The number of nodes is a try and error but usually it is the average.

# Adding the second hidden layer

classifier.add(Dense(6, activation='relu', kernel_initializer='glorot_uniform',input_dim=11))

# Adding the output layer


classifier.add(Dense(6, activation='sigmoid', kernel_initializer='glorot_uniform',input_dim=11))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Optimizer, It is how the weights and variables are going to be changed.
#Adam: Computationally efficient and Little memory requirements.
#Binary_cross_entropy is the way to evaluate Sigmoid and is used to update weights.
#Accuracy is a metric used to evaluate our model.

# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)#Error 1

history = classifier.fit(X_train, Y_train, epochs=100, validation_data=(X_test,Y_test))#Erroer 2




classifier.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))
# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
