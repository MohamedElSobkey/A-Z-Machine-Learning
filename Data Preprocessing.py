#Data _PreProcessing

#Importing_the_libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

#Importing the dataset

dataset = pd.read_csv('Data.csv')

X=dataset.iloc[: , :-1].values
Y=dataset.iloc[: , 3].values


#Taking care of missing data 

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[: , 1:3 ])
X[: , 1:3 ] = imputer.transform(X[: , 1:3 ])


#Encoding Categorical Data
#Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[: , 0]=labelencoder_X.fit_transform(X[: , 0])

onehotencoder = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = "passthrough")
X = onehotencoder.fit_transform(X)

#Encoding the Dependent Variable
labelencoder_Y = LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)


#Splitting the data set to Training set & Testing set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)










