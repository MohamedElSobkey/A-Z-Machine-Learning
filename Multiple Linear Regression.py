#Multiple Linear Regression
#Importing_the_libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X=dataset.iloc[: , :-1].values
Y=dataset.iloc[: , 4].values

#Encoding Categorical Data
#Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[: , 3]=labelencoder_X.fit_transform(X[: , 3])

onehotencoder = ColumnTransformer([("Country", OneHotEncoder(), [3])], remainder = "passthrough")
X = onehotencoder.fit_transform(X)

# Avoiding the Dummy Variable Trap
X=X[:,1:]



#Splitting the data set to Training set & Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , Y_train) # Model is created

# Predicting the Test set results
Y_pred=regressor.predict(X_test)

# Buliding The optimal model using Backword Elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values= X , axis = 1 )
X_opt = X[: , [0,1,2,3,4,5]]
#OrdinaryLeastSquares
regressor_OLS.summary()


#X_opt = np.array(X_opt, dtype=float)
#regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()#error 


#Rmoving the highest P ,Here is index = 2
X_opt = X[: , [0,1,3,4,5]]
#OrdinaryLeastSquares
#regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
regressor_OLS.summary()



# Buliding The optimal model using Backword Elimination
import statsmodels.api as sm
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()
