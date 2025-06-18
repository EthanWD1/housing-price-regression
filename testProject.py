import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cahousing.csv")
#print(df.head)
#print(df.isnull().sum())
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
df = pd.get_dummies(df, drop_first=True)
X = df.drop('median_house_value', axis=1)
Y = df['median_house_value']

xTrain, xTest, yTrain, yTest = train_test_split(X,Y, test_size=.2,random_state = 12345)

mlrModel = LinearRegression()

cvScores = cross_val_score(mlrModel,xTrain,yTrain,cv = 5, scoring ='r2')
mlrModel.fit(xTrain,yTrain)

print("\nLinear Model:")

yPred=mlrModel.predict(xTest)
predRMSE = np.sqrt(mean_squared_error(yTest,yPred))
predR2 = r2_score(yTest,yPred)

print("Cross-Validation R^2 Scores:", cvScores)
print("Average Cross-Validation R^2 Score:", cvScores.mean())
print("Model intercept: ", mlrModel.intercept_)
print("Model coefficients: ", mlrModel.coef_)
print("Model RMSE: ",predRMSE)
print("Model R2:", predR2)

##################

print("\nRidge Model:")

ridgeModel = Ridge()

paramDict = {'alpha': [.01,.1,.5,1,2,10,100]}
grid_search = GridSearchCV(ridgeModel, paramDict, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(xTrain, yTrain)

bestRidge = grid_search.best_estimator_
ridgeCV = cross_val_score(bestRidge,xTrain,yTrain,cv=5,scoring = 'r2')
gridPred = grid_search.predict(xTest)

gridRMSE = np.sqrt(mean_squared_error(yTest,gridPred))
gridR2 = r2_score(yTest,gridPred)

print("Grid best params: ", grid_search.best_params_)
print("Ridge Cross-Validation R^2 Scores:", ridgeCV)
print("Average Ridge Cross-Validation R^2 Score:", ridgeCV.mean())
print("Grid Root MSE: ",gridRMSE)
print("Grid R2:", gridR2)