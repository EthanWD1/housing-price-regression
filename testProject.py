import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cahousing.csv") ### load csv
#print(df.head)
#print(df.isnull().sum())
df['total_bedrooms'] = df['total_bedrooms'].fillna(0)  ### Replace null bedroom counts with 0 bedrooms
df = pd.get_dummies(df, drop_first=True)  ### qualitative variables to binary categorical variables
X = df.drop('median_house_value', axis=1) ### predictor variables everything aside from house price
Y = df['median_house_value'] ### predicted price is the estimated variable

xTrain, xTest, yTrain, yTest = train_test_split(X,Y, test_size=.2,random_state = 12345)  ### create random 80/20 training and test splits

mlrModel = LinearRegression() ### create empty linear regression model

cvScores = cross_val_score(mlrModel,xTrain,yTrain,cv = 5, scoring ='r2') ### splits training and test data into 5 diff folds, creates temp mlr models and returns r2 of each fold
mlrModel.fit(xTrain,yTrain) ### creates full mlr model

print("\nLinear Model:")

yPred=mlrModel.predict(xTest) ### generate estimated y values given each entry
predRMSE = np.sqrt(mean_squared_error(yTest,yPred)) ### square root of mean squared error, roughly mean error
predR2 = r2_score(yTest,yPred) ### what percent of variability in data can be explained by model

print("Cross-Validation R^2 Scores:", cvScores)
print("Average Cross-Validation R^2 Score:", cvScores.mean())
print("Model intercept: ", mlrModel.intercept_)
print("Model coefficients: ", mlrModel.coef_)
print("Model RMSE: ",predRMSE)
print("Model R2:", predR2)

##################

print("\nRidge Model:")

ridgeModel = Ridge()

paramDict = {'alpha': [.01,.1,.5,1,2,10,100]} ### different coefficient scalars to test
grid_search = GridSearchCV(ridgeModel, paramDict, cv=5, scoring='neg_mean_squared_error') ### find best scalar

grid_search.fit(xTrain, yTrain) ### create grid search model with best hyperparameters

bestRidge = grid_search.best_estimator_ ###Save the best alpha value for the ridge model
ridgeCV = cross_val_score(bestRidge,xTrain,yTrain,cv=5,scoring = 'r2') ### cross validation with 5 folds
gridPred = grid_search.predict(xTest) ### Create predicted values with 20% of the data set aside for testing

gridRMSE = np.sqrt(mean_squared_error(yTest,gridPred)) ### find root of MSE by comparing real y test values to predicted values
gridR2 = r2_score(yTest,gridPred) ### generate r^2 score, percent of variation in y that can be explained by the model

print("Grid best params: ", grid_search.best_params_)
print("Ridge Cross-Validation R^2 Scores:", ridgeCV)
print("Average Ridge Cross-Validation R^2 Score:", ridgeCV.mean())
print("Grid Root MSE: ",gridRMSE)
print("Grid R2:", gridR2)

ridgeResid=gridPred-yTest
plt.scatter(yTest, ridgeResid)
plt.xlabel("Actual block median value")
plt.ylabel("Ridge model residual")
plt.title("Ridge regression residual plot")
plt.axhline(y=0,color='r')
plt.show()