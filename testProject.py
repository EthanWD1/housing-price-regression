import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cahousing.csv") ### load csv
#print(df.head)
#print(df.isnull().sum())
df['total_bedrooms'] = df['total_bedrooms'].fillna(0)  ### Replace null bedroom counts with 0 bedrooms
df = pd.get_dummies(df, drop_first=True)  ### qualitative variables to binary categorical variables
df=df[df['median_house_value']<500000] ### remove all blocks with average house price at $500,000, as dataset caps median price
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
print("Intercept: ", mlrModel.intercept_)
print("Coefficients: ", mlrModel.coef_)
print("RMSE: ",predRMSE)
print("R2:", predR2)

##################

print("\nRidge Model:")

ridgeModel = Ridge()

paramDict = {'alpha': [.01,.1,.5,1,2,10,100]} ### different coefficient scalars to test
grid_search = GridSearchCV(ridgeModel, paramDict, cv=5, scoring='neg_mean_squared_error') ### find best scalar

grid_search.fit(xTrain, yTrain) ### create grid search model with best hyperparameters

bestRidge = grid_search.best_estimator_ ###Save the best alpha value for the ridge model
ridgeCV = cross_val_score(bestRidge,xTrain,yTrain,cv=5,scoring = 'r2') ### cross validation with 5 folds
gridPred = bestRidge.predict(xTest) ### Create predicted values with 20% of the data set aside for testing

gridRMSE = np.sqrt(mean_squared_error(yTest,gridPred)) ### find root of MSE by comparing real y test values to predicted values
gridR2 = r2_score(yTest,gridPred) ### generate r^2 score, percent of variation in y that can be explained by the model

print("Best params: ", bestRidge)
print("Cross-Validation R^2 Scores:", ridgeCV)
print("Average Cross-Validation R^2 Score:", ridgeCV.mean())
print("Root MSE: ",gridRMSE)
print("R2:", gridR2)

#########
print("\nLasso Model:")
lassoModel = Lasso()
lasso_search = GridSearchCV(lassoModel,paramDict,cv=5,scoring='neg_mean_squared_error')
lasso_search.fit(xTrain,yTrain)
bestLasso=lasso_search.best_estimator_
lasso_CV=cross_val_score(bestLasso,xTrain,yTrain,cv=5,scoring='r2')
lassoPred=bestLasso.predict(xTest)
lassoRMSE = np.sqrt(mean_squared_error(yTest,lassoPred))
lassoR2 = r2_score(yTest,lassoPred)

print("Best params: ", bestLasso)
print("Cross-Validation R^2 Scores:", lasso_CV)
print("Average Cross-Validation R^2 Score:", lasso_CV.mean())
print("Root MSE: ",lassoRMSE)
print("R2:", lassoR2)
print("Coefficients: ", bestLasso.coef_)

models = ['Linear', 'Ridge', 'Lasso']
r2_scores = [predR2, gridR2, lassoR2]
rmse_scores = [predRMSE, gridRMSE, lassoRMSE]

summary = pd.DataFrame({'Model': models, 'R2': r2_scores, 'RMSE': rmse_scores})
print(summary)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, cbar=False)
plt.title("Feature Correlation Heatmap")

ridgeResid=gridPred-yTest
plt.subplot(1, 3, 2)
plt.scatter(yTest, ridgeResid)
plt.xlabel("Actual block median value")
plt.ylabel("Ridge model residual")
plt.title("Ridge regression residual plot")
plt.axhline(y=0,color='r')

lassoResid = lassoPred - yTest
plt.subplot(1, 3, 3)
plt.scatter(yTest, lassoResid)
plt.xlabel("Actual block median value")
plt.ylabel("Lasso model residual")
plt.title("Lasso regression residual plot")
plt.axhline(y=0, color='r')
plt.tight_layout()
plt.show()