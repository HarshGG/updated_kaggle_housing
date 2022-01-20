import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
import re
# from xgboost import XGBRegressor
# import seaborn as sns

print('imported')
# uploaded = files.upload()

df_train = pd.read_csv("https://raw.githubusercontent.com/HarshGG/Kaggle-Housing-Prices/main/train.csv")
df_train.head()




#making HouseStyle values numerical. 
#Still not ok because some columns are all non numerical, such as 75, SVL. 
#Some are also unfinished i assume like column 9, 1.5Unf
# non_decimal = re.compile(r'[^\d.]+')
# for i in range(len(df_train['HouseStyle'])):
#   df_train['HouseStyle'][i] = non_decimal.sub('',df_train['HouseStyle'][i])
# df_train.loc[367,'HouseStyle']
# df_train.head()

numerical=['LotFrontage', 'BsmtFinSF1', 'WoodDeckSF', 'MSSubClass', '2ndFlrSF', 'TotRmsAbvGrd', 'MasVnrArea', 'YearRemodAdd', 'ScreenPorch', 'OverallCond', 'EnclosedPorch', 'GrLivArea', 'GarageArea', 'LowQualFinSF', 'OverallQual', 'TotalBsmtSF', 'KitchenAbvGr', 'GarageYrBlt', 'BedroomAbvGr', '1stFlrSF', 'YearBuilt', 'OpenPorchSF', 'LotArea', 'BsmtUnfSF']
categorical=['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'YrSold', 'Fireplaces','PoolQC','Alley','Fence']

# temp = df_train[categorical]
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse = False)

# df_train.dropna(subset=numerical,inplace=True)
num_imputer = SimpleImputer(strategy = 'median')
cat_imputer = SimpleImputer(strategy = 'most_frequent')
df_train[categorical] = cat_imputer.fit_transform(df_train[categorical])
df_train[numerical] = num_imputer.fit_transform(df_train[numerical])

for cat in categorical:
    integer_encoded = label_encoder.fit_transform(df_train[cat])
    integer_encoded = integer_encoded.reshape(-1,1)
    df_train[cat] = one_hot_encoder.fit_transform(integer_encoded)

# removing categories with little to no information
df_train = df_train.drop("Street", axis = 1)
df_train = df_train.drop("Utilities", axis = 1)
df_train = df_train.drop("LandSlope", axis = 1)
df_train = df_train.drop("MiscFeature", axis = 1)

# neigh = NearestNeighbors(n_neighbors = 4)
# neigh.fit(df_train)
# A = neigh.kneighbors_graph(df_train)
# A = A.toarray()
# for row in A: 
#     for col in A:
#         if (A[row][col] != 1):
#             print('!!!')

df_train = df_train[(np.abs(stats.zscore(df_train)) < 4).all(axis=1)]

X_train, X_val, y_train, y_val = train_test_split(df_train, df_train['SalePrice'], test_size=0.2, random_state = 42)
X_train = X_train.drop('SalePrice', axis = 1)
X_val = X_val.drop('SalePrice', axis = 1)

X_train = df_train.drop('SalePrice', axis = 1)
y_train = df_train['SalePrice']

# print(y_train.describe())
# plt.hist(y_train, bins = 20)



gbr = GradientBoostingRegressor()
param_grid = {'n_estimators' : [1000, 3000, 5000, 7000], 'learning_rate' : [.01, .02, .03, .04, .05], 'max_depth': [1,2,3,4]}
grid_search = GridSearchCV(gbr, param_grid, n_jobs = 4)
grid_search.fit(X_train,y_train)
print('BEST PARAMS:', grid_search.best_params_)
# regr = linear_model.LinearRegression()
# regr.fit(X_train, y_train)
print('regr model made')


# yPred = regr.predict(X_val)

# print("yPred:\n", pd.Series(yPred))
# print("y_val:\n", y_val)

# print("Coefficients: \n", regr.coef_)
# The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(y_val, yPred))
# The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.4f" % r2_score(y_val, yPred))

# scores = cross_val_score(gbr, X_train, y_train, cv = 10)
# avg_score = sum(scores)/len(scores)
# print('CV SCORES:', scores)
# print('AVG SCORE:', avg_score)
# df_test = pd.read_csv('https://raw.githubusercontent.com/HarshGG/Kaggle-Housing-Prices/main/test.csv')
# df_test.dropna(subset=numerical,inplace=True)
# for cat in categorical:
#     integer_encoded = label_encoder.fit_transform(df_test[cat])
#     integer_encoded = integer_encoded.reshape(-1,1)
#     df_test[cat] = one_hot_encoder.fit_transform(integer_encoded)
# # removing categories with little to no information
# df_test = df_test.drop("Street", axis = 1)
# df_test = df_test.drop("Utilities", axis = 1)
# df_test = df_test.drop("LandSlope", axis = 1)
# df_test = df_test.drop("MiscFeature", axis = 1)