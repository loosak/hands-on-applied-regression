# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# # %matplotlib inline
from io import StringIO
import zipfile

from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn import (dummy, ensemble, linear_model, metrics,
                     model_selection, neighbors, neural_network, 
                     preprocessing, svm, tree)
from yellowbrick import features, regressor
from yellowbrick import model_selection as ms_yb
import xgbfir
import xgboost as xgb
# -

# ## Data

# https://archive.ics.uci.edu/ml/datasets/Automobile
auto_cols = '''symboling
normalized-losses
make
fuel-type
aspiration
num-of-doors
body-style
drive-wheels
engine-location
wheel-base
length
width
height
curb-weight
engine-type
num-of-cylinders
engine-size
fuel-system
bore
stroke
compression-ratio
horsepower
peak-rpm
city-mpg
highway-mpg
price'''.split('\n')
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
auto = pd.read_csv('../data/imports-85.data', names=auto_cols)

auto.T

auto.dtypes

auto['num-of-doors']



# ## Linear Regression

# +
def tweak_cars(auto):
    return (auto
            .query('horsepower != "?" and price != "?"'
                  ' and bore != "?" and stroke != "?"')
            .rename({c:c.replace('-', '_') for c in auto.columns}, axis=1)
            .replace({'num_of_doors': {'?': 4, 'four': 4, 'two': 2}})
            .replace({'num_of_cylinders': {'two': 2, 'three': 3,'four': 4, 'five': 5,
                                           'six': 6, 'eight':8, 'twelve':12}})
            .assign(horsepower=lambda df: pd.to_numeric(df.horsepower),
                    peak_rpm=lambda df: pd.to_numeric(df.peak_rpm),
                    price=lambda df: pd.to_numeric(df.price),
                    bore=lambda df: pd.to_numeric(df.bore),
                    stroke=lambda df: pd.to_numeric(df.stroke))
            .drop(columns=['normalized_losses', 'highway_mpg'])
            .pipe(lambda df: pd.get_dummies(df, drop_first=True))
           )

def getX_y(df, y_col):
    return df.drop(columns=[y_col]), df[y_col]

auto2 = tweak_cars(auto)
auto_X, auto_y = getX_y(auto2, 'city_mpg')
# -

auto2.plot.scatter(x='horsepower', y='city_mpg')

# very simple
lr = linear_model.LinearRegression()
lr.fit(auto2[['horsepower']], auto2.city_mpg)
lr.score(auto2[['horsepower']], auto2.city_mpg)


ax = auto2.plot.scatter(x='horsepower', y='city_mpg')
xs = np.arange(40, 200)
ax.plot(xs, xs*lr.coef_ + lr.intercept_)

# Let's use all of the columns
# Baseline model - default strategy is to always predict mean
dm = dummy.DummyRegressor()
dm.fit(auto_X, auto_y)
dm.score(auto_X, auto_y)

# Score is R2 score - coefficient of determintation
# Usually between 0-1 - .92 amount that answer is explained by features
# 1 - 100% of answer is explained by features
lr = linear_model.LinearRegression()
lr.fit(auto_X, auto_y)
lr.score(auto_X, auto_y)

pd.Series(lr.coef_, auto_X.columns)

lr.intercept_



# ## Lab Data
# Ames Housing Dataset
# http://www.amstat.org/publications/jse/v19n3/decock/AmesHousing.xls

# Ames Housing Dataset
ames_url = '../data/AmesHousing.xls'
ames_df = pd.read_excel(ames_url)


ames_df

ames_df2 = (ames_df
           .select_dtypes('number')
           .dropna()
           )
ames_X, ames_y = getX_y(ames_df2, 'SalePrice')                           

# ## Regression Exercise
# * Create a regression model for the Ames dataset
# * What is the "score" of your model?
#

# Let's use all of the columns
# Baseline model - default strategy is to always predict mean
dm = dummy.DummyRegressor()
dm.fit(ames_X, ames_y)
dm.score(ames_X, ames_y)


# Score is R2 score - coefficient of determintation
# Usually between 0-1 - .92 amount that answer is explained by features
# 1 - 100% of answer is explained by features
lr = linear_model.LinearRegression()
lr.fit(ames_X, ames_y)
lr.score(ames_X, ames_y)



# ## Splitting Data
#

# +
def get_train_test_X_y(auto, y_col, size=.3, standardize=True):
    """We don't want to impute or standardize on the whole dataset
    else we are 'leaking' data"""
    y = auto[y_col]
    X = auto.drop(columns=y_col)
    X_train, X_test, y_train, y_test = \
       model_selection.train_test_split(
       X, y, test_size=size, random_state=42)
    cols = X.columns
    X_train = pd.DataFrame(X_train, columns=cols)
    X_test = pd.DataFrame(X_test, columns=cols)
    if standardize:
        std = preprocessing.StandardScaler()
        X_train = pd.DataFrame(std.fit_transform(X_train), columns=cols,
                              index=y_train.index)
        X_test = pd.DataFrame(std.transform(X_test), columns=cols,
                             index=y_test.index)

    return X_train, X_test, y_train, y_test

auto_X_train, auto_X_test, auto_y_train, auto_y_test = \
    get_train_test_X_y(auto2, 'city_mpg') 
# -

# Baseline model - default strategy is to always predict mean
dm = dummy.DummyRegressor()
dm.fit(auto_X_train, auto_y_train)
dm.score(auto_X_test, auto_y_test)

# Score is R2 score - coefficient of determintation
# Usually between 0-1 - .76 amount that answer is explained by features
# 1 - 100% of answer is explained by features
lr = linear_model.LinearRegression()
lr.fit(auto_X_train, auto_y_train)
lr.score(auto_X_test, auto_y_test)

lr.score(auto_X_train, auto_y_train)



# ## Splitting Exercise
# * Split the Ames data into a training and testing set
# * Run a regression model against the new data, what is the score?

ames_X_train, ames_X_test, ames_y_train, ames_y_test = \
    get_train_test_X_y(ames_df2, 'SalePrice') 

lr = linear_model.LinearRegression()
lr.fit(ames_X_train, ames_y_train)
lr.score(ames_X_test, ames_y_test)



# ## Model Evaluation

lr = linear_model.LinearRegression()
lr.fit(auto_X_train, auto_y_train)
lr.score(auto_X_test, auto_y_test)

metrics.r2_score(auto_y_test, lr.predict(auto_X_test))

# average error (but can't indicate direction of error)
metrics.mean_squared_error(auto_y_test, lr.predict(auto_X_test))

# penalizes large errors
metrics.mean_absolute_error(auto_y_test, lr.predict(auto_X_test))

dt = tree.DecisionTreeRegressor(max_depth=4)
dt.fit(auto_X_train, auto_y_train)
dt.score(auto_X_test, auto_y_test)

metrics.mean_absolute_error(auto_y_test, dt.predict(auto_X_test))

metrics.mean_squared_error(auto_y_test, dt.predict(auto_X_test))

# Residuals plot 
# Good for looking at homoskedasticity - variance of errors
regressor.residuals_plot(lr, auto_X_train, auto_y_train, auto_X_test, auto_y_test)

# Residuals plot 
# Good for looking at homoskedasticity - variance of errors
regressor.residuals_plot(dt, auto_X_train, auto_y_train, auto_X_test, auto_y_test)

# Prediction Error 
# Good for looking at variance and performance at different ends
regressor.prediction_error(lr, auto_X_train, auto_y_train, auto_X_test, auto_y_test)

# Prediction Error 
# Good for looking at variance and performance at different ends
regressor.prediction_error(dt, auto_X_train, auto_y_train, auto_X_test, auto_y_test)



# ## Evaluation Exercise
# * Create a decision tree model for the housing data
# * Compare the r2 and MSE for the decision tree and linear regression models.
# * Plot a residuals plot for both models.

lr = linear_model.LinearRegression()
lr.fit(ames_X_train, ames_y_train)
lr.score(ames_X_test, ames_y_test)

metrics.mean_squared_error(ames_y_test, lr.predict(ames_X_test))

f'{metrics.mean_squared_error(ames_y_test, lr.predict(ames_X_test)):,.2f}'

dt = tree.DecisionTreeRegressor()
dt.fit(ames_X_train, ames_y_train)
dt.score(ames_X_test, ames_y_test)

metrics.mean_squared_error(ames_y_test, dt.predict(ames_X_test))

f'{metrics.mean_squared_error(ames_y_test, dt.predict(ames_X_test)):,.2f}'

regressor.residuals_plot(lr, ames_X_train, ames_y_train, ames_X_test, ames_y_test)

regressor.residuals_plot(dt, ames_X_train, ames_y_train, ames_X_test, ames_y_test)



# ## Tuning the Model

dt = tree.DecisionTreeRegressor(max_depth=None)
dt.fit(auto_X_train, auto_y_train)
dt.score(auto_X_test, auto_y_test)

_=tree.plot_tree(dt, filled=True)

fig, ax = plt.subplots(figsize=(14,6))
_=tree.plot_tree(dt, max_depth=2, filled=True, feature_names=auto_X.columns, ax=ax)

print(tree.export_text(dt, feature_names=list(auto_X.columns)))

# This model is overfitting
# (hint: performs well on training data, but worse on testing)
dt.score(auto_X_train, auto_y_train)

dt.score(auto_X_test, auto_y_test)

stump = tree.DecisionTreeRegressor(max_depth=1)
stump.fit(auto_X_train, auto_y_train)
stump.score(auto_X_test, auto_y_test)

fig, ax = plt.subplots(figsize=(14,6))
_=tree.plot_tree(stump, max_depth=2, filled=True, feature_names=auto_X.columns, ax=ax)

for i in range(1, 10):
    model = tree.DecisionTreeRegressor(max_depth=i)
    model.fit(auto_X_train, auto_y_train)
    print(i, model.score(auto_X_test, auto_y_test) )

ms_yb.validation_curve(tree.DecisionTreeRegressor(),  
    auto_X, auto_y, param_name='max_depth', param_range=range(1, 20))

# many "hyperparameters"
stump

# +
param_grid = {'random_state': [42],
             'max_depth': [1,2,5,10,20],
             'min_impurity_decrease': [0, .1, .2, .5]}
grid = model_selection.GridSearchCV(tree.DecisionTreeRegressor(), param_grid=param_grid)
grid.fit(auto_X, auto_y)


# -

grid.best_params_

tuned_dt = tree.DecisionTreeRegressor(**grid.best_params_)



# ## Tuning Exercise
# * Tune a decision tree for the Ames data
# * What is the score?

ms_yb.validation_curve(tree.DecisionTreeRegressor(random_state=42),  
    ames_X, ames_y, param_name='max_depth', param_range=range(1, 20))

# +
# different from above? - This uses 5-fold validation above is 3
param_grid = {'random_state': [42],
             'max_depth': [1,2,5,8,10,20],
             'min_impurity_decrease': [0, .1, .2, .5]}
grid = model_selection.GridSearchCV(tree.DecisionTreeRegressor(), param_grid=param_grid)
grid.fit(ames_X, ames_y)


# -

grid.best_params_

tuned_ames = tree.DecisionTreeRegressor(**grid.best_params_)
tuned_ames.fit(ames_X_train, ames_y_train)
tuned_ames.score(ames_X_test, ames_y_test)

dt_ames = tree.DecisionTreeRegressor()  # value will change w/o random state
dt_ames.fit(ames_X_train, ames_y_train)
dt_ames.score(ames_X_test, ames_y_test)



# ## Explaining the Model

dt = tree.DecisionTreeRegressor(max_depth=10)
dt.fit(auto_X_train, auto_y_train)
dt.score(auto_X_test, auto_y_test)

fig, ax = plt.subplots(figsize=(14,6))
_=tree.plot_tree(dt, max_depth=3, filled=True, feature_names=auto_X.columns, ax=ax)

dt.feature_importances_

pd.Series(dt.feature_importances_, auto_X.columns).sort_values().tail(10)

ms_yb.feature_importances(dt, auto_X, auto_y)

# Shap - Interpret black box models
shap_ex = shap.TreeExplainer(dt)
vals = shap_ex.shap_values(auto_X_test)

dt.predict(auto_X_test.iloc[[0]])  # prediction

auto_y_test.iloc[0]  # actual

auto_X_test.iloc[[0]].T

auto.loc[[148]]

# Horsepower (negative in sample)/price (also negative) increasing mpg
shap.initjs()
shap.force_plot(shap_ex.expected_value, vals[0, :], feature_names=auto_X_test.columns)

# Summary of HP
# As HP goes up SHAP goes down (lower mpg)
# Shap choose to show bore along with HP (more bore -> more HP)
shap.dependence_plot('horsepower', shap_values=vals, features=auto_X_test)

# +

# Summary of HP (show w/ price)
# As HP goes up SHAP goes down (lower mpg)
# Price appears to go up as HP goes up
shap.dependence_plot('horsepower', shap_values=vals, features=auto_X_test,
                    interaction_index='price')
# -

# Summary of features - global view
# HP is most important
#   As HP goes lower (more blue) shap goes up (as does MPG)
shap.summary_plot(vals, auto_X_test)



# ## Exercise - Explaining the model
# Using a model for the Ames dataset:
# * What are the most useful features?
# * Use shap to inspect an individual result
# * Use shap to inspect the summary of the features
# * Use shap to inspect an individual feature

dt_ames = tree.DecisionTreeRegressor(max_depth=10)  # value will change w/o random state
dt_ames.fit(ames_X_train, ames_y_train)
dt_ames.score(ames_X_test, ames_y_test)

dt_ames.feature_importances_

pd.Series(dt_ames.feature_importances_, ames_X.columns)

pd.Series(dt_ames.feature_importances_, ames_X.columns).plot.barh()



# Shap - Interpret black box models
shap_ex = shap.TreeExplainer(dt_ames)
vals = shap_ex.shap_values(ames_X_test)

# Horsepower (negative in sample)/price (also negative) increasing mpg
shap.initjs()
shap.force_plot(shap_ex.expected_value, vals[0, :], feature_names=ames_X_test.columns)

ames_X_test.iloc[[0]]

ames_df.loc[[2661]]

shap.summary_plot(vals, ames_X_test)

shap.dependence_plot('Overall Qual', shap_values=vals, features=ames_X_test)



# ## XGBoost
# Powerful algorithm using "boosting" (like golfing) to predict target

dt = tree.DecisionTreeRegressor(max_depth=10)
dt.fit(auto_X_train, auto_y_train)
dt.score(auto_X_test, auto_y_test)

xg = xgb.XGBRFRegressor()
xg.fit(auto_X_train, auto_y_train)
xg.score(auto_X_test, auto_y_test)

xg

xgb.plot_importance(xg)

booster = xg.get_booster()
print(booster.get_dump()[0])

booster = xg.get_booster()
print(booster.get_dump()[1])

booster = xg.get_booster()
print(booster.get_dump()[-1])

# Residuals plot 
regressor.residuals_plot(xg, auto_X_train, auto_y_train, auto_X_test, auto_y_test)

# viewing interactions
xgbfir.saveXgbFI(xg, feature_names=auto_X.columns, OutputXlsxFile='fir-auto.xlsx')

# column impmortance
# Gain - total gain of each feature
# Fscore - number of splits
# wFscore - weighted number of splits (by probability of split taking place)
pd.read_excel('fir-auto.xlsx').head(3).T

# column impmortance
# Gain - total gain of each feature
# Fscore - number of splits
# wFscore - weighted number of splits (by probability of split taking place)
pd.read_excel('fir-auto.xlsx', sheet_name='Interaction Depth 1').head(3).T

# column impmortance
# Gain - total gain of each feature
# Fscore - number of splits
# wFscore - weighted number of splits (by probability of split taking place)
pd.read_excel('fir-auto.xlsx', sheet_name='Interaction Depth 2').head(3).T





# ## XGBoost Exercise
# With the Ames data
# * Create an XGBoost model
# * Evaluate the performance. What is the score?

xg_ames = xgb.XGBRFRegressor()
xg_ames.fit(ames_X_train, ames_y_train)
xg_ames.score(ames_X_test, ames_y_test)

dt_ames = tree.DecisionTreeRegressor(max_depth=10)  
dt_ames.fit(ames_X_train, ames_y_train)
dt_ames.score(ames_X_test, ames_y_test)


