%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')
import sklearn
from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize = True)

from sklearn.datasets import load_boston
sklearn.set_config(print_changed_only = True)
boston = load_boston()
X, y = boston.data, boston.target
boston_data = pd.DataFrame(data = boston.data, columns = boston.feature_names)
boston_data['Price'] = y

#Plotting this data into a scatterplot
fig, axes = plt.subplots(3, 5,figsize=(30,15))

for i, ax in enumerate(axes.ravel()):
    if (i > 12):
        ax.set_visible(False)
        continue
    ax.plot(X[:,i], y, 'o', alpha = 0.5)
    ax.set_title("{}:{}".format(i, boston.feature_names[i]))
    ax.set_ylabel("MEDV")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
train_score_lr = model.score(X_train, y_train)
test_score_lr = model.score(X_test, y_test)

#Installation for yellowbrick if not installed
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(LinearRegression())
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()  #Produces a residual plot for the model. Uses R-squared values of the training and test data points.

# By GridSearchCV

param_grid = {'alpha': np.logspace(-3,3,14)}

grid = GridSearchCV(Ridge(), param_grid, cv=10, return_train_score=True)
grid.fit(X_train,y_train)
#Plotting the grid
results = pd.DataFrame(grid.cv_results_)

results.plot('param_alpha', 'mean_train_score',ax=plt.gca())
results.plot('param_alpha', 'mean_test_score', ax=plt.gca())

plt.legend()
plt.xscale("log")

ridge = Ridge(alpha = 0.07017038286703829).fit(X_train,y_train)
y_pred = ridge.predict(X_test)

train_score_ridge = ridge.score(X_train,y_train)
test_score_ridge = ridge.score(X_test, y_test)

#Feature Engineering for better analysis and model improvement

from sklearn.preprocessing import PolynomialFeatures,scale

X_poly = PolynomialFeatures(include_bias = False).fit_transform(scale(X))
X_train, X_test, y_train, y_test = train_test_split(X_poly,y, random_state=42)

linear_score = np.mean(cross_val_score(LinearRegression(), X_train, y_train, cv=10))

grid = GridSearchCV(Ridge(), param_grid, cv=10, return_train_score = True)

grid.fit(X_train,y_train)
ridge1 = Ridge(alpha = 1).fit(X_train,y_train)
ridge14 = Ridge(alpha = 14.251026703029993).fit(X_train,y_train)
ridge50 = Ridge(alpha = 50).fit(X_train,y_train)
ridge100 = Ridge(alpha = 100).fit(X_train,y_train)
y_pred = ridge.predict(X_test)

plt.plot(ridge14.coef_, 'o', label = "alpha=14.25")
plt.plot(ridge1.coef_, 'o', label = "alpha=1")
plt.plot(ridge50.coef_, 'o', label = "alpha=50", color = "yellow")
plt.plot(ridge100.coef_, 'o', label = "alpha=100",color = "black")
plt.legend()
