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
