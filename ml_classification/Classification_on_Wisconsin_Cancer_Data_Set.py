#Importing basic packages
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
x,y = cancer.data, cancer.target
print(cancer.DESCR)

df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df ['Diagnosis'] = cancer.target
fig,axis = plt.subplots(nrows = 6, ncols = 5, figsize = (20,25), dpi = 150)
count = 0

#Plotting distribution of attributes

for i in range(0,6):
    for j in range(0,5):
        column = df.iloc[:,count]
        sns.distplot(column, hist = True, ax = axis[i,j])
        count += 1
plt.show()

#Splitting the data from training models
from sklearn.model_selection import train_test_split

x= df.drop(['Diagnosis'], axis = 1)
y = df['Diagnosis']
X_train,X_test,y_train,y_test = train_test_split(x,y, stratify = y, random_state = 42)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,plot_confusion_matrix,classification_report,f1_score,confusion_matrix,ConfusionMatrixDisplay
log_model = LogisticRegression()
log_model.fit(X_train,y_train)
y_pred = log_model.predict(X_test)
results_df = pd.DataFrame(data = [["Logistic Regression",accuracy_score(y_test,y_pred),precision_score(y_test,y_pred),recall_score(y_test,y_pred), f1_score(y_test,y_pred)]],columns = ['Model','Accuracy','Precision','Recall','F1 Score'])

conf_mat = confusion_matrix(y_test,y_pred)
display_log = ConfusionMatrixDisplay(confusion_matrix = conf_mat)

display_log = display_log.plot(cmap = plt.cm.Blues, values_format = 'g')
plt.title("Logistic Regression", fontweight = "bold")
plt.show()
results_df

#Scaling the data for the model and adding it to the dataframe along with plotting the confusion matrix
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model_scaled = LogisticRegression()
log_model_scaled.fit(X_train_scaled,y_train)
y_pred_scaled = log_model_scaled.predict(X_test_scaled)

results_df_scaled = pd.DataFrame(data = [["Logistic Regression Scaled",accuracy_score(y_test,y_pred_scaled),precision_score(y_test,y_pred_scaled),recall_score(y_test,y_pred_scaled), f1_score(y_test,y_pred_scaled)]],columns = ['Model','Accuracy','Precision','Recall','F1 Score'])

results_df = results_df.append(results_df_scaled, ignore_index = True)

conf_mat = confusion_matrix(y_test,y_pred_scaled)
display_log = ConfusionMatrixDisplay(confusion_matrix = conf_mat)

display_log = display_log.plot(cmap = plt.cm.Blues, values_format = 'g')
plt.title("Logistic Regression Scaled", fontweight = "bold")
plt.show()
results_df

#Analysing data with KNN Classifier
#K-Nearest Neighbor classifier
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()

knn_model.fit(X_train,y_train)

y_pred = knn_model.predict(X_test)

knn_results_df = pd.DataFrame(data = [["K Nearest Neighbor",accuracy_score(y_test,y_pred),precision_score(y_test,y_pred),recall_score(y_test,y_pred), f1_score(y_test,y_pred)]],columns = ['Model','Accuracy','Precision','Recall','F1 Score'])
results_df = results_df.append(knn_results_df, ignore_index = True)

conf_mat = confusion_matrix(y_test,y_pred)
display_log = ConfusionMatrixDisplay(confusion_matrix = conf_mat)

display_log = display_log.plot(cmap = plt.cm.Blues, values_format = 'g')
plt.title("KNN", fontweight = "bold")
plt.show()
results_df

#Scaling datapoints for KNN 
knn_model.fit(X_train_scaled, y_train)
y_pred = knn_model.predict(X_test_scaled)

knn_results_df = pd.DataFrame(data = [["K Nearest Neighbor Scaled",accuracy_score(y_test,y_pred),precision_score(y_test,y_pred),recall_score(y_test,y_pred), f1_score(y_test,y_pred)]],columns = ['Model','Accuracy','Precision','Recall','F1 Score'])
results_df = results_df.append(knn_results_df, ignore_index = True)

conf_mat = confusion_matrix(y_test,y_pred)
display_log = ConfusionMatrixDisplay(confusion_matrix = conf_mat)

display_log = display_log.plot(cmap = plt.cm.Blues, values_format = 'g')
plt.title("KNN Scaled", fontweight = "bold")
plt.show()
results_df

#Using SVC for classification
#Support Vector Classifier

from sklearn.svm import SVC

svc_model = SVC(gamma = 0.1).fit(X_train,y_train)

y_pred = svc_model.predict(X_test)

svc_df = pd.DataFrame(data = [["Support Vector Classifier",accuracy_score(y_test,y_pred),precision_score(y_test,y_pred),recall_score(y_test,y_pred), f1_score(y_test,y_pred)]],columns = ['Model','Accuracy','Precision','Recall','F1 Score'])
results_df = results_df.append(svc_df, ignore_index = True)

conf_mat = confusion_matrix(y_test,y_pred)
display_log = ConfusionMatrixDisplay(confusion_matrix = conf_mat)

display_log = display_log.plot(cmap = plt.cm.Blues, values_format = 'g')
plt.title("SVC", fontweight = "bold")
plt.show()
results_df

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
y_pred = dt_model.predict(X_test)

dt_model_df = pd.DataFrame(data = [["Decision Tree Clasifier",accuracy_score(y_test,y_pred),precision_score(y_test,y_pred),recall_score(y_test,y_pred), f1_score(y_test,y_pred)]],columns = ['Model','Accuracy','Precision','Recall','F1 Score'])
results_df = results_df.append(dt_model_df, ignore_index = True)

conf_mat = confusion_matrix(y_test,y_pred)
display_log = ConfusionMatrixDisplay(confusion_matrix = conf_mat)

display_log = display_log.plot(cmap = plt.cm.Blues, values_format = 'g')
plt.title("Decision Tree Classifier", fontweight = "bold")
plt.show()
results_df
# Pruning the tree with depth of 3

dt_model_depth_3 = DecisionTreeClassifier(max_depth = 3)
dt_model_depth_3.fit(X_train,y_train)
y_pred = dt_model_depth_3.predict(X_test)

dt_model_depth_3_df = pd.DataFrame(data = [["Decision Tree Clasifier With Depth = 3",accuracy_score(y_test,y_pred),precision_score(y_test,y_pred),recall_score(y_test,y_pred), f1_score(y_test,y_pred)]],columns = ['Model','Accuracy','Precision','Recall','F1 Score'])
results_df = results_df.append(dt_model_depth_3_df, ignore_index = True)

conf_mat = confusion_matrix(y_test,y_pred)
display_log = ConfusionMatrixDisplay(confusion_matrix = conf_mat)

display_log = display_log.plot(cmap = plt.cm.Blues, values_format = 'g')
plt.title("Decision Tree Classifier with depth = 3", fontweight = "bold")
plt.show()
results_df

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest_df = pd.DataFrame(data = [["Random Forest Classifier", accuracy_score(y_test,y_pred), precision_score(y_test,y_pred),recall_score(y_test,y_pred), f1_score(y_test,y_pred)]], columns = ['Model','Accuracy','Precision','Recall', 'F1 Score'])
results_df = results_df.append(random_forest_df, ignore_index = True)

conf_mat = confusion_matrix(y_test,y_pred)
display_log = ConfusionMatrixDisplay(confusion_matrix = conf_mat)

display_log = display_log.plot(cmap = plt.cm.Blues, values_format = 'g')
plt.title("Random Forest Classifier", fontweight = "bold")
plt.show()

results_df

# Plot to analyse the performance of the models
plt.figure(figsize=(20,20))
ax= plt.gca()
results_df.plot(kind = 'line', x = 'Model', y = 'Accuracy', color = 'blue', ax = ax)
results_df.plot(kind = 'line', x = 'Model', y = 'Precision', color = 'orange', ax = ax)
results_df.plot(kind = 'line', x = 'Model', y = 'Recall', color = 'red', ax = ax)
results_df.plot(kind = 'line', x = 'Model', y = 'F1 Score', color = 'green',ax = ax)
plt.grid(visible = True)

plt.show()
