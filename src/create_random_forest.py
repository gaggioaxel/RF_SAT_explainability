"""
This file is used to create the random forest model as explained in "On Explaining Random Forests with SAT" by
Yacine Izza and Joao Marques-Silva
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from random import sample
import os

dataset = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/data/iris.data").to_numpy()

# 80% of the dataset used for training
# 20% of the dataset used for testing
n_tr = int(0.8 * len(dataset))
n_ts = int(0.2 * len(dataset))

# shuffle the dataset
j = sample(range(len(dataset)), len(dataset))
dataset = dataset[j, :]

X_tr = dataset[range(0, n_tr), :-1]
X_ts = dataset[range(n_tr, n_tr + n_ts), :-1]
Y_tr = dataset[range(0, n_tr), -1]
Y_ts = dataset[range(n_tr, n_tr + n_ts), -1]

Y_tr[Y_tr == 'Iris-setosa'] = 1
Y_tr[Y_tr == 'Iris-versicolor'] = 2
Y_tr[Y_tr == 'Iris-virginica'] = 3
Y_tr = Y_tr.astype('int')

Y_ts[Y_ts == 'Iris-setosa'] = 1
Y_ts[Y_ts == 'Iris-versicolor'] = 2
Y_ts[Y_ts == 'Iris-virginica'] = 3
Y_ts = Y_ts.astype('int')

n_trees = 100
d = 8
M = RandomForestClassifier(n_estimators=n_trees, max_depth=d)
M.fit(X_tr, Y_tr)

Y_pred = M.predict(X_ts)

print("Accuracy: {}".format(accuracy_score(Y_ts, Y_pred)))
print(confusion_matrix(Y_ts, Y_pred))

with open(os.path.dirname(os.path.realpath(__file__))+'/model/iris_model', 'wb') as f:
    joblib.dump(M, f)
