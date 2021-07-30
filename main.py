# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv

import pandas as pd
from numpy import mean, std
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

training_set = pd.read_csv('../tabular-playground-series-jun-2021/train.csv')
test_set = pd.read_csv('../tabular-playground-series-jun-2021/test.csv')

training_set.drop(columns=['id'], axis=1, inplace=True)
test_set.drop(columns=['id'], axis=1, inplace=True)


# plotting
df = pd.DataFrame(training_set)
ax = df.plot.hist(bins=12, alpha=0.5)
plt.show()

exit()

x_train = training_set.iloc[:,training_set.columns != 'target'].values
y_train = training_set.iloc[:,training_set.columns == 'target'].values.ravel()

x_test = test_set



#pd.set_option('display.max_columns',20)
#pd.set_option('display.width', 1000)
#print (training_set.head())
scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1)
model.fit(x_train_scaled, y_train)

# define the model evaluation procedure
#cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=1)

# evaluate the model and collect the scores
#n_scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

# report the model performance
#print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

scaler = preprocessing.StandardScaler().fit(x_test)
x_test_scaled = scaler.transform(x_test)
test_y_classes = model.predict_proba(x_test_scaled)

print(test_y_classes[0])

header = ["id"]
for i in range(1, 10):
    header.append("Class_" + str(i))

print(header)

test_set = pd.read_csv('../tabular-playground-series-jun-2021/test.csv')

with open('submission.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, delimiter=',')
    wr.writerow(header)
    for i in range(0, len(test_y_classes)):
        rr = []
        rr.append(test_set.id[i])
        for j in range(0, 9):
            rr.append(test_y_classes[i][j])
        wr.writerow(rr)

