# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from numpy import mean, std
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

training_set = pd.read_csv('../tabular-playground-series-jun-2021/train.csv')
test_set = pd.read_csv('../tabular-playground-series-jun-2021/test.csv')

training_set.drop(columns=['id'], axis=1, inplace=True)
test_set.drop(columns=['id'], axis=1, inplace=True)

x_train = training_set.iloc[:,training_set.columns != 'target']
y_train = training_set.iloc[:,training_set.columns == 'target']

x_test = test_set

#pd.set_option('display.max_columns',20)
#pd.set_option('display.width', 1000)
#print (training_set.head())

print(x_train.head())
print(y_train.head())

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100)

# define the model evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)

# evaluate the model and collect the scores
n_scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

# report the model performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))