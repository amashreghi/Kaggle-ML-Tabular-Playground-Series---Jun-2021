######################### KAGGLE CHALLENGE - https://www.kaggle.com/c/tabular-playground-series-jun-2021

######################### Results ###################
# The score that I got was 1.82426 while the best score was around 1.738. This still has a lot of room for improvement

import csv
import math
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

# since the error function is custom we need to write our own cross validation
def custom_cross_validation(cv, x_train_scaled, y_train, model):
    pie_len = len(x_train_scaled) / cv
    for i in range(0, cv):
        b = int(i * pie_len)
        e = int(b + pie_len)
        model.fit(x_train_scaled[b:e], y_train[b:e])
        predicted_probs = model.predict_proba(x_train_scaled[b:e])

        error = 0
        for j in range(0, len(predicted_probs)):
            norm = [float(pr) / sum(predicted_probs[j]) for pr in predicted_probs[j]]
            correct_class = int(y_train[b:e][j][6])  # Class_9
            error += math.log(norm[correct_class - 1])

        error /= (-1 * len(predicted_probs))

        print(error)

do_cross_validation = True

training_set = pd.read_csv('../tabular-playground-series-jun-2021/train.csv')
test_set = pd.read_csv('../tabular-playground-series-jun-2021/test.csv')

training_set.drop(columns=['id'], axis=1, inplace=True)
test_set.drop(columns=['id'], axis=1, inplace=True)


# plotting -- for visualization only
#df = pd.DataFrame(training_set)
#ax = df.plot.hist(bins=12, alpha=0.5)
#plt.show()

x_train = training_set.iloc[:,training_set.columns != 'target'].values
y_train = training_set.iloc[:,training_set.columns == 'target'].values.ravel()

x_test = test_set

# scale to make sure the features are following a Gaussian distribution -- this didn't change results by much though
scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)

# pick the model
#model = LogisticRegression(multi_class='multinomial', solver='saga', penalty='elasticnet', l1_ratio=0.5, C=0.01, max_iter=10000)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

if do_cross_validation:
    custom_cross_validation(10, x_train_scaled, y_train, model)

# final fit and prediction
model.fit(x_train_scaled, y_train)
scaler = preprocessing.StandardScaler().fit(x_test)
x_test_scaled = scaler.transform(x_test)
test_y_classes = model.predict_proba(x_test_scaled)

# prepare header row for writing the results into csv
header = ["id"]
for i in range(1, 10):
    header.append("Class_" + str(i))

# reassign test_set since we dropped a column previously
test_set = pd.read_csv('../tabular-playground-series-jun-2021/test.csv')

# save to csv
with open('submission.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, delimiter=',')
    wr.writerow(header)
    for i in range(0, len(test_y_classes)):
        rr = []
        rr.append(test_set.id[i])
        for j in range(0, 9):
            rr.append(test_y_classes[i][j])
        wr.writerow(rr)

