#!/usr/bin/python
from sklearn import neighbors
import csv
import numpy as np

print "reading training data"

with open('data/train.csv', 'rb') as f:
    reader = csv.reader(f)
    X = []
    Y = []

    for row in reader:
        if reader.line_num > 1:
            X.append(row[1:])
            Y.append(row[0])

    #X = 784 x N, Y = N
    X = np.array(X)
    Y = np.array(Y)

print "reading test data"

with open('data/test.csv', 'rb') as f:
    reader = csv.reader(f)
    X_test = []

    for row in reader:
        if reader.line_num > 1:
            X_test.append(row)

    X_test = np.array(X_test)

print "fitting model"

k_neighbors = 10
clf = neighbors.KNeighborsClassifier(k_neighbors, algorithm = 'auto')
clf.fit(X, Y)

print "making predictions"

Z = clf.predict(X_test)

print "writing predictions to file"

with open('submission.csv', 'w+') as f:
    header = 'ImageId,Label\n'
    f.write(header)
    ImageId = 0
    for prediction in Z:
        ImageId += 1
        line = str(ImageId) + ',' + str(prediction) + '\n'
        f.write(line)
