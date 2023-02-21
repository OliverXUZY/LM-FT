import json
import pandas as pd
from pandas import DataFrame

ftrain = open('./TREC.train.all')
train_val = []
for line in ftrain:
    line = line.strip()
    label = int(line[0])
    text = line[2:]
    train_val.append([label, text])
val = train_val[:500]
train = train_val[500:]
train = DataFrame(train)
val = DataFrame(val)

train.to_csv('train.csv', header=False, index=False)
val.to_csv('validation.csv', header=False, index=False)


ftest = open('./TREC.test.all')
test = []
for line in ftest:
    line = line.strip()
    label = int(line[0])
    text = line[2:]
    test.append([label, text])
test = DataFrame(test)
test.to_csv('test.csv', header=False, index=False)
