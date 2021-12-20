from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import csv
import numpy as np
import time

tm1 = time.time()

with open('data.csv', 'r') as f:
    data = np.array(list(csv.reader(f, delimiter=",")))

ans = list(np.array(data[1:, 0], dtype=float))
data = np.array(data[1:])
data = list(normalize(np.array(data[:, 1:7], dtype=float)))

train_data, val_data, train_ans, val_ans = train_test_split(data, ans, test_size=0.7)


def accur_score(a, b, c):
    n = len(a)
    pr = 0  # процент
    for j in range(n):
        pr += (abs(a[j] - b[j])) / n

    return pr


best = 0
best_k = 0
best_predicted = 0

for k in range(1, 2):
    knn = KNeighborsRegressor(n_neighbors=1)
    knn.fit(train_data, train_ans)
    predicted = list(knn.predict(val_data))
    val_ans = list(val_ans)
    tmp = accur_score(predicted, val_ans, predicted + val_ans)
    if best < tmp:
        best = tmp
        best_k = k
        best_predicted = predicted

for i in range(len(best_predicted)):
    best_predicted[i] = round(best_predicted[i], 1)

print(f'{best} -- average absolute error\n', f'best k for this val_data is: {best_k}\n\n\n\n')
#print(val_ans)
#print(best_predicted)

#for j in range(len(best_predicted)):
  #  print(f'{val_ans[j]} -> {best_predicted[j]}')

tm2 = time.time()

print(f'{round(tm2 - tm1, 2)} seconds')
