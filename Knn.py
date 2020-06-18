import pandas as pd
from sklearn.model_selection import train_test_split


def distance(cord1, cord2, p):
    res = 0
    for i in range(len(cord1)):
        res += abs(cord1[i] - cord2[i]) ** p

    return res ** (1 / p)


def frechet_dist(cord1, cord2):
    max_diff = float('-inf')
    for i in range(len(cord1)):
        max_diff = max(abs(cord1[i] - cord2[i]), max_diff)

    return max_diff


def calc_dist(x_train, y_train, x_test, dist_fun):
    distances = []
    for cord1 in range(x_test.shape[0]):
        cord1_dist = []
        for cord2 in range(x_train.shape[0]):
            cord1_dist.append((dist_fun(x_test[cord1], x_train[cord2]), y_train[cord2]))

        cord1_dist.sort(key=lambda x: x[0])
        distances.append(cord1_dist)

    return distances


def get_classifier(k, distance_list):
    dic = {}
    for i in range(k):
        label = distance_list[i][1]
        dic[label] = dic[label] + 1 if label in dic else 1

    return max(dic, key=dic.get)


def compute_err(k, dist_list, y_test):
    err = 0
    rows = y_test.shape[0]
    for i in range(rows):
        classifier = get_classifier(k, dist_list[i])
        if classifier != y_test[i]:
            err += 1
    return err / rows


def start_knn(x, y):
    averages_manhattan = [0] * 5
    averages_euclidaen = [0] * 5
    averages_frechet = [0] * 5

    repeats = 500
    for r in range(500):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
        manhattan_dist_list = calc_dist(x_train, y_train, x_test, lambda cord1, cord2: distance(cord1, cord2, 1))
        euclidaen_dist_list = calc_dist(x_train, y_train, x_test, lambda cord1, cord2: distance(cord1, cord2, 2))
        frechet_dist_list = calc_dist(x_train, y_train, x_test, frechet_dist)

        for k in range(1, 10, 2):
            index = int(k / 2)
            averages_manhattan[index] += compute_err(k, manhattan_dist_list, y_test) / repeats
            averages_euclidaen[index] += compute_err(k, euclidaen_dist_list, y_test) / repeats
            averages_frechet[index] += compute_err(k, frechet_dist_list, y_test) / repeats

    print(averages_manhattan)
    print(averages_euclidaen)
    print(averages_frechet)


dataSet = pd.read_csv('HC_Body_Temperature', delim_whitespace=True, header=None)
X = dataSet.iloc[:, [0, 2]].values
Y = dataSet.iloc[:, 1].values

start_knn(X, Y)

d ={1:2, 2:4,5:6}

t = max(d, key=d.get)