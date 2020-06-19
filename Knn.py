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


def print_results(averages, p):
    print("------ [ P =", p, "] ------")
    for i in range(len(averages)):
        print("[ K = ", i * 2 + 1, "], [ Avg =", averages[i], "]")


def find_best_results(results):
    min_err = results[0][0]
    r_c = [0, 0]

    for i in range(len(results)):
        for j in range(len(results[i])):
            if results[i][j] < min_err:
                r_c = [i, j]
                min_err = results[i][j]

    return r_c, min_err


def start_knn(x, y, n):
    tests = int(n / 2)
    averages_manhattan = [0] * tests
    averages_euclidean = [0] * tests
    averages_frechet = [0] * tests

    repeats = 500
    for r in range(repeats):
        ratio = (r + 1) / repeats
        print("\r[%-25s] %d%%" % ('=' * int(ratio * 25), ratio * 100), end='')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
        manhattan_dist_list = calc_dist(x_train, y_train, x_test, lambda cord1, cord2: distance(cord1, cord2, 1))
        euclidean_dist_list = calc_dist(x_train, y_train, x_test, lambda cord1, cord2: distance(cord1, cord2, 2))
        frechet_dist_list = calc_dist(x_train, y_train, x_test, frechet_dist)

        for k in range(1, n, 2):
            index = int(k / 2)
            averages_manhattan[index] += compute_err(k, manhattan_dist_list, y_test) / repeats  # p = 1
            averages_euclidean[index] += compute_err(k, euclidean_dist_list, y_test) / repeats  # p = 2
            averages_frechet[index] += compute_err(k, frechet_dist_list, y_test) / repeats      # p = inf

    dist_names = ["1", "2", "inf"]
    print()
    print("------ Averages results ------")
    print_results(averages_manhattan, dist_names[0])
    print_results(averages_euclidean, dist_names[1])
    print_results(averages_frechet, dist_names[2])
    r_c, min_err = find_best_results([averages_manhattan, averages_euclidean, averages_frechet])
    print("------ Best result for this training ------")
    print("[ P =", dist_names[r_c[0]], "], [ K = ", r_c[1] * 2 + 1, "], [ Avg =", min_err, "]")


dataSet = pd.read_csv('HC_Body_Temperature', delim_whitespace=True, header=None)
X = dataSet.iloc[:, [0, 2]].values
Y = dataSet.iloc[:, 1].values

start_knn(X, Y, 10)
