import csv
import matplotlib.pyplot
import math
import operator
import random


def read_file(file_name):
    with open(file_name, 'rt') as data_file:
        rows = csv.reader(data_file)
        table = list(rows)
        for row in rows:
            print(row)
        return table


def plot_data(data_table):
    x = []
    y = []
    for row in data_table:
        x.append(row[0])
        y.append(row[1])
    plot = matplotlib.pyplot.plot(x, y, 'r.')
    matplotlib.pyplot.show()


def euclidean_distance(first, second):
    distance = pow(second[0] - first[0], 2) + pow(second[1] - first[1], 2)
    return math.sqrt(distance)


def convert_data(data_table):
    for row in data_table:
        for i in range(len(row)):
            row[i] = float(row[i])


def find_neighbors(data_table, test, k):
    distance_list = []
    neighbor_list = []
    for data_row in data_table:
        distance = euclidean_distance(test, data_row)
        distance_list.append([distance, data_row])
    distance_list.sort(key=operator.itemgetter(0))
    first = True
    for x in range(k + 1):
        if first:
            first = False
        else:
            neighbor_list.append(distance_list[x])
    return neighbor_list


def label_neighbor(neighbor_list, k):
    score = 0
    for neighbor in neighbor_list:
        score += neighbor[1][2]
    return score >= k//2+1


def split_data(whole_data, training_set, test_set):
    for row in whole_data:
        if random.random() > 0.67:
            test_set.append(row)
        else:
            training_set.append(row)


def knn(data_set, k):
    for row in data_set:
        neighbors = find_neighbors(train_set, row, k)
        label = label_neighbor(neighbors, k)
        row.append(label)


def accuracy(table):
    score = 0
    for row in table:
        if bool(row[2]) is row[3]:
            score += 1
    return score/len(table)


if __name__ == "__main__":
    # TODO Plotting
    # TODO Fix appending bug with data/find more int
    accuracy_per_epoch = []
    k_parameter = 5
    for x in range(50):
        data = read_file("data.csv")
        convert_data(data)
        train_set = []
        test_set = []
        split_data(data, train_set, test_set)
        # plot_data(data)
        knn(test_set, k_parameter)
        knn(train_set, k_parameter)
        accuracy_per_epoch.append([accuracy(test_set), accuracy(train_set)])
    print("hello")
