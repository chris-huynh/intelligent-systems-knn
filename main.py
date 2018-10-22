import csv
import matplotlib.pyplot as plt
import math
import operator
import random
import numpy as np


def read_file(file_name):
    with open(file_name, 'rt') as data_file:
        rows = csv.reader(data_file)
        table = list(rows)
        for row in rows:
            print(row)
        return table


def plot_data(data_table, metric_list):
    # Plot Bar Graph for metrics
    plt.figure(1)
    fig, met_plot = plt.subplots()
    index = range(5)
    ticks = [0, 1, 2, 3, 4]
    ax = met_plot.bar(index, metric_list, 0.7)
    met_plot.set_xlabel('Metrics')
    met_plot.set_ylabel('Percentage')
    met_plot.set_title('Metrics for k nearest neighbors')
    met_plot.set_xticks(ticks)
    met_plot.set_xticklabels(('Hit Rate', 'Sensitivity', 'Specificity', 'PPV', 'NPV'))
    plt.show()
    # Plot Decision Boundary
    plt.figure(2)
    true_list = []
    xt_list = []
    yt_list = []
    false_list = []
    xf_list = []
    yf_list = []
    for row in data_table:
        if row[2]:
            true_list.append(row)
        else:
            false_list.append(row)
    for row in true_list:
        xt_list.append(row[0])
        yt_list.append(row[1])
    for row in false_list:
        xf_list.append(row[0])
        yf_list.append(row[1])
    plt.plot(xt_list, yt_list, 'r.')
    plt.plot(xf_list, yf_list, 'b.')
    plt.title('Husband vs. Wife Income - Stressed & Unstresssed')
    plt.xlabel('Wife Income ($)')
    plt.ylabel('Husband Income ($)')
    plt.legend(['Stressed', 'Not Stressed'])
    plt.show()


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


def knn(testing_set, data_set, k):
    for row in testing_set:
        neighbors = find_neighbors(data_set, row, k)
        label = label_neighbor(neighbors, k)
        row.append(label)


def calculate_metrics(data_table):
    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0
    data_points = len(data_table)
    for row in data_table:
        if bool(row[2]) & row[3]:
            true_positive = true_positive + 1
        if not bool(row[2]) and not row[3]:
            true_negative = true_negative + 1
        if not bool(row[2]) and row[3]:
            false_positive = false_positive + 1
        if (row[2]) and not row[3]:
            false_negative = false_negative + 1
    hit_rate = 100 * (true_negative + true_positive) / data_points
    sensitivity = 100 * true_positive / (true_positive + false_negative)
    specificity = 100 * true_negative / (true_negative + false_positive)
    ppv = 100 * true_positive / (true_positive + false_positive)
    npv = 100 * true_negative / (true_negative + false_negative)
    return [hit_rate, sensitivity, specificity, ppv, npv]


def accuracy(table):
    score = 0
    for row in table:
        if bool(row[2]) is row[3]:
            score += 1
    return score/len(table)


def min_max_data(data_table):
    x_list = []
    y_list = []
    for row in data_table:
        y_list.append(row[1])
        x_list.append(row[0])
    points = [min(y_list), min(x_list), max(y_list), max(x_list)]
    return points


def create_sample(endpoints):
    x = np.linspace(endpoints[0], endpoints[2], 100)
    y = np.linspace(endpoints[1], endpoints[3], 100)
    sample = []
    for a in x:
        for b in y:
            sample.append([a, b])
    return sample


if __name__ == "__main__":
    k_parameter = 5
    data = read_file("data.csv")
    convert_data(data)
    train_set = []
    test_set = []
    split_data(data, train_set, test_set)
    # plot_data(data)
    knn(test_set, train_set, k_parameter)
    metrics = calculate_metrics(test_set)
    endpoints = min_max_data(data)
    sampling = create_sample(endpoints)
    knn(sampling, data, k_parameter)
    plot_data(sampling, metrics)
