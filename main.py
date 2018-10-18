import csv
import matplotlib.pyplot
import math
import operator


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


def accuracy(table):
    score = 0
    for row in table:
        if bool(row[2]) is row[3]:
            score += 1
    print(score/len(table))


if __name__ == "__main__":
    # TODO Run for 50 epochs
    # TODO Split data into two sets
    data = read_file("data.csv")
    convert_data(data)
    # plot_data(data)
    k_parameter = 3
    # Find neighbors, find the score, and assign label based on score
    for row in data:
        neighbors = find_neighbors(data, row, k_parameter)
        label = label_neighbor(neighbors, k_parameter)
        row.append(label)
    accuracy(data)
