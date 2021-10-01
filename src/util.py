import math
import random
from typing import Tuple, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sting.data import AbstractDataSet, DataSet

"""
This is where you will implement helper functions and utility code which you will reuse from project to project.
Feel free to edit the parameters if necessary or if it makes it more convenient.
Make sure you read the instruction clearly to know whether you have to implement a function for a specific assignment.
"""

# Creates the same dataframe every time due to the random seed
def create_fake_data():
    random.seed(2468)
    df = pd.DataFrame()
    df['a'] = [random.choice(('+', '-')) for i in range(0, 10)]
    df['b'] = [random.choice(('+', '-', '-')) for i in range(0, 10)]
    df['c'] = [random.choice(("red", "white", "blue")) for i in range(0, 10)]
    df['d'] = [10 * random.random() for i in range(0, 10)]
    df['label'] = [random.choice((1, 1, 0)) for i in range(0, 10)]
    return df

#creates a contingency table
def contingencyTable(y: Iterable[int], y_hat: Iterable[int]) -> list:
    # table will hold the attributes in this order (tp,fp,fn,tn)
    y = np.array(y)
    CT = [0, 0, 0, 0]
    for i in range(0, len(y)):
        predicted_label = y_hat[i]
        true_label = y[i]
        # True Positive
        if (predicted_label == 1) and (true_label == 1):
            CT[0] += 1
        # False Positive
        elif (predicted_label == 1) and (true_label == 0):
            CT[1] += 1
        # False Negative
        elif (predicted_label == 0) and (true_label == 1):
            CT[2] += 1
        # True Negative
        elif (predicted_label == 0) and (true_label == 0):
            CT[3] += 1
    return CT


def cv_split(dataset: AbstractDataSet, folds: int, stratified: bool = True) -> Tuple[AbstractDataSet, ...]:
    """
    You will implement this function.
    Perform a cross validation split on a dataset and return the cross validation folds.
    :param dataset: Dataset to be split.
    :param folds: Number of folds to create.
    :param stratified: If True, create the splits with stratified cross validation.
    :return: A tuple of the dataset splits.
    """

    df, y = dataset.unzip()
    df['label'] = y
    foldedData = ()
    if stratified:
        #two dataframes, one holding 1's, the other holding 0's
        df1 = df[df.label == 1]
        df0 = df[df.label == 0]


        datasize1 = math.floor(len(df1)/folds)
        datasize0 = math.floor(len(df0) / folds)

        for i in range(0, folds):
            newdf0 = df0.loc[i*datasize0:(i+1)*datasize0, :]
            newdf1 = df1.loc[i*datasize1:(i+1)*datasize1, :]
            newdf = newdf1.append(newdf0)
            newdf.reset_index(inplace=True)
            foldedData += (DataSet(newdf.drop(newdf.columns[-1], axis=1), newdf.label.array, dataset.schema),)

    return foldedData


# Build a dataset with all but the ith fold
def build_cv_data(folds, index):
    res = DataSet(None, None, folds[0].schema)
    for i in range(0, len(folds)):
        if i == index:
            continue
        else:
           res = res.append(folds[i])
    res, y = res.unzip()
    res['label'] = y.values

    res.reset_index(inplace=True)
    return DataSet(res.drop('index', axis=1))


def accuracy(y: Iterable[int], y_hat: Iterable[int]) -> float:
    """  You will implement this function.
     Evaluate the accuracy of a set of predictions.
     :param y: Labels (true data)
     :param y_hat: Predictions
     :return: Accuracy of predictions
     """
    return sum(y == y_hat)/len(y)


def precision(y: Iterable[int], y_hat: Iterable[int]) -> float:
    """  You will implement this function.
    Evaluate the precision of a set of predictions.
    :param y: Labels (true data)
    :param y_hat: Predictions
    :return: Precision of predictions
    """

    ct = contingencyTable(y, y_hat)
    if ct[0] == 0 and ct[1] == 0:
        return 0
    else:
        return ct[0] / (ct[0] + ct[1])


def recall(y: Iterable[int], y_hat: Iterable[int]) -> float:
    """
    You will implement this function.
    Evaluate the recall of a set of predictions.
    :param y: Labels (true data)
    :param y_hat: Predictions
    :return: Recall of predictions
    """
    ct = contingencyTable(y, y_hat)
    if ct[0] == 0 and ct[2] == 0:
        return 0
    else:
        return ct[0] / (ct[0] + ct[2])


def roc_curve_pairs(y: Iterable[int], p_y_hat: Iterable[int]) -> Iterable[Tuple[float, float]]:
    """
    You will implement this function.
    Find pairs of FPR and TPR of prediction probabilities based on different decision thresholds.
    You can use this function to implement plot_roc_curve and auc.
    :param y: Labels (true data)
    :param p_y_hat: Classifier predictions (probabilities)
    :return: pairs of FPR and TPR
    """

    pairs = []
    for iteration, conf in enumerate(p_y_hat):
        predict = []
        for i in range(0, len(y)):
            if iteration >= i:
                predict.append(1)
            else:
                predict.append(0)

        ct = contingencyTable(y, predict)
        if ct[3] == 0 and ct[1] == 0:
            fpr = 0
        else:
            fpr = ct[1] / (ct[3] + ct[1])
        if ct[2] == 0 and ct[0] == 0:
            tpr = 0
        else:
            tpr = ct[0] / (ct[0] + ct[2])

        pairs.append((fpr, tpr))

    pairs.sort(key=lambda x: x[0])

    if pairs[0] != (0, 0):
        pairs.insert(0, (0, 0))
    if pairs[-1] != (1, 1):
        pairs.append((1, 1))

    return pairs


def plot_roc_curve(y: Iterable[int], p_y_hat: Iterable[int]):
    """
    plots the roc graph
    :param y: Labels (true data)
    :param p_y_hat: Classifier predictions (probabilities)
    """

    roc_pairs = roc_curve_pairs(y, p_y_hat)
    xpoints = []
    ypoints = []
    for point in roc_pairs:
        xpoints.append(point[0])
        ypoints.append(point[1])

    plt.close('all')
    plt.plot(xpoints, ypoints)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Graph")
    plt.show()


def auc(y: Iterable[int], p_y_hat: Iterable[int]) -> float:
    """
    You will implement this function.
    Calculate the AUC score of a set of prediction probabilities.
    :param y: Labels (true data)
    :param p_y_hat: Classifier predictions (probabilities)
    :return: AUC score of the predictions
    """
    lhs = 0
    rhs = 0
    roc_pairs = roc_curve_pairs(y, p_y_hat)
    i = 0
    j = 1
    while (j < len(roc_pairs)):
        rhs += roc_pairs[j][1] * (roc_pairs[j][0] - roc_pairs[i][0])
        lhs += roc_pairs[i][1] * (roc_pairs[j][0] - roc_pairs[i][0])
        i += 1
        j += 1

    return (rhs + lhs) / 2
