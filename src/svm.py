import argparse
import os.path

from sting.classifier import Classifier
from sklearn.model_selection import train_test_split

from numpy import linalg
import cvxopt
import cvxopt.solvers

from util import *


# Kernel function definitions
def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x1, x2, p=2):
    return (1 + np.dot(x1, x2)) ** p


def gaussian_kernel(x1, x2, sigma=4.0):
    return np.exp(-linalg.norm(x1-x2)**2 / sigma)


def exponential_kernel(x1, x2, sigma=4.0):
    return np.exp(-(x1-x2) / sigma)


def rational_quadratic_kernel(x1, x2, sigma=4.0):
    return 1-(linalg.norm(x1-x2)**2)/((linalg.norm(x1-x2)**2)+sigma)


def wave_kernel(x1, x2, sigma=4.0):
    return sigma/(math.abs(x1-x2))*math.sin(math.abs(x1-x2)/sigma)


class Svm(Classifier):

    def __init__(self, kernel):
        self.kernel_type = kernel
        self.iter = 0

    def train(self, x_train, y_train) -> None:

        # Initializing values
        num_samples, num_features = x_train.shape
        y = y_train.reshape(-1, 1) * 1.
        X_dash = y * x_train
        H = np.dot(X_dash, X_dash.T) * 1.

        # Converting into cvxopt format
        P = cvxopt.matrix(H)
        q = cvxopt.matrix(-np.ones((num_samples, 1)))
        G = cvxopt.matrix(-np.eye(num_samples))
        h = cvxopt.matrix(np.zeros(num_samples))
        A = cvxopt.matrix(y.reshape(1, -1))
        b = cvxopt.matrix(np.zeros(1))

        # Relaxing the constraints
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['abstol'] = 1e-10
        cvxopt.solvers.options['reltol'] = 1e-10
        cvxopt.solvers.options['feastol'] = 1e-10

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        self.alphas = np.ravel(solution['x']).reshape(num_samples)

        # w parameter in vectorized form
        self.w = ((y_train * self.alphas).T @ x_train).reshape(-1, 1)

        # Selecting the set of indices S corresponding to non zero parameters
        SV = np.where(self.alphas > 1e-4)[0][0]

        # Computing b
        self.b = y[SV] - np.dot(x_train[SV], self.w)


    def predict(self, x_test):
        predicted_label = np.array([])
        for sample in x_test:
            sum= 0
            for i in range(0, len(sample)):
                sum += self.w[i] * sample[i]
            if sum + self.b >= 0:
                predicted_label = np.append(predicted_label, 1)
            else:
                predicted_label = np.append(predicted_label, 0)
        return predicted_label


def evaluate_svm(Accuracy: float, Precision: float, Recall: float):
    print('----------------------')
    print('Accuracy: ', Accuracy)
    print('Precision: ', Precision)
    print('Recall: ', Recall)


def svm(data_path, online, classifier_type = linear_kernel):
    """
    svm function to easily run from jupyter notebook

    :param data_path: The path to the data.
    :param tree_depth_limit: Depth limit of the decision tree
    :param use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
    :param information_gain: If true, use information gain as the split criterion. Otherwise use gain ratio.
    :return:
    """

    path = os.path.normpath(data_path)
    df = pd.read_csv(path)
    df = df.sample(frac=1)

    df_y = df.label
    df_x = df.drop('label', axis=1)
    if 'Name' in df.columns:
        df_x = df_x.drop('Name', axis=1)

    df_x = df_x.drop_duplicates()
    test_size = .8
    classifier = Svm(classifier_type)

    if online:
        online_batch_percent = .2
        percent_not_in_batch = 1

        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=test_size, shuffle=True)

        for i in range(0, math.ceil(1/online_batch_percent)):
            if percent_not_in_batch > 0:
                test_size1 = test_size*percent_not_in_batch
                x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train, y_train, test_size=test_size1, shuffle=True)
            else:
                x_train1 = x_train
                y_train1 = y_train

            classifier.train(x_train1.values, y_train1.values)
            predicted_labels = classifier.predict(x_test.values)
            Accuracy = accuracy(y_test.values, predicted_labels)
            Precision = precision(y_test.values, predicted_labels)
            Recall = recall(y_test.values, predicted_labels)
            evaluate_svm(Accuracy, Precision, Recall)

            percent_not_in_batch -= online_batch_percent

    else:
        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=test_size)
        classifier.train(x_train.values, y_train.values)
        predicted_labels = classifier.predict(x_test.values)
        Accuracy = accuracy(y_test.values, predicted_labels)
        Precision = precision(y_test.values, predicted_labels)
        Recall = recall(y_test.values, predicted_labels)
        evaluate_svm(Accuracy, Precision, Recall)

if __name__ == '__main__':

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Runs an online binary classification SVM algorithm, the default kernel is linear')

    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')

    # online vs batch data
    parser.add_argument('--online', dest='online', action='store_true', help='uses online data vs batch data')

    # Flags for types of kernel being called
    parser.add_argument('--polynomial-kernel', dest='poly_kernel', action='store_true',
                        help='enables the polynomial kernel')
    parser.add_argument('--exponential-kernel', dest='expo_kernel', action='store_true',
                        help='enables the exponential kernel')
    parser.add_argument('--gaussian-kernel', dest='gauss_kernel', action='store_true',
                        help='enables the gaussian kernel')
    parser.add_argument('--rational-quadratic-kernel', dest='rat_kernel', action='store_true',
                        help='enables the rational quadratic kernel')
    parser.add_argument('--wave-kernel', dest='wav_kernel', action='store_true',
                        help='enables the wave kernel')

    parser.set_defaults(kernel=linear_kernel, online=False, poly_kernel=False, expo_kernel=False, gauss_kernel=False, rat_kernel= False, wav_kernel=False)
    args = parser.parse_args()

    # storing that argument values
    data_path = os.path.expanduser(args.path)
    go_online = args.online

    if args.poly_kernel:
        kernel = polynomial_kernel
    elif args.expo_kernel:
        kernel = exponential_kernel
    elif args.gauss_kernel:
        kernel = gaussian_kernel
    elif args.rat_kernel:
        kernel = rational_quadratic_kernel
    elif args.wav_kernel:
        kernel = wave_kernel
    else:
        kernel = linear_kernel

    svm(data_path, go_online, kernel)
