import pandas as pd
import numpy as np
import data_pipeline as p1


def get_tuning_data(data, fraction=0.2) -> pd.DataFrame:
    """
    Randomly sample 20% of the data
    @param data: The data set from which data needs to be sampled
    @param fraction: The fraction of data that needs to be sampled. By default 20% is sampled if no value is provided
    @return: Returns the sampled data
    """
    data = data.sample(frac=fraction)
    return data


def normalize(train, test):
    """
    Apply Mini-Max Normalization to the data
    @param train: Training set that needs to be normalized
    @param test: Test set that needs to be normalized
    @return: Normalized Training & Test set
    """
    min = train.min()
    max = train.max()

    if test is None:
        return np.divide(np.subtract(train, min), np.subtract(max, min))
    return np.divide(np.subtract(train, min), np.subtract(max, min)), np.divide(np.subtract(test, min),
                                                                                np.subtract(max, min))


def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean Distance between point1 and point2
    @param point1:
    @param point2:
    @return:
    """
    return np.power(np.sum(np.power(np.subtract(point2, point1), 2)), 0.5)


def knn(train, test, num_of_neighbors, class_label):
    """
    K-Nearest Algorithm
    @param train: The training data set
    @param test: The test data set
    @param num_of_neighbors: K value or the number of neighbors to check
    @param class_label: The class label
    @return: Returns a prediction for point(s) in the test set
    """
    # Creating an empty set for storing predictions
    prediction = pd.DataFrame()
    # Separating the class label feature
    train_class = train[class_label]
    # Removing the class label feature from the training data
    train = train.drop(columns=[class_label])
    test_class = test[class_label]
    test = test.drop(columns=[class_label])
    for idx_test, row_test in test.iterrows():
        # Creating an empty set to store the distance and index between the query point and
        # all the points in the training set
        dist_label = pd.DataFrame(columns=['idx', 'distance', 'class_label'])
        for idx_train, row_train in train.iterrows():
            # Distance between every row in the test with every row in the train
            distance = euclidean_distance(row_train, row_test)
            # Getting the label
            label = train_class[idx_train]
            # Storing the index, distance and the class label
            dist_label = dist_label.append({'idx': idx_train, 'distance': distance, 'class_label': label},
                                           ignore_index=True)
        # Sorting the set in ascending order of distance and getting the k closed data points
        dist_label = dist_label.sort_values(by='distance')[:num_of_neighbors]
        # Getting the majority class within the k neighbors
        pred_label = dist_label['class_label'].mode()
        prediction = prediction.append(pred_label, ignore_index=True)
    return prediction


def enn(train_set, validation_set, class_label):
    """
    Edited Nearest Neighbor
    @param train_set: The training set which needs to be edited
    @param validation_set: Validation set against whcih the edited training set is evaluated
    @param class_label: The class label
    @return: Returns the edited training set
    """
    Z = train_set
    validation_labels = validation_set[class_label]
    previous_score = -1
    previous_size = len(Z)
    curr_score = 0
    curr_size = -1
    # print('Current Score: {}, Previous Score: {}'.format(curr_score, previous_score))
    # Pass through the training set until the performance doesn't improve
    while curr_score > previous_score:  # or curr_size == previous_size:
        # previous_score = p1.evaluation_metrics(validation_labels, knn(Z, validation_set, k, class_label), 0)
        previous_score = p1.evaluation_metrics(validation_labels, knn(Z, validation_set, 1, class_label), 0)
        # print('Current Score: {}, Previous Score: {}'.format(curr_score, previous_score))
        for idx, row in Z.iterrows():
            remaining_data = Z.iloc[lambda x: x.index != idx]
            # Edited nearest neighbor with k=1
            # Prediction a data point from the set using the rest of the set
            pred = knn(remaining_data, row.to_frame().T, 1, class_label).to_numpy()[0][0]
            # Getting the true/correct label for the data point
            true_label = row[class_label]
            # If the prediction matches the rest of the data, we drop that record from the training set
            if pred == true_label:
                Z.drop(idx, inplace=True)
        # Evaluating the edited set against the validation set
        new_pred = knn(Z, validation_set, 1, class_label)
        curr_score = p1.evaluation_metrics(validation_labels, new_pred, 0)
    return Z


def cnn(data, class_label):
    """
    Condensed Nearest Neighbor
    @param data: The training set that needs to be condensed
    @param class_label: The class label (column header) for the training set
    @return: Returns a condensed training set
    """
    data_cpy = data.copy()
    # Randomizing the data
    data_cpy = data_cpy.sample(frac=1)
    # Creating an empty set to store condensed values
    Z = pd.DataFrame()
    previous_size = -1
    curr_size = 0
    # print('Previous: {}, Current: {}'.format(previous_size, curr_size))
    # Keep condensing the set until the size of Z doesn't change
    while curr_size != previous_size:
        # For every data point in the data set, evaluating it using rest of the points
        for idx, row in data_cpy.iterrows():
            # Initial condition when Z is empty, so adding the first point
            if len(Z) == 0:
                Z = Z.append(row)
            previous_size = curr_size
            # Predicting the query point using the rest of the points
            pred = knn(train=Z, test=row.to_frame().T, num_of_neighbors=1, class_label=class_label).to_numpy()[0][0]
            true_label = row[class_label]
            # If the predicted class is not equal to the true value, removing it from the training set
            if pred != true_label:
                Z = Z.append(row)
                data_cpy.drop(idx, inplace=True)
            curr_size = len(Z)
            # print('Previous: {}, Current: {}'.format(previous_size, curr_size))
        # If no points are removed at all, return the original training set
        if len(Z) == 0:
            Z = data
    return Z


def gaussianKernel(sigma, point1, point2):
    """
    Gaussian Kernel for Regression to smooth the outputs
    K = exp((-1/2*sigma) * Distance(point1, point2))
    @param sigma: The spread of Sigma value for Gaussian Kernel.
    @param point1: Used to calculate distance between two points
    @param point2: Used to calculate distance between two points
    @return: The kernel value for point1 and point2 or the weight
    """
    euclid_dist = euclidean_distance(point1=point1, point2=point2)
    return np.exp(np.divide(-1, 2 * sigma) * euclid_dist)


def smoother_knn(train, test, sigma, class_label, k=2):
    """
    Smoother KNN for regression
    @param train: The training set for regression
    @param test: The test set for which output needs to be predicted
    @param sigma: The h or sigma value for the Gaussian kernel and the regression function
    @param class_label: The class label (column header)
    @param k: Number of neighbors to check
    @return: Returns the predicted output for point(s) in the test set
    """
    # Storing the true outputs
    true_output_train = train[class_label]
    train = train.drop(columns=[class_label])
    # Creating empty set for storing predicted outputs
    pred_output = []
    true_output_test = test[class_label]
    test = test.drop(columns=[class_label])
    for idx_test, row_test in test.iterrows():
        # For every test point get the neighbors first, creating an empty frame to store the k neighbors
        # neighbors will store the the index of the training data point close to the test point and its distance
        neighbors = pd.DataFrame(columns=['idx', 'distance'])
        pred_value = 0
        # First getting the neighbors from the training set for they query point
        for idx_train, row_train in train.iterrows():
            dist = euclidean_distance(row_test, row_train)
            neighbors = neighbors.append({'idx': idx_train, 'distance': dist}, ignore_index=True)
        # Getting the first k neighbors
        neighbors = neighbors.sort_values(by='distance')[:k]
        # print(neighbors)
        # Getting the complete data for the first k neighbors
        new_train = train.loc[neighbors['idx'].to_list()]
        # Getting the output of the k training neighbors
        true_output_new = true_output_train.loc[neighbors['idx'].to_list()]
        # Applying g(x) along with Gaussian kernel to each of the neighboring data points to get the predicted output
        numerator = 0
        denominator = 0
        # print(neighbors)
        for idx_neighbor, row_neighbor in new_train.iterrows():
            # Getting the distance between the query point and the neighbor
            euclid_dist = euclidean_distance(row_test, row_neighbor)
            # kernel = np.exp(np.divide(-1, 2*sigma) * euclid_dist)
            kernel = gaussianKernel(sigma, row_test, row_neighbor)
            # Adding this condition because I noticed if the points were similar/duplicates in the data set, the
            # distance would be zero and we get division by zero
            if euclid_dist == 0:
                euclid_dist = 0.00001
            # Getting the output value for neighbor (r^t)
            true_output_value = true_output_new.loc[idx_neighbor]
            numerator = numerator + (kernel * (euclid_dist / sigma) * true_output_value)
            denominator = denominator + (kernel * (euclid_dist / sigma))

        pred_value = np.divide(numerator, denominator)
        # print('Pred: ', (pred_value))
        pred_output.append(pred_value)
    return pred_output


def smoother_enn(train_set, validation_set, class_label, sigma, epsilon):
    """

    @param train_set: The training set that needs to be edited
    @param validation_set: The set against which the edited set is validated
    @param class_label: The class label (column header)
    @param sigma: The h or sigma value for the Gaussian kernel and the regression function
    @param epsilon: The error threshold. The prediction needs to be within this threshold to be edited
    @return: Returns the edited training set
    """
    # Shuffling the training set
    train_set = train_set.sample(frac=1)
    Z = train_set
    validation_labels = validation_set[class_label]
    previous_score = 0
    curr_score = -1
    # Editing the training set until the performance improves i.e., the MSE is low
    while curr_score < previous_score:
        print('Current: {}, Previous: {}'.format(curr_score, previous_score))
        previous_score = p1.evaluation_metrics(validation_labels, smoother_knn(Z, validation_set, sigma=sigma,
                                                                               class_label=class_label), 1)
        print('Previous MSE: {}'.format(previous_score))
        for idx, row in Z.iterrows():
            remaining_data = Z.iloc[lambda x: x.index != idx]
            # Edited nearest neighbor with k=1
            # Prediction a data point from the set using the rest of the set
            pred = smoother_knn(remaining_data, row.to_frame().T, sigma=sigma, class_label=class_label, k=1)
            # Getting the true/correct label for the data point
            true_output = row[class_label]
            # If the prediction matches the rest of the data, we drop that record from the training set
            if (true_output - epsilon) < pred < (true_output + epsilon):
                Z.drop(idx, inplace=True)
        new_pred = smoother_knn(Z, validation_set, sigma, class_label)
        curr_score = p1.evaluation_metrics(validation_labels, new_pred, 1)
        print('Current MSE: {}'.format(curr_score))
    return Z


def smoother_cnn(data, sigma, epsilon, class_label):
    """
    Condensed Nearest Neighbor for regression
    @param data: The training set that needs to be condensed
    @param sigma: The h or sigma value for the Gaussian kernel and the regression function
    @param epsilon: The error threshold. The prediction needs to be within this threshold to be edited
    @param class_label: The class label (column header)
    @return: Returns condensed training set
    """
    data_cpy = data.copy()
    # Randomizing the data
    data_cpy = data_cpy.sample(frac=1)
    # Creating an empty set to store training samples
    Z = pd.DataFrame()
    previous_size = -1
    curr_size = 0
    # Keep editing the training set until the size doesn't change
    while curr_size != previous_size:
        # For every data point in the data set, evaluating it using rest of the points
        for idx, row in data_cpy.iterrows():
            # Initial condition when Z is empty, so adding the first point
            if len(Z) == 0:
                Z = Z.append(row)
            previous_size = curr_size
            # Predicting the output for the query point using the rest of the points in Z
            pred = smoother_knn(train=Z, test=row.to_frame().T, sigma=sigma, class_label=class_label, k=1)
            true_output = row[class_label]
            # If the predicted output is not within the threshold of the ture output, it is added to Z
            if not ((true_output - epsilon) < pred < (true_output + epsilon)):
                Z = Z.append(row)
                data_cpy.drop(idx, inplace=True)
            curr_size = len(Z)
        # If none of the points are added to Z, the whole un-condensed training set is returned
        if len(Z) == 0:
            Z = data
    return Z



