import pandas as pd
import numpy as np


def replace_na_with_feature_mean(dataframe) -> pd.DataFrame:
    """
    The function will replace any NA values in the dataset with the mean of the feature. This will check all features
    in the dataset and see if any of them have NA values, if yes, it will replace all the NA values with the mean of
    that particular feature
    :param dataframe: Pandas dataframe with one or more features
    :return: Return the dataset with NA values imputed
    """
    # Getting NA counts by each feature
    na_counts_by_feature = dataframe.isna().sum().to_dict()
    for key in na_counts_by_feature:
        # Imputing NAs with the mean if the NA counts for any feature is > 0
        if na_counts_by_feature[key] > 0:
            dataframe[key] = dataframe[key].fillna(value=dataframe[key].mean())
    return dataframe


def drop_non_feature_columns(dataframe, column_labels) -> pd.DataFrame:
    """
    Drops the specified columns form the dataframe
    :param dataframe: dataframe from which column(s) need to be dropped
    :param column_labels: A list exact column label(s) to be dropped
    :return: dataframe with the dropped feature/column
    """
    return dataframe.drop(labels=column_labels, axis=1)


def load_data(file_path, has_column_headers, column_headers=None, has_na=False, na_values=None) -> pd.DataFrame:
    """
    The function reads in the data from a CSV file
    :param file_path: Path to the CSV file stored locally on your machine that you want to read in
    :param has_column_headers: Boolean value if the CSV has column headers or not.
    :param column_headers: List of column headers to add to the dataframe. Column headers are only added
    if has_column_headers = False
    :param has_na: Boolean value to indicate if the data has NA or empty values
    :param na_values: If the data has NA values, you can provide a list of different NA values in the data.
    e.g. ?, NaN, NULL
    :return: returns a dataframe of the data
    """
    if has_column_headers:
        if has_na:
            dataframe = pd.read_csv(file_path, header=0, na_values=na_values)
        else:
            dataframe = pd.read_csv(file_path, header=0)
    else:
        if has_na:
            dataframe = pd.read_csv(file_path, header=None, names=column_headers, na_values=na_values)
        else:
            dataframe = pd.read_csv(file_path, header=None, names=column_headers)
    return dataframe


def categorical_encoding(dataframe, categorical_data_type, encoding_mapping=None) -> pd.DataFrame:
    """
    This method converts categorical data to proper encoding
    :param dataframe: Dataframe on which categorical encoding needs to be applied
    :param categorical_data_type: 0 - Ordinal (Order matters), 1 - Nominal (Order doesn't matter - One hot enconding)
    :param encoding_mapping: If ordinal encoding is selected, the encoding
    mapping (a dictionary) needs to be provided. E.g. {'high': 1, 'medium': 2, 'low':3}
    :return: Returns the
    dataframe with the encoding applied
    """
    if categorical_data_type == 0:
        # Applying Ordinal encoding based on the mapping provided by the user
        dataframe = dataframe.replace(encoding_mapping)
    else:
        # Applying one-hot encoding
        dataframe = pd.get_dummies(dataframe)
    return dataframe


def get_bin_range(min_value, max_value, total_bins):
    """
    This is a helper function for data discretization
    :param min_value: A minimum value in the range
    :param max_value: A max value ni the range
    :param total_bins: Total bins needed in the range
    :return: Based on the min, max and total bins, returns a list of starting points for the bins
    """
    steps = int((max_value - min_value) / total_bins)
    rng = []
    for x in range(min_value, max_value, steps):
        rng.append(x)
    return rng


def data_discretization(dataset, equal_width, bins, labels=None):
    """
    Converts continuous data into discrete data based on the number of bins needed :param dataset: Dataset which needs
    to be discretized :param equal_width: Boolean value to indicate if equal width discretization is need. Equal
    frequency discretization is applied if false :param bins: Number of bins needed for the discretized column :param
    labels: Label of the new discretized column :return: Returns a new discretized column
    """
    if equal_width:
        # Getting the starting points for the bins for equal width and giving labels to the bins if provided by the user
        # the lowest value is included in the bins
        rng = get_bin_range(min_value=dataset.min(), max_value=dataset.max(), total_bins=bins)
        discrete_data = pd.cut(dataset, bins=rng, retbins=True, include_lowest=True, labels=labels)
    else:
        # Creating equal sized bins and giving the bins any labels if provided by the user
        discrete_data = pd.qcut(dataset, bins, retbins=True, labels=labels, precision=0)
    return discrete_data


def z_score_standardization(train, test):
    """
    Applies Z-score standardization to the train and test sets. Please note: the test set is standardized based on the
    mean and standard deviation from the train set
    :param train: The train set that needs to be standardized
    :param test: The test set that needs to be standardized
    :return: A tuple of standardized train and test set
    """
    # Getting the means and standard deviations based on the training dataset
    mean = train.mean()
    std = train.std()
    # Applying Z-score standardization and returning both the train and test set
    return np.divide(np.subtract(train, mean), std), np.divide(np.subtract(test, mean), std)


def get_stratified_sample(dataset, no_of_samples, class_header):
    """
    This is a helper function for K-Fold cross validation to get stratified samples per fold.
    :param dataset: Dataset on which stratified sampling needs to be applied
    :param no_of_samples: Number of samples needed per fold
    :param class_header: Column header which represents the classes
    :return: returns sampled dataset maintaining class proportions
    """
    # Getting unique classes in the dataset
    unique_classes = dataset[class_header].unique().tolist()
    stratified_fold = pd.DataFrame()
    for uclass in unique_classes:
        # Getting all the datapoints for a specific class which will be used for sampling
        datapoints_by_class = dataset[dataset[class_header] == uclass]
        proportion = round(no_of_samples * len(datapoints_by_class)/ len(dataset))
        if proportion > len(datapoints_by_class):
            # print('Proportions: {}, Data by class: {}'.format(proportion, len(datapoints_by_class)))
            proportion = proportion - 1
        # Sampling class specific data based on the proportions (or frequency)
        fold = datapoints_by_class.sample(proportion)
        # print('Fold size for class {}: {}'.format(uclass, len(fold)))
        stratified_fold = stratified_fold.append(fold)
    return stratified_fold


def k_fold_cross_validation(dataset, folds=2, stratified=False, class_label=None):
    """
    Applies K-Fold cross validation on the dataset and returns a list of subset of the dataset based on
    the number of folds
    :param dataset: Input dataset for K-Fold cross validation
    :param folds: Number of folds to create on the dataset. By default creates 2 folds
    :param stratified: Boolean value if the class distribution needs to be maintained across each fold/subset
    :param class_label: Class on which the data needs to be stratified.
    :return: A list of subset of the dataset based on the number of folds
    """
    data_size = len(dataset)
    copy_dataset = dataset.copy(deep=True)
    # Need to round to the nearest integer
    samples_per_fold = round(data_size / folds)
    # By rounding to the nearest integer I noticed for few instances where samples per fold * folds was greater than
    # the total dataset size which is not valid. Hence, subtracting 1 from the samples per fold for such instances
    if (samples_per_fold * folds) > len(dataset):
        samples_per_fold = samples_per_fold - 1
    split_data = []

    for k in range(folds):
        if stratified:
            # For stratified sampling, grouping the groups together first and then for each group I'm getting the
            # proportion of datapoints per class and sampling based on the proportions
            fold = get_stratified_sample(copy_dataset, samples_per_fold, class_label)
            # print(fold.index.tolist()) # To compare the indices for each fold to check for duplicates
        else:
            # Sample data randomly from the dataset
            fold = copy_dataset.sample(samples_per_fold)
        index_of_sample = fold.index.values.tolist()
        # Dropping data points from the copy dataset that have already been sampled, so there is no duplicates
        copy_dataset.drop(labels=index_of_sample, axis=0, inplace=True)
        split_data.append(fold)
    # If the data doesn't split evenly across k folds, the remaining datapoints are distributed in each of the folds
    while len(copy_dataset) != 0:
        for elements_remaining in range(len(copy_dataset)):
            # print('Elements remaining: ', len(copy_dataset))
            datapoint = copy_dataset.sample(1)
            datapoint_idx = datapoint.index.values.tolist()
            split_data[elements_remaining] = split_data[elements_remaining].append(datapoint)
            copy_dataset.drop(labels=datapoint_idx, axis=0, inplace=True)
    return split_data


def evaluation_metrics(true_output, predicted_output, metric_type):
    """
    0 - Classification Score
    1 - Mean Squared Error
    2 - Mean Absolute Error
    :param true_output: The true values
    :param predicted_output: The predicted values 
    :param metric_type: 0 - Classification Score, 1 - Mean Squared Error, 2 - Mean Absolute Error
    :return: Returns the evaluated metric i.e. Classification Score, Mean Squared Error or Mean Absolute Error
    """
    true_output = pd.DataFrame(true_output)
    predicted_output = pd.DataFrame(predicted_output)
    if metric_type == 0:
        # Classification Score
        # Converting to Numpy Array as a warning was thrown if they were not converted
        true_output = true_output.to_numpy()
        predicted_output = predicted_output.to_numpy()
        # Returns classification score as a %
        return np.divide(len(np.where(predicted_output == true_output)[0]), len(true_output)) * 100
    elif metric_type == 1:
        # # Converting to Numpy Array as a warning was thrown if they were not converted
        true_output = true_output.to_numpy()
        predicted_output = predicted_output.to_numpy()
        # Mean Squared Error
        sq_sum = np.sum(np.square(np.subtract(predicted_output, true_output)))
        total_data_points = len(true_output)
        return np.divide(sq_sum, total_data_points)
    elif metric_type == 2:
        # Converting to Numpy Array as a warning was thrown if they were not converted
        true_output = true_output.to_numpy()
        predicted_output = predicted_output.to_numpy()
        # Mean Absolute Error
        absolute_sum = np.sum(np.absolute(np.subtract(true_output, predicted_output)))
        total_data_points = len(true_output)
        return absolute_sum / total_data_points


def majority_predictor(train, test, is_classification):
    """
    Returns the majority class (classification) from the training set as the predicted class for the test set or the
    average of the outputs (regression) from the training set as the predicted output for the test set
    :parm train: The training dataset
    :parm test: The test dataset
    :parm is_classification: Boolean value to see if this is a classification problem or not.
    If False, it is considered to be a regression problem
    """
    if is_classification:
        majority_class = train.mode()
        pred = []
        print('Majority class for Classification: {}'.format(majority_class))
        # For every datapoint in the test set, predicting the class label (the majority class) and appending to pred
        for _ in range(len(test)):
            pred.append(majority_class)
        prediction = pd.DataFrame(pred)
    else:
        avg_output = train.mean()
        print('Regression Avg Output: {}'.format(avg_output))
        pred = []
        # For every datapoint in the test set, predicting the output (mean output from train) and appending to pred
        for _ in range(len(test)):
            pred.append(avg_output)
        prediction = pd.DataFrame(pred)
    return prediction


def avg_output(metrics):
    """
    Takes in a list and returns the average. Helper function for K-Fold cross validation to return the average of
    the evaluation metrics after running K folds
    :param metrics: A list of evaluation metrics returned for each fold during K-Fold cross validation
    :return: Return the average of the metrics
    """
    total = 0
    for val in metrics:
        total += val
    return total / len(metrics)
