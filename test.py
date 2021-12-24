import numpy as np
import pandas as pd

import knn as knn
import data_pipeline as pipe
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Classification datasets
    breast_cancer = 'https://raw.githubusercontent.com/jay619/Datasets/main/breast-cancer-wisconsin.data'
    cars = 'https://raw.githubusercontent.com/jay619/Datasets/main/car.data'
    house_votes84 = 'https://raw.githubusercontent.com/jay619/Datasets/main/house-votes-84.data'

    # Regression datasets
    abalone = 'https://raw.githubusercontent.com/jay619/Datasets/main/abalone.data'
    comp_hardware = 'https://raw.githubusercontent.com/jay619/Datasets/main/machine.data'
    forest_fires = 'https://raw.githubusercontent.com/jay619/Datasets/main/forestfires.data'

    breast_cancer_headers = ['sample_code_number', 'clump_thickness', 'uniformity_of_cell_size',
                             'uniformity_of_cell_shape', 'marginal_adhesion', 'single_epithelial_cell_size',
                             'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
    car_headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']
    house_votes_headers = ['class', 'handicapped_infants', 'water_project_cost_sharing',
                           'adoption_of_the_budget_resoluton', 'physician_fee_freeze', 'el_salvador_aid',
                           'religious_groups_in_schools', 'anti_satellite_test_ban', 'aid_to_nicaraguan_contras',
                           'mx_missile', 'immigration', 'synfuels_corporation_cutback', 'education_spending',
                           'superfund_right_to_sue', 'crime', 'duty_free_exports',
                           'export_administration_act_south_africa']
    abalone_headers = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight',
                       'shell_weight', 'rings']
    comp_hardware_headers = ['vendor', 'model', 'myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'prp', 'erp']

    breast_cancer_data = pipe.load_data(file_path=breast_cancer, has_column_headers=False,
                                        column_headers=breast_cancer_headers, has_na=True, na_values=['?'])
    # Dropping non-feature columns form the dataframe
    breast_cancer_data = pipe.drop_non_feature_columns(breast_cancer_data, column_labels=['sample_code_number'])
    # Replacing NA values
    breast_cancer_data = pipe.replace_na_with_feature_mean(breast_cancer_data)

    cars_data = pipe.load_data(file_path=cars, has_column_headers=False, column_headers=car_headers, has_na=False)
    # Cars categorical mapping
    buying_to_numerical = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
    maint_to_numerical = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
    doors_to_numerical = {'2': 0, '3': 1, '4': 2, '5more': 3}
    persons_to_numerical = {'2': 0, '4': 1, 'more': 2}
    lug_boot_to_numerical = {'small': 0, 'med': 1, 'big': 2}
    safety_to_numerical = {'low': 0, 'med': 1, 'high': 2}
    accept_to_numerical = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
    car_encoding = {'buying': buying_to_numerical, 'maint': maint_to_numerical, 'doors': doors_to_numerical,
                    'persons': persons_to_numerical, 'lug_boot': lug_boot_to_numerical, 'safety': safety_to_numerical,
                    'acceptability': accept_to_numerical}
    # Applying Ordinal encoding
    cars_data = pipe.categorical_encoding(dataframe=cars_data, categorical_data_type=0, encoding_mapping=car_encoding)

    house_votes84_data = pipe.load_data(file_path=house_votes84, has_column_headers=False,
                                        column_headers=house_votes_headers, has_na=False)
    house_votes_data = house_votes84_data[
        ['handicapped_infants', 'water_project_cost_sharing', 'adoption_of_the_budget_resoluton',
         'physician_fee_freeze', 'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban',
         'aid_to_nicaraguan_contras', 'mx_missile', 'immigration', 'synfuels_corporation_cutback', 'education_spending',
         'superfund_right_to_sue', 'crime', 'duty_free_exports', 'export_administration_act_south_africa']]
    house_votes_data = pipe.categorical_encoding(house_votes_data, 1, None)
    house_votes_class = house_votes84_data[['class']]
    house_votes_data['class'] = house_votes_class

    abalone_data = pipe.load_data(file_path=abalone, has_column_headers=False, column_headers=abalone_headers,
                                  has_na=False)
    abalone_data = pipe.categorical_encoding(dataframe=abalone_data, categorical_data_type=1)

    comp_hardware_data = pipe.load_data(file_path=comp_hardware, has_column_headers=False,
                                        column_headers=comp_hardware_headers, has_na=False)
    # Dropping non-feature columns
    comp_hardware_data = pipe.drop_non_feature_columns(comp_hardware_data, column_labels=['vendor', 'model'])
    comp_erp = comp_hardware_data['erp']  ## Will be used later
    comp_hardware_data = pipe.drop_non_feature_columns(comp_hardware_data, column_labels=['erp'])

    forest_fires_data = pipe.load_data(file_path=forest_fires, has_column_headers=True, has_na=False)
    # Applying the log transformation to the area as suggested by the author
    forest_fires_data['area'] = forest_fires_data['area'].apply(lambda a: np.log(a + 1))
    # Forest Fires categorical mapping
    month_to_numerical = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
                          'oct': 10, 'nov': 11, 'dec': 12}
    day_to_numerical = {'sun': 0, 'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6}
    forest_fires_encoding = {'month': month_to_numerical, 'day': day_to_numerical}
    # Uncomment to apply ordinal encoding to the forest fires dataset based on the forest_fires_encoding
    forest_fires_data = pipe.categorical_encoding(dataframe=forest_fires_data, categorical_data_type=0,
                                                  encoding_mapping=forest_fires_encoding)

    ####### For Tuning K ########
    # For tuning K value for breast cancer dataset
    tune_data = knn.get_tuning_data(breast_cancer_data, 0.20)
    # label = 'acceptability'
    label = 'class'
    remaining_data = breast_cancer_data.drop(index=tune_data.index.tolist())
    folds = pipe.k_fold_cross_validation(tune_data, folds=5, stratified=True, class_label=label)

    average_k_score = []
    ks = []
    k_values = [1, 2, 3, 5, 7, 13, 20]
    for k_val in k_values:
        average_pred = []
        ks.append(k_val)
        for i, x in enumerate(folds):
            test = x
            train_set = [l for idx, l in enumerate(folds) if idx != i]
            train = pd.DataFrame()
            for val in train_set:
                train = train.append(val)
            print('Running fold: {}'.format(i))
            print('Test size: {}, Train size: {}'.format(len(test), len(train)))
            pred = knn(train, test, k_val, label).to_numpy()
            true_labels = test[label].to_numpy().reshape(len(test), 1)
            accuracy = pipe.evaluation_metrics(true_output=true_labels, predicted_output=pred, metric_type=0)
            average_pred.append(accuracy)
            print('Classification Accuracy: {:.3f} %'.format(accuracy))
            print('*' * 50)
        fold_average = pipe.avg_output(average_pred)
        average_k_score.append(fold_average)
        print('Average classification score across 5-folds: {:.3f}% for K: {}'.format(fold_average, k_val))
        print('-' * 50)

    fig = plt.figure()
    fig.set_figwidth(8)
    fig.set_figheight(5)
    ax1 = fig.add_subplot(111)
    ax1.plot(ks, average_k_score, label='accuracy')
    ax1.legend()
    ax1.set_title('Accuracy v/s K values (KNN) - Cars Evaluation')
    plt.xticks(ks)

    for x, y in zip(ks, average_k_score):
        lab = "{:.2f}".format(y)
        ax1.annotate(lab, (x, y))

    ############ Running KNN with tuned K value
    best_k = 2
    folds = pipe.k_fold_cross_validation(remaining_data, folds=5, stratified=True, class_label=label)
    average_prediction = []

    for i, x in enumerate(folds):
        test = x
        train_set = [l for idx, l in enumerate(folds) if idx != i]
        train = pd.DataFrame()
        for val in train_set:
            train = train.append(val)
        print('Running fold: {}'.format(i))
        print('Test size: {}, Train size: {}'.format(len(test), len(train)))
        pred = knn(train, test, best_k, label).to_numpy()
        true_labels = test[label].to_numpy().reshape(len(test), 1)
        accuracy = pipe.evaluation_metrics(true_output=true_labels, predicted_output=pred, metric_type=0)
        average_pred.append(accuracy)
        print('Classification Accuracy: {} %'.format(accuracy))
        print('*' * 50)
    average_prediction.append(accuracy)
    print('Average classification score across 5-folds: {:.3f}% for K: {}'.format(pipe.avg_output(average_prediction),
                                                                                  best_k))
    print('-' * 50)

    ########## Running Edited Nearest Neighbor ############
    average_pred = []

    for i, x in enumerate(folds):
        test = x
        train_set = [l for idx, l in enumerate(folds) if idx != i]
        train = pd.DataFrame()
        for val in train_set:
            train = train.append(val)
        print('Running fold: {}'.format(i))
        print('Test size: {}, Train size: {}'.format(len(test), len(train)))
        # Editing the training set
        print('Size before editing the set: {}'.format(train.shape))
        z = knn.enn(train, tune_data, label)
        print('Size after editing the set: {}'.format(z.shape))
        # Predicting based on the newly edited training set
        new_pred = knn.knn(z, test, best_k, label)
        accuracy = pipe.evaluation_metrics(test[label], new_pred, 0)
        average_pred.append(accuracy)
        print('Classification Score: {:.2f} %'.format(accuracy))
        print('*' * 50)
    print('Average Classification Score across 5-folds: {:.2f} %'.format(pipe.avg_output(average_pred)))
    print('-' * 50)

    ########## Running Condensed Nearest Neighbor ############
    average_pred = []
    for i, x in enumerate(folds):
        test = x
        train_set = [l for idx, l in enumerate(folds) if idx != i]
        train = pd.DataFrame()
        for val in train_set:
            train = train.append(val)
        print('Running fold: {}'.format(i))
        print('Test size: {}, Train size: {}'.format(len(test), len(train)))
        # Editing the training set
        print('Size before editing the set: {}'.format(train.shape))
        z = knn.cnn(train, label)
        print('Size after editing the set: {}'.format(z.shape))
        # Predicting based on the newly edited training set
        new_pred = knn.knn(z, test, best_k, label)
        accuracy = pipe.evaluation_metrics(test[label], new_pred, 0)
        average_pred.append(accuracy)
        print('Classification Score: {:.2f} %'.format(accuracy))
        print('*' * 50)
    print('Average Classification Score across 5-folds: {:.2f} %'.format(pipe.avg_output(average_pred)))
    print('-' * 50)

    ########################################### Regression ###################################################
    # For tuning K
    # Normalizing the tuning data, can be changes as per the data set
    tune_data = pipe.normalize(pipe.get_tuning_data(comp_hardware_data, fraction=0.2), None)
    rest_data = comp_hardware_data.drop(index=tune_data.index.to_list())
    label = 'prp'

    # Running 5-fold cross validation on the tuning set  for tuning sigma and K
    # Look at the variance in the data among all features to get an idea for the range of sigma
    folds = pipe.k_fold_cross_validation(tune_data, folds=5, stratified=False, class_label=label)
    k_values = [3, 5, 7, 13]
    sigmas = [0.02, 0.05, 0.2, 0.5]
    avg_k_sigs = []
    for s in sigmas:
        avg_ks = []
        for k in k_values:
            avg_sigs = []
            for i, x in enumerate(folds):
                test = x
                train_set = [l for idx, l in enumerate(folds) if idx != i]
                train = pd.DataFrame()
                for val in train_set:
                    train = train.append(val)
                # train_std, test_std = normalize(train, test)
                train_std, test_std = train, test
                print('Running Fold: {}'.format(i))
                print('Train size: {}, Test size: {}'.format(train_std.shape, test_std.shape))
                pred = knn.smoother_knn(train_std, test_std, s, label, k)
                true_output = test_std[label]
                mse = pipe.evaluation_metrics(true_output, pred, 1)
                print('MSE: {:.5f} for K: {}, Sigma: {}'.format(mse, k, s))
                avg_sigs.append(mse)
                print('-' * 50)
            print('Average for 5 folds: {:.5f}'.format(pipe.avg_output(avg_sigs)))
            print('-' * 50)
            avg_ks.append(pipe.avg_output(avg_sigs))
        print('*' * 50)
        avg_k_sigs.append(avg_ks)

    ## Running 5-fold cross validation on the remaining 80% of the data set after tuning k & sigma
    ## Change the values for sigma and k as required
    avg_sigs = []
    s = 0.2
    k = 7
    folds = pipe.k_fold_cross_validation(rest_data, folds=5, stratified=False, class_label=label)
    for i, x in enumerate(folds):
        test = x
        train_set = [l for idx, l in enumerate(folds) if idx != i]
        train = pd.DataFrame()
        for val in train_set:
            train = train.append(val)
        print('Running fold: ', i)
        print('Test Size: {} Train Size: {}'.format(test.shape, train.shape))
        # train_std, test_std = train, test
        # Normalizing the data
        train_std, test_std = pipe.normalize(train, test)
        pred = knn.smoother_knn(train_std, test_std, s, label, k)
        true_output = test_std[label]
        mse = pipe.evaluation_metrics(true_output, pred, 1)
        print('MSE: {:.5f} for K: {}, Sigma: {}'.format(mse, k, s))
        avg_sigs.append(mse)
    print('Average MSE across 5-folds: {:.5f}'.format(pipe.avg_output(avg_sigs)))

    ## For tuning Epsilon, running different values of epsilon with tuned values of k and sigma
    ## Look at the variance in the output values for the range of epsilon
    avg_sigs = []
    # s = 0.09
    # k = 13
    epsilons = [0.05, 0.09, 0.12, 0.2]
    folds = pipe.k_fold_cross_validation(tune_data, folds=5, stratified=False, class_label=label)
    eps_avg = []
    for e in epsilons:
        for i, x in enumerate(folds):
            test = x
            train_set = [l for idx, l in enumerate(folds) if idx != i]
            train = pd.DataFrame()
            for val in train_set:
                train = train.append(val)
            print('Running fold: ', i)
            print('Test Size: {} Train Size: {}'.format(test.shape, train.shape))
            train_std, test_std = pipe.normalize(train, test)
            # Run either ENN or CNN to edit the training set
            z = knn.smoother_enn(train_std, tune_data, label, s, e)
            # z = smoother_cnn(train_std, s, epsilons, label)
            print('After editing: ', z.shape)
            pred = knn.smoother_knn(z, test_std, s, label, k)
            true_output = test_std[label]
            mse = pipe.evaluation_metrics(true_output, pred, 1)
            print('MSE: {:.5f} for K: {}, Sigma: {}, Epsilon: {}'.format(mse, k, s, e))
            avg_sigs.append(mse)
            print('-' * 50)
        print('Average MSE across 5-folds: {:.5f}'.format(pipe.avg_output(avg_sigs)))
        eps_avg.append(pipe.avg_output(avg_sigs))
        print('*' * 50)

    ## Running ENN on tuned epsilon, K and sigma
    avg_sigs = []
    # s = 0.07
    # k = 7
    epsilons = 0.12
    folds = pipe.k_fold_cross_validation(rest_data, folds=5, stratified=False, class_label=label)
    for i, x in enumerate(folds):
        test = x
        train_set = [l for idx, l in enumerate(folds) if idx != i]
        train = pd.DataFrame()
        for val in train_set:
            train = train.append(val)
        print('Running fold: ', i)
        print('Test Size: {} Train Size: {}'.format(test.shape, train.shape))
        train_std, test_std = pipe.normalize(train, test)
        # Editing the training set
        z = knn.smoother_enn(train_std, tune_data, label, s, epsilons)
        print('After editing: ', z.shape)
        # KNN on the edited training set
        pred = knn.smoother_knn(z, test_std, s, label, k)
        true_output = test_std[label]
        mse = pipe.evaluation_metrics(true_output, pred, 1)
        print('MSE: {:.5f} for K: {}, Sigma: {}, Epsilon: {}'.format(mse, k, s, epsilons))
        avg_sigs.append(mse)
        print('-' * 50)
    print('Average MSE across 5-folds: {:.5f}'.format(pipe.avg_output(avg_sigs)))
    print('*' * 50)

    ## Running ENN on tuned epsilon, K and sigma
    avg_sigs = []
    # s = 0.09
    # k = 13
    epsilons = 0.12
    folds = pipe.k_fold_cross_validation(rest_data, folds=5, stratified=False, class_label=label)
    for i, x in enumerate(folds):
        test = x
        train_set = [l for idx, l in enumerate(folds) if idx != i]
        train = pd.DataFrame()
        for val in train_set:
            train = train.append(val)
        print('Running fold: ', i)
        print('Test Size: {} Train Size: {}'.format(test.shape, train.shape))
        train_std, test_std = pipe.normalize(train, test)
        # Condensing the training set
        z = knn.smoother_cnn(train_std, s, epsilons, label)
        print('After condensing: ', z.shape)
        # KNN on the condensed training set
        pred = knn.smoother_knn(z, test_std, s, label, k)
        true_output = test_std[label]
        mse = pipe.evaluation_metrics(true_output, pred, 1)
        print('MSE: {:.5f} for K: {}, Sigma: {}, Epsilon: {}'.format(mse, k, s, epsilons))
        avg_sigs.append(mse)
        print('-' * 50)
    print('Average MSE across 5-folds: {:.5f}'.format(pipe.avg_output(avg_sigs)))
    print('*' * 50)
