# Import things
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers.recurrent import LSTM

np.set_printoptions(threshold=np.nan)

INDEX_COLS = ['subject_id', 'icustay_id', 'hours_in', 'hadm_id']

# where the X, Y, static raw data is
data_path = 'data/'

# where you will save the processed data matrices
save_data_path = 'data/mortality/'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default='mortality_test',
                        help="This will become the name of the folder where are the models and results \
        are stored. Type: String. Default: 'mortality_test'.")
    parser.add_argument("--data_hours", type=int, default=24,
                        help="The number of hours of data to use in making the prediction. \
        Type: int. Default: 24.")
    parser.add_argument("--gap_time", type=int, default=12, \
                        help="The gap between data and when we are making predictions. Type: int. Default: 12.")
    parser.add_argument("--model_type", type=str, default='GLOBAL',
                        help="One of {'GLOBAL', MULTITASK', 'SEPARATE'} indicating \
        which type of model to run. Type: String.")
    parser.add_argument("--num_lstm_layers", type=int, default=1,
                        help="Number of beginning LSTM layers, applies to all model types. \
        Type: int. Default: 1.")
    parser.add_argument("--lstm_layer_size", type=int, default=16,
                        help="Number of units in beginning LSTM layers, applies to all model types. \
        Type: int. Default: 16.")
    parser.add_argument("--num_dense_shared_layers", type=int, default=0,
                        help="Number of shared dense layers following LSTM layer(s), applies to \
        all model types. Type: int. Default: 0.")
    parser.add_argument("--dense_shared_layer_size", type=int, default=0,
                        help="Number of units in shared dense layers, applies to all model types. \
        Type: int. Default: 0.")
    parser.add_argument("--num_multi_layers", type=int, default=0,
                        help="Number of separate-task dense layers, only applies to multitask models. Currently \
        only 0 or 1 separate-task dense layers are supported. Type: int. Default: 0.")
    parser.add_argument("--multi_layer_size", type=int, default=0,
                        help="Number of units in separate-task dense layers, only applies to multitask \
        models. Type: int. Default: 0.")
    parser.add_argument("--cohorts", type=str, default='careunit',
                        help="One of {'careunit', 'saps', 'custom'}. Indicates whether to use pre-defined cohorts \
        (careunits or saps quartile) or use a custom cohort membership (i.e. result of clustering). \
        Type: String. Default: 'careunit'. ")
    parser.add_argument("--cohort_filepath", type=str, help="This is the filename containing a numpy \
        array of length len(X), containing the cohort membership for each example in X. This file should be \
        saved in the folder 'cluster_membership'. Only applies to cohorts == 'custom'. Type: str.")
    parser.add_argument("--sample_weights", action="store_true", default=False, help="This is an indicator \
        flag to weight samples during training by their cohort's inverse frequency (i.e. smaller cohorts will be \
        more highly weighted during training).")
    parser.add_argument("--include_cohort_as_feature", action="store_true", default=False,
                        help="This is an indicator flag to include cohort membership as an additional feature in the matrix.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train for. Type: int. Default: 30.")
    parser.add_argument("--train_val_random_seed", type=int, default=0,
                        help="Random seed to use during train / val / split process. Type: int. Default: 0.")
    parser.add_argument("--repeats_allowed", action="store_true", default=False,
                        help="Indicator flag allowing training and evaluating of existing models. Without this flag, \
        if you run a configuration for which you've already saved models & results, it will be skipped.")
    parser.add_argument("--no_val_bootstrap", action="store_true", default=False,
                        help="Indicator flag turning off bootstrapping evaluation on the validation set. Without this flag, \
        minimum, maximum and average AUCs on bootstrapped samples of the validation dataset are saved. With the flag, \
        just one AUC on the actual validation set is saved.")
    parser.add_argument("--num_val_bootstrap_samples", type=int, default=100,
                        help="Number of bootstrapping samples to evaluate on for the validation set. Type: int. Default: 100. ")
    parser.add_argument("--test_time", action="store_true", default=False,
                        help="Indicator flag of whether we are in testing time. With this flag, we will load in the already trained model \
        of the specified configuration, and evaluate it on the test set. ")
    parser.add_argument("--test_bootstrap", action="store_true", default=False,
                        help="Indicator flag of whether to evaluate on bootstrapped samples of the test set, or just the single \
        test set. Adding the flag will result in saving minimum, maximum and average AUCs on bo6otstrapped samples of the validation dataset. ")
    parser.add_argument("--num_test_bootstrap_samples", type=int, default=100,
                        help="Number of bootstrapping samples to evaluate on for the test set. Type: int. Default: 100. ")
    parser.add_argument("--gpu_num", type=str, default='0', 
                        help="Limit GPU usage to specific GPUs. Specify multiple GPUs with the format '0,1,2'. Type: String. Default: '0'.")

    args = parser.parse_args()
    print(args)
    return args

################ HELPER FUNCTIONS ###############################################
####################################################################################


def load_phys_data():
    """ 
    Loads X, Y, and static matrices into Pandas DataFrames 
    Returns:
        X: Pandas DataFrame containing one row per patient per hour. 
           Each row should include the columns {'subject_id', 'icustay_id', 'hours_in', 'hadm_id'}
           along with any additional features.
        static: Pandas DataFrame containing one row per patient. 
                Should include {'subject_id', 'hadm_id', 'icustay_id'}.
    """

    X = pd.read_hdf(data_path + 'X.h5', 'X')
    # Y = pd.read_hdf(data_path + 'Y.h5', 'Y')
    static = pd.DataFrame.from_csv(data_path + 'static.csv')

    if 'subject_id' not in X.columns:
        X = X.reset_index()
        X.columns = [fix_byte_data(c) for c in X.columns]
    # if 'subject_id' not in Y.columns:
    #     Y = Y.reset_index()
    #     Y.columns = [fix_byte_data(c) for c in Y.columns]

    static = static[static.subject_id.isin(np.unique(X.subject_id))]
    return X, static


def categorize_ethnicity(ethnicity):
    """ 
    Groups ethnicity sub-categories into 5 major categories.
    Args:
        ethnicity (str): string indicating patient ethnicity.
    Returns:
        string: ethnicity. Categorized into 5 main categories. 
    """

    if 'ASIAN' in ethnicity:
        ethnicity = 'ASIAN'
    elif 'WHITE' in ethnicity:
        ethnicity = 'WHITE'
    elif 'HISPANIC' in ethnicity:
        ethnicity = 'HISPANIC/LATINO'
    elif 'BLACK' in ethnicity:
        ethnicity = 'BLACK'
    else:
        ethnicity = 'OTHER'
    return ethnicity


def make_discrete_values(mat):
    """ 
    Converts numerical values into one-hot vectors of number of z-scores 
    above/below the mean, aka physiological words (see Suresh et al 2017).
    Args:
        mat (Pandas DataFrame): Matrix of feature values including columns in
        INDEX_COLS as the first columns.
    Returns:
        DataFrame: X_categorized. A DataFrame where each features is a set of
        indicator columns signifying number of z-scores above or below the mean.
    """

    normal_dict = mat.groupby(['subject_id']).mean().mean().to_dict()
    std_dict = mat.std().to_dict()
    feature_cols = mat.columns[len(INDEX_COLS):]
    print(feature_cols)
    X_words = mat.loc[:, feature_cols].apply(
        lambda x: transform_vals(x, normal_dict, std_dict), axis=0)
    mat.loc[:, feature_cols] = X_words
    X_categorized = pd.get_dummies(mat, columns=mat.columns[len(INDEX_COLS):])
    na_columns = [col for col in X_categorized.columns if '_9' in col]
    X_categorized.drop(na_columns, axis=1, inplace=True)
    return X_categorized


def transform_vals(x, normal_dict, std_dict):
    """ 
    Helper function to convert values to z-scores between -4 and 4. 
    Missing values are assigned 9. 
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """

    x = 1.0*(x - normal_dict[x.name])/std_dict[x.name]
    x = x.round()
    x = x.clip(-4, 4)
    x = x.fillna(9)
    x = x.round(0).astype(int)
    return x


def categorize_age(age):
    """ 
    Categorize age into windows. 
    Args:
        age (int): A number.
    Returns:
        int: cat. The age category.
    """

    if age > 10 and age <= 30:
        cat = 1
    elif age > 30 and age <= 50:
        cat = 2
    elif age > 50 and age <= 70:
        cat = 3
    else:
        cat = 4
    return cat


def _pad_df(df, max_hr, pad_value=np.nan):
    """ Add dataframe with padding up to max stay. """

    existing = set(df.index.get_level_values(1))
    fill_hrs = set(range(max_hr)) - existing
    if len(fill_hrs) > 0:
        return fill_hrs
    else:
        return 0


def fix_byte_data(s):
    """ Python 2/3 fix """

    try:
        s = s.decode()
    except AttributeError:
        pass
    return s


def stratified_split(X, Y, cohorts, train_val_random_seed=0):
    """ 
    Return stratified split of X, Y, and a cohort membership array, stratified by outcome. 
    Args:
        X (Numpy array): X matrix, shape = num patients x num timesteps x num features.
        Y (Numpy array): Y matrix, shape = num_patients.
        cohorts (Numpy array): array of cohort membership, shape = num_patients.
        train_val_random_seed (int): random seed for splitting.
    Returns:
        Numpy arrays: X_train, X_val, X_test, y_train, y_val, y_test, 
        cohorts_train, cohorts_val, cohorts_test. 
    """

    X_train_val, X_test, y_train_val, y_test, \
        cohorts_train_val, cohorts_test = \
        train_test_split(X, Y, cohorts, test_size=0.2,
                         random_state=train_val_random_seed, stratify=Y)

    X_train, X_val, y_train, y_val, \
        cohorts_train, cohorts_val = \
        train_test_split(X_train_val, y_train_val, cohorts_train_val, test_size=0.125,
                         random_state=train_val_random_seed, stratify=y_train_val)

    return X_train, X_val, X_test, \
        y_train, y_val, y_test, \
        cohorts_train, cohorts_val, cohorts_test


def generate_bootstrap_indices(X, y, split, num_bootstrap_samples=100):
    """ 
    Generates and saves to file sets of indices for val or test bootstrapping. 
    Args:
        X (Numpy array): X matrix, shape = num patients x num timesteps x num features.
        y (Numpy array): Y matrix, shape = num_patients.
        split (string): 'val' or 'test' indicating for which split to generate indices.
        num_bootstrap_samples (int): number indicating how many sets of bootstrap samples to generate.
    Returns:
        Numpy arrays: all_pos_samples, all_neg_samples. Contains num_bootstrap_samples indices 
        of positive and negative examples. 
    """

    positive_X = X[np.where(y == 1)]
    negative_X = X[np.where(y == 0)]
    all_pos_samples = []
    all_neg_samples = []
    for i in range(num_bootstrap_samples):
        pos_samples = np.random.choice(
            len(positive_X), replace=True, size=len(positive_X))
        neg_samples = np.random.choice(
            len(negative_X), replace=True, size=len(negative_X))
        all_pos_samples.append(pos_samples)
        all_neg_samples.append(neg_samples)

    np.save(split + '_pos_bootstrap_samples_' +
            str(num_bootstrap_samples), np.array(all_pos_samples))
    np.save(split + '_neg_bootstrap_samples_' +
            str(num_bootstrap_samples), np.array(all_neg_samples))
    return all_pos_samples, all_neg_samples


def get_bootstrapped_dataset(X, y, cohorts, index=0, test=False, num_bootstrap_samples=100):
    """ 
    Returns a bootstrapped (sampled w replacement) dataset. 
    Args:
        X (Numpy array): X matrix, shape = num patients x num timesteps x num features.
        y (Numpy array): Y matrix, shape = num_patients.
        cohorts (Numpy array): array of cohort membership, shape = num_patients.
        index (int): which bootstrap sample to look at. 
        test (bool): 
        num_bootstrap_samples (int):
    Returns:
        Numpy arrays: all_pos_samples, all_neg_samples. Contains num_bootstrap_samples indices 
        of positive and negative examples. 
    """

    if index == 0:
        return X, y, cohorts

    positive_X = X[np.where(y == 1)]
    negative_X = X[np.where(y == 0)]
    positive_cohorts = cohorts[np.where(y == 1)]
    negative_cohorts = cohorts[np.where(y == 0)]
    positive_y = y[np.where(y == 1)]
    negative_y = y[np.where(y == 0)]

    split = 'test' if test else 'val'
    try:
        pos_samples = np.load(
            split + '_pos_bootstrap_samples_' + str(num_bootstrap_samples) + '.npy')[index]
        neg_samples = np.load(
            split + '_neg_bootstrap_samples_' + str(num_bootstrap_samples) + '.npy')[index]
    except:
        all_pos_samples, all_neg_samples = generate_bootstrap_indices(
            X, y, split, num_bootstrap_samples)
        pos_samples = all_pos_samples[index]
        neg_samples = all_neg_samples[index]

    positive_X_bootstrapped = positive_X[pos_samples]
    negative_X_bootstrapped = negative_X[neg_samples]
    all_X_bootstrappped = np.concatenate(
        (positive_X_bootstrapped, negative_X_bootstrapped))
    all_y_bootstrapped = np.concatenate(
        (positive_y[pos_samples], negative_y[neg_samples]))
    all_cohorts_bootstrapped = np.concatenate(
        (positive_cohorts[pos_samples], negative_cohorts[neg_samples]))

    return all_X_bootstrappped, all_y_bootstrapped, all_cohorts_bootstrapped


def bootstrap_predict(X_orig, y_orig, cohorts_orig, task, model, return_everything=False, test=False, all_tasks=[], num_bootstrap_samples=100):
    """ 
    Evaluates model on each of the num_bootstrap_samples sets. 
    Args: 
        X_orig (Numpy array): The X matrix.
        y_orig (Numpy array): The y matrix. 
        cohorts_orig (Numpy array): List of cohort membership for each X example.
        task (String/Int): task to evalute on (either 'all' to evalute on the entire dataset, or a specific task). 
        model (Keras model): the model to evaluate.
        return_everything (bool): if True, return list of AUCs on all bootstrapped samples. If False, return [min auc, max auc, avg auc].
        test (bool): if True, use the test bootstrap indices.
        all_tasks (list): list of the tasks (used for evaluating multitask model).
        num_bootstrap_samples (int): number of bootstrapped samples to evalute on.
    Returns: 
        all_aucs OR min_auc, max_auc, avg_auc depending on the value of return_everything.
    """

    all_aucs = []

    for i in range(num_bootstrap_samples):
        X_bootstrap_sample, y_bootstrap_sample, cohorts_bootstrap_sample = get_bootstrapped_dataset(
            X_orig, y_orig, cohorts_orig, index=i, test=test, num_bootstrap_samples=num_bootstrap_samples)
        if task != 'all':
            X_bootstrap_sample_task = X_bootstrap_sample[cohorts_bootstrap_sample == task]
            y_bootstrap_sample_task = y_bootstrap_sample[cohorts_bootstrap_sample == task]
            cohorts_bootstrap_sample_task = cohorts_bootstrap_sample[cohorts_bootstrap_sample == task]
        else:
            X_bootstrap_sample_task = X_bootstrap_sample
            y_bootstrap_sample_task = y_bootstrap_sample
            cohorts_bootstrap_sample_task = cohorts_bootstrap_sample

        preds = model.predict(X_bootstrap_sample_task, batch_size=128)
        if len(preds) < len(y_bootstrap_sample_task):
            preds = get_correct_task_mtl_outputs(
                preds, cohorts_bootstrap_sample_task, all_tasks)

        try:
            auc = roc_auc_score(y_bootstrap_sample_task, preds)
            all_aucs.append(auc)
        except Exception as e:
            print(e)
            print('Skipped this sample.')

    avg_auc = np.mean(all_aucs)
    min_auc = min(all_aucs)
    max_auc = max(all_aucs)

    if return_everything:
        return all_aucs
    else:
        return min_auc, max_auc, avg_auc

################ MODEL DEFINITIONS ###############################################
####################################################################################


def create_single_task_model(n_layers, units, num_dense_shared_layers, dense_shared_layer_size, input_dim, output_dim):
    """ 
    Create a single task model with LSTM layer(s), shared dense layer(s), and sigmoided output. 
    Args:
        n_layers (int): Number of initial LSTM layers.
        units (int): Number of units in each LSTM layer.
        num_dense_shared_layers (int): Number of dense layers following LSTM layer(s).
        dense_shared_layer_size (int): Number of units in each dense layer.
        input_dim (int): Number of features in the input.
        output_dim (int): Number of outputs (1 for binary tasks).
    Returns: 
        final_model (Keras model): A compiled model with the provided architecture. 
    """

    # global model
    model = Sequential()

    # first layer
    if n_layers > 1:
        return_seq = True
    else:
        return_seq = False

    model.add(LSTM(units=units, activation='relu',
                   input_shape=input_dim, return_sequences=return_seq))

    # additional hidden layers
    for l in range(n_layers - 1):
        model.add(LSTM(units=units, activation='relu'))

    # additional dense layers
    for l in range(num_dense_shared_layers):
        model.add(Dense(units=dense_shared_layer_size, activation='relu'))

    # output layer
    model.add(Dense(units=output_dim, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=.0001),
                  metrics=['accuracy'])

    return model


def create_multitask_model(input_dim, n_layers, units, num_dense_shared_layers, dense_shared_layer_size, n_multi_layers, multi_units, output_dim, tasks):
    """ 
    Create a multitask model with LSTM layer(s), shared dense layer(s), separate dense layer(s) 
    and separate sigmoided outputs. 
    Args: 
        input_dim (int): Number of features in the input.
        n_layers (int): Number of initial LSTM layers.
        units (int): Number of units in each LSTM layer.
        num_dense_shared_layers (int): Number of dense layers following LSTM layer(s).
        dense_shared_layer_size (int): Number of units in each dense layer.
        n_multi_layers (int): Number of task-specific dense layers. 
        multi_layer_size (int): Number of units in each task-specific dense layer.
        output_dim (int): Number of outputs (1 for binary tasks).
        tasks (list): list of the tasks.
    Returns: 
        final_model (Keras model): A compiled model with the provided architecture. 
    """

    tasks = [str(t) for t in tasks]
    n_tasks = len(tasks)

    # Input layer
    x_inputs = Input(shape=input_dim)

    # first layer
    if n_layers > 1:
        return_seq = True
    else:
        return_seq = False

    # Shared layers
    combined_model = LSTM(units, activation='relu',
                          input_shape=input_dim,
                          name='combined', return_sequences=return_seq)(x_inputs)

    for l in range(n_layers - 1):
        combined_model = LSTM(units, activation='relu')(combined_model)

    for l in range(num_dense_shared_layers):
        combined_model = Dense(dense_shared_layer_size,
                               activation='relu')(combined_model)

    # Individual task layers
    if n_multi_layers == 0:
        # Only create task-specific output layer.
        output_layers = []
        for task_num in range(n_tasks):
            output_layers.append(Dense(output_dim, activation='sigmoid',
                                       name=tasks[task_num])(combined_model))

    else:
        # Also create task-specific dense layer.
        task_layers = []
        for task_num in range(n_tasks):
            task_layers.append(Dense(multi_units, activation='relu',
                                     name=tasks[task_num])(combined_model))

        output_layers = []
        for task_layer_num in range(len(task_layers)):
            output_layers.append(Dense(output_dim, activation='sigmoid',
                                       name=str(tasks[task_layer_num]) + '_output')(task_layers[task_layer_num]))

    loss_fn = 'binary_crossentropy'
    learning_rate = 0.0001
    final_model = Model(inputs=x_inputs, outputs=output_layers)
    final_model.compile(loss=loss_fn,
                        optimizer=Adam(lr=learning_rate),
                        metrics=['accuracy'])

    return final_model


def get_mtl_sample_weights(y, cohorts, all_tasks, sample_weights=None):
    """ 
    Generates a dictionary of sample weights for the multitask model that masks out 
    (and prevents training on) outputs corresponding to cohorts to which a given sample doesn't belong. 
    Args: 
        y (Numpy array): The y matrix.
        cohorts (Numpy array): cohort membership corresponding to each example, in the same order as y.
        all_tasks (list/Numpy array): list of all unique tasks.
        sample_weights (list/Numpy array): if samples should be weighted differently during training, 
                                           provide a list w len == num_samples where each value is how much 
                                           that value should be weighted.
    Returns: 
        sw_dict (dictionary): Dictionary mapping task to list w len == num_samples, where each value is 0 if 
                              the corresponding example does not belong to that task, and either 1 or a sample weight
                              value (if sample_weights != None) if it does.
    """

    sw_dict = {}
    for task in all_tasks:
        task_indicator_col = (cohorts == task).astype(int)
        if sample_weights:
            task_indicator_col = np.array(
                task_indicator_col) * np.array(sample_weights)
        sw_dict[task] = task_indicator_col
    return sw_dict


def get_correct_task_mtl_outputs(mtl_output, cohorts, all_tasks):
    """ 
    Gets the output corresponding to the right task given the multitask output.  Necessary since 
    the MTL model will produce an output for each cohort's output, but we only care about the one the example
    actually belongs to. 
    Args: 
        mtl_output (Numpy array/list): the output of the multitask model. Should be of size n_tasks x n_samples.
        cohorts (Numpy array): list of cohort membership for each sample.
        all_tasks (list): unique list of tasks (should be in the same order that corresponds with that of the MTL model output.)
    Returns:
        mtl_output (Numpy array): an array of size n_samples x 1 where each value corresponds to the MTL model's
                                  prediction for the task that that sample belongs to.
    """

    n_tasks = len(all_tasks)
    cohort_key = dict(zip(all_tasks, range(n_tasks)))
    mtl_output = np.array(mtl_output)
    mtl_output = mtl_output[[cohort_key[c]
                             for c in cohorts], np.arange(len(cohorts))]
    return mtl_output

################ RUNNING MODELS ###############################################
####################################################################################


def run_separate_models(X_train, y_train, cohorts_train,
                        X_val, y_val, cohorts_val,
                        X_test, y_test, cohorts_test,
                        all_tasks, fname_keys, fname_results,
                        FLAGS):
    """
    Train and evaluate separate models for each task. 
    Results are saved in FLAGS.experiment_name/results:
        - The numpy file ending in '_keys' contains the parameters for the model, 
          and the numpy file ending in '_results' contains the validation AUCs for that 
          configuration. 
        - If you run multiple configurations for the same experiment name, 
          those parameters and results will append to the same files.
        - At test time, results are saved into the file beginning 'test_auc_on_separate_'. 
          The format of results will depend on whether you use bootstrapping or not. With bootstrapping, 
          minimum, maximum and average AUCs are saved. Without, just the single AUC on the actual 
          val / test dataset is saved. 
    Args:
        X_train (Numpy array): The X matrix w training examples.
        y_train (Numpy array): The y matrix w training examples. 
        cohorts_train (Numpy array): List of cohort membership for each validation example. 
        X_val (Numpy array): The X matrix w validation examples.
        y_val (Numpy array): The y matrix w validation examples. 
        cohorts_val (Numpy array): List of cohort membership for each validation example.
        X_test (Numpy array): The X matrix w testing examples.
        y_test (Numpy array): The y matrix w testing examples. 
        cohorts_test (Numpy array): List of cohort membership for each testing example.
        all_tasks (Numpy array/list): List of tasks.
        fname_keys (String): filename where the model parameters will be saved.
        fname_results (String): filename where the model AUCs will be saved.
        FLAGS (dictionary): all the arguments.
    """

    cohort_aucs = []

    # if we're testing, just load the model and save results
    if FLAGS.test_time:
        for task in all_tasks:
            model_fname_parts = ['separate', str(task), 'lstm_shared', str(FLAGS.num_lstm_layers), 'layers', str(FLAGS.lstm_layer_size), 'units',
                                 str(FLAGS.num_dense_shared_layers), 'dense_shared', str(FLAGS.dense_shared_layer_size), 'dense_units', 'mortality']
            model_path = FLAGS.experiment_name + \
                '/models/' + "_".join(model_fname_parts)
            model = load_model(model_path)

            if FLAGS.test_bootstrap:
                all_aucs = bootstrap_predict(X_test, y_test, cohorts_test, task, model, return_everything=True,
                                             test=True, num_bootstrap_samples=FLAGS.num_test_bootstrap_samples)
                cohort_aucs.append(np.array(all_aucs))

            else:
                x_test_in_task = X_test[cohorts_test == task]
                y_test_in_task = y_test[cohorts_test == task]

                y_pred = model.predict(x_test_in_task)
                auc = roc_auc_score(y_test_in_task, y_pred)
                cohort_aucs.append(auc)

        suffix = 'single' if not FLAGS.test_bootstrap else 'all'
        test_auc_fname = 'test_auc_on_separate_' + suffix
        np.save(FLAGS.experiment_name + '/results/' +
                test_auc_fname, cohort_aucs)
        return

    # otherwise, create and train a model
    for task in all_tasks:

        # get training data from cohort
        x_train_in_task = X_train[cohorts_train == task]
        y_train_in_task = y_train[cohorts_train == task]

        x_val_in_task = X_val[cohorts_val == task]
        y_val_in_task = y_val[cohorts_val == task]

        # create & fit model
        model = create_single_task_model(FLAGS.num_lstm_layers, FLAGS.lstm_layer_size,
                                         FLAGS.num_dense_shared_layers, FLAGS.dense_shared_layer_size, X_train.shape[1:], 1)
        model_fname_parts = ['separate', str(task), 'lstm_shared', str(FLAGS.num_lstm_layers), 'layers', str(FLAGS.lstm_layer_size), 'units',
                             str(FLAGS.num_dense_shared_layers), 'dense_shared', str(FLAGS.dense_shared_layer_size), 'dense_units', 'mortality']
        model_dir = FLAGS.experiment_name + \
            '/checkpoints/' + "_".join(model_fname_parts)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_fname = model_dir + '/{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpointer = ModelCheckpoint(
            model_fname, monitor='val_loss', verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        model.fit(x_train_in_task, y_train_in_task, epochs=FLAGS.epochs, batch_size=100,
                  callbacks=[checkpointer, early_stopping],
                  validation_data=(x_val_in_task, y_val_in_task))

        # make validation predictions & evaluate
        preds_for_cohort = model.predict(x_val_in_task, batch_size=128)

        print('AUC of separate model for ', task, ':')
        if FLAGS.no_val_bootstrap:
            try:
                auc = roc_auc_score(y_val_in_task, preds_for_cohort)
            except:
                auc = np.nan

            cohort_aucs.append(auc)
        else:
            min_auc, max_auc, avg_auc = bootstrap_predict(
                X_val, y_val, cohorts_val, task, model, return_everything=False, num_bootstrap_samples=FLAGS.num_val_bootstrap_samples)
            cohort_aucs.append(np.array([min_auc, max_auc, avg_auc]))
            auc = avg_auc
            print("(min/max/average):")

        print(cohort_aucs[-1])

        model.save(FLAGS.experiment_name + '/models/' +
                   "_".join(model_fname_parts))

    # save results to a file
    current_run_params = [FLAGS.num_lstm_layers, FLAGS.lstm_layer_size,
                          FLAGS.num_dense_shared_layers, FLAGS.dense_shared_layer_size]
    try:
        separate_model_results = np.load(fname_results)
        separate_model_key = np.load(fname_keys)
        separate_model_results = np.concatenate(
            (separate_model_results, np.expand_dims(cohort_aucs, 0)))
        separate_model_key = np.concatenate(
            (separate_model_key, np.array([current_run_params])))

    except:
        separate_model_results = np.expand_dims(cohort_aucs, 0)
        separate_model_key = np.array([current_run_params])

    np.save(fname_results, separate_model_results)
    np.save(fname_keys, separate_model_key)
    print('Saved separate results.')


def run_global_model(X_train, y_train, cohorts_train,
                     X_val, y_val, cohorts_val,
                     X_test, y_test, cohorts_test,
                     all_tasks, fname_keys, fname_results,
                     FLAGS):
    """
    Train and evaluate global model. 
    Results are saved in FLAGS.experiment_name/results:
        - The numpy file ending in '_keys' contains the parameters for the model, 
          and the numpy file ending in '_results' contains the validation AUCs for that 
          configuration. 
        - If you run multiple configurations for the same experiment name, 
          those parameters and results will append to the same files.
        - At test time, results are saved into the file beginning 'test_auc_on_global_'. 
          The format of results will depend on whether you use bootstrapping or not. With bootstrapping, 
          minimum, maximum and average AUCs are saved. Without, just the single AUC on the actual 
          val / test dataset is saved. 
    Args:
        X_train (Numpy array): The X matrix w training examples.
        y_train (Numpy array): The y matrix w training examples. 
        cohorts_train (Numpy array): List of cohort membership for each validation example. 
        X_val (Numpy array): The X matrix w validation examples.
        y_val (Numpy array): The y matrix w validation examples. 
        cohorts_val (Numpy array): List of cohort membership for each validation example.
        X_test (Numpy array): The X matrix w testing examples.
        y_test (Numpy array): The y matrix w testing examples. 
        cohorts_test (Numpy array): List of cohort membership for each testing example.
        all_tasks (Numpy array/list): List of tasks.
        fname_keys (String): filename where the model parameters will be saved.
        fname_results (String): filename where the model AUCs will be saved.
        FLAGS (dictionary): all the arguments.
    """

    model_fname_parts = ['global', 'lstm_shared', str(FLAGS.num_lstm_layers), 'layers', str(FLAGS.lstm_layer_size), 'units',
                         str(FLAGS.num_dense_shared_layers), 'dense_shared', str(FLAGS.dense_shared_layer_size), 'dense_units', 'mortality']

    if FLAGS.test_time:
        model_path = FLAGS.experiment_name + \
            '/models/' + "_".join(model_fname_parts)
        model = load_model(model_path)
        cohort_aucs = []
        y_pred = model.predict(X_test)

        # all bootstrapped AUCs
        for task in all_tasks:
            if FLAGS.test_bootstrap:
                all_aucs = bootstrap_predict(X_test, y_test, cohorts_test, task, model, return_everything=True,
                                             test=True, num_bootstrap_samples=FLAGS.num_test_bootstrap_samples)
                cohort_aucs.append(np.array(all_aucs))
            else:
                y_pred_in_cohort = y_pred[cohorts_test == task]
                y_true_in_cohort = y_test[cohorts_test == task]
                auc = roc_auc_score(y_true_in_cohort, y_pred_in_cohort)
                cohort_aucs.append(auc)

        if FLAGS.test_bootstrap:
            # Macro AUC
            cohort_aucs = np.array(cohort_aucs)
            cohort_aucs = np.concatenate(
                (cohort_aucs, np.expand_dims(np.mean(cohort_aucs, axis=0), 0)))

            # Micro AUC
            all_micro_aucs = bootstrap_predict(X_test, y_test, cohorts_test, 'all', model,
                                               return_everything=True, test=True, num_bootstrap_samples=FLAGS.num_test_bootstrap_samples)
            cohort_aucs = np.concatenate(
                (cohort_aucs, np.array([all_micro_aucs])))

        else:
            # Macro AUC
            macro_auc = np.mean(cohort_aucs)
            cohort_aucs.append(macro_auc)

            # Micro AUC
            micro_auc = roc_auc_score(y_test, y_pred)
            cohort_aucs.append(micro_auc)

        suffix = 'single' if not FLAGS.test_bootstrap else 'all'
        test_auc_fname = 'test_auc_on_global_' + suffix
        np.save(FLAGS.experiment_name + '/results/' +
                test_auc_fname, cohort_aucs)
        return

    model = create_single_task_model(FLAGS.num_lstm_layers, FLAGS.lstm_layer_size,
                                     FLAGS.num_dense_shared_layers, FLAGS.dense_shared_layer_size, X_train.shape[1:], 1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    model_dir = FLAGS.experiment_name + \
        '/checkpoints/' + "_".join(model_fname_parts)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_fname = model_dir + '/{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpointer = ModelCheckpoint(model_fname, monitor='val_loss', verbose=1)

    model.fit(X_train, y_train,
              epochs=FLAGS.epochs, batch_size=100,
              sample_weight=samp_weights,
              callbacks=[checkpointer, early_stopping],
              validation_data=(X_val, y_val))

    model.save(FLAGS.experiment_name + '/models/' +
               "_".join(model_fname_parts))

    cohort_aucs = []
    y_pred = model.predict(X_val)
    for task in all_tasks:
        print('Global Model AUC on ', task, ':')
        if FLAGS.no_val_bootstrap:
            try:
                auc = roc_auc_score(
                    y_val[cohorts_val == task], y_pred[cohorts_val == task])
            except:
                auc = np.nan
            cohort_aucs.append(auc)
        else:
            min_auc, max_auc, avg_auc = bootstrap_predict(
                X_val, y_val, cohorts_val, task, model, num_bootstrap_samples=FLAGS.num_val_bootstrap_samples)
            cohort_aucs.append(np.array([min_auc, max_auc, avg_auc]))
            print ("(min/max/average): ")

        print(cohort_aucs[-1])

    cohort_aucs = np.array(cohort_aucs)

    # Add Macro AUC
    cohort_aucs = np.concatenate(
        (cohort_aucs, np.expand_dims(np.nanmean(cohort_aucs, axis=0), 0)))

    # Add Micro AUC
    if FLAGS.no_val_bootstrap:
        micro_auc = roc_auc_score(y_val, y_pred)
        cohort_aucs = np.concatenate((cohort_aucs, np.array([micro_auc])))
    else:
        min_auc, max_auc, avg_auc = bootstrap_predict(
            X_val, y_val, cohorts_val, 'all', model, num_bootstrap_samples=FLAGS.num_val_bootstrap_samples)
        cohort_aucs = np.concatenate(
            (cohort_aucs, np.array([[min_auc, max_auc, avg_auc]])))

    # Save Results
    current_run_params = [FLAGS.num_lstm_layers, FLAGS.lstm_layer_size,
                          FLAGS.num_dense_shared_layers, FLAGS.dense_shared_layer_size]
    try:
        print('appending results.')
        global_model_results = np.load(fname_results)
        global_model_key = np.load(fname_keys)
        global_model_results = np.concatenate(
            (global_model_results, np.expand_dims(cohort_aucs, 0)))
        global_model_key = np.concatenate(
            (global_model_key, np.array([current_run_params])))

    except Exception as e:
        global_model_results = np.expand_dims(cohort_aucs, 0)
        global_model_key = np.array([current_run_params])

    np.save(fname_results, global_model_results)
    np.save(fname_keys, global_model_key)
    print('Saved global results.')


def run_multitask_model(X_train, y_train, cohorts_train,
                        X_val, y_val, cohorts_val,
                        X_test, y_test, cohorts_test,
                        all_tasks, fname_keys, fname_results,
                        FLAGS):
    """
    Train and evaluate multitask model. 
    Results are saved in FLAGS.experiment_name/results:
        - The numpy file ending in '_keys' contains the parameters for the model, 
          and the numpy file ending in '_results' contains the validation AUCs for that 
          configuration. 
        - If you run multiple configurations for the same experiment name, 
          those parameters and results will append to the same files.
        - At test time, results are saved into the file beginning 'test_auc_on_separate_'. 
          The format of results will depend on whether you use bootstrapping or not. With bootstrapping, 
          minimum, maximum and average AUCs are saved. Without, just the single AUC on the actual 
          val / test dataset is saved. 
    Args:
        X_train (Numpy array): The X matrix w training examples.
        y_train (Numpy array): The y matrix w training examples. 
        cohorts_train (Numpy array): List of cohort membership for each validation example. 
        X_val (Numpy array): The X matrix w validation examples.
        y_val (Numpy array): The y matrix w validation examples. 
        cohorts_val (Numpy array): List of cohort membership for each validation example.
        X_test (Numpy array): The X matrix w testing examples.
        y_test (Numpy array): The y matrix w testing examples. 
        cohorts_test (Numpy array): List of cohort membership for each testing example.
        all_tasks (Numpy array/list): List of tasks.
        fname_keys (String): filename where the model parameters will be saved.
        fname_results (String): filename where the model AUCs will be saved.
        FLAGS (dictionary): all the arguments.
    """

    model_fname_parts = ['mtl', 'lstm_shared', str(FLAGS.num_lstm_layers), 'layers', str(FLAGS.lstm_layer_size), 'units',
                         'dense_shared', str(FLAGS.num_dense_shared_layers), 'layers', str(
                             FLAGS.dense_shared_layer_size), 'dense_units',
                         'specific', str(FLAGS.num_multi_layers), 'layers', str(FLAGS.multi_layer_size), 'specific_units', 'mortality']

    n_tasks = len(np.unique(cohorts_train))
    cohort_key = dict(zip(all_tasks, range(n_tasks)))

    if FLAGS.test_time:
        model_path = FLAGS.experiment_name + \
            '/models/' + "_".join(model_fname_parts)
        model = load_model(model_path)
        y_pred = model.predict(X_test)
        
        cohort_aucs = []
        for task in all_tasks:
            if FLAGS.test_bootstrap:
                all_aucs = bootstrap_predict(X_test, y_test, cohorts_test,
                                             task=task, model=model, return_everything=True, test=True,
                                             all_tasks=all_tasks,
                                             num_bootstrap_samples=FLAGS.num_test_bootstrap_samples)
                cohort_aucs.append(np.array(all_aucs))
            else:
                y_pred_in_cohort = y_pred[cohorts_test ==
                                          task, cohort_key[task]]
                y_true_in_cohort = y_test[cohorts_test == task]
                auc = roc_auc_score(y_true_in_cohort, y_pred_in_cohort)
                cohort_aucs.append(auc)

        if FLAGS.test_bootstrap:
            cohort_aucs = np.array(cohort_aucs)
            cohort_aucs = np.concatenate(
                (cohort_aucs, np.expand_dims(np.mean(cohort_aucs, axis=0), 0)))

            all_micro_aucs = bootstrap_predict(X_test, y_test, cohorts_test, 'all', model, return_everything=True, test=True,
                                               all_tasks=all_tasks, num_bootstrap_samples=FLAGS.num_test_bootstrap_samples)
            cohort_aucs = np.concatenate(
                (cohort_aucs, np.array([all_micro_aucs])))

        else:
            macro_auc = np.mean(cohort_aucs)
            cohort_aucs.append(macro_auc)
            micro_auc = roc_auc_score(y_test, y_pred[np.arange(len(y_test)), [
                                      cohort_key[c] for c in cohorts_test]])
            cohort_aucs.append(micro_auc)

        suffix = 'single' if not FLAGS.test_bootstrap else 'all'
        test_auc_fname = 'test_auc_on_multitask_' + suffix
        np.save(FLAGS.experiment_name + '/results/' +
                test_auc_fname, cohort_aucs)
        return

    # model
    mtl_model = create_multitask_model(X_train.shape[1:], FLAGS.num_lstm_layers,
                                       FLAGS.lstm_layer_size, FLAGS.num_dense_shared_layers, FLAGS.dense_shared_layer_size,
                                       FLAGS.num_multi_layers, FLAGS.multi_layer_size, output_dim=1, tasks=all_tasks)

    early_stopping = EarlyStopping(monitor='val_loss', patience=4)

    model_dir = FLAGS.experiment_name + \
        '/checkpoints/' + "_".join(model_fname_parts)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_fname = model_dir + '/{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpointer = ModelCheckpoint(model_fname, monitor='val_loss', verbose=1)
    mtl_model.fit(X_train, [y_train for i in range(n_tasks)],
                  batch_size=100,
                  epochs=FLAGS.epochs,
                  verbose=1,
                  sample_weight=get_mtl_sample_weights(
                      y_train, cohorts_train, all_tasks, sample_weights=samp_weights),
                  callbacks=[early_stopping, checkpointer],
                  validation_data=(X_val, [y_val for i in range(n_tasks)]))

    mtl_model.save(FLAGS.experiment_name + '/models/' +
                   "_".join(model_fname_parts))

    cohort_aucs = []

    y_pred = get_correct_task_mtl_outputs(
        mtl_model.predict(X_val), cohorts_val, all_tasks)

    # task aucs
    for task in all_tasks:
        print('Multitask AUC on', task, ': ')
        if FLAGS.no_val_bootstrap:
            y_pred_in_task = y_pred[cohorts_val == task]
            try:
                auc = roc_auc_score(y_val[cohorts_val == task], y_pred_in_task)
            except:
                auc = np.nan
            cohort_aucs.append(auc)
        else:
            min_auc, max_auc, avg_auc = bootstrap_predict(
                X_val, y_val, cohorts_val, task, mtl_model, all_tasks=all_tasks, num_bootstrap_samples=FLAGS.num_val_bootstrap_samples)
            cohort_aucs.append(np.array([min_auc, max_auc, avg_auc]))
            print("(min/max/average):")

        print(cohort_aucs[-1])

    # macro average
    cohort_aucs = np.array(cohort_aucs)
    cohort_aucs = np.concatenate(
        (cohort_aucs, np.expand_dims(np.nanmean(cohort_aucs, axis=0), 0)))

    # micro average
    if FLAGS.no_val_bootstrap:
        cohort_aucs = np.concatenate(
            (cohort_aucs, np.array([roc_auc_score(y_val, y_pred)])))
    else:
        min_auc, max_auc, avg_auc = bootstrap_predict(
            X_val, y_val, cohorts_val, 'all', mtl_model, all_tasks=all_tasks, num_bootstrap_samples=FLAGS.num_val_bootstrap_samples)
        cohort_aucs = np.concatenate(
            (cohort_aucs, np.array([[min_auc, max_auc, avg_auc]])))

    current_run_params = [FLAGS.num_lstm_layers, FLAGS.lstm_layer_size, FLAGS.num_dense_shared_layers,
                          FLAGS.dense_shared_layer_size, FLAGS.num_multi_layers, FLAGS.multi_layer_size]

    try:
        multitask_model_results = np.load(fname_results)
        multitask_model_key = np.load(fname_keys)
        multitask_model_results = np.concatenate(
            (multitask_model_results, np.expand_dims(cohort_aucs, 0)))
        multitask_model_key = np.concatenate(
            (multitask_model_key, np.array([current_run_params])))

    except:
        multitask_model_results = np.expand_dims(cohort_aucs, 0)
        multitask_model_key = np.array([current_run_params])

    np.save(fname_results, multitask_model_results)
    np.save(fname_keys, multitask_model_key)
    print('Saved multitask results.')

################ LOAD & PROCESS DATA ###############################################
####################################################################################


def load_processed_data(data_hours=24, gap_time=12):
    """
    Either read pre-processed data from a saved folder, or load in the raw data and preprocess it.
    Should have the files 'saps.csv' (with columns 'subject_id', 'hadm_id', 'icustay_id', 'sapsii')
    and 'code_status.csv' (with columns 'subject_id', 'hadm_id', 'icustay_id', 'timecmo_chart', 'timecmo_nursingnote')
    in the local directory.
    
    Args: 
        data_hours (int): hours of data to use for predictions.
        gap_time (int): gap between last data hour and time of prediction.
    Returns: 
        X (Numpy array): matrix of data of size n_samples x n_timesteps x n_features.  
        Y (Numpy array): binary array of len n_samples corresponding to in hospital mortality after the gap time.
        careunits (Numpy array): array w careunit membership of each sample.
        saps_quartile (Numpy array): array w SAPS quartile of each sample.
        subject_ids (Numpy array): subject_ids corresponding to each row of the X/Y/careunits/saps_quartile arrays.
    """
    save_data_path = 'data/mortality_' + str(data_hours) + '/'

    # see if we already have the data matrices saved
    try:
        X = np.load(save_data_path + 'X.npy')
        careunits = np.load(save_data_path + 'careunits.npy')
        saps_quartile = np.load(save_data_path + 'saps_quartile.npy')
        subject_ids = np.load(save_data_path + 'subject_ids.npy')
        Y = np.load(save_data_path + 'Y.npy')
        print('Loaded data from ' + save_data_path)
        print('shape of X: ', X.shape)

    # otherwise create them
    except Exception as e:
        data_cutoff = data_hours
        mort_cutoff = data_hours + gap_time

        X, static = load_phys_data()

        # Add SAPS Score to static matrix
        saps = pd.read_csv('data/saps.csv')
        ser, bins = pd.qcut(saps.sapsii, 4, retbins=True, labels=False)
        saps['sapsii_quartile'] = pd.cut(
            saps.sapsii, bins=bins, labels=False, include_lowest=True)
        saps = saps[['subject_id', 'hadm_id', 'icustay_id', 'sapsii_quartile']]
        static = pd.merge(static, saps, how='left', on=[
                          'subject_id', 'hadm_id', 'icustay_id'])

        # Add Mortality Outcome
        deathtimes = static[['subject_id', 'hadm_id',
                             'icustay_id', 'deathtime', 'dischtime']].dropna()
        deathtimes_valid = deathtimes[deathtimes.dischtime >=
                                      deathtimes.deathtime]
        deathtimes_valid['mort_hosp_valid'] = True
        cmo = pd.read_csv('data/code_status.csv')
        cmo = cmo[cmo.cmo > 0]
        cmo['timecmo_chart'] = pd.to_datetime(cmo.timecmo_chart)
        cmo['timecmo_nursingnote'] = pd.to_datetime(cmo.timecmo_nursingnote)
        cmo['cmo_min_time'] = cmo.loc[:, [
            'timecmo_chart', 'timecmo_nursingnote']].min(axis=1)
        all_mort_times = pd.merge(deathtimes_valid, cmo, on=['subject_id', 'hadm_id', 'icustay_id'], how='outer')[
            ['subject_id', 'hadm_id', 'icustay_id', 'deathtime', 'dischtime', 'cmo_min_time']]
        all_mort_times['deathtime'] = pd.to_datetime(all_mort_times.deathtime)
        all_mort_times['cmo_min_time'] = pd.to_datetime(
            all_mort_times.cmo_min_time)
        all_mort_times['min_mort_time'] = all_mort_times.loc[:,
                                                             ['deathtime', 'cmo_min_time']].min(axis=1)
        min_mort_time = all_mort_times[[
            'subject_id', 'hadm_id', 'icustay_id', 'min_mort_time']]
        static = pd.merge(static, min_mort_time, on=[
                          'subject_id', 'hadm_id', 'icustay_id'], how='left')
        static['mort_hosp_valid'] = np.invert(np.isnat(static.min_mort_time))

        # For those who died, filter for at least 36 hours of data
        static['time_til_mort'] = pd.to_datetime(
            static.min_mort_time) - pd.to_datetime(static.intime)
        static['time_til_mort'] = static.time_til_mort.apply(
            lambda x: x.total_seconds()/3600)

        static['time_in_icu'] = pd.to_datetime(
            static.dischtime) - pd.to_datetime(static.intime)
        static['time_in_icu'] = static.time_in_icu.apply(
            lambda x: x.total_seconds()/3600)

        static = static[((static.time_in_icu >= data_cutoff) & (
            static.mort_hosp_valid == False)) | (static.time_til_mort >= mort_cutoff)]

        # Make discrete values and cut off stay at 24 hours
        X_discrete = make_discrete_values(X)
        X_discrete = X_discrete[X_discrete.hours_in < data_cutoff]
        X_discrete = X_discrete[[
            c for c in X_discrete.columns if c not in ['hadm_id', 'icustay_id']]]

        # Pad people whose records stop early
        test = X_discrete.set_index(['subject_id', 'hours_in'])
        extra_hours = test.groupby(level=0).apply(_pad_df, data_cutoff)
        extra_hours = extra_hours[extra_hours != 0].reset_index()
        extra_hours.columns = ['subject_id', 'pad_hrs']
        pad_tuples = []
        for s in extra_hours.subject_id:
            for hr in list(extra_hours[extra_hours.subject_id == s].pad_hrs)[0]:
                pad_tuples.append((s, hr))
        pad_df = pd.DataFrame(0, index=pd.MultiIndex.from_tuples(
            pad_tuples, names=('subject_id', 'hours_in')), columns=test.columns)
        new_df = pd.concat([test, pad_df], axis=0)

        # get the static vars we want, make them discrete columns
        static_to_keep = static[['subject_id', 'gender', 'age', 'ethnicity',
                                 'sapsii_quartile', 'first_careunit', 'mort_hosp_valid']]
        static_to_keep.loc[:, 'ethnicity'] = static_to_keep['ethnicity'].apply(
            categorize_ethnicity)
        static_to_keep.loc[:, 'age'] = static_to_keep['age'].apply(
            categorize_age)
        static_to_keep = pd.get_dummies(static_to_keep, columns=[
                                        'gender', 'age', 'ethnicity'])

        # merge the phys with static
        X_full = pd.merge(new_df.reset_index(), static_to_keep,
                          on='subject_id', how='inner')
        X_full = X_full.set_index(['subject_id', 'hours_in'])

        # print mortality per careunit
        mort_by_careunit = X_full.groupby(
            'subject_id')['first_careunit', 'mort_hosp_valid'].max()
        for cu in mort_by_careunit.first_careunit.unique():
            print(cu + ": " + str(np.sum(mort_by_careunit[mort_by_careunit.first_careunit == cu].mort_hosp_valid)) + ' out of ' + str(
                len(mort_by_careunit[mort_by_careunit.first_careunit == cu])))

        # create Y and cohort matrices
        subject_ids = X_full.index.get_level_values(0).unique()
        Y = X_full[['mort_hosp_valid']].groupby(level=0).max()
        careunits = X_full[['first_careunit']].groupby(level=0).max()
        saps_quartile = X_full[['sapsii_quartile']].groupby(level=0).max()
        Y = Y.reindex(subject_ids)
        careunits = careunits.reindex(subject_ids)
        saps_quartile = saps_quartile.reindex(subject_ids)

        # delete those columns from the X matrix
        X_full = X_full.loc[:, X_full.columns != 'mort_hosp_valid']
        X_full = X_full.loc[:, X_full.columns != 'sapsii_quartile']
        X_full = X_full.loc[:, X_full.columns != 'first_careunit']
       
        feature_names = X_full.columns
        np.save('feature_names.npy', feature_names)

        # get the data as a np matrix of size num_examples x timesteps x features
        X_full_matrix = np.reshape(
            X_full.as_matrix(), (len(subject_ids), data_cutoff, -1))
        print("Shape of X: ")
        print(X_full_matrix.shape)

        # print feature values
        print("Features : ")
        print(np.array(X_full.columns))

        print(subject_ids)
        print(Y.index)
        print(careunits.index)

        print("Number of positive examples : ", len(Y[Y == 1]))

        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path)

        np.save(save_data_path + 'X.npy', X_full_matrix)
        np.save(save_data_path + 'careunits.npy',
                np.squeeze(careunits.as_matrix(), axis=1))
        np.save(save_data_path + 'saps_quartile.npy',
                np.squeeze(saps_quartile.as_matrix(), axis=1))
        np.save(save_data_path + 'subject_ids.npy', np.array(subject_ids))
        np.save(save_data_path + 'Y.npy', np.squeeze(Y.as_matrix(), axis=1))

        X = X_full_matrix

    return X, Y, careunits, saps_quartile, subject_ids


################ RUN THINGS ####################################################
####################################################################################
if __name__ == "__main__":

    FLAGS = get_args()

    # Limit GPU usage.
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_num
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Don't use all GPUs
    config.allow_soft_placement = True  # Enable manual control
    K.tensorflow_backend.set_session(tf.Session(config=config))

    # Make folders for the results & models
    for folder in ['results', 'models', 'checkpoints']:
        if not os.path.exists(os.path.join(FLAGS.experiment_name, folder)):
            os.makedirs(os.path.join(FLAGS.experiment_name, folder))

    # The file that we'll save model configurations to
    sw = 'with_sample_weights' if FLAGS.sample_weights else 'no_sample_weights'
    sw = '' if FLAGS.model_type == 'SEPARATE' else sw
    fname_keys = FLAGS.experiment_name + '/results/' + \
        '_'.join([FLAGS.model_type.lower(), 'model_keys', sw]) + '.npy'
    fname_results = FLAGS.experiment_name + '/results/' + \
        '_'.join([FLAGS.model_type.lower(), 'model_results', sw]) + '.npy'

    # Check that we haven't already run this configuration
    if os.path.exists(fname_keys) and not FLAGS.repeats_allowed:
        model_key = np.load(fname_keys)
        current_run = [FLAGS.num_lstm_layers, FLAGS.lstm_layer_size,
                       FLAGS.num_dense_shared_layers, FLAGS.dense_shared_layer_size]
        if FLAGS.model_type == "MULTITASK":
            current_run = current_run + \
                [FLAGS.num_multi_layers, FLAGS.multi_layer_size]
        print('Now running :', current_run)
        print('Have already run: ', model_key.tolist())
        if current_run in model_key.tolist():
            print('Have already run this configuration. Now skipping this one.')
            sys.exit(0)

    # Load Data
    X, Y, careunits, saps_quartile, subject_ids = load_processed_data(
        FLAGS.data_hours, FLAGS.gap_time)
    Y = Y.astype(int)

    # Split
    if FLAGS.cohorts == 'careunit':
        cohort_col = careunits
    elif FLAGS.cohorts == 'saps':
        cohort_col = saps_quartile
    elif FLAGS.cohorts == 'custom':
        cohort_col = np.load('cluster_membership/' + FLAGS.cohort_filepath)
        cohort_col = np.array([str(c) for c in cohort_col])

    # Include cohort membership as an additional feature
    if FLAGS.include_cohort_as_feature:
        cohort_col_onehot = pd.get_dummies(cohort_col).as_matrix()
        cohort_col_onehot = np.expand_dims(cohort_col_onehot, axis=1)
        cohort_col_onehot = np.tile(cohort_col_onehot, (1, 24, 1))
        X = np.concatenate((X, cohort_col_onehot), axis=-1)

    # Train, val, test split
    X_train, X_val, X_test, \
        y_train, y_val, y_test, \
        cohorts_train, cohorts_val, cohorts_test = stratified_split(
            X, Y, cohort_col, train_val_random_seed=FLAGS.train_val_random_seed)

    # Sample Weights
    task_weights = dict()
    all_tasks = np.unique(cohorts_train)
    for cohort in all_tasks:
        num_in_cohort = len(np.where(cohorts_train == cohort)[0])
        print("Number of people in cohort " +
              str(cohort) + ": " + str(num_in_cohort))
        task_weights[cohort] = len(X_train)*1.0/num_in_cohort

    if FLAGS.sample_weights:
        samp_weights = np.array([task_weights[cohort]
                                 for cohort in cohorts_train])

    else:
        samp_weights = None

    # Run model
    run_model_args = [X_train, y_train, cohorts_train,
                      X_val, y_val, cohorts_val,
                      X_test, y_test, cohorts_test,
                      all_tasks, fname_keys, fname_results,
                      FLAGS]

    if FLAGS.model_type == 'SEPARATE':
        run_separate_models(*run_model_args)
    elif FLAGS.model_type == 'GLOBAL':
        run_global_model(*run_model_args)
    elif FLAGS.model_type == 'MULTITASK':
        run_multitask_model(*run_model_args)
