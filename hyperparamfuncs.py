from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model.logistic import LogisticRegression

def preprocess_data(df):
    """
    function to remove non-numeric features and null values
    :param df: dataframe
    :return: df & var_class=Series with class labels popped from df
    """
    # select only columns with int or float data types
    df = df.select_dtypes(['number'])
    # drop any columns with null values
    df.dropna(axis=1, inplace=True)
    return df


def scale_data(df, scaler=MinMaxScaler):
    """
    function to scale the input dataframe and return scaled dataframe
    :param df: dataframe
    :param scaler: scaling function
    :return: scaled df
    """
    scaler = scaler()
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
    return scaled_df


def compare_dicts(a, b, ignore=['test_score', 'train_score', 'tn', 'fn', 'tp', 'fp',
                                'f1_score', 'precision', 'recall', 'feature_importances']):
    """
    function to compare if current hyperparameters have already been run in a model
    :param a: hyperparameter entry
    :param b: current hyperparameters you would like to test
    :param ignore: list of hyperparameter terms to ignore in comparison
    :return: boolean; True if the current hyperparameters have been run already, and False if they have not
    """
    a = dict(a)
    b = dict(b)
    for k in ignore:
        a.pop(k, None)
        b.pop(k, None)

    return tuple(a.items()) == tuple(b.items())


def make_comparison(hyperparam_table, hyper_dict, compare_func=compare_dicts):
    """
    function to compare current hyperparameters (hyper_dict) to existing hyperparam_table
    :param hyperparam_table: existing hyperparam_table
    :param hyper_dict: current hyperparameters you would like to run in a model
    :param compare_func: function used to compare hyperparameter table to current hyperparams
    :return: exists: boolean; True if hyper_dict has been run before and False if it hasn't
    """
    exists = any([compare_func(a, b=hyper_dict) for a in hyperparam_table])
    return exists, hyper_dict


def train_test_write(x_train, x_test, y_train, y_test, filename, scaled=None):
    """
    function to write train and test sets to files
    :param x_train: df
    :param x_test: df
    :param y_train: Series
    :param y_test: Series
    :param filename: filename of original dataset
    :param scaled: scaling function that is used
    :return: None
    """

    if scaled:
        x_train.to_csv(filename[:-4] + '_scaledxtrain.csv')
        x_test.to_csv(filename[:-4] + '_scaledxtest.csv')
        y_train.to_csv(filename[:-4] + '_scaledytrain.csv', header=False)
        y_test.to_csv(filename[:-4] + '_scaledytest.csv', header=False)
    else:
        x_train.to_csv(filename[:-4] + '_xtrain.csv')
        x_test.to_csv(filename[:-4] + '_xtest.csv')
        y_train.to_csv(filename[:-4] + '_ytrain.csv', header=False)
        y_test.to_csv(filename[:-4] + '_ytest.csv', header=False)


def train_model(x_train, y_train, x_test, y_test, hyper_dict, hyperparam_table):
    """
    function to train model with given training/test sets and hyperparameters
    :param x_train: training set df
    :param y_train: training set Series
    :param x_test: test set df
    :param y_test: test set Series
    :param hyper_dict: dictionary of current hyperparameters you would like to run
    :param hyperparam_table: hyperparameter table of all models run with their respective hyperparameters
    :return: clf=classifier trained & hyperparam_table updated
    """

    hyperparam_table += [hyper_dict]
    try:
        clf = hyper_dict['model'](class_weight=hyper_dict['class_weight'], random_state=hyper_dict['random_state'])
    except:
        clf = hyper_dict['model']().set_params(**hyper_dict['params'])
        
    clf.fit(x_train, y_train)

    predictions_test = clf.predict(x_test)

    score = clf.score(x_test, y_test)
    hyperparam_table[-1]['test_score'] = score
    training_score = clf.score(x_train, y_train)
    hyperparam_table[-1]['train_score'] = training_score

    tn, fp, fn, tp = confusion_matrix(y_test, predictions_test).ravel()
    hyperparam_table[-1]['tn'] = tn
    hyperparam_table[-1]['fp'] = fp
    hyperparam_table[-1]['fn'] = fn
    hyperparam_table[-1]['tp'] = tp

    f1 = f1_score(y_test, predictions_test)
    hyperparam_table[-1]['f1_score'] = f1
    precision = precision_score(y_test, predictions_test)
    hyperparam_table[-1]['precision'] = precision
    recall = recall_score(y_test, predictions_test)
    hyperparam_table[-1]['recall'] = recall
    try:
        hyperparam_table[-1]['feature_importances'] = clf.coef_
    except:
        try:
            hyperparam_table[-1]['feature_importances'] = clf.feature_importances_
        except:
            hyperparam_table[-1]['feature_importances'] = None
    return clf, hyperparam_table


def train_eval(x_train, y_train, x_test, y_test, hyper_dict, hyperparam_table):
    """
    function to train and evaluate models with given hyperparamters using helper functions defined above
    :param filename: filename of dataset
    :param hyper_dict: dictionary of hyperparameters to run model with
    :return: updated hyperparameter table, classifier
    """

    exists, hyper_dict = make_comparison(hyperparam_table, hyper_dict, compare_func=compare_dicts)
    if not exists:
        clf, hyperparam_table = train_model(x_train, y_train, x_test, y_test, hyper_dict, hyperparam_table)
        return clf, hyperparam_table
    return None, hyperparam_table
    