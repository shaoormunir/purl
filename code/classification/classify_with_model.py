from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn import preprocessing
import pandas as pd
import os
import sys
from yaml import load, dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from treeinterpreter import treeinterpreter as ti
import json
from collections import Counter
import random
import numpy as np
import collections
import tldextract
import pickle
from tqdm.auto import tqdm

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def report_feature_importance(feature_importances, result_dir, file_name="featimpcomplete"):
    """
    Function to make classification stats report.

    Args:
    report, result_dir, avg='macro avg', stats=['mean', 'std']
    Returns:
    Nothing, writes to a file

    This functions does the following:

    1. Splits the visit IDs into sets for each fold (one set will be used as test data).
    2. Creates test and train data.
    3. Performs training/classification.
    """

    fname = os.path.join(result_dir, file_name)
    with open(fname, "a") as f:
        f.write(feature_importances.to_string())
        f.write("\n")


def report_true_pred(
    y_true,
    y_pred,
    name,
    vid,
    i,
    result_dir,
):
    """
    Function to make truth/prediction output file.

    Args:
    y_true: Truth values
    y_pred: Predicted values
    name: Classified resource URLs
    vid: Visit IDs
    i: Fold number
    result_dir: Output directory
    Returns:
    Nothing, writes to a file

    This functions does the following:

    1. Obtains the classification metrics for each fold.
    """

    fname = os.path.join(result_dir, "tp_%s" % str(i))
    with open(fname, "w") as f:
        for i in range(0, len(y_true)):
            f.write(
                "%s |$| %s |$| %s |$| %s\n" % (y_true[i], y_pred[i], name[i], vid[i])
            )

    fname = os.path.join(result_dir, "confusion_matrix")
    with open(fname, "a") as f:
        f.write(
            np.array_str(
                confusion_matrix(y_true, y_pred, labels=["Positive", "Negative"])
            )
            + "\n\n"
        )

def get_domain(url):

    try:
        if (isinstance(url, list)):
            domains = []
            for u in url:
                u = tldextract.extract(u)
                domains.append(u.domain+"."+u.suffix)
            return domains
        else:
            u = tldextract.extract(url)
            return u.domain+"."+u.suffix
    except:
        #traceback.print_exc()
        return None
    

def label_party(name):
    parts = name.split("||")

    if get_domain(parts[0].strip()) == get_domain(parts[1].strip()):
        return "First"
    else:
        return "Third"

def classify(df, result_dir, file_name, model_name):
    if os.path.exists(result_dir) == False:
        os.makedirs(result_dir)
    # load the pickled model
    clf = pickle.load(open(model_name, "rb"))

    df["party"] = df["name"].apply(label_party)

    df = df[df.party == "Third"]

    fields_to_remove = ["visit_id", "name", "label", "party", "Unnamed: 0"]

    df_features = df.drop(fields_to_remove, axis=1, errors="ignore")

    columns = df_features.columns

    df_features = df_features.to_numpy()

    # predict the labels
    y_pred = clf.predict(df_features)

    # predict the probabilities
    y_pred_proba = clf.predict_proba(df_features)

    # add the predicted labels to the dataframe
    df["clabel"] = y_pred

    # add the predicted probabilities to the dataframe
    df["clabel_prob"] = y_pred_proba[:, 1]

    # save the results
    df.to_csv(os.path.join(result_dir, file_name), index=False)

    # feature importance
    feature_importances = pd.DataFrame(
        clf.feature_importances_, index=columns, columns=["importance"]
    ).sort_values("importance", ascending=False)

    report_feature_importance(feature_importances, result_dir, file_name.split(".")[0] + "_featimp")


if __name__ == "__main__":
    FEATURE_FOLDER = "../../"
    LABEL_FOLDER = "../../"
    RESULT_DIR = "results/20k-run-6-19-23"
    MODEL_NAME = "results/20k-run-6-19-23/model_2.sav"

    # df_features = pd.DataFrame()
    # df_features_dflow = pd.DataFrame()
    # df_labels = pd.DataFrame()

    # fnames = os.listdir(FEATURE_FOLDER)
    # for fname in fnames:
    # 	fpath = os.path.join(FEATURE_FOLDER, fname)
    # 	df = pd.read_csv(fpath)
    # 	df_features = df_features.append(df)

    # df_features = df_features.drop_duplicates()

    # fnames = os.listdir(LABEL_FOLDER)
    # for fname in fnames:
    # 	fpath = os.path.join(LABEL_FOLDER, fname)
    # 	df = pd.read_csv(fpath)
    # 	df_labels = df_labels.append(df)

    df_features = []

    for i in tqdm(range(0, 20000, 1000)):
        df_features.append(pd.read_csv(f"../features_{i}.csv"))
    
    df_features = pd.concat(df_features)

    # df_labels = []

    # for i in tqdm(range(0, 20000, 1000)):
    #     df_labels.append(pd.read_csv(f"../labels_{i}.csv", on_bad_lines='skip', usecols=['name', 'label']))

    # df_labels = pd.concat(df_labels)

    # df_labels = df_labels[['name', 'label']].drop_duplicates()

    df_labels = pd.read_parquet("../labels_cleaned.parquet")

    df = df_features.merge(df_labels[["name", "label"]], on=["name"])

    df = df.drop_duplicates()

    df_labelled = df[df["label"] != "Unknown"]
    df_unknown = df[df["label"] == "Unknown"]

    classify(df_labelled, RESULT_DIR, "labelled_results.csv", MODEL_NAME)
    classify(df_unknown, RESULT_DIR, "unknown_results.csv", MODEL_NAME)

    classify(df, RESULT_DIR, "all_results.csv", MODEL_NAME)