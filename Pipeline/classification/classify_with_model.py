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
import argparse
import datetime

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def report_feature_importance(
    feature_importances, result_dir, file_name="featimpcomplete"
):
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
        if isinstance(url, list):
            domains = []
            for u in url:
                u = tldextract.extract(u)
                domains.append(u.domain + "." + u.suffix)
            return domains
        else:
            u = tldextract.extract(url)
            return u.domain + "." + u.suffix
    except:
        # traceback.print_exc()
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

    report_feature_importance(
        feature_importances, result_dir, file_name.split(".")[0] + "_featimp"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_base_path", type=str, default="../features_")
    parser.add_argument("--label_base_path", type=str, default="../labels_new_")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--result_dir",
        type=str,
        default=f"results/{datetime.datetime.now().strftime('%m-%d-%H-%M')}",
    )
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--generate-filterlist", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    FEATURE_PATH = args.feature_base_path
    RESULT_DIR = args.result_dir
    LABEL_PATH = args.label_base_path
    ITERATIONS = args.iterations
    FILTERLIST = args.generate_filterlist
    MODEL_NAME = args.model_path

    # recursively create result_dir if it does not exist
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    df_features = []
    df_labels = []

    for i in tqdm(range(0, ITERATIONS * 1000, 1000)):
        df_features.append(pd.read_csv(f"{FEATURE_PATH}{i}.csv"))
        df_labels.append(pd.read_csv(f"{LABEL_PATH}{i}.csv"))

    df_features = pd.concat(df_features)
    df_labels = pd.concat(df_labels)

    df = df_features.merge(df_labels[["name", "label"]], on=["name"])

    df = df.drop_duplicates()

    classify(df, RESULT_DIR, "results.csv", MODEL_NAME)

    if FILTERLIST:
        df_results = pd.read_csv(os.path.join(RESULT_DIR, "results.csv"))
        df_results = df_results[df_results.clabel == "Positive"]
        df_results = df_results[["name"]]
        df_results.to_csv(
            os.path.join(RESULT_DIR, "filterlist.txt"), index=False, header=False
        )
