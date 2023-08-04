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
import pickle
from tqdm.auto import tqdm
import tldextract

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


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

def print_stats(report, result_dir, avg="macro avg", stats=["mean", "std"]):
    """
    Function to make classification stats report.

    Args:
    report,
    result_dir,
    avg='macro avg',
    stats=['mean', 'std']
    Returns:
    Nothing, writes to a file

    This functions does the following:

    1. Splits the visit IDs into sets for each fold (one set will be used as test data).
    2. Creates test and train data.
    3. Performs training/classification.
    """

    by_label = report.groupby("label").describe()
    fname = os.path.join(result_dir, "scores")
    with open(fname, "w") as f:
        for stat in stats:
            print(by_label.loc[avg].xs(stat, level=1))
            x = by_label.loc[avg].xs(stat, level=1)
            f.write(by_label.loc[avg].xs(stat, level=1).to_string())
            f.write("\n")


def report_feature_importance(feature_importances, result_dir):
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

    fname = os.path.join(result_dir, "featimp")
    with open(fname, "a") as f:
        f.write(feature_importances.to_string())
        f.write("\n")


def report_true_pred(y_true, y_pred, name, vid, i, result_dir):
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


def describe_classif_reports(results, result_dir):
    """
    Function to make classification stats report.

    Args:
    results: Results of classification
    result_dir: Output directory
    Returns:
    all_folds: DataFrame of results

    This functions does the following:

    1. Obtains the classification metrics for each fold.
    """

    true_vectors, pred_vectors, name_vectors, vid_vectors = (
        [r[0] for r in results],
        [r[1] for r in results],
        [r[2] for r in results],
        [r[3] for r in results],
    )
    fname = os.path.join(result_dir, "scores")

    all_folds = pd.DataFrame(
        columns=["label", "fold", "precision", "recall", "f1-score", "support"]
    )
    for i, (y_true, y_pred, name, vid) in enumerate(
        zip(true_vectors, pred_vectors, name_vectors, vid_vectors)
    ):
        report_true_pred(y_true, y_pred, name, vid, i, result_dir)
        output = classification_report(y_true, y_pred)
        with open(fname, "a") as f:
            f.write(output)
            f.write("\n\n")
        # df = pd.DataFrame(output).transpose().reset_index().rename(columns={'index': 'label'})
        # df['fold'] = i
        # all_folds = all_folds.append(df)
    return all_folds


def log_prediction_probability(
    clf, df_feature_test, cols, test_mani, y_pred, result_dir, tag
):
    y_pred_prob = clf.predict_proba(df_feature_test)

    fname = os.path.join(result_dir, "predict_prob_" + str(tag))
    with open(fname, "w") as f:
        class_names = [str(x) for x in clf.classes_]
        class_names = " |$| ".join(class_names)
        f.write("Truth |$| Pred |$| " + class_names + " |$| Name |$| VID" + "\n")

        truth_labels = [str(x) for x in list(test_mani.label)]
        pred_labels = [str(x) for x in list(y_pred)]
        truth_names = [str(x) for x in list(test_mani.name)]
        truth_vids = [str(x) for x in list(test_mani.visit_id)]

        for i in range(0, len(y_pred_prob)):
            preds = [str(x) for x in y_pred_prob[i]]
            preds = " |$| ".join(preds)
            f.write(
                "%s |$| %s |$| %s |$| %s |$| %s\n"
                % (
                    truth_labels[i],
                    pred_labels[i],
                    preds,
                    truth_names[i],
                    truth_vids[i],
                )
            )

    preds, bias, contributions = ti.predict(clf, df_feature_test)
    fname = os.path.join(result_dir, "interpretation_" + str(tag))
    with open(fname, "w") as f:
        data_dict = {}
        for i in range(len(df_feature_test)):
            name = test_mani.iloc[i]["name"]
            vid = str(test_mani.iloc[i]["visit_id"])
            key = str(name) + "_" + str(vid)
            data_dict[key] = {}
            data_dict[key]["name"] = name
            data_dict[key]["vid"] = vid
            c = list(contributions[i, :, 0])
            c = [round(float(x), 2) for x in c]
            fn = list(cols)
            fn = [str(x) for x in fn]
            feature_contribution = list(zip(c, fn))
            # feature_contribution = list(zip(contributions[i,:,0], df_feature_test.columns))
            data_dict[key]["contributions"] = feature_contribution
        f.write(json.dumps(data_dict, indent=4))


def classify(train, test, result_dir, tag, sample, log_pred_probability):
    train_mani = train.copy()
    test_mani = test.copy()
    clf = RandomForestClassifier(n_estimators=100)
    # clf = AdaBoostClassifier(n_estimators=100)
    fields_to_remove = ["visit_id", "name", "label", "party", "Unnamed: 0"]
    #'ascendant_script_length',
    #'ascendant_script_has_fp_keyword',
    #'ascendant_has_ad_keyword',
    #'ascendant_script_has_eval_or_function']
    # 'num_exfil', 'num_infil',
    #                   'num_url_exfil', 'num_header_exfil', 'num_body_exfil',
    #                   'num_ls_exfil', 'num_ls_infil',
    #                   'num_ls_url_exfil', 'num_ls_header_exfil', 'num_ls_body_exfil', 'num_cookieheader_exfil',
    # #                   # 'num_get_storage', 'num_set_storage',
    # #                  'num_script_predecessors',
    #                   'indirect_in_degree', 'indirect_out_degree',
    #                   'indirect_ancestors', 'indirect_descendants',
    #                   'indirect_closeness_centrality', 'indirect_average_degree_connectivity',
    #                   'indirect_eccentricity', 'indirect_all_in_degree',
    #                   'indirect_all_out_degree', 'indirect_all_ancestors',
    #                   'indirect_all_descendants', 'indirect_all_closeness_centrality',
    #                   'indirect_all_average_degree_connectivity', 'indirect_all_eccentricity']
    # #  #'num_nodes', 'num_edges',
    # #'nodes_div_by_edges', 'edges_div_by_nodes']
    df_feature_train = train_mani.drop(fields_to_remove, axis=1, errors="ignore")
    df_feature_test = test_mani.drop(fields_to_remove, axis=1, errors="ignore")

    columns = df_feature_train.columns
    df_feature_train = df_feature_train.to_numpy()
    train_labels = train_mani.label.to_numpy()

    if sample:
        oversample = RandomOverSampler(sampling_strategy=0.5)
        df_feature_train, train_labels = oversample.fit_resample(
            df_feature_train, train_labels
        )
        undersample = RandomUnderSampler(sampling_strategy=0.5)
        df_feature_train, train_labels = undersample.fit_resample(
            df_feature_train, train_labels
        )

        fname = os.path.join(result_dir, "composition")
        with open(fname, "a") as f:
            counts = collections.Counter(train_labels)
            f.write(
                "\nAfter sampling, new composition: "
                + str(counts["Positive"])
                + " "
                + get_perc(counts["Positive"], len(train_labels))
                + "\n"
            )

    # Perform training
    clf.fit(df_feature_train, train_labels)

    # save the model to disk
    filename = os.path.join(result_dir, "model_" + str(tag) + ".sav")
    pickle.dump(clf, open(filename, "wb"))

    # Obtain feature importances
    feature_importances = pd.DataFrame(
        clf.feature_importances_, index=columns, columns=["importance"]
    ).sort_values("importance", ascending=False)
    report_feature_importance(feature_importances, result_dir)

    # Perform classification and get predictions
    cols = df_feature_test.columns
    df_feature_test = df_feature_test.to_numpy()
    y_pred = clf.predict(df_feature_test)

    acc = accuracy_score(test_mani.label, y_pred)
    prec_binary = precision_score(test_mani.label, y_pred, pos_label="Positive")
    rec_binary = recall_score(test_mani.label, y_pred, pos_label="Positive")
    prec_micro = precision_score(test_mani.label, y_pred, average="micro")
    rec_micro = recall_score(test_mani.label, y_pred, average="micro")
    prec_macro = precision_score(test_mani.label, y_pred, average="macro")
    rec_macro = recall_score(test_mani.label, y_pred, average="macro")

    # Write accuracy score
    fname = os.path.join(result_dir, "accuracy")
    with open(fname, "a") as f:
        f.write("\nAccuracy score: " + str(round(acc * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: binary " + str(round(prec_binary * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: binary " + str(round(rec_binary * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: micro " + str(round(prec_micro * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: micro " + str(round(rec_micro * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: macro " + str(round(prec_macro * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: macro " + str(round(rec_macro * 100, 3)) + "%" + "\n")

    print("Accuracy Score:", acc)

    if log_pred_probability:
        log_prediction_probability(
            clf, df_feature_test, cols, test_mani, y_pred, result_dir, tag
        )

    return (
        list(test_mani.label),
        list(y_pred),
        list(test_mani.name),
        list(test_mani.visit_id),
    )


def classify_unknown(df_train, df_test, result_dir):
    train_mani = df_train.copy()
    test_mani = df_test.copy()
    # test_mani = test_mani[test_mani['single'] != "NegBinary"]
    # print(test_mani['single'].value_counts())
    clf = RandomForestClassifier(n_estimators=100)
    # clf = AdaBoostClassifier(n_estimators=100)
    fields_to_remove = ["visit_id", "name", "label", "party", "Unnamed: 0"]
    #'ascendant_script_length', 'ascendant_script_has_fp_keyword',
    #'ascendant_has_ad_keyword',
    #'ascendant_script_has_eval_or_function']
    # 'num_exfil', 'num_infil',
    #                   'num_url_exfil', 'num_header_exfil', 'num_body_exfil',
    #                   'num_ls_exfil', 'num_ls_infil',
    #                   'num_ls_url_exfil', 'num_ls_header_exfil', 'num_ls_body_exfil', 'num_cookieheader_exfil',
    #                   'indirect_in_degree', 'indirect_out_degree',
    #                   'indirect_ancestors', 'indirect_descendants',
    #                   'indirect_closeness_centrality', 'indirect_average_degree_connectivity',
    #                   'indirect_eccentricity', 'indirect_all_in_degree',
    #                   'indirect_all_out_degree', 'indirect_all_ancestors',
    #                   'indirect_all_descendants', 'indirect_all_closeness_centrality',
    #                   'indirect_all_average_degree_connectivity', 'indirect_all_eccentricity'
    # ]
    #'num_nodes', 'num_edges',
    #'nodes_div_by_edges', 'edges_div_by_nodes']
    df_feature_train = train_mani.drop(fields_to_remove, axis=1, errors="ignore")
    df_feature_test = test_mani.drop(fields_to_remove, axis=1, errors="ignore")

    columns = df_feature_train.columns
    df_feature_train = df_feature_train.to_numpy()
    train_labels = train_mani.label.to_numpy()

    # Perform training
    clf.fit(df_feature_train, train_labels)

    # Obtain feature importances
    feature_importances = pd.DataFrame(
        clf.feature_importances_, index=columns, columns=["importance"]
    ).sort_values("importance", ascending=False)
    report_feature_importance(feature_importances, result_dir)

    df_feature_test = df_feature_test.to_numpy()
    y_pred = clf.predict(df_feature_test)
    y_pred = list(y_pred)
    name = list(test_mani.name)
    vid = list(test_mani.visit_id)

    fname = os.path.join(result_dir, "predictions")
    with open(fname, "w") as f:
        for i in range(0, len(y_pred)):
            f.write("%s |$| %s |$| %s\n" % (y_pred[i], name[i], vid[i]))

    preds, bias, contributions = ti.predict(clf, df_feature_test)
    fname = os.path.join(result_dir, "interpretations")
    with open(fname, "w") as f:
        data_dict = {}
        for i in range(len(df_feature_test)):
            name = test_mani.iloc[i]["name"]
            vid = str(test_mani.iloc[i]["visit_id"])
            key = str(name) + "_" + str(vid)
            data_dict[key] = {}
            data_dict[key]["name"] = name
            data_dict[key]["vid"] = vid
            c = list(contributions[i, :, 0])
            c = [round(float(x), 2) for x in c]
            fn = list(columns)
            fn = [str(x) for x in fn]
            feature_contribution = list(zip(c, fn))
            # feature_contribution = list(zip(contributions[i,:,0], df_feature_test.columns))
            data_dict[key]["contributions"] = feature_contribution
        f.write(json.dumps(data_dict, indent=4))


def classify_validation(
    df_train, df_validation, result_dir, sample=False, log_pred_probability=False
):
    i = 0
    result = classify(
        df_train, df_validation, result_dir, i, sample, log_pred_probability
    )
    results = [result]

    return results


def classify_crossval(
    df_labelled, result_dir, sample=False, log_pred_probability=False
):
    vid_list = df_labelled["visit_id"].unique()
    num_iter = 10
    num_test_vid = int(len(vid_list) / num_iter)
    print("VIDs", len(vid_list))
    print("To use!", num_test_vid)
    used_test_ids = []
    results = []

    for i in range(0, num_iter):
        print("Fold", i)
        vid_list_iter = list(set(vid_list) - set(used_test_ids))
        chosen_test_vid = random.sample(vid_list_iter, num_test_vid)
        used_test_ids += chosen_test_vid

        df_train = df_labelled[~df_labelled["visit_id"].isin(chosen_test_vid)]
        df_test = df_labelled[df_labelled["visit_id"].isin(chosen_test_vid)]

        fname = os.path.join(result_dir, "composition")
        train_pos = len(df_train[df_train["label"] == "Positive"])
        test_pos = len(df_test[df_test["label"] == "Positive"])

        with open(fname, "a") as f:
            f.write("\nFold " + str(i) + "\n")
            f.write(
                "Train: "
                + str(train_pos)
                + " "
                + get_perc(train_pos, len(df_train))
                + "\n"
            )
            f.write(
                "Test: " + str(test_pos) + " " + get_perc(test_pos, len(df_test)) + "\n"
            )
            f.write("\n")

        result = classify(
            df_train, df_test, result_dir, i, sample, log_pred_probability
        )
        results.append(result)

    return results


def get_perc(num, den):
    return str(round(num / den * 100, 2)) + "%"

def label_party(name):
    parts = name.split("||")

    if get_domain(parts[0].strip()) == get_domain(parts[1].strip()):
        return "First"
    else:
        return "Third"

def pipeline(df_features, df_labels, result_dir):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    df = df_features.merge(df_labels[['name', 'label']], on=["name"])
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df['party'] = df['name'].apply(label_party)

    print(df['party'].value_counts())

    df = df[df['party'] == 'Third']
    # df = df.fillna(0)
    print(len(df))
    df_labelled = df[df["label"] != "Unknown"]
    df_unknown = df[df["label"] == "Unknown"]
    df_positive = df[df["label"] == "Positive"]
    df_negative = df[df["label"] == "Negative"]

    # find nan values
    print("Nan values")
    print(df.isnull().values.any())

    # remove nan
    df_labelled = df_labelled.dropna()
    df_unknown = df_unknown.dropna()
    df_positive = df_positive.dropna()
    df_negative = df_negative.dropna()

    fname = os.path.join(result_dir, "composition")
    with open(fname, "a") as f:
        f.write("Number of samples: " + str(len(df)) + "\n")
        f.write(
            "Labelled samples: "
            + str(len(df_labelled))
            + " "
            + get_perc(len(df_labelled), len(df))
            + "\n"
        )
        f.write(
            "Positive samples: "
            + str(len(df_positive))
            + " "
            + get_perc(len(df_positive), len(df))
            + "\n"
        )
        f.write(
            "Negative samples: "
            + str(len(df_negative))
            + " "
            + get_perc(len(df_negative), len(df))
            + "\n"
        )
        f.write("\n")
        
		# sample negative labels to match positive labels
    df_negative = df_negative.sample(n=len(df_positive), random_state=1)
    df_labelled = pd.concat([df_positive, df_negative])
    vid_list = df_labelled["visit_id"].unique()
    num_iter = 10
    num_test_vid = int(len(vid_list) / num_iter)
    chosen_test_vid = random.sample(list(vid_list), num_test_vid)
    df_validation = df_labelled[df_labelled["visit_id"].isin(chosen_test_vid)]
    df_crossval = df_labelled[~df_labelled["visit_id"].isin(chosen_test_vid)]

    results = classify_crossval(
        df_labelled, result_dir, sample=False, log_pred_probability=True
    )
    report = describe_classif_reports(results, result_dir)
    # print(report)
    # print_stats(report, result_dir)

    valid_result_dir = os.path.join(result_dir, "validation")
    os.mkdir(valid_result_dir)
    results = classify_validation(df_crossval, df_validation, valid_result_dir, sample=False, log_pred_probability=True)
    report = describe_classif_reports(results, valid_result_dir)

    # Unknown labels
    unknown_result_dir = os.path.join(result_dir, "unlabelled")
    os.mkdir(unknown_result_dir)
    classify_unknown(df_labelled, df_unknown, unknown_result_dir)


def change_label_setter(setter_label):
    if setter_label == True:
        return "Positive"
    if setter_label == False:
        return "Negative"
    return "Unknown"


def change_label_discrepancy(row):
    if (row["setter_label"] == False) & (row["declared_label"] == 3):
        print("here")
        return "Unknown"
    return row["label"]


def change_label_oldcookiepedia(row):
    if row["category"] == "Targeting/Advertising":
        return "Positive"
    elif row["setter_label"] == "False":
        return "Negative"

    return "Unknown"


def change_label_ga(row):
    exclude_list = ["_ga", "_gid", "_gat"]
    if row["name"] in exclude_list:
        # if row['label'] != 'Positive':
        # 	return 'Positive'
        return "GA"
    return row["label"]


def fix_conflict(row):
    try:
        if row["clabel"] != "N/A":
            return row["clabel"]
    except:
        return row["label"]
    return row["label"]


def resolve_labels(df):
    labels = df["label"].unique()
    if len(labels) > 1:
        if "Positive" in labels:
            data = (df["visit_id"].iloc[0], df["name"].iloc[0], "Positive")
        elif "Negative" in labels:
            data = (df["visit_id"].iloc[0], df["name"].iloc[0], "Negative")
    else:
        data = (df["visit_id"].iloc[0], df["name"].iloc[0], df["label"].iloc[0])
    return data


def check_labels(df):
    labels = df["label"].unique()
    if len(labels) > 1:
        print("error!")


def change_label(row, checks):
    split_name = row["name"].split("|$$|")[0]
    if (split_name in checks) and (row["label"] == "Negative"):
        return "Positive"
    return row["label"]


if __name__ == "__main__":
    FEATURE_FOLDER = "../../"
    LABEL_FOLDER = "../../"
    RESULT_DIR = "results/20k-run-6-19-23"

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
    #     df_labels.append(pd.read_csv(f"../labels_new_{i}.csv", on_bad_lines='skip', usecols=['name', 'label']))

    # df_labels = pd.concat(df_labels)

    # df_labels = df_labels[['name', 'label']].drop_duplicates()

    df_labels = pd.read_parquet("../labels_unique.parquet")

    # df_features = pd.read_csv("../features_0.csv")
    # df_labels = pd.read_csv("../labels_0.csv", on_bad_lines='skip')

    pipeline(df_features, df_labels, RESULT_DIR)