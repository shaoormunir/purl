import re
import json
import tldextract
from urllib.parse import urlparse
import pandas as pd
from pandarallel import pandarallel

# initialize pandarallel to use maxmimum number of cores
pandarallel.initialize(progress_bar=True)
import os
import requests
from tqdm.auto import tqdm

tqdm.pandas()

from adblockparser import AdblockRules
import json
import re


from collections import Counter
from math import log

import enchant

eng_dict = enchant.Dict("en_US")
eng_uk_dict = enchant.Dict("en_UK")
eng_ca_dict = enchant.Dict("en_CA")
eng_au_dict = enchant.Dict("en_AU")


def get_decoration_type(row):
    if "||path||" in row["name"]:
        return "path"
    elif "||fragment" in row["name"]:
        return "fragment"
    else:
        return "queryparam"


def get_decoration_value(row):
    try:
        attr = json.loads(row.decoration_attr)
        return attr["value"]
    except:
        return ""


def check_if_path_english_word(row):
    try:
        if row["value"] is None:
            return False
        if row["value"] == "":
            return False
        return (
            eng_dict.check(row["value"])
            | eng_uk_dict.check(row["value"])
            | eng_au_dict.check(row["value"])
            | eng_ca_dict.check(row["value"])
        )
    except:
        return False


def calculate_shannon_entropy(str):
    counts = Counter(str)
    frequencies = ((i / len(str)) for i in counts.values())
    return -sum(f * log(f, 2) for f in frequencies)


def check_if_file(value):
    # if the value ends with a file extension and is not a website address, return true, use regex
    if re.search(r"\.[a-zA-Z]{2,4}$", value) and not re.search(
        r"^(http|https|www)", value
    ):
        return True
    else:
        return False


def find_identifiers(row):
    # the value matches the regex [^a-zA-Z0-9_=-] it is an identifier
    if len(row.value) > 6:
        if row.entropy > 3 and not row.is_word and not row.is_file:
            return True
    return False


def get_domain(url):
    u = tldextract.extract(url)
    return u.domain + "." + u.suffix


def get_path(url):
    u = urlparse(url)
    return u.path


def check_pathname(pathname, path):
    if pathname.startswith("/") and pathname.endswith("/"):
        return bool(re.search(pathname[1:-1], path))
    else:
        return pathname in path


def label_query_by_regex(rule, param):
    rule = r"{}".format(rule[1:-1])

    if bool(re.search(rule, param)):
        return True

    return False


def label_query(rule, param):
    if rule.startswith("/") and rule.endswith("/"):
        return label_query_by_regex(rule, param)
    else:
        return rule == param


def label_decoration(param, URL):
    # url_parsed = url.Url(URL)

    try:
        domain = get_domain(URL)
        path = get_path(URL)
        # print(path)

        with open("query_parameter_rules.json") as json_file:
            rules = json.load(json_file)

        General_rules = rules["GENERAL"]

        for rule in General_rules:
            if isinstance(rule, str):
                if label_query(rule, param):
                    return True
        # remove GENERAL from rules
        rules.pop("GENERAL")

        for rule in rules:
            if re.search(domain, rule):
                for r in rules[rule]:
                    if isinstance(r, str):
                        if label_query(r, param):
                            return True
                    elif isinstance(r, dict):
                        if check_pathname(r["pathname"], path):
                            if label_query(r["search"], param):
                                return True
    except Exception as e:
        print(param, URL)
        print(e)
        raise e
    return False


def read_file_newline_stripped(fname):
    with open(fname) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def get_resource_type(attr):
    try:
        attr = json.loads(attr)
        return attr["content_policy_type"]
    except Exception as e:
        # print("error in type", e)
        return None
    return None


def download_lists(FILTERLIST_DIR):
    """
    Function to download the lists used in AdGraph.
    Args:
        FILTERLIST_DIR: Path of the output directory to which filter lists should be written.
    Returns:
        Nothing, writes the lists to a directory.
    This functions does the following:
    1. Sends HTTP requests for the lists used in AdGraph.
    2. Writes to an output directory.
    """

    raw_lists = {
        "easylist": "https://easylist.to/easylist/easylist.txt",
        "easyprivacy": "https://easylist.to/easylist/easyprivacy.txt",
        "antiadblock": "https://raw.github.com/reek/anti-adblock-killer/master/anti-adblock-killer-filters.txt",
        "blockzilla": "https://raw.githubusercontent.com/annon79/Blockzilla/master/Blockzilla.txt",
        "fanboyannoyance": "https://easylist.to/easylist/fanboy-annoyance.txt",
        "fanboysocial": "https://easylist.to/easylist/fanboy-social.txt",
        "peterlowe": "http://pgl.yoyo.org/adservers/serverlist.php?hostformat=adblockplus&mimetype=plaintext",
        "squid": "http://www.squidblacklist.org/downloads/sbl-adblock.acl",
        "warning": "https://easylist-downloads.adblockplus.org/antiadblockfilters.txt",
    }

    for listname, url in raw_lists.items():
        with open(os.path.join(FILTERLIST_DIR, "%s.txt" % listname), "wb") as f:
            f.write(requests.get(url).content)


def create_filterlist_rules(filterlist_dir):
    filterlist_rules = {}
    filterlists = os.listdir(filterlist_dir)
    for fname in filterlists:
        rule_dict = {}
        rules = read_file_newline_stripped(os.path.join(filterlist_dir, fname))
        rule_dict["script"] = AdblockRules(
            rules,
            use_re2=False,
            max_mem=1024 * 1024 * 1024,
            supported_options=["script", "domain", "subdocument"],
            skip_unsupported_rules=False,
        )
        rule_dict["script_third"] = AdblockRules(
            rules,
            use_re2=False,
            max_mem=1024 * 1024 * 1024,
            supported_options=["third-party", "script", "domain", "subdocument"],
            skip_unsupported_rules=False,
        )
        rule_dict["image"] = AdblockRules(
            rules,
            use_re2=False,
            max_mem=1024 * 1024 * 1024,
            supported_options=["image", "domain", "subdocument"],
            skip_unsupported_rules=False,
        )
        rule_dict["image_third"] = AdblockRules(
            rules,
            use_re2=False,
            max_mem=1024 * 1024 * 1024,
            supported_options=["third-party", "image", "domain", "subdocument"],
            skip_unsupported_rules=False,
        )
        rule_dict["css"] = AdblockRules(
            rules,
            use_re2=False,
            max_mem=1024 * 1024 * 1024,
            supported_options=["stylesheet", "domain", "subdocument"],
            skip_unsupported_rules=False,
        )
        rule_dict["css_third"] = AdblockRules(
            rules,
            use_re2=False,
            max_mem=1024 * 1024 * 1024,
            supported_options=["third-party", "stylesheet", "domain", "subdocument"],
            skip_unsupported_rules=False,
        )
        rule_dict["xmlhttp"] = AdblockRules(
            rules,
            use_re2=False,
            max_mem=1024 * 1024 * 1024,
            supported_options=["xmlhttprequest", "domain", "subdocument"],
            skip_unsupported_rules=False,
        )
        rule_dict["xmlhttp_third"] = AdblockRules(
            rules,
            use_re2=False,
            max_mem=1024 * 1024 * 1024,
            supported_options=[
                "third-party",
                "xmlhttprequest",
                "domain",
                "subdocument",
            ],
            skip_unsupported_rules=False,
        )
        rule_dict["third"] = AdblockRules(
            rules,
            use_re2=False,
            max_mem=1024 * 1024 * 1024,
            supported_options=["third-party", "domain", "subdocument"],
            skip_unsupported_rules=False,
        )
        rule_dict["domain"] = AdblockRules(
            rules,
            use_re2=False,
            max_mem=1024 * 1024 * 1024,
            supported_options=["domain", "subdocument"],
            skip_unsupported_rules=False,
        )
        filterlist_rules[fname] = rule_dict

    return filterlists, filterlist_rules


def match_url(domain_top_level, current_domain, current_url, resource_type, rules_dict):
    try:
        if domain_top_level == current_domain:
            third_party_check = False
        else:
            third_party_check = True

        if resource_type == "sub_frame":
            subdocument_check = True
        else:
            subdocument_check = False

        if resource_type == "script":
            if third_party_check:
                rules = rules_dict["script_third"]
                options = {
                    "third-party": True,
                    "script": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }
            else:
                rules = rules_dict["script"]
                options = {
                    "script": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }

        elif resource_type == "image" or resource_type == "imageset":
            if third_party_check:
                rules = rules_dict["image_third"]
                options = {
                    "third-party": True,
                    "image": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }
            else:
                rules = rules_dict["image"]
                options = {
                    "image": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }

        elif resource_type == "stylesheet":
            if third_party_check:
                rules = rules_dict["css_third"]
                options = {
                    "third-party": True,
                    "stylesheet": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }
            else:
                rules = rules_dict["css"]
                options = {
                    "stylesheet": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }

        elif resource_type == "xmlhttprequest":
            if third_party_check:
                rules = rules_dict["xmlhttp_third"]
                options = {
                    "third-party": True,
                    "xmlhttprequest": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }
            else:
                rules = rules_dict["xmlhttp"]
                options = {
                    "xmlhttprequest": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }

        elif third_party_check:
            rules = rules_dict["third"]
            options = {
                "third-party": True,
                "domain": domain_top_level,
                "subdocument": subdocument_check,
            }

        else:
            rules = rules_dict["domain"]
            options = {"domain": domain_top_level, "subdocument": subdocument_check}

        return rules.should_block(current_url, options)

    except Exception as e:
        print("Exception encountered", e)
        print("top url", domain_top_level)
        print("current url", current_domain)
        return False


def label_request(row, filterlists, filterlist_rules):
    try:
        top_domain = row["top_level_domain"]
        current_url = row["request_url"]
        current_domain = row["request_domain"]
        resource_type = row["resource_type"]
        data_label = False

        for fl in filterlists:
            if top_domain and current_domain:
                list_label = match_url(
                    top_domain,
                    current_domain,
                    current_url,
                    resource_type,
                    filterlist_rules[fl],
                )
                data_label = data_label | list_label
            else:
                data_label = "Error"
    except Exception as e:
        print("Exception encountered", e)
        data_label = "Error"

    return data_label


def get_decoration_name(decoration):
    if "||path||" in decoration or "||fragment" in decoration:
        return None
    else:
        return decoration.split("||")[-1]


def get_final_label(row):
    if (row["decoration_label"] == True or row["cookiepedia_label"] == True) and (
        row["is_identifier"] == True
    ):
        return "Positive"
    elif row["request_label"] == False:
        return "Negative"
    else:
        return "Unknown"


def get_cookie_name(name):
    try:
        return name.split("|$$|")[0].strip()
    except Exception as e:
        print("Exception encountered", e)
        print("name", name)
        return pd.NA


def get_cookie_domain(name):
    try:
        return name.split("|$$|")[1].strip()
    except Exception as e:
        print("Exception encountered", e)
        print("name", name)
        return pd.NA


def clean_up_label(label):
    if label < 0:
        return pd.NA
    if label > 3:
        return pd.NA
    return label


def get_single_label(label):
    if label >= 2:
        return True
    else:
        return False


def label_cookiepedia(df_exfils, df_decoration_edges):
    df_dec = pd.read_csv("declared_cookie_labels.csv")

    df_exfils["cookie_name"] = df_exfils["src"].apply(get_cookie_name)
    df_exfils["cookie_domain"] = df_exfils["src"].apply(get_cookie_domain)

    df_exfils = df_exfils.merge(
        df_dec,
        left_on=["cookie_name", "cookie_domain"],
        right_on=["name", "domain"],
        how="left",
    )

    df_exfils["cookiepedia_label"] = df_exfils["declared_label"].apply(clean_up_label)
    df_exfils["cookiepedia_label"] = df_exfils["cookiepedia_label"].apply(
        get_single_label
    )

    df_exfils = df_exfils[["visit_id", "dst", "cookiepedia_label"]]

    df_decoration_edges = df_decoration_edges.merge(
        df_exfils,
        left_on=["visit_id", "name"],
        right_on=["visit_id", "dst"],
        how="left",
    )

    return df_decoration_edges


def label_decorations(df, df_exfils, filterlists, filterlist_rules):
    df_requests = df[df["type"] == "Request"]
    df_decorations = df[df["type"] == "Decoration"]
    df_edges = df[df["graph_attr"] == "Edge"]
    df_decoration_edges = pd.merge(
        df_decorations,
        df_edges,
        left_on=["visit_id", "name"],
        right_on=["visit_id", "dst"],
        how="left",
    )
    df_decoration_edges = df_decoration_edges[
        ["visit_id", "name_x", "src_y", "top_level_domain_x", "attr_x"]
    ]
    df_decoration_edges = df_decoration_edges.rename(
        columns={
            "name_x": "name",
            "src_y": "src",
            "top_level_domain_x": "top_level_domain",
            "attr_x": "attr",
        }
    )

    df_decoration_edges = df_decoration_edges[df_decoration_edges["src"].notna()]
    # print(df_decoration_edges.head())
    df_decoration_edges = pd.merge(
        df_decoration_edges,
        df_requests,
        left_on=["visit_id", "src"],
        right_on=["visit_id", "name"],
        how="left",
    )
    # print(df_decoration_edges.columns)
    df_decoration_edges = df_decoration_edges[
        ["visit_id", "name_x", "src_x", "top_level_domain_x", "attr_y", "attr_x"]
    ]
    df_decoration_edges = df_decoration_edges.rename(
        columns={
            "name_x": "name",
            "top_level_domain_x": "top_level_domain",
            "attr_y": "attr",
            "src_x": "request_url",
            "attr_x": "decoration_attr",
        }
    )
    df_decoration_edges = df_decoration_edges.drop_duplicates()
    df_decoration_edges["resource_type"] = df_decoration_edges["attr"].apply(
        get_resource_type
    )
    df_decoration_edges_unique = df_decoration_edges[
        df_decoration_edges["resource_type"].notna()
    ]
    df_decoration_edges_unique = df_decoration_edges[
        ["request_url", "top_level_domain", "resource_type"]
    ].drop_duplicates()

    df_decoration_edges_unique["request_domain"] = df_decoration_edges_unique[
        "request_url"
    ].apply(get_domain)
    df_decoration_edges_unique["request_label"] = (
        df_decoration_edges_unique.parallel_apply(
            label_request,
            filterlists=filterlists,
            filterlist_rules=filterlist_rules,
            axis=1,
        )
    )
    # df_decoration_edges_unique['request_label'] = False

    df_decoration_edges = df_decoration_edges.merge(
        df_decoration_edges_unique,
        on=["request_url", "top_level_domain", "resource_type"],
        how="left",
    )

    df_decoration_edges.drop_duplicates(inplace=True)
    # for the requests that are not in the filterlists, we label them as "Unknown"
    df_decoration_edges["request_label"] = df_decoration_edges["request_label"].fillna(
        "Unknown"
    )
    df_decoration_edges_unique = df_decoration_edges[
        ["name", "request_url"]
    ].drop_duplicates()

    df_decoration_edges_unique["decoration_name"] = df_decoration_edges_unique[
        "name"
    ].apply(get_decoration_name)
    df_decoration_edges_unique = df_decoration_edges_unique[
        df_decoration_edges_unique["decoration_name"].notna()
    ]
    df_decoration_edges_unique = df_decoration_edges_unique.drop_duplicates()
    df_decoration_edges_unique["decoration_label"] = (
        df_decoration_edges_unique.parallel_apply(
            lambda x: label_decoration(x["decoration_name"], x["request_url"]), axis=1
        )
    )

    # print(df_decoration_edges.columns)
    # print(df_decoration_edges_unique.columns)

    df_decoration_edges = df_decoration_edges.merge(
        df_decoration_edges_unique[["name", "request_url", "decoration_label"]],
        on=["name", "request_url"],
        how="left",
    )

    df_decoration_edges = label_cookiepedia(df_exfils, df_decoration_edges)

    # pri(df_decoration_edges.attr)

    df_decoration_edges["value"] = df_decoration_edges.parallel_apply(
        get_decoration_value, axis=1
    )

    # pri("Value")
    # print(df_decoration_edges.value.value_counts())

    df_decoration_edges["type"] = df_decoration_edges.parallel_apply(
        get_decoration_type, axis=1
    )

    df_decoration_edges["is_word"] = df_decoration_edges.parallel_apply(
        check_if_path_english_word, axis=1
    )

    # print("is_word")
    # print(df_decoration_edges.is_word.value_counts())

    df_decoration_edges["is_file"] = df_decoration_edges.value.parallel_apply(
        check_if_file
    )

    # print("is_file")
    # print(df_decoration_edges.is_file.value_counts())

    df_decoration_edges["entropy"] = df_decoration_edges.value.parallel_apply(
        calculate_shannon_entropy
    )

    # print("entropy")
    # # print(df_decoration_edges.entropy.describe())

    df_decoration_edges["is_identifier"] = df_decoration_edges.parallel_apply(
        find_identifiers, axis=1
    )

    # # print(df_decoration_edges.is_identifier.value_counts())

    # print('request_label')
    # print(df_decoration_edges.request_label.value_counts())

    # print('decoration_label')
    # print(df_decoration_edges.decoration_label.value_counts())

    # print('cookiepedia_label')
    # print(df_decoration_edges.cookiepedia_label.value_counts())

    # print('is_identifier')
    # print(df_decoration_edges.is_identifier.value_counts())

    df_decoration_edges["label"] = df_decoration_edges.parallel_apply(
        get_final_label, axis=1
    )

    df_decoration_edges = df_decoration_edges[
        [
            "visit_id",
            "name",
            "request_url",
            "top_level_domain",
            "request_label",
            "decoration_label",
            "cookiepedia_label",
            "label",
            "attr",
            "decoration_attr",
        ]
    ]

    df_decoration_edges.drop_duplicates(inplace=True)

    print(df_decoration_edges.label.value_counts())

    return df_decoration_edges


if __name__ == "__main__":
    df_graph = pd.read_csv("../graph_test.csv")

    FILTERLIST_DIR = "filterlists"
    if not os.path.isdir(FILTERLIST_DIR):
        os.makedirs(FILTERLIST_DIR)
        download_lists(FILTERLIST_DIR)

    filterlists, filterlist_rules = create_filterlist_rules(FILTERLIST_DIR)

    df_exfils = pd.read_csv("../exfils_test.csv")

    df_labels = label_decorations(df_graph, df_exfils, filterlists, filterlist_rules)

    # df_labels.to_csv('../labels_test.csv', index=False)
