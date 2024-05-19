import graph_scripts as gs
import labelling_scripts as ls
from tqdm import tqdm
import pandas as pd
import gc
from yaml import full_load
from feature_extraction import extract_graph_features
from networkx.readwrite import json_graph
import json
import leveldb
import argparse

import timeit
from resource import getrusage, RUSAGE_SELF
import traceback
import time
import os
import tldextract

pd.set_option("display.max_rows", None, "display.max_columns", None)


def read_sites_visit(db_file, conn):
    """Read the list of sites visited by crawler and their information
    :return: pandas df of site_visits table in scraper SQL file.
    """
    # conn = gs.get_local_db_connection(db_file)
    # Return a dataframe of the sites visited (stored in sites_visits of scraper SQL)
    return gs.get_sites_visit(conn)


def create_graph(df):
    """Function to build a graph on each visit_id/site. Complete as required.
    :param df: pandas dataframe of nodes and edges.
    :return: graph object.
    :rtype: Graph
    """
    G = gs.build_graph(df)
    tqdm.write(f"Built graph of {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G


def load_features_info(filename):
    """Load features from features.yaml file
    :param filename: yaml file name containing feature names
    :return: list of features to use.
    """
    with open(filename) as file:
        return full_load(file)


def get_features(pdf, G, features_file):
    """Getter to generate the features of each node in a graph.
    :param pdf: pandas df of nodes and edges in a graph.
    :param G: Graph object representation of the pdf.
    :return: dataframe of features per node in the graph
    """
    # Generate features for each node in our graph
    feature_config = load_features_info(features_file)
    df_features = extract_graph_features(pdf, G, pdf.visit_id[0], None, feature_config)
    return df_features


def find_setter_domain(setter):
    try:
        domain = gs.get_domain(setter)
        return domain
    except:
        return None


def find_domain(row):
    domain = None

    try:
        node_type = row["type"]
        if (
            (node_type == "Document")
            or (node_type == "Request")
            or (node_type == "Script")
        ):
            domain = gs.get_domain(row["name"])
        elif node_type == "Element":
            return domain
        else:
            return row["domain"]
        return domain
    except Exception as e:
        traceback.print_exc()
        return None


def find_tld(top_level_url):
    try:
        if top_level_url:
            tld = gs.get_domain(top_level_url)
            return tld
        else:
            return None
    except:
        return None


def update_storage_names(row):
    name = row["name"]
    try:
        if row["type"] == "Storage":
            name = name + "|$$|" + row["domain"]
    except Exception as e:
        return name
    return name


def find_setters(
    df_all_storage_nodes,
    df_http_cookie_nodes,
    df_all_storage_edges,
    df_http_cookie_edges,
):
    df_setter_nodes = pd.DataFrame(
        columns=[
            "visit_id",
            "name",
            "type",
            "attr",
            "top_level_url",
            "domain",
            "setter",
            "setting_time_stamp",
        ]
    )

    try:
        df_storage_edges = pd.concat([df_all_storage_edges, df_http_cookie_edges])
        if len(df_storage_edges) > 0:
            df_storage_sets = df_storage_edges[
                (df_storage_edges["action"] == "set")
                | (df_storage_edges["action"] == "set_js")
            ]
            df_setters = gs.get_original_cookie_setters(df_storage_sets)
            df_storage_nodes = pd.concat([df_all_storage_nodes, df_http_cookie_nodes])
            df_setter_nodes = df_storage_nodes.merge(
                df_setters, on=["visit_id", "name"], how="outer"
            )

    except Exception as e:
        print("Error getting setter:", e)
        traceback.print_exc()

    return df_setter_nodes


def get_party(row):
    if row["type"] == "Storage":
        if row["domain"] and row["top_level_domain"]:
            if row["domain"] == row["top_level_domain"]:
                return "first"
            else:
                return "third"
    return "N/A"


def read_sql_crawl_data(visit_id, db_file, conn):
    """Read SQL data from crawler for a given visit_ID.
    :param visit_id: visit ID of a crawl URL.
    :return: Parsed information (nodes and edges) in pandas df.
    """
    # Read tables from DB and store as DataFrames
    df_requests, df_responses, df_redirects, call_stacks, javascript = gs.read_tables(
        conn, visit_id
    )
    df_js_nodes, df_js_edges = gs.build_html_components(javascript)
    df_request_nodes, df_request_edges = gs.build_request_components(
        df_requests, df_responses, df_redirects, call_stacks
    )

    df_decoration_nodes, df_decoration_edges = gs.build_decoration_components(
        df_request_nodes
    )
    df_all_storage_nodes, df_all_storage_edges = gs.build_storage_components(javascript)
    df_http_cookie_nodes, df_http_cookie_edges = gs.build_http_cookie_components(
        df_request_edges, df_request_nodes
    )
    df_storage_node_setter = find_setters(
        df_all_storage_nodes,
        df_http_cookie_nodes,
        df_all_storage_edges,
        df_http_cookie_edges,
    )
    # Concatenate to get all nodes and edges
    df_request_nodes["domain"] = None
    df_decoration_nodes["domain"] = None
    df_all_nodes = pd.concat(
        [df_js_nodes, df_request_nodes, df_storage_node_setter, df_decoration_nodes]
    )
    df_all_nodes["domain"] = df_all_nodes.apply(find_domain, axis=1)
    df_all_nodes["top_level_domain"] = df_all_nodes["top_level_url"].apply(find_tld)
    df_all_nodes["setter_domain"] = df_all_nodes["setter"].apply(find_setter_domain)
    # df_all_nodes['name'] = df_all_nodes.apply(update_storage_names, axis=1)
    df_all_nodes = df_all_nodes.drop_duplicates()
    df_all_nodes["graph_attr"] = "Node"

    df_all_edges = pd.concat(
        [
            df_js_edges,
            df_request_edges,
            df_all_storage_edges,
            df_http_cookie_edges,
            df_decoration_edges,
        ]
    )
    df_all_edges = df_all_edges.drop_duplicates()
    df_all_edges["top_level_domain"] = df_all_edges["top_level_url"].apply(find_tld)
    df_all_edges["graph_attr"] = "Edge"

    df_all_graph = pd.concat([df_all_nodes, df_all_edges])
    df_all_graph = df_all_graph.astype(
        {"type": "category", "response_status": "category"}
    )

    return df_all_graph


def load_features_info(filename):
    """Load features from features.yaml file
    :param filename: yaml file name containing feature names
    :return: list of features to use.
    """
    with open(filename) as file:
        return full_load(file)


def get_features(pdf, G, visit_id, features_file, ldb_file, tag):
    """Getter to generate the features of each node in a graph.
    :param pdf: pandas df of nodes and edges in a graph.
    :param G: Graph object representation of the pdf.
    :return: dataframe of features per node in the graph
    """
    # Generate features for each node in our graph
    feature_config = load_features_info(features_file)
    # ldb = leveldb.LevelDB(ldb_file)
    # ldb = plyvel.DB(ldb_file)
    ldb = None

    df_features = extract_graph_features(pdf, G, visit_id, ldb, feature_config, tag)
    return df_features


def apply_tasks(
    df,
    visit_id,
    features_file,
    ldb_file,
    graph_columns,
    feature_columns,
    tag,
):
    try:
        start = time.time()
        graph_fname = "graph_" + str(tag) + ".csv"
        if not os.path.exists(graph_fname):
            df.reindex(columns=graph_columns).to_csv(graph_fname)
        else:
            df.reindex(columns=graph_columns).to_csv(
                graph_fname, mode="a", header=False
            )
        G = create_graph(df)
        df_features = get_features(df, G, visit_id, features_file, ldb_file, tag)
        features_fname = "features_" + str(tag) + ".csv"
        if not os.path.exists(features_fname):
            df_features.reindex(columns=feature_columns).to_csv(features_fname)
        else:
            df_features.reindex(columns=feature_columns).to_csv(
                features_fname, mode="a", header=False
            )
        end = time.time()
        print("Extracted features:", end - start)

    except Exception as e:
        print("Errored in pipeline:", e)
        traceback.print_exc()


def pipeline(db_file, features_file, ldb_file, tag):
    start = time.time()
    conn = gs.get_local_db_connection(db_file)
    try:
        sites_visits = read_sites_visit(db_file, conn)
    except Exception as e:
        tqdm.write(f"Problem reading the sites_visits or the scraper data: {e}")
        exit()

    end = time.time()
    print("read site visits", end - start)

    fail = 0
    start = time.time()

    graph_columns = [
        "visit_id",
        "name",
        "top_level_url",
        "type",
        "attr",
        "domain",
        "document_url",
        "setter",
        "setting_time_stamp",
        "top_level_domain",
        "setter_domain",
        "graph_attr",
        "party",
        "src",
        "dst",
        "action",
        "time_stamp",
        "reqattr",
        "respattr",
        "response_status",
        "content_hash",
        "post_body",
        "post_body_raw",
    ]

    feature_columns = [
        "visit_id",
        "name",
        "num_nodes",
        "num_edges",
        "nodes_div_by_edges",
        "edges_div_by_nodes",
        "in_degree",
        "out_degree",
        "in_out_degree",
        "ancestors",
        "descendants",
        "closeness_centrality",
        "average_degree_connectivity",
        "eccentricity",
        "clustering",
        "is_parent_script",
        "is_ancestor_script",
        "ascendant_has_ad_keyword",
        "descendant_of_eval_or_function",
        "parent_num_get_storage",
        "parent_num_set_storage",
        "parent_num_get_storage_js",
        "parent_num_set_storage_js",
        "parent_num_get_storage_ls",
        "parent_num_set_storage_ls",
        "parent_num_get_storage_ls_js",
        "parent_num_set_storage_ls_js",
        "parent_num_cookieheader_exfil",
        "num_script_predecessors",
        "indirect_in_degree",
        "indirect_out_degree",
        "indirect_ancestors",
        "indirect_descendants",
        "indirect_closeness_centrality",
        "indirect_average_degree_connectivity",
        "indirect_eccentricity",
        "num_exfil",
        "indirect_all_in_degree",
        "indirect_all_out_degree",
        "indirect_all_ancestors",
        "indirect_all_descendants",
        "indirect_all_closeness_centrality",
        "indirect_all_average_degree_connectivity",
        "indirect_all_eccentricity",
        "sender_exfil",
        "sender_redirects_sent",
        "sender_redirects_rec",
        "shannon_entropy",
        "max_depth_decoration",
    ]

    # feature_columns = [
    #     "visit_id",
    #     "name",
    #     "num_nodes",
    #     "num_edges",
    #     "nodes_div_by_edges",
    #     "edges_div_by_nodes",
    #     "in_degree",
    #     "out_degree",
    #     "in_out_degree",
    #     "ancestors",
    #     "descendants",
    #     "closeness_centrality",
    #     "average_degree_connectivity",
    #     "eccentricity",
    #     "clustering",
    #     "is_parent_script",
    #     "is_ancestor_script",
    #     "ascendant_has_ad_keyword",
    #     "descendant_of_eval_or_function",
    #     "parent_num_get_storage",
    #     "parent_num_set_storage",
    #     "parent_num_get_storage_js",
    #     "parent_num_set_storage_js",
    #     "parent_num_get_storage_ls",
    #     "parent_num_set_storage_ls",
    #     "parent_num_get_storage_ls_js",
    #     "parent_num_set_storage_ls_js",
    #     "parent_num_cookieheader_exfil",
    #     "num_script_predecessors",
    #     "indirect_in_degree",
    #     "indirect_out_degree",
    #     "indirect_ancestors",
    #     "indirect_descendants",
    #     "indirect_closeness_centrality",
    #     "indirect_average_degree_connectivity",
    #     "indirect_eccentricity",
    #     "parent_num_exfil",
    #     # "parent_num_infil",
    #     # "parent_num_infil_content",
    #     # "parent_num_url_exfil",
    #     # "parent_num_header_exfil",
    #     # "parent_num_body_exfil",
    #     # "parent_num_ls_exfil",
    #     # "parent_num_ls_infil",
    #     # "parent_num_ls_infil_content",
    #     # "parent_num_ls_url_exfil",
    #     # "parent_num_ls_header_exfil",
    #     # "parent_num_ls_body_exfil",
    #     # "parent_num_ls_src",
    #     # "parent_num_ls_dst",
    #     "indirect_all_in_degree",
    #     "indirect_all_out_degree",
    #     "indirect_all_ancestors",
    #     "indirect_all_descendants",
    #     "indirect_all_closeness_centrality",
    #     "indirect_all_average_degree_connectivity",
    #     "indirect_all_eccentricity",
    #     "sender_exfil",
    #     "sender_redirects_sent",
    #     "sender_redirects_rec",
    #     "sender_requests_sent",
    #     "sender_responses_received",
    #     "max_depth_redirect",
    #     "max_depth_decoration",
    #     "decoration_entropy",
    #     "decoration_exfil",
    #     "decoration_infil",
    # ]

    # for i, row in tqdm(
    #     sites_visits.iterrows(),
    #     total=len(sites_visits),
    #     position=0,
    #     leave=True,
    #     ascii=True,
    # ):
    #     # For each visit, grab the visit_id and the site_url
    #     visit_id = row["visit_id"]
    #     site_url = row["site_url"]
    #     tqdm.write("")
    #     tqdm.write(f"• Visit ID: {visit_id} | Site URL: {site_url}")

    #     try:
    #         start = time.time()
    #         # this cannot be parallelized as it is reading from the sqlite file, only one process at a time can do that
    #         pdf = read_sql_crawl_data(visit_id, db_file, conn)
    #         end = time.time()
    #         print("Built graph of shape: ", pdf.shape, "in :", end - start)
    #         pdf = pdf[pdf["top_level_domain"].notnull()]
    #         pdf.groupby(["visit_id", "top_level_domain"]).apply(
    #             apply_tasks,
    #             visit_id,
    #             features_file,
    #             ldb_file,
    #             graph_columns,
    #             feature_columns,
    #             tag,
    #         )

    #         end = time.time()
    #         print("Finished processing graph: ", row["visit_id"], "in :", end - start)

    #     except Exception as e:
    #         fail += 1
    #         tqdm.write(f"Fail: {fail}")
    #         tqdm.write(f"Error: {e}")
    #         traceback.print_exc()
    #         pass

    # Label the graph
    print("Labeling graph")

    FILTERLIST_DIR = "filterlists"
    if not os.path.isdir(FILTERLIST_DIR):
        os.makedirs(FILTERLIST_DIR)
        ls.download_lists(FILTERLIST_DIR)

    filterlists, filterlist_rules = ls.create_filterlist_rules(FILTERLIST_DIR)
    end = time.time()

    print("Finished making filterlist rules", end - start)
    start = time.time()
    exfils_fname = "exfils_" + str(tag) + ".csv"
    df_exfils = pd.read_csv(exfils_fname)

    graph_fname = "graph_" + str(tag) + ".csv"
    df = pd.read_csv(graph_fname)
    df_labelled = ls.label_decorations(df, df_exfils, filterlists, filterlist_rules)
    labels_fname = "labels_new_" + str(tag) + ".csv"
    if not os.path.exists(labels_fname):
        df_labelled.to_csv(labels_fname)
    else:
        df_labelled.to_csv(labels_fname, mode="a", header=False)
    end = time.time()
    print("Labelled graph:", end - start)

    percent = (fail / len(sites_visits)) * 100
    print(f"Fail: {fail}, Total: {len(sites_visits)}, Percentage:{percent}", db_file)


if __name__ == "__main__":
    # get the features file, the dataset folder and the output folder from the command line\
    parser = argparse.ArgumentParser(description="Process the Graph features for PURL.")
    parser.add_argument(
        "--features", type=str, default="features_new.yaml", help="the features file"
    )
    parser.add_argument("--folder", type=str, default="data", help="the dataset folder")
    parser.add_argument(
        "--output", type=str, default="output", help="the output folder"
    )
    parser.add_argument(
        "--tag", type=str, default="", help="the tag for the output file"
    )
    args = parser.parse_args()

    FEATURES_FILE = args.features
    FOLDER = args.folder
    OUTPUT = args.output
    TAG = args.tag

    # DB_FILE = os.path.join(FOLDER, "crawl-data.sqlite")
    # LDB_FILE = os.path.join(FOLDER, "content.ldb")

    # pipeline(DB_FILE, FEATURES_FILE, LDB_FILE, TAG)

    for i in range(0, 20000, 1000):
        print("Processing:", i)
        DB_FILE = os.path.join(FOLDER, f"datadir-{i}/crawl-data.sqlite")
        LDB_FILE = os.path.join(FOLDER, f"datadir-{i}/content.ldb")
        print(DB_FILE, LDB_FILE)
        TAG = str(i)
        pipeline(DB_FILE, FEATURES_FILE, LDB_FILE, TAG)

    # folders = os.listdir(FOLDER)
    # print(folders)
    # for folder in folders:
    #     tag = folder[-1:]
    #     print("Processing:", folder, tag)
    #     DB_FILE = os.path.join(FOLDER, folder, "crawl-data.sqlite")
    #     ldb_file = os.path.join(FOLDER, folder, "content.ldb")
    #     pipeline(DB_FILE, FEATURES_FILE, ldb_file, tag)
    #     first = True
