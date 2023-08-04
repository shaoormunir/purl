import pandas as pd
import json
import networkx as nx
from yaml import load, dump
import os
import time
import multiprocessing as mp
from tqdm.auto import tqdm

# pool = mp.Pool(32)

import feature_scripts_cookies as fs


def extract_graph_node_features(
    G,
    df_graph,
    G_indirect,
    df_indirect_graph,
    G_indirect_all,
    node,
    ldb,
    selected_features,
    vid,
):
    all_features = []
    all_feature_names = ["visit_id", "name"]
    content_features = []
    structure_features = []
    dataflow_features = []
    additional_features = []
    content_feature_names = []
    structure_feature_names = []
    dataflow_feature_names = []
    additional_feature_names = []

    # calcualte time it takes to get each content feature

    start = time.time()
    if "content" in selected_features:
        content_features, content_feature_names = fs.get_content_features(
            G, df_graph, node
        )
    end = time.time()
    # print("Time to get content features:", end - start)

    start = time.time()
    if "structure" in selected_features:
        structure_features, structure_feature_names = fs.get_structure_features(
            G, df_graph, node, ldb
        )
    end = time.time()
    # print("Time to get structure features:", end - start)

    start = time.time()
    if "dataflow" in selected_features:
        dataflow_features, dataflow_feature_names = fs.get_dataflow_features(
            G, df_graph, node, G_indirect, G_indirect_all, df_indirect_graph
        )
    end = time.time()
    # print("Time to get dataflow features:", end - start)

    start = time.time()
    if "additional" in selected_features:
        additional_features, additional_feature_names = fs.get_additional_features(
            G, df_graph, node
        )
    end = time.time()
    # print("Time to get additional features:", end - start)

    all_features = (
        content_features + structure_features + dataflow_features + additional_features
    )
    all_feature_names += (
        content_feature_names
        + structure_feature_names
        + dataflow_feature_names
        + additional_feature_names
    )

    df = pd.DataFrame([[vid] + [node] + all_features], columns=all_feature_names)

    return df


def extract_graph_features(df_graph, G, vid, ldb, feature_config, tag):
    """
    Function to extract features.

    Args:
      df_graph_vid: DataFrame of nodes/edges for.a site
      G: networkX graph of site
      vid: Visit ID
      ldb: Content LDB
      feature_config: Feature config
    Returns:
      df_features: DataFrame of features for each URL in the graph

    This functions does the following:

    1. Reads the feature config to see which features we want.
    2. Creates a graph of indirect edges if we want to calculate dataflow features.
    3. Performs feature extraction based on the feature config. Feature extraction is per node of graph.
    """

    exfil_columns = [
        "visit_id",
        "src",
        "dst",
        "dst_domain",
        "attr",
        "time_stamp",
        "direction",
        "type",
    ]

    df_features = []
    nodes = G.nodes(data=True)
    G_indirect = nx.DiGraph()
    G_indirect_all = nx.DiGraph()
    df_indirect_graph = pd.DataFrame()

    df_graph["src_domain"] = df_graph["src"].apply(fs.get_domain)
    df_graph["dst_domain"] = df_graph["dst"].apply(fs.get_domain)

    selected_features = feature_config["features_to_extract"]

    if "dataflow" in selected_features:
        G_indirect, G_indirect_all, df_indirect_graph = fs.pre_extraction(
            G, df_graph, ldb
        )
        exfil_fname = "exfils_" + str(tag) + ".csv"
        if not os.path.exists(exfil_fname):
            df_indirect_graph.reindex(columns=exfil_columns).to_csv(exfil_fname)
        else:
            df_indirect_graph.reindex(columns=exfil_columns).to_csv(
                exfil_fname, mode="a", header=False
            )
            
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for node in nodes:

            if ("type" in node[1]) and (node[1]["type"] == "Decoration"):
                result = pool.apply_async(
                    extract_graph_node_features,
                    args=(
                        G,
                        df_graph,
                        G_indirect,
                        df_indirect_graph,
                        G_indirect_all,
                        node[0],
                        ldb,
                        selected_features,
                        vid,
                    ),
                )
                results.append(result)

        for result in tqdm(results):
            df_features.append(result.get())

        df_features = pd.concat(df_features)
    return df_features
