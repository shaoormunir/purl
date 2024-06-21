import pandas as pd
from urllib.parse import urlparse, parse_qs
import json


def get_link_decorations(url, top_level_url):
    try:
        link_decorations = []
        if url is None:
            return None
        parsed = urlparse(url)
        parsed_top_level = urlparse(top_level_url)
        query_params = parsed.query
        query_params = parse_qs(query_params)
        # add the query params to the link decorations, combine the dict values
        for k, v in query_params.items():
            # link_decorations[k] = v[0]
            link_decorations.append(
                {parsed_top_level.netloc + "||" + parsed.netloc + "||" + k: v[0]}
            )
        fragment = parsed.fragment
        if fragment != "":
            if "=" in fragment:
                if "&" in fragment:
                    # treat the fragment as a query param
                    fragment = fragment.split("&")
                    for f in fragment:
                        f = f.split("=")
                        if len(f) == 1:
                            # link_decorations[f[0]] = ""
                            link_decorations.append(
                                {
                                    parsed_top_level.netloc
                                    + "||"
                                    + parsed.netloc
                                    + "||"
                                    + f[0]: ""
                                }
                            )
                        else:
                            # link_decorations[f[0]] = f[1]
                            link_decorations.append(
                                {
                                    parsed_top_level.netloc
                                    + "||"
                                    + parsed.netloc
                                    + "||"
                                    + f[0]: f[1]
                                }
                            )
                else:
                    # treat the fragment as a query param
                    fragment = fragment.split("=")
                    # link_decorations[fragment[0]] = fragment[1]
                    link_decorations.append(
                        {
                            parsed_top_level.netloc
                            + "||"
                            + parsed.netloc
                            + "||"
                            + fragment[0]: fragment[1]
                        }
                    )
            else:
                # link_decorations[parsed.netloc + "||fragment"] = fragment
                link_decorations.append(
                    {
                        parsed_top_level.netloc
                        + "||"
                        + parsed.netloc
                        + "||fragment": fragment
                    }
                )
        path = parsed.path
        path = path.split("/")
        if path[0] == "":
            path = path[1:]
        paths = {}
        for p, i in zip(path, range(len(path))):
            if p == "":
                continue
            # link_decorations[parsed.netloc + "||path||" + str(i)] = p
            link_decorations.append(
                {
                    parsed_top_level.netloc
                    + "||"
                    + parsed.netloc
                    + "||path||"
                    + str(i): p
                }
            )
        return link_decorations
    except Exception as e:
        print(url)
        raise e


def build_decoration_components(df_request_nodes):
    df_decoration_nodes = pd.DataFrame()
    df_decoration_edges = pd.DataFrame()

    df_decoration_nodes = df_request_nodes[["visit_id", "name", "top_level_url"]]

    df_decoration_nodes["decoration"] = df_decoration_nodes.apply(
        lambda x: get_link_decorations(x["name"], x["top_level_url"]), axis=1
    )
    df_decoration_nodes["decoration"] = df_decoration_nodes["decoration"].apply(
        lambda x: [] if x is None else x
    )

    df_decoration_nodes = df_decoration_nodes.explode("decoration")
    df_decoration_nodes.dropna(inplace=True)
    # print(df_decoration_nodes.head())

    df_decoration_nodes["type"] = "Decoration"

    df_decoration_nodes["attr"] = df_decoration_nodes["decoration"].apply(
        lambda x: json.dumps({"value": list(x.values())[0]})
    )

    df_decoration_nodes.rename(columns={"name": "req_name"}, inplace=True)

    df_decoration_nodes["name"] = df_decoration_nodes["decoration"].apply(
        lambda x: list(x.keys())[0]
    )

    df_decoration_nodes.drop(columns=["decoration"], inplace=True)

    df_decoration_nodes = df_decoration_nodes[
        ["visit_id", "name", "req_name", "top_level_url", "type", "attr"]
    ]

    df_decoration_nodes.drop_duplicates(inplace=True)

    df_decoration_edges = df_decoration_nodes.copy()

    df_decoration_edges = pd.merge(
        df_decoration_edges,
        df_request_nodes,
        left_on=["visit_id", "req_name", "top_level_url"],
        right_on=["visit_id", "name", "top_level_url"],
        how="left",
    )

    # print(df_decoration_edges.columns)
    print("Decoration edges: ", df_decoration_edges.shape)

    df_decoration_edges = df_decoration_edges[
        ["visit_id", "name_x", "name_y", "top_level_url", "type_x", "attr_x"]
    ]

    df_decoration_edges.rename(
        columns={
            "name_x": "dst",
            "name_y": "src",
            "top_level_url_x": "top_level_url",
            "type_x": "type",
            "attr_x": "attr",
        },
        inplace=True,
    )

    df_decoration_edges.drop(columns=["type"], inplace=True)

    df_decoration_nodes.drop(columns=["req_name"], inplace=True)

    return df_decoration_nodes, df_decoration_edges
