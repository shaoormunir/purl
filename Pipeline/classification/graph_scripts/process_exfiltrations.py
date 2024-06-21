# Imports
import sys
import pymysql
from utils import *
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool, cpu_count
import math
tqdm.pandas()
pd.set_option('mode.chained_assignment', None)


def parallelize_dataframe(df, func, output, n_cores=4):
    df_split = np.array_split(df, n_cores)
    with Pool(processes=n_cores) as p:
        with tqdm(total=n_cores, desc="Processing: Exfiltrations") as pbar:
            for df_result in p.imap_unordered(func, df_split):
                pbar.update()
                with open(output, 'a') as f:
                    df_result.to_csv(f, header=f.tell() == 0)


def process_exfiltrations(conn, visit_id, disconnect_entity_map, duck_entity_map):

    load_requests_table(database, host, username, password,
                        ca_cert, client_key, client_cert, port)

    df_js_cookie = get_cookies_info(conn, visit_id)
    df_js_cookie = df_js_cookie[df_js_cookie.call_stack != 'undefined']
    df_js_cookie = df_js_cookie[df_js_cookie.document_url != 'about:blank']
    df_js_cookie = df_js_cookie[df_js_cookie.script_url != '']

    tqdm.pandas(desc='Processing: Document Domain')
    df_js_cookie['document_domain'] = df_js_cookie.progress_apply(
        lambda x: get_domain(x['document_url']), axis=1)

    tqdm.pandas(desc='Processing: Document Entity')
    df_js_cookie['document_entity'] = df_js_cookie.progress_apply(
        lambda x: get_entity(x['document_domain'], disconnect_entity_map, duck_entity_map), axis=1)

    tqdm.pandas(desc='Processing: Script Domain')
    df_js_cookie['script_domain'] = df_js_cookie.progress_apply(
        lambda x: get_domain(x['script_url']), axis=1)

    tqdm.pandas(desc='Processing: Script Entity')
    df_js_cookie['script_entity'] = df_js_cookie.progress_apply(
        lambda x: get_entity(x['script_domain'], disconnect_entity_map, duck_entity_map), axis=1)

    tqdm.pandas(desc='Processing: Party')
    df_js_cookie['party'] = df_js_cookie.progress_apply(
        find_party, axis=1)

    tqdm.pandas(desc='Processing: Cookie values')
    df_js_cookie['value_processed'] = df_js_cookie.progress_apply(
        process_cookie_js, axis=1)
    df_js_cookie = df_js_cookie.explode('value_processed')
    df_js_cookie = df_js_cookie[df_js_cookie.value_processed.notna(
    )].reset_index()

    tqdm.pandas(desc='Processing: Cookie Keys')
    df_js_cookie['cookie_key'] = df_js_cookie.progress_apply(
        get_cookie_key, axis=1)

    tqdm.pandas(desc='Processing: Document Domain')
    df_js_cookie['cookie_value'] = df_js_cookie.progress_apply(
        get_cookie_value, axis=1)
    df_js_cookie.drop(['value_processed'], axis=1, inplace=True)

    tqdm.pandas(desc='Processing: Execution Context')
    df_js_cookie['execution_context'] = df_js_cookie.progress_apply(
        get_context, axis=1)

    df_js_cookie = df_js_cookie[df_js_cookie.execution_context.values == 'first']

    df_js_cookie.drop_duplicates(inplace=True)
    df_js_cookie.reset_index(inplace=True)
    
    parallelize_dataframe(
        df_js_cookie, find_exfiltrations, output, cores)

    print('Processing: Removing duplicates')
    df_exfil = pd.read_csv(output, index_col=0)
    df_exfil.drop_duplicates(inplace=True)
    df_exfil.to_csv(output)

    print('Processing complete')

if __name__ == "__main__":
    
    process_exfiltrations(database, ssl_ca, ssl_cert,
                       ssl_key, host_name, username, password, port, cores, output)