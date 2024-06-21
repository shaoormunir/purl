import pandas as pd
import json
import networkx as nx
import re
#import plyvel
from openwpm_utils import domain as du
from sklearn import preprocessing
from yaml import load, dump
import numpy as np
import traceback
import tldextract

import graph_scripts as gs
import base64
import hashlib
from itertools import combinations
from Levenshtein import distance

from urllib.parse import parse_qs
import urllib.parse as urlparse

# import leveldb
import warnings
warnings.filterwarnings("ignore")

def extract_data(data, key=None):

  if isinstance(data, dict):
      for k, v in data.items():
          if isinstance(v, (dict, list)):
              for d in extract_data(v, k):
                  yield d
          else:
              yield {k:v}
  elif isinstance(data, list):
      for d in data:
          for d in extract_data(d, key):
              yield d
  elif data is not None:
      yield {key: data}
  else:
      pass


def get_response_content(content_hash, ldb):
    try:
        content = ldb.Get(content_hash.encode('utf-8'))
        jsonData = content.decode('utf-8')
        return json.loads(jsonData)
    except Exception as e:
      return None
    return None

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

def parse_url_arguments(url):

    parsed = urlparse.urlparse(url)
    sep1 = parse_qs(parsed.query)
    return {**sep1}

def has_ad_keyword(node, G):

  keyword_raw = ["ad", "ads", "advert", "popup", "banner", "sponsor", "iframe", "googlead", "adsys", "adser", "advertise", "redirect",
                 "popunder", "punder", "popout", "click", "track", "play", "pop", "prebid", "bid", "pb.min", "affiliate", "ban",
                 "delivery", "promo","tag", "zoneid", "siteid", "pageid", "size", "viewid", "zone_id", "google_afc" , "google_afs"]
  has_ad_keyword = 0
  node_type = G.nodes[node]['type']
  if node_type != "Element" and node_type != "Storage":
    for key in keyword_raw:
        key_matches = [m.start() for m in re.finditer(key, node, re.I)]
        for key_match in key_matches:
          has_ad_keyword = 1
          break
        if has_ad_keyword == 1:
          break
  return has_ad_keyword

def ad_keyword_ascendants(node, G):
  
  keyword_raw = ["ad", "ads", "advert", "popup", "banner", "sponsor", "iframe", "googlead", "adsys", "adser", "advertise", "redirect", 
                 "popunder", "punder", "popout", "click", "track", "play", "pop", "prebid", "bid", "pb.min", "affiliate", "ban", "delivery", 
                 "promo","tag", "zoneid", "siteid", "pageid", "size", "viewid", "zone_id", "google_afc" , "google_afs"]
  
  ascendant_has_ad_keyword = 0
  ascendants = nx.ancestors(G, node)
  for ascendant in ascendants:
    try:
      node_type = nx.get_node_attributes(G, 'type')[ascendant]
      if node_type != "Element" and node_type != "Storage":
        for key in keyword_raw:
          key_matches = [m.start() for m in re.finditer(key, ascendant, re.I)]
          for key_match in key_matches:
            ascendant_has_ad_keyword = 1
            break
          if ascendant_has_ad_keyword == 1:
            break
      if ascendant_has_ad_keyword == 1:
            break
    except:
      continue
  return ascendant_has_ad_keyword

def get_num_attr_changes(df_modified_attr, G):

  num_attr_changes = 0
  source_list = df_modified_attr['src'].tolist()
  for source in source_list:
    if nx.get_node_attributes(G, 'type')[source] == 'Script':
      num_attr_changes += 1
  return num_attr_changes

def get_cookieval(attr):

  try:
    attr = json.loads(attr)
    if 'value' in attr:
      return attr['value']
    else:
      return None
  except:
    return None

def get_cookiename(attr):

  try:
    attr = json.loads(attr)
    if 'name' in attr:
      return attr['name']
    else:
      return None
  except:
    return None

def find_urls(df):
  src_urls = df['src'].tolist()
  dst_urls = df['dst'].tolist()
  return list(set(src_urls + dst_urls))

def check_full_cookie(cookie_value, dest):
  return True if len([item for item in cookie_value if item in dest and len(item) > 3]) > 0 else False

def check_partial_cookie(cookie_value, dest):
  for value in cookie_value:
      split_cookie = re.split(r'\.+|;+|]+|\!+|\@+|\#+|\$+|\%+|\^+|\&+|\*+|\(+|\)+|\-+|\_+|\++|\~+|\`+|\@+=|\{+|\}+|\[+|\]+|\\+|\|+|\:+|\"+|\'+|\<+|\>+|\,+|\?+|\/+', value)
      return True if len([item for item in split_cookie if item in dest and len(item) > 3]) > 0 else False
  return False

def check_base64_cookie(cookie_value, dest):
  return True if len([item for item in cookie_value if base64.b64encode(item.encode('utf-8')).decode('utf8') in dest and len(item) > 3]) > 0 else False


def check_md5_cookie(cookie_value, dest):
  return True if len([item for item in cookie_value if hashlib.md5(item.encode('utf-8')).hexdigest() in dest and len(item) > 3]) > 0 else False


def check_sha1_cookie(cookie_value, dest):
  return True if len([item for item in cookie_value if hashlib.sha1(item.encode('utf-8')).hexdigest() in dest and len(item) > 3]) > 0 else False

def check_full_cookie_set(cookie_value, dest):
  if (len(cookie_value) > 3) and (cookie_value in dest):
    return True
  else:
    return False

def check_partial_cookie_set(cookie_value, dest):
  split_cookie = re.split(r'\.+|;+|]+|\!+|\@+|\#+|\$+|\%+|\^+|\&+|\*+|\(+|\)+|\-+|\_+|\++|\~+|\`+|\@+=|\{+|\}+|\[+|\]+|\\+|\|+|\:+|\"+|\'+|\<+|\>+|\,+|\?+|\/+', cookie_value)
  for item in split_cookie:
    if len(item) > 3 and item in dest:
      return True
  return False


def check_base64_cookie_set(cookie_value, dest):
  if (len(cookie_value) > 3) and (base64.b64encode(cookie_value.encode('utf-8')).decode('utf8') in dest):
    return True
  else:
    return False

def check_md5_cookie_set(cookie_value, dest):
  if (len(cookie_value) > 3) and (hashlib.md5(cookie_value.encode('utf-8')).hexdigest() in dest):
    return True
  else:
    return False

def check_sha1_cookie_set(cookie_value, dest):
  if (len(cookie_value) > 3) and (hashlib.sha1(cookie_value.encode('utf-8')).hexdigest() in dest):
    return True
  else:
    return False

def check_cookie_presence(http_attr, dest):

  check_value = False

  try:
    http_attr = json.loads(http_attr)

    for item in http_attr:
      if 'Cookie' in item[0]:
          cookie_pairs = item[1].split(';')
          for cookie_pair in cookie_pairs:
            cookie_value = cookie_pair.strip().split('=')[1:]
            full_cookie = check_full_cookie(cookie_value, dest)
            partial_cookie = check_partial_cookie(cookie_value, dest)
            base64_cookie = check_base64_cookie(cookie_value, dest)
            md5_cookie = check_md5_cookie(cookie_value, dest)
            sha1_cookie = check_sha1_cookie(cookie_value, dest)
            check_value = check_value | full_cookie | partial_cookie | base64_cookie | md5_cookie | sha1_cookie
            if check_value:
              return check_value
  except:
    check_value = False
  return check_value

def compare_with_obfuscation(cookie_value, param_value, threshold):

    if (cookie_value == param_value or distance(cookie_value, param_value) < threshold):
        return True
    encoded_value = hashlib.md5(cookie_value.encode('utf-8')).hexdigest()
    if (encoded_value == param_value):
        return True

    encoded_value = hashlib.sha1(cookie_value.encode('utf-8')).hexdigest()
    if (encoded_value == param_value):
        return True

    encoded_value = base64.b64encode(
        cookie_value.encode('utf-8')).decode('utf8')
    if (encoded_value == param_value):
        return True

    return False

def check_in_header(cookie_value, headers, threshold):

  headers = [x for x in headers if len(x) > 5]
  for header in headers:
    difference = abs(len(cookie_value) - len(header))
    if difference > 10:
        continue
    offset = len(cookie_value) - difference if len(cookie_value) > len(
            header) else len(header)-difference
    for i in range(0, difference+1):
      if len(header) > len(cookie_value):
          if (compare_with_obfuscation(cookie_value, header[i:i+offset], threshold)):
              return True
      else:
          if (compare_with_obfuscation(cookie_value[i:i+offset], headers, threshold)):
              return True
  return False

def check_in_body(cookie_value, body):

  if (cookie_value in body):
        return True
  encoded_value = hashlib.md5(cookie_value.encode('utf-8')).hexdigest()
  if (encoded_value in body):
      return True

  encoded_value = hashlib.sha1(cookie_value.encode('utf-8')).hexdigest()
  if (encoded_value in body):
      return True

  encoded_value = base64.b64encode(
      cookie_value.encode('utf-8')).decode('utf8')
  if (encoded_value in body):
      return True
  return False


def check_if_cookie_value_exists_post_body(cookie_key, cookie_value, post_body, threshold):

  cookie_value = str(cookie_value)
  if (len(cookie_value) <= 5):
        return False, None
  if check_in_body(cookie_value, post_body):
    return "True", "out"
  return False, None


def check_if_cookie_value_exists_header(cookie_key, cookie_value, reqheader_values, respheader_values, threshold):

  cookie_value = str(cookie_value)
  if (len(cookie_value) <= 5):
        return False, None

  # if cookie_key in reqheader_values:
  #   return True, "out"
  # if cookie_key in respheader_values:
  #   return True, "in"

  if check_in_header(cookie_value, reqheader_values, threshold):
    return True, "out"
  if check_in_header(cookie_value, respheader_values, threshold):
    return True, "in"
  return False, None


def check_if_cookie_value_exists_content(cookie_key, cookie_value, json_data, threshold):

  cookie_value = str(cookie_value)
  if (len(cookie_value) <= 5):
    return False, None

  try:
    for data in json_data: 
      key = list(data.keys())[0]
      if (cookie_value == data[key]):
          return True, key
      if not isinstance(data[key], str):
          continue
      if(len(data[key]) < 5):
          continue
      difference = abs(len(cookie_value) - len(data[key]))
      if difference > 10:
          continue
      offset = len(cookie_value) - difference if len(cookie_value) > len(
          data[key]) else len(data[key])-difference
      for i in range(0, difference+1):
          if len(data[key]) > len(cookie_value):
              if (compare_with_obfuscation(cookie_value, data[key][i:i+offset], threshold)):
                  return True, key
          else:
              if (compare_with_obfuscation(cookie_value[i:i+offset], data[key][0], threshold)):
                  return True, key
  except:
    return False, None
  return False, None

def check_if_cookie_value_exists(cookie_key, cookie_value, param_dict, threshold):
    
    cookie_value = str(cookie_value)
    if (len(cookie_value) <= 5):
        return False, None

    for key in param_dict:
        if (cookie_key == param_dict[key]):
            return True, key
        if(len(param_dict[key]) < 5):
            continue
        difference = abs(len(cookie_value) - len(param_dict[key]))
        if difference > 10:
            continue
        offset = len(cookie_value) - difference if len(cookie_value) > len(
            param_dict[key]) else len(param_dict[key])-difference
        for i in range(0, difference+1):
            if len(param_dict[key]) > len(cookie_value):
                if (compare_with_obfuscation(cookie_value, param_dict[key][i:i+offset], threshold)):
                    return True, key
            else:
                if (compare_with_obfuscation(cookie_value[i:i+offset], param_dict[key], threshold)):
                    return True, key

    return False, None

def get_header_values(header_string):

  values = []
  try:
    headers = json.loads(header_string)
    for header in headers:
      if (str(header[0].lower()) != "cookie") and (str(header[0].lower()) != "set-cookie"):
        values.append(header[1])
  except Exception as e:
    #print(e, header_string)
    #traceback.print_exc()
    return values
  return values

def get_ls_name(name):

  try:
    parts = name.split("|$$|")
    if len(parts) == 3:
      return name.rsplit("|$$|", 1)[0]
  except:
    return name
  return name

def find_exfiltrations(G, df_graph, ldb):

  find_exfiltrations.df_edges = pd.DataFrame(columns=['visit_id', 'src', 'dst', 'atrr', 'time_stamp', 'direction'])

  try:
    
    df_cookie_set = df_graph[(df_graph['action'] == 'set') | \
                  (df_graph['action'] == 'set_js')].copy()
    df_cookie_set['cookie_value'] = df_cookie_set['attr'].apply(get_cookieval)
    df_cookie_set = df_cookie_set[['dst', 'cookie_value']].drop_duplicates()
    df_cookie_set = df_cookie_set.rename(columns={'dst' : 'cookie_key'})
    cookie_names = df_cookie_set['cookie_key'].unique().tolist()

    #Check LS with same cookie name
    df_ls_set = df_graph[(df_graph['action'] == 'set_storage_js')].copy()
    df_ls_set['split_name'] = df_ls_set['dst'].apply(get_ls_name)
    df_ls_set = df_ls_set[df_ls_set['split_name'].isin(cookie_names)]
    df_ls_set['cookie_value'] = df_ls_set['attr'].apply(get_cookieval)
    df_ls_set = df_ls_set[['dst', 'cookie_value']].drop_duplicates()
    df_ls_set = df_ls_set.rename(columns={'dst' : 'cookie_key'})

    # df_requests = df_graph[(df_graph['graph_attr'] == 'Node') & \
    #                     ((df_graph['type'] == 'Request') | \
    #                     (df_graph['type'] == 'Script') | \
    #                     (df_graph['type'] == 'Document'))]
    
    df_decorations = df_graph[(df_graph['graph_attr'] == 'Node') & \
                        ((df_graph['type'] == 'Decoration'))]
  

    # df_headers = df_graph[(df_graph['reqattr'].notnull()) & (df_graph['reqattr'] != "N/A")]

    # df_post_bodies = df_graph[(df_graph['post_body'].notnull()) | (df_graph['post_body_raw'].notnull())]

    # df_content = df_graph[(df_graph['content_hash'].notnull()) | (df_graph['content_hash'] != 'N/A')]
    # df_content['content'] = df_content['content_hash'].apply(get_response_content, ldb=ldb)
    # df_content = df_content[df_content['content'].notnull()]

    def process_cookie(cookie_row):

      cookie_key = cookie_row['cookie_key']
      cookie_value = cookie_row['cookie_value']
      cookie_key_stripped = cookie_key.split("|$$|")[0].strip()

      def process_decoration(row):
         key = row['name']
         value = json.loads(row['attr'])['value']
         value_dict = {key : value}

        #  print(value_dict)
         
         exists, key = check_if_cookie_value_exists(cookie_key_stripped, cookie_value, value_dict, 2)

         if exists:
            find_exfiltrations.df_edges = find_exfiltrations.df_edges.append({'visit_id' : row['visit_id'], 'src' : cookie_key, 'dst' : row['name'], 'dst_domain' : row['domain'], 'attr': {
              'src_attr' : cookie_value, 'dst_attr':row['attr']}, 'time_stamp' : row['time_stamp'], 'direction' : 'out', 'type' : 'decoration'}, ignore_index=True)


      # def process_request(row):
      #   header_cookies = {}
      #   url_parameters = parse_url_arguments(row['name'])
      #   values_dict = {**header_cookies, **url_parameters}
      #   exists, key = check_if_cookie_value_exists(cookie_key_stripped, cookie_value, values_dict, 2)
      #   if exists:
      #     find_exfiltrations.df_edges = find_exfiltrations.df_edges.append({'visit_id' : row['visit_id'], 'src' : cookie_key, 'dst' : row['name'], 'dst_domain' : row['domain'],
      #       'attr' : cookie_value, 'time_stamp' : row['time_stamp'], 'direction' : 'out', 'type' : 'url'}, ignore_index=True)

      # def process_header(row):

      #   reqheader_values = get_header_values(row['reqattr'])
      #   respheader_values = get_header_values(row['respattr'])
       
      #   if (len(reqheader_values) > 0) & (len(respheader_values) > 0):
      #     exists, direction = check_if_cookie_value_exists_header(cookie_key_stripped, cookie_value, reqheader_values, respheader_values, 2)
      #     if exists:
      #       if direction == "in":
      #         find_exfiltrations.df_edges = find_exfiltrations.df_edges.append({'visit_id' : row['visit_id'], 'src' : cookie_key, 'dst' : row['dst'], 'dst_domain' : row['dst_domain'],
      #           'attr' : cookie_value, 'time_stamp' : row['time_stamp'], 'direction' : 'in', 'type' : 'header'}, ignore_index=True)
      #       else:
      #         find_exfiltrations.df_edges = find_exfiltrations.df_edges.append({'visit_id' : row['visit_id'], 'src' : cookie_key, 'dst' : row['dst'], 'dst_domain' : row['dst_domain'],
      #           'attr' : cookie_value, 'time_stamp' : row['time_stamp'], 'direction' : 'out', 'type' : 'header'}, ignore_index=True)            

      # def process_post_bodies(row):

      #   exists = False
      #   body_value = ""
      #   if (row['post_body']) and (row['post_body'] != "CS"):
      #     body_value = row['post_body']
      #   elif row['post_body_raw'] and (row['post_body_raw'] != "CS"):
      #     try:
      #       body = json.loads(row['post_body_raw'])
      #       if len(body) > 0:
      #         body_value = body[0][1]
      #         body_value = base64.b64decode(body_value).decode()
      #     except:
      #       traceback.print_exc()
      #       body_value = ""
      #   if len(body_value) > 1:
      #     exists, direction = check_if_cookie_value_exists_post_body(cookie_key_stripped, cookie_value, body_value, 2)
      #   if exists:
      #     find_exfiltrations.df_edges = find_exfiltrations.df_edges.append({'visit_id' : row['visit_id'], 'src' : cookie_key, 'dst' : row['dst'], 'dst_domain' : row['dst_domain'],
      #       'attr' : cookie_value, 'time_stamp' : row['time_stamp'], 'direction' : 'out', 'type' : 'postbody'}, ignore_index=True)

      # def process_response_content(row):

      #   exists = False
      #   content = row['content']
      #   json_data = []
      #   try:
      #     for d in extract_data(content):
      #       json_data.append(d)
      #   except:
      #     pass

      #   if (len(json_data) > 0) & (cookie_value is not None):     
      #     exists, key = check_if_cookie_value_exists_content(cookie_key_stripped, cookie_value, json_data, 2)
      #     if exists:
      #       find_exfiltrations.df_edges = find_exfiltrations.df_edges.append({'visit_id' : row['visit_id'], 'src' : cookie_key, 'dst' : row['dst'], 'dst_domain' : row['dst_domain'],
      #       'attr' : cookie_value, 'time_stamp' : row['time_stamp'], 'direction' : 'in', 'type' : 'content'}, ignore_index=True)
          
      # df_requests.apply(process_request, axis=1)
      # df_headers.apply(process_header, axis=1)
      # df_post_bodies.apply(process_post_bodies, axis=1)
      # if len(df_content) > 0:
      #   df_content.apply(process_response_content, axis=1)
      df_decorations.apply(process_decoration, axis=1)
    df_cookie_set.apply(process_cookie, axis=1)
    df_ls_set.apply(process_cookie, axis=1)

  except Exception as e:
    print("Error in exfiltration edges")
    traceback.print_exc()
    return find_exfiltrations.df_edges

  return find_exfiltrations.df_edges

def get_common_storage_alt(df):

  df = df.reset_index()
  df = df.sort_values(by='time_stamp')
  data = []

  names = df['dst'].unique().tolist()
  cookie_names = []
  ls_names = []
  for name in names:
    if len(name.split("|$$|")) == 3:
      ls_names.append(name)
    else:
      cookie_names.append(name)

  for cn in cookie_names:
    for lsn in ls_names:
      src_script = df.iloc[0]['src']
      vid = df.iloc[0]['visit_id']
      ts = df.iloc[0]['time_stamp']
      data.append([vid, cn, lsn, src_script, ts, 'local'])
  df_edges = pd.DataFrame(data, columns=['visit_id', 'src', 'dst', 'attr', 'time_stamp', 'direction'])
  return df_edges  

def get_common_storage(df):

  df = df.reset_index()
  df = df.sort_values(by='time_stamp')
  data = []

  if len(df) > 1:
    for i in range(1, len(df)):
      src_entry = df.iloc[i-1]
      dst_entry = df.iloc[i]
      if src_entry['dst'] != dst_entry['dst']:
        data.append([src_entry['visit_id'], src_entry['dst'], dst_entry['dst'], src_entry['src'], src_entry['time_stamp'], 'local'])

  df_edges = pd.DataFrame(data, columns=['visit_id', 'src', 'dst', 'attr', 'time_stamp', 'direction'])
 
  return df_edges

def find_local_storage(G, df_graph):

  df_edges = pd.DataFrame(columns=['visit_id', 'src', 'dst', 'attr', 'time_stamp', 'direction'])
  try:
    df_sets = df_graph[(df_graph['action'].values == 'set') | \
                  (df_graph['action'].values == 'set_js') | \
                  (df_graph['action'] == 'set_storage_js')].copy()
    df_edges = df_sets.groupby(['src'], as_index=False).apply(get_common_storage_alt)
    if len(df_edges) > 0:
      df_edges = df_edges[['visit_id', 'src', 'dst', 'attr', 'time_stamp', 'direction']].reset_index()
    else:
      df_edges = pd.DataFrame(columns=['visit_id', 'src', 'dst', 'attr', 'time_stamp', 'direction'])
  except:
    print("Error in local storage edges")
    traceback.print_exc()
  return df_edges

