import pandas as pd
from urllib.parse import unquote
import numpy as np
import json
import os
import datetime

def get_value(row):

	value = 'N/A'
	attr_dict = {}
	try:
		attr = row['attr']
		if not pd.isna(attr):
			attr_dict = json.loads(attr)
		else:
			attr = row['attr_x']
			if not pd.isna(attr):
				attr_dict = json.loads(attr)
		new_attr_dict = {}
		for k, v in attr_dict.items():
			new_attr_dict[k.strip().lower()] = v
		value = new_attr_dict['value']
	except Exception as e:
		print(e)
		return value
	return value

def get_expiry(attr):

	expiry = None
	try:
		attr_dict = json.loads(attr)
		new_attr_dict = {}
		for k, v in attr_dict.items():
			new_attr_dict[k.strip().lower()] = v
		expiry = new_attr_dict.get('expires')
	except Exception as e:
		print(e)
		return expiry
	return expiry

def check_identifier(df):

	#things to check
	# incomplete measurements -- '' in domain, name, value
	# expiry -- < 1 month expiry, session cookies
	# values between 6 and 100 char
	# constant value across crawl

	num_days = 30
	
	try:

		vid = df['visit_id'].iloc[0]
		name = df['dst'].iloc[0]

		values = df['value'].unique().tolist()

		#if (len(values) > 1):
		#	return (vid, name, False)
		value = values[0]
		if (len(value) < 6) or (len(value) > 100):
			return (vid, name, False)
		df_sorted = df.sort_values(by=['time_stamp'])
		access = df_sorted.iloc[0]['time_stamp']
		expiry = df_sorted.iloc[0]['expiry']

		print(name, expiry)

		if (expiry is None) or (expiry == ''):
			return (vid, name, False)

		access = datetime.datetime.strptime(access, "%Y-%m-%dT%H:%M:%S.%fZ")
		expiry = datetime.datetime.strptime(expiry, "%a, %d %b %Y %H:%M:%S GMT")

		if (expiry - access).days < num_days:
			return (vid, name, False)
		
		return (vid, name, True)

	except Exception as e:
		print(e)

def get_single_value_ls(df):

	try:
		vid = df['visit_id'].iloc[0]
		name = df['split_name'].iloc[0]
		label = df['label'].iloc[0]
		values = df['value'].unique().tolist()
		values = [unquote(x).strip() for x in values]
		#if label == 'Positive':
		if len(values) <= 2:
			if ('true' in values) or ('false') in values:
				if label == "Positive":
					return (vid, name, values[0], "Binary")
				else:
					return (vid, name, values[0], "NegBinary")
		value_len = [len(x) for x in values if x != 'N/A']
		mean_len = np.mean(value_len)
		if mean_len <= 1:
			if label == "Positive":
				return (vid, name, values[0], "Binary")
			else:
				return (vid, name, values[0], "NegBinary")

		return (vid, name, values[0], "Multi")
		#else:
		#	return (vid, name, False)
	except Exception as e:
		print(df)

def get_single_value(df):

	try:
		vid = df['visit_id'].iloc[0]
		name = df['dst'].iloc[0]
		label = df['label'].iloc[0]
		values = df['value'].unique().tolist()
		values = [unquote(x).strip() for x in values]
		#if label == 'Positive':
		if len(values) <= 2:
			if ('true' in values) or ('false') in values:
				if label == "Positive":
					return (vid, name, values[0], "Binary")
				else:
					return (vid, name, values[0], "NegBinary")
		value_len = [len(x) for x in values if x != 'N/A']
		mean_len = np.mean(value_len)
		if mean_len <= 1:
			if label == "Positive":
				return (vid, name, values[0], "Binary")
			else:
				return (vid, name, values[0], "NegBinary")

		return (vid, name, values[0], "Multi")
		#else:
		#	return (vid, name, False)
	except Exception as e:
		print(df)

def identifier_cookies(df_graph, df_labelled):

	df_labelled = df_labelled[['name', 'visit_id']]
	df_merge = df_graph.merge(df_labelled, left_on=['visit_id', 'dst'], right_on=['visit_id', 'name'])
	df_merge = df_merge[(df_merge['action'] == 'set_js')]
	df_merge['value'] = df_merge['attr'].apply(get_value)
	df_merge['expiry'] = df_merge['attr'].apply(get_expiry)

	df_identifier = df_merge.groupby(['visit_id', 'dst']).apply(check_identifier).reset_index()
	identifier_data = df_identifier[0].tolist()
	df_identifier = pd.DataFrame(identifier_data, columns=['visit_id', 'name', 'identifier'])
	df_identifier = df_identifier[df_identifier['identifier'] == True]

	return df_identifier

def single_value_cookies(df_graph, df_labelled):

	df_merge = df_graph.merge(df_labelled, left_on=['visit_id', 'dst'], right_on=['visit_id', 'name'])
	df_merge = df_merge[(df_merge['action'] == 'set_js') | (df_merge['action'] == 'set')]
	df_merge['value'] = df_merge.apply(get_value, axis=1)

	df_single = df_merge.groupby(['visit_id', 'dst']).apply(get_single_value).reset_index()
	single_data = df_single[0].tolist()
	df_single = pd.DataFrame(single_data, columns=['visit_id', 'name', 'value', 'single'])

	#df_multi = df_single[df_single['single'] == False]
	
	return df_single

def get_ls_name(name):

  try:
    parts = name.split("|$$|")
    if len(parts) == 3:
      return name.rsplit("|$$|", 1)[0]
  except:
    return name
  return name


def single_value_cookies_ls(df_graph, df_labelled):

	df_sets = df_graph[(df_graph['action'] == 'set_js') | (df_graph['action'] == 'set') | (df_graph['action'] == 'set_storage_js')]
	df_sets['split_name'] = df_sets['dst'].apply(get_ls_name)
	df_merge = df_sets.merge(df_labelled, left_on=['visit_id', 'split_name'], right_on=['visit_id', 'name'])
	df_merge['value'] = df_merge.apply(get_value, axis=1)

	df_single = df_merge.groupby(['visit_id', 'split_name']).apply(get_single_value_ls).reset_index()
	single_data = df_single[0].tolist()
	df_single = pd.DataFrame(single_data, columns=['visit_id', 'name', 'value', 'single'])

	return df_single


if __name__ == "__main__":

	df_allmulti = pd.DataFrame()

	#Folder of graph files
	graph_folder = "/home/siby/webgraph_optimized/run9_data/graph_data"
	#Processed label data
	labelled_file = "/home/siby/webgraph_optimized/newlabels_processed.pkl"
	df_labelled = pd.read_pickle(labelled_file)

	for i in range(1, 11):
		
		print("processing:", i) #Change as required to read graph files
		graph_file = os.path.join(graph_folder, "graph_" + str(i) + ".csv")
		df_graph = pd.read_csv(graph_file)
		
		df_multi = single_value_cookies_ls(df_graph, df_labelled)
		df_allmulti = df_allmulti.append(df_multi)

		#break

	#df_allmulti = df_allmulti.append(df_other)
	print(df_allmulti.shape)

	df_allmulti.to_pickle("multi_cookies.pkl")


