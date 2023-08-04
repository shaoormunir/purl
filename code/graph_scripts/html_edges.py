import re
import json
import pandas as pd
import traceback
import numpy as np

def check_tag_dict(row, tagdict):
		
	"""Function to check if an element is inline."""
	vid = row['visit_id']
	url = row['script_url']

	try:
		if tagdict.get(vid).get(url):
			return "Present"
		else:
			return "Absent"
	except Exception as e:
		return "Absent"
	
def get_inline_name(row, tagdict):
		
	"""Function to set name of an inline script using its location in a file."""
	vid = row['visit_id']
	url = row['script_url']
	line = row['script_line']

	try:
		name = ""
		line_tags = tagdict.get(vid).get(url)
		if line_tags:
			for item in line_tags:
				if line >= item[0] and line < item[1]:
					name = "inline_" + url + "_" + str(item[0]) + "_" + str(item[1])
					break
			if len(name) == 0:
				name = "inline_" + url + "_0_0"
		else:
			name = "inline_" + url + "_0_0"
		return name
	except Exception as e:
		name = "inline_" + url + "_0_0"
		return name
	
def convert_attr(row):
		
	"""Function to convert attribute of an element."""
	attr = {}
	try:
		attr["openwpm"] = json.loads(row['attributes'])["0"]["openwpm"]
		attr['subtype'] = row['subtype_list']
		if row['script_loc_eval'] != "":
			attr['eval'] = True
		else:
			attr['eval'] = False
		attr = json.dumps(attr)
		return attr
	except Exception as e:
		print(e)
		return json.dumps(attr)

def convert_subtype(x):
		
	"""Function to convert subtype of an element."""

	try:
		return json.loads(x)[0]
	except Exception as e:
		return ""

def get_eval_name(script, line):
		
	"""Function to get name of an eval node based on the location in a script."""

	try:
		return line.split()[3] + "_" + line.split()[1] + "_" + script
	except:
		return "Eval_error" 

def get_tag(record, key):
	try:
		val = json.loads(record)

		if key == "fullopenwpm":
			openwpm = val.get("0").get("openwpm")
			return str(openwpm)
		else:
			return str(val.get(key))
	except Exception as e:
		return ""
	return ""
	
def get_attr_action(symbol):
		
	"""Function to find attribute modification (new, deleted, changed)."""

	try:
		result = re.search('.attr_(.*)', symbol)
		return "attr_" + result.group(1)
	except Exception as e:
		return ""

def find_parent_elem(src_elements, df_element):
	
	"""Function to find parent elements."""
	
	src_elements['new_attr'] = src_elements['attributes'].apply(get_tag, key="fullopenwpm")
	df_element['new_attr'] = df_element['attr'].apply(get_tag, key="openwpm")
	result = src_elements.merge(df_element[['new_attr', 'name']], on='new_attr', how='left')
	return result


def find_modified_elem(df_element, df_javascript):
	
	"""Function to find modified elements where the attribute has changed."""
	
	df_javascript['new_attr'] = df_javascript['attributes'].apply(get_tag, key="openwpm")
	df_element['new_attr'] = df_element["attr"].apply(get_tag, key="openwpm")
	result = df_javascript.merge(df_element[['new_attr', 'name']], on='new_attr', how='left')
	return result


def build_html_components(df_javascript):

	df_js_nodes = pd.DataFrame()
	df_js_edges = pd.DataFrame()

	try:
		#Find all created elements
		created_elements = df_javascript[df_javascript['symbol'] == 'window.document.createElement'].copy()
		created_elements['name'] = created_elements.index.to_series().apply(lambda x: "Element_" + str(x))
		created_elements['type'] = 'Element'

		created_elements['subtype_list'] = created_elements['arguments'].apply(convert_subtype)
		#created_elements['attr'] = created_elements[['attributes', 'subtype_list']].apply(
		#				lambda x: convert_attr(*x), axis=1)
		created_elements['attr'] = created_elements.apply(convert_attr, axis=1)
		created_elements['action'] = 'create'

		# #Get evals from the createElement calls
		# eval_elements = created_elements[created_elements['script_loc_eval'] != ""].copy()
		# if len(eval_elements) > 0:
		# 	eval_elements['eval_name'] = eval_elements[['script_url', 'script_loc_eval']].apply(
		# 					lambda x: get_eval_name(*x), axis=1)
		# 	eval_elements['eval_type'] = "Script"
		# 	eval_elements['eval_attr'] = "eval"
		# 	eval_elements['eval_action'] = "eval"

		# 	#Eval nodes and edges (to be inserted)
		# 	df_eval_nodes = eval_elements[['visit_id', 'eval_name', 'eval_type', 'eval_attr']].drop_duplicates()
		# 	df_eval_nodes = df_eval_nodes.rename(columns={'eval_name': 'name', 'eval_type': 'type',
		# 		'eval_attr': 'attr'})
		# 	df_script_eval_edges = eval_elements[['visit_id', 'script_url', 'eval_name', 'eval_action', 'time_stamp']].drop_duplicates()
		# 	df_script_eval_edges = df_script_eval_edges.rename(
		# 					columns={'eval_action' : 'action', 'script_url' : 'src', 'eval_name' : 'dst'})

		# 	#Created element edges (with eval parents)
		# 	df_eval_element_edges = eval_elements[['visit_id', 'eval_name', 'name', 'action', 'time_stamp']]
		# 	df_eval_element_edges = df_eval_element_edges.rename(columns={'eval_name': 'src', 'name': 'dst'})
		# else:
		# 	df_eval_nodes = pd.DataFrame()
		# 	df_script_eval_edges = pd.DataFrame()
		# 	df_eval_element_edges = pd.DataFrame()
			
		# #Created Element nodes and edges (to be inserted)
		# created_non_eval_elements = created_elements[created_elements['script_loc_eval'] == ""].copy()
		# df_script_created_edges = created_non_eval_elements[['visit_id', 'script_url', 'name', 'action', 'time_stamp']]
		# df_script_created_edges = df_script_created_edges.rename(columns={'script_url' : 'src', 'name' : 'dst'})

		#Created Element nodes and edges (to be inserted)
		df_element_nodes = created_elements[['visit_id', 'name', 'top_level_url', 'type', 'attr']]
		df_create_edges = created_elements[['visit_id', 'script_url', 'name', 'top_level_url', 'action', 'time_stamp']]
		df_create_edges = df_create_edges.rename(columns={'script_url' : 'src', 'name' : 'dst'})
		
		src_elements = df_javascript[(df_javascript['symbol'].str.contains("Element.src")) & (df_javascript['operation'].str.contains('set'))].copy()
		src_elements['type'] = "Request"
		src_elements = find_parent_elem(src_elements, df_element_nodes)
		src_elements['action'] = "setsrc"

		#Src Element nodes and edges (to be inserted)
		df_src_nodes = src_elements[['visit_id', 'value', 'top_level_url', 'type', 'attributes']].copy()
		df_src_nodes = df_src_nodes.rename(columns={'value': 'name', 'attributes': 'attr'})
		df_src_nodes = df_src_nodes.dropna(subset=["name"])
		
		#df_src_nodes = df_src_nodes['attr'].groupby(['visit_id', 'name', 'type']).apply(list) 
		#df_src_nodes['attr'] = df_src_nodes['attr'].apply(lambda x: json.dumps(x))
		
		df_src_edges = src_elements[['visit_id', 'name', 'value', 'top_level_url', 'action', 'time_stamp']]
		df_src_edges = df_src_edges.dropna(subset=["name"])
		df_src_edges = df_src_edges.rename(columns={'name': 'src', 'value': 'dst'})
		
		#df_js_nodes = pd.concat([df_element_nodes, df_src_nodes, df_eval_nodes]).drop_duplicates()
		#df_js_edges = pd.concat([ df_script_created_edges, df_script_eval_edges, df_eval_element_edges, df_src_edges])
		df_js_nodes = pd.concat([df_element_nodes, df_src_nodes]).drop_duplicates()
		df_js_nodes = df_js_nodes.drop(columns=['new_attr'])
		df_js_edges = pd.concat([df_create_edges, df_src_edges])
		
		df_js_edges['reqattr'] = pd.NA
		df_js_edges['respattr'] = pd.NA
		df_js_edges['response_status'] = pd.NA
		df_js_edges['attr'] = pd.NA

	except Exception as e:
		print("Error in build_html_components:", e)
		traceback.print_exc()
		return df_js_nodes, df_js_edges

	return df_js_nodes, df_js_edges

def build_html_components_bk(df_javascript, tag_dict):

	df_js_nodes = pd.DataFrame()
	df_js_edges = pd.DataFrame()

	try:
	
		"""Function to create JS nodes and edges in WebGraph."""

		#Find all created elements
		created_elements = df_javascript[df_javascript['symbol'] == 'window.document.createElement'].copy()
		created_elements['name'] = created_elements.index.to_series().apply(lambda x: "Element_" + str(x))
		created_elements['type'] = 'Element'

		created_elements['subtype_list'] = created_elements['arguments'].apply(convert_subtype)
		created_elements['attr'] = created_elements[['attributes', 'subtype_list']].apply(
						lambda x: convert_attr(*x), axis=1)
		created_elements['action'] = 'create'
		
		#Get evals from the createElement calls
		eval_elements = created_elements[created_elements['script_loc_eval'] != ""]
		eval_elements['eval_name'] = eval_elements[['script_url', 'script_loc_eval']].apply(
						lambda x: get_eval_name(*x), axis=1)
		eval_elements['eval_type'] = "Script"
		eval_elements['eval_attr'] = "eval"
		eval_elements['eval_action'] = "eval"

		#Eval nodes and edges (to be inserted)
		df_eval_nodes = eval_elements[['visit_id', 'eval_name', 'eval_type', 'eval_attr']].drop_duplicates()
		df_eval_nodes = df_eval_nodes.rename(columns={'eval_name': 'name', 'eval_type': 'type',
			'eval_attr': 'attr'})

		eval_elements['tagdict_check'] = eval_elements.apply(check_tag_dict, args=(tag_dict,), axis=1)
		#Eval elements (with non-inline parents)
		eval_non_inline_elements = eval_elements[eval_elements['tagdict_check'] == "Absent"].copy()
		df_eval_non_inline_edges = eval_non_inline_elements[['visit_id', 'script_url', 'eval_name', 'eval_action', 'time_stamp']].drop_duplicates()
		df_eval_non_inline_edges = df_eval_non_inline_edges.rename(
						columns={'eval_action' : 'action', 'script_url' : 'src', 'eval_name' : 'dst'})

		eval_inline_elements = eval_elements[eval_elements['tagdict_check'] == "Present"]
		if len(eval_inline_elements) > 0:
			df_eval_inline = eval_inline_elements[['visit_id', 'script_url', 'script_line', 'eval_name', 'eval_action', 'time_stamp']].copy()
			df_eval_inline['inline_name'] = df_eval_inline.apply(get_inline_name, args=(tag_dict,), axis=1)
			df_eval_inline_nodes = df_eval_inline[['visit_id', 'inline_name']].drop_duplicates()
			df_eval_inline_nodes = df_eval_inline_nodes.rename(
						columns={'inline_name' : 'name'})
			df_eval_inline_nodes['type'] = 'Script'
			df_eval_inline_nodes['attr'] = 'inline'
		
			df_eval_inline_edges = df_eval_inline[['visit_id', 'script_url', 'inline_name', 'eval_action', 'time_stamp']]
			df_eval_inline_edges = df_eval_inline_edges.rename(
						columns={'script_url' : 'src', 'inline_name' : 'dst', 'eval_action': 'action'})
		
			#Eval element edges (with inline parents)
			df_eval_inline_parent_edges = df_eval_inline[['visit_id', 'inline_name', 'eval_name', 'eval_action', 'time_stamp']]
			df_eval_inline_parent_edges = df_eval_inline_parent_edges.rename(
						columns={'eval_name': 'dst', 'inline_name': 'src', 'eval_action': 'action'})

		else:
			df_eval_inline_nodes = pd.DataFrame()
			df_eval_inline_edges = pd.DataFrame()

		#Created Element nodes and edges (to be inserted)
		df_element_nodes = created_elements[['visit_id', 'name', 'type', 'attr']]
		created_elements['tagdict_check'] = created_elements.apply(check_tag_dict, args=(tag_dict,), axis=1)
		created_non_eval_inline_elements = created_elements[(created_elements['script_loc_eval'] == "") & (created_elements['tagdict_check'] == "Present")]
		created_non_eval_non_inline_elements = created_elements[(created_elements['script_loc_eval'] == "") & (created_elements['tagdict_check'] == "Absent")]
		df_created_non_inline_edges = created_non_eval_non_inline_elements[['visit_id', 'script_url', 'name', 'action', 'time_stamp']]
		df_created_non_inline_edges = df_created_non_inline_edges.rename(
						columns={'script_url' : 'src', 'name' : 'dst'})

		df_created_inline = created_non_eval_inline_elements[['visit_id', 'script_url', 'script_line', 'name', 'action', 'time_stamp']]
		df_created_inline['inline_name'] = df_created_inline.apply(get_inline_name, args=(tag_dict,), axis=1)
		
		df_created_inline_nodes = df_created_inline[['visit_id', 'inline_name']].drop_duplicates()
		df_created_inline_nodes = df_created_inline_nodes.rename(columns={'inline_name':'name'})
		df_created_inline_nodes['type'] = 'Script'
		df_created_inline_nodes['attr'] = 'inline'
		
		df_inline_edges = df_created_inline[['visit_id', 'script_url', 'inline_name', 'action', 'time_stamp']]
		df_inline_edges = df_inline_edges.rename(columns={'script_url': 'src', 'inline_name': 'dst'})
		
		#Created element edges (with inline parents)
		df_created_inline_edges = df_created_inline[['visit_id', 'inline_name', 'name', 'action', 'time_stamp']]
		df_created_inline_edges = df_created_inline_edges.rename(columns={'name': 'dst', 'inline_name': 'src'})

		#Created element edges (with eval parents)
		df_created_eval_edges = eval_elements[['visit_id', 'eval_name', 'name', 'action', 'time_stamp']]
		df_created_eval_edges = df_created_eval_edges.rename(columns={'eval_name': 'src', 'name': 'dst'})
		
		src_elements = df_javascript[(df_javascript['symbol'].str.contains("Element.src")) & (df_javascript['operation'].str.contains('set'))]
		src_elements['type'] = "Request"
		src_elements = find_parent_elem(src_elements, df_element_nodes)
		src_elements['action'] = "setsrc"

		#Src Element nodes and edges (to be inserted)
		df_src_nodes = src_elements[['visit_id', 'value', 'type', 'attributes']]
		df_src_nodes = df_src_nodes.rename(columns={'value': 'name', 'attributes': 'attr'})
		df_src_nodes = df_src_nodes.dropna(subset=["name"])
		df_src_nodes = df_src_nodes['attr'].groupby(['visit_id', 'name', 'type']).apply(list) #XX
		df_src_nodes['attr'] = df_src_nodes['attr'].apply(lambda x: json.dumps(x))
		
		df_src_edges = src_elements[['visit_id', 'name', 'value', 'action', 'time_stamp']]
		df_src_edges = df_src_edges.dropna(subset=["name"])
		df_src_edges = df_src_edges.rename(columns={'name': 'src', 'value': 'dst'})
		
		#Modified elements
		modified_elements = df_javascript[df_javascript['symbol'].str.contains("attr_")]
		df_modified = find_modified_elem(df_element_nodes, modified_elements)
		df_modified['action'] = df_modified['symbol'].apply(get_attr_action)
		df_modified_edges = df_modified[['visit_id', 'name', 'script_url', 'action', 'time_stamp']]
		df_modified_edges = df_modified_edges.rename(columns={'script_url': 'src', 'name': 'dst'})
		
		"""Node descriptions
		df_element_nodes: 
		df_src_nodes: Set src nodes 
		df_eval_nodes: Eval nodes
		df_eval_inline_nodes: Inline script nodes within an eval call
		df_created_inline_nodes: 
		"""

		"""Edge descriptions
		df_inline_edges: Between a script and an inline script node
		df_created_non_inline_edges: Between a script and a created element node
		df_created_inline_edges: Between an inline script node and a created element node
		df_eval_inline_edges: Between a script and an inline script node (when there is an eval child)
		df_eval_inline_parent_edges: Between an inline script node and an eval node
		df_eval_non_inline_edges: Between a script and an eval element
		df_created_eval_edges: Between an eval node and a created element node
		df_src_edges: Between a created element node and a set src node
		df_modified_edges: Between a URL and a created element node
		"""

		df_js_nodes = pd.concat([df_element_nodes, df_src_nodes, df_eval_nodes, 
									df_eval_inline_nodes, df_created_inline_nodes]).drop_duplicates()
		
		df_js_edges = pd.concat([df_inline_edges, df_created_non_inline_edges, 
									df_created_inline_edges, df_eval_inline_edges, df_eval_inline_parent_edges,
									df_eval_non_inline_edges, df_created_eval_edges, df_src_edges, df_modified_edges])
		
		df_js_edges['reqattr'] = pd.NA
		df_js_edges['respattr'] = pd.NA
		df_js_edges['response_status'] = pd.NA
		df_js_edges['attr'] = pd.NA
		df_js_edges['post_body'] = pd.NA
		df_js_edges['post_body_raw'] = pd.NA

	except Exception as e:
		print("Error in build_html_components:", e)
		traceback.print_exc()

	return df_js_nodes, df_js_edges
