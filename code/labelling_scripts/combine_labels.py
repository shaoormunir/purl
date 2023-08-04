import pandas as pd

def get_combined_label(row):

  setter_label = row['setter_label']
  declared_label = row['declared_label']
  
  try:
    if (pd.notna(declared_label)) and (declared_label == 3):
      return 'Positive'
    elif setter_label == False:
      return 'Negative'
  except:
    return 'Unknown'
  return 'Unknown'

def label_storage_nodes(df_setters, df_declared):

  try:
    df_combined_data = pd.merge(df_setters, df_declared, on=['visit_id', 'name'], how='outer')
    df_combined_data['label'] = df_combined_data.apply(get_combined_label, axis=1)
    df_combined_data = df_combined_data.drop_duplicates()
  except:
    return pd.DataFrame()

  return df_combined_data

