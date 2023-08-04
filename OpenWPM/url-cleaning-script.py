import os
import json
import shutil

root = 'urls'

def check_if_json_is_empty(file_name):
    # load json file
    # surround with try/except to handle file not found
    try:
      with open(file_name, 'r') as f:
          data = json.load(f)
          keys = list(data.keys())
          urls = data[keys[0]]

          if len(urls) == 0:
              return True
          else:
              return False
    except:
      return True
    

for path, subdirest, files in os.walk(root):
    for file in files:
      if (check_if_json_is_empty(os.path.join(path, file))):
        print(os.path.join(path, file))
        os.remove(os.path.join(path, file))
        os.rmdir(path)
