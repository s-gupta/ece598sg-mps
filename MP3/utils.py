import os
import pickle

def save_variables(pickle_file_name, var, info, overwrite=False):
  if os.path.exists(pickle_file_name) and overwrite == False:
    raise Exception('{:s} exists and overwrite is false.'.format(pickle_file_name))
  # Construct the dictionary
  assert(type(var) == list); assert(type(info) == list);
  d = {}
  for i in range(len(var)):
    d[info[i]] = var[i]
  with open(pickle_file_name, 'wb') as f:
    pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

def load_variables(pickle_file_name):
  if os.path.exists(pickle_file_name):
    with open(pickle_file_name, 'rb') as f:
      d = pickle.load(f)
    return d
  else:
    raise Exception('{:s} does not exists.'.format(pickle_file_name))
