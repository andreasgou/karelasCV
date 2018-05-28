# File system helper funtions
import pickle
import os

# Deserialize object from disk
def read_object(path):
    f = open(path, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

# Serialize object to disk
def write_object(path, obj):
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()

# Create path if not exists
def get_create_path(path):
    if os.path.exists(path):
        print("Path {} exists.", format(path))
    else:
        print("Path {} does not exists and it will be created.", format(path))
        os.makedirs(path)
        