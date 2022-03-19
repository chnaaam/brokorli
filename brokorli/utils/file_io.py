import os


def make_dir(dir):
    if not is_existed_dir(dir):
        os.mkdir(dir)

def is_existed_dir(dir):
    if os.path.isdir(dir):
        return True
    
    return False

def is_existed_file(file_full_path):
    if os.path.isfile(file_full_path):
        return True
    
    return False

def save_label_file(path, data):
    pass

def load_label_file(path):
    with open(path):
        pass