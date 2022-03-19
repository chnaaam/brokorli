import os
import yaml


class ConfigBase:    
    """
    ConfigBase
    Change the YAML data to class
    """
    
    def __init__(self, data):
        for key, value in data.items():
            if type(value) is dict:
                self.__setattr__(key, ConfigBase(value))
            else:
                self.__setattr__(key, value)


def get_config(cfg_path):
    
    data = None
    with open(cfg_path, "r", encoding="utf-8") as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
    
    if data:
        return ConfigBase(data=data)
    else:
        return None