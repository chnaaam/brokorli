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


def get_data_gene_config(cfg_path, cfg_fn):
    """
    Load configuration file

    Args:
        cfg_path (str): Configuration directory
        cfg_fn (str): Configuration file name

    Returns:
        ConfigBase Class
        
    Example:
        cfg = get_data_gene_config(config_path="./", config_file="datagene.cfg")
        
        print(cfg.parameters)
    """
    config_full_path = os.path.join(cfg_path, cfg_fn)

    with open(config_full_path, "r", encoding="utf-8") as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)

    return ConfigBase(data=data)