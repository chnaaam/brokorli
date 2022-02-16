import os
import yaml


class ConfigBase:    
    """
    ConfigBase
    YAML 파일을 클래스화 합니다.
    """
    
    def __init__(self, data):
        for key, value in data.items():
            if type(value) is dict:
                self.__setattr__(key, ConfigBase(value))
            else:
                self.__setattr__(key, value)


def get_data_gene_config(cfg_path, cfg_fn):
    """
    지정된 위치의 Configuration 파일을 불러와서 클래스화 한 후, 리턴 합니다.

    Args:
        cfg_path (str): Configuration 파일이 있는 디렉터리
        cfg_fn (str): Configuration 파일 이름

    Returns:
        클래스화된 Configuration 
        
    Example:
        cfg = get_data_gene_config(config_path="./", config_file="datagene.cfg")
        
        print(cfg.parameters)
    """
    config_full_path = os.path.join(cfg_path, cfg_fn)

    with open(config_full_path) as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)

    return ConfigBase(data=data)