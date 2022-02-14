import os

from .config import get_data_gene_config
from .tasks import TASKS
from .tokenizers import TOKENIZERS
# from .tokenizers.word_piece_tokenizer import WordPieceTokenizer

class DataGene:
    """
    DataGene
    
    """
    def __init__(self, cfg_path, cfg_fn):
        cfg_full_path = os.path.join(cfg_path, cfg_fn)
        
        if not os.path.exists(cfg_full_path):
            raise OSError("Configuration directory or file is not existed")
        
        self.cfg = get_data_gene_config(cfg_path=cfg_path, cfg_fn=cfg_fn)

    def train(self, task_name):
        task = TASKS[task_name]
        
        # model_name = task["model"]
        # tokenizer_name = task["tokenizer"]
        
        # model = TASKS[model_name]
        # tokenizer = TOKENIZERS[tokenizer_name]
        
        
    
    def predict(self):
        pass