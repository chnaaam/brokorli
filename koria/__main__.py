import torch
import numpy as np
import random
import argparse
import logging

logger = logging.getLogger("koria")

from koria import KoRIA

def init_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s|%(levelname)s] > %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
def fix_torch_seed(random_seed=42):
    torch.manual_seed(random_seed)

    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    np.random.seed(random_seed)
    random.seed(random_seed)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cfg_path", type=str, default="./")
    parser.add_argument("--cfg_fn", type=str, default="koria.cfg")
    
    parser.add_argument("--type", choices=["train", "predict"])
    
    # Train specific task
    parser.add_argument("--task_name", type=str, default="srl", choices=["srl", "ner"])
    
    args = parser.parse_args()
    
    init_logger()
    fix_torch_seed()
    
    logger.info("KoRIA package")
    
    koria = KoRIA(
        cfg_path=args.cfg_path, 
        cfg_fn=args.cfg_fn
    )
    
    if args.type == "train":
        koria.train(task_name=args.task_name)
    # elif args.type == "predict":
    #     koria.predict()
    else:
        raise NotImplementedError()
    