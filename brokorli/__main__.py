import torch
import numpy as np
import random
import argparse
import logging

from brokorli import KoRIA
    
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
    parser.add_argument("--type", choices=["train", "task_demo", "cli"])
    parser.add_argument("--task_name", type=str, default="qg", choices=["srl", "ner", "mrc", "sm", "qg"])
    
    args = parser.parse_args()
    
    fix_torch_seed()
    
    koria = KoRIA(
        cfg_path=args.cfg_path, 
        cfg_fn=args.cfg_fn,
        run_type=args.type,
        task_name=args.task_name
    )
    
    if args.type == "train":
        koria.train()
    elif args.type == "task_demo":
        koria.task_demo()
    elif args.type == "cli":
        koria.cli()
    else:
        raise NotImplementedError()
    