import argparse

from datagene import DataGene

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cfg_path", type=str, default="./")
    parser.add_argument("--cfg_fn", type=str, default="datagene.cfg")
    
    parser.add_argument("--type", choices=["train", "predict"])
    
    # Train specific task
    parser.add_argument("--task", type=str, default="SRL", choices=["SRL", ])
    
    args = parser.parse_args()
    
    datagene = DataGene(
        cfg_path=args.cfg_path, 
        cfg_fn=args.cfg_fn
    )
    
    if args.type == "train":
        datagene.train(task=args.task)
    elif args.type == "predict":
        datagene.predict()
    else:
        raise NotImplementedError()
    