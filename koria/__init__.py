import torch.optim as optim

MODEL_NAME_LIST = {
    "bert": "monologg/kobert",
    "electra": "monologg/koelectra-base-v3-discriminator",
    "charelectra": "monologg/kocharelectra-base-discriminator",
    "roberta": "klue/roberta-base"
}

OPTIMIZER_LIST = {
    "adam": optim.Adam,
    "adamw": optim.AdamW
}


from .koria import KoRIA