import torch.optim as optim

MODEL_NAME_LIST = {
    "bert": "bert-base-multilingual-cased",
    "kobert": "monologg/kobert",
    "koelectra": "monologg/koelectra-base-v3-discriminator",
    "kocharelectra": "monologg/kocharelectra-base-discriminator",
    "klueroberta": "klue/roberta-base"
}

OPTIMIZER_LIST = {
    "adam": optim.Adam,
    "adamw": optim.AdamW
}


from .koria import KoRIA