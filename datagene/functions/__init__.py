import torch.nn as nn
import torch.optim as optim

CRITERION_LIST = {
    "cross-entropy": nn.CrossEntropyLoss
}

OPTIMIZER_LIST = {
    "adam": optim.Adam,
    "adamw": optim.AdamW
}

SCHEDULER_LIST = {
    "lambda": optim.lr_scheduler.LambdaLR
}