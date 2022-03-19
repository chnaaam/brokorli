MODEL_NAME_LIST = {
    "bert": "bert-base-multilingual-cased",
    "kobert": "monologg/kobert",
    "koelectra": "monologg/koelectra-base-v3-discriminator",
    "kocharelectra": "monologg/kocharelectra-base-discriminator",
    "klueroberta": "klue/roberta-base"
}




from .brokorli_unit import BrokorliUnit
from .brokorli import Brokorli
from .tasks import UserTemplate