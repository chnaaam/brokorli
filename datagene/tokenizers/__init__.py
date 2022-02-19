from .tokenization_kobert import KoBertTokenizer
from .tokenization_kocharelectra import KoCharElectraTokenizer
from transformers import AutoTokenizer

SPECIAL_TOKEN_LIST = {
    "srl": {
        "predicate_begin": "<PREDICATE>",
        "predicate_end": "</PREDICATE>"
    }
}

TOKENIZER_LIST = {
    "bert": KoBertTokenizer,
    "electra": AutoTokenizer,
    "charelectra": KoCharElectraTokenizer,
    "roberta": AutoTokenizer
}