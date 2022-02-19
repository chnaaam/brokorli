from .tokenization_kobert import KoBertTokenizer
from .tokenization_kocharelectra import KoCharElectraTokenizer
from transformers import AutoTokenizer

TOKENIZER_LIST = {
    "bert": KoBertTokenizer,
    "electra": AutoTokenizer,
    "charelectra": KoCharElectraTokenizer,
    "roberta": AutoTokenizer
}