from .tokenization_kobert import KoBertTokenizer
from .tokenization_kocharelectra import KoCharElectraTokenizer
from transformers import AutoTokenizer

TOKENIZER_LIST = {
    "bert": KoBertTokenizer.from_pretrained("monologg/kobert"),
    "electra": AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator"),
    "charelectra": KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator"),
    "roberta": AutoTokenizer.from_pretrained("klue/roberta-base")
}