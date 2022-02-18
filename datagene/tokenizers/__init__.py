from .tokenization_kobert import KoBertTokenizer
from transformers import ElectraTokenizer

TOKENIZER_LIST = {
    "bert": KoBertTokenizer.from_pretrained("monologg/kobert"),
    "electra": ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
}