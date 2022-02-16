from .tokenization_kobert import KoBertTokenizer

TOKENIZER_LIST = {
    "word-piece": KoBertTokenizer.from_pretrained("monologg/kobert")
}