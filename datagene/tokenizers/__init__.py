from .tokenization_kobert import KoBertTokenizer


TOKENIZERS = {
    "word-piece": KoBertTokenizer.from_pretrained("monologg/kobert")
}