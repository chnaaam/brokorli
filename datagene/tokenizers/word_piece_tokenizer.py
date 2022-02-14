from .tokenizer import Tokenizer

class WordPieceTokenizer:
    def __init__(self, type):
        self.tokenizer = Tokenizer(type=type)
        
    def __call__(self, sentence, predicate, arguments):
        pass