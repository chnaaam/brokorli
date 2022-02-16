from . import TokenizerFactories

class Tokenizer:
    def __init__(self, type):
        self.tokenizer = TokenizerFactories[type]
        
    def tokenize(self, data):
        return self.tokenizer.tokenize(data)