from .tokenization_kobert import KoBertTokenizer
from .tokenization_kocharelectra import KoCharElectraTokenizer
from transformers import AutoTokenizer

"""
SPECIAL_TOKEN_LIST는 Task 별 학습 또는 추론에 필요한 토큰들이 정의되어 있습니다.
새로운 토큰을 추가하고자 하는 경우, 아래 형식에 맞게 값을 추가해주세요.
주의 사항
 - Special Token은 Tokenizer에서 사용되는 토큰들을 의미합니다.
 - Label(Class)는 data_loaders/dataset 내 각 Task Dataset 클래스 내부에서 정의해주세요.

Example
    SPECIAL_TOKEN_LIST = {
        "task_name": {
            "new token name 1": "new token 1",
            "new token name 2": "new token 2",
            ...
        },
    }
"""

SPECIAL_TOKEN_LIST = {
    "srl": {
        "predicate_begin": "<PREDICATE>",
        "predicate_end": "</PREDICATE>"
    },
    "ner": {
        
    }
}

"""
TOKENIZER_LIST는 model에서 사용되는 Tokenizer들이 정의되어 있습니다.
새로운 Tokenizer를 추가하고자 하는 경우, 아래 형식에 맞게 값을 추가해주세요.
주의 사항
 - 새로운 Tokenizer가 monologg/kobert/와 같이 새로운 코드를 추가한 후 사용해야하는 경우,
   tokenizers 디렉터리 내부에 신규 코드를 추가한 후, 본 파일 상단에 import 한 후, 정의해주세요

Example
    TOKENIZER_LIST = {
        "model name 1": {NEW TOKENIZER 1},
        "model name 2": {NEW TOKENIZER 2},
        ...
    }
"""

TOKENIZER_LIST = {
    "bert": AutoTokenizer,
    "kobert": KoBertTokenizer,
    "koelectra": AutoTokenizer,
    "kocharelectra": KoCharElectraTokenizer,
    "klueroberta": AutoTokenizer
}