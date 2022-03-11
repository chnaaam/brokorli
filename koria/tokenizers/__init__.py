from .tokenization_kobert import KoBertTokenizer
from .tokenization_kocharelectra import KoCharElectraTokenizer
from transformers import AutoTokenizer

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