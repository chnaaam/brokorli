from datagene.models.ko_electra import KoElectraLayer
from .ko_bert import KoBERTLayer
from .ko_electra import KoElectraLayer
from .klue_roberta import KlueRobertaLayer
from .crf import CRFLayer
from .model_builder import ModelBuilder

LAYER_LIST = {
    "kobert": KoBERTLayer,
    "koelectra": KoElectraLayer,
    "klueroberta": KlueRobertaLayer,
    "crf": CRFLayer
}