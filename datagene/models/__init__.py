from .ko_bert import KoBERTLayer
from .crf import CRFLayer
from .model_builder import ModelBuilder

LAYER_LIST = {
    "kobert": KoBERTLayer,
    "crf": CRFLayer
}