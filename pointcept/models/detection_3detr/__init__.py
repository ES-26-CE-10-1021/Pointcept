from .dataset_config import ScanNetDetectionConfig, AgcoBBoxConfig
from .components import (
    IdentityEncoder3DETR,
    PointnetSAPreEncoder,
    VanillaTransformerEncoder3DETR,
    MaskedTransformerEncoder3DETR,
    TransformerDecoder3DETR,
)
from .criterion import SetCriterion3DETR
from .model import Model3DETRDetector, dense2point, point2dense
try:
    from .ptv3 import PTv3PreEncoder
except ImportError:
    PTv3PreEncoder = None
