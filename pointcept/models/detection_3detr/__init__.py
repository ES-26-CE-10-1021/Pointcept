from .dataset_config import ScanNetDetectionConfig, AgcoBBoxConfig
from .components import (
    IdentityEncoder3DETR,
    PointnetSAPreEncoder,
    PointnetSAPreEncoderWithDino,
    VanillaTransformerEncoder3DETR,
    MaskedTransformerEncoder3DETR,
    TransformerDecoder3DETR,
)
from .criterion import SetCriterion3DETR
from .model import Model3DETRDetector, dense2point, point2dense
from .dino_injection import Model3DETRDetectorWithDino

try:
    from .ptv3 import (
        PTv3PreEncoder,
        PTv3m3PreEncoder,
        PTv3PreEncoderWithDino,
        PTv3PreEncoderWithDinoTrueFps,
        PTv3m3PreEncoderWithDinoTrueFps,
        PTv3DinoMixin,
    )
except ImportError:
    PTv3PreEncoder = None
    PTv3m3PreEncoder = None
try:
    from .multi_task import MultiTask3DETRSegmentor
except ImportError:
    MultiTask3DETRSegmentor = None
try:
    from .seg_only import Dense3DETRSegmentor
except ImportError:
    Dense3DETRSegmentor = None
