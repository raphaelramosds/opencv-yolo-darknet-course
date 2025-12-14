from typing import Sequence
import cv2

from dataclasses import dataclass


@dataclass
class YOLO:
    labels_path: str
    weights_path: str
    config_path: str
    labels: list


@dataclass
class ResultadoDeteccao:
    net: cv2.dnn.Net
    imagem: cv2.typing.MatLike
    layer_outputs: Sequence[cv2.typing.MatLike]
