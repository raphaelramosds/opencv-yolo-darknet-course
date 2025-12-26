import os
from .media import buscar_imagem

def detectar_objetos(imagem: str, conf: float = 0.25) -> None:
    imagem = buscar_imagem(imagem)
    yolo_cmd = f"yolo task=detect mode=predict model=yolov8n.pt conf={conf} source='{imagem}' save=True"
    os.system(yolo_cmd)

def detectar_objetos_url(url_imagem: str, conf: float = 0.25) -> None:
    yolo_cmd = f"yolo task=detect mode=predict model=yolov8n.pt conf={conf} source='{url_imagem}' save=True"
    os.system(yolo_cmd)

def detectar_objetos_imagens(pasta: str, conf: float = 0.25) -> None:
    yolo_cmd = f"yolo task=detect mode=predict model=yolov8n.pt conf={conf} source='{pasta}' save=True"
    os.system(yolo_cmd)