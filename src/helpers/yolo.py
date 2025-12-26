import os
from .media import buscar_imagem

def detectar_objetos(imagem: str) -> None:
    imagem = buscar_imagem(imagem)
    yolo_cmd = f"yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source='{imagem}' save=True"
    os.system(yolo_cmd)

def detectar_objetos_url(url_imagem: str) -> None:
    yolo_cmd = f"yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source='{url_imagem}' save=True"
    os.system(yolo_cmd)