import os


from .settings import *
from .models import YOLO
from .media import buscar_imagem

def compactar_yolo() -> None:
    """
    Rotina para compactar os arquivos relevantes do modelo YOLO

    Apenas tres arquivos sao importantes para fazer deteccoes com o YOLO

        - yolov3.weights    pesos da rede neural
        - yolov3.cfg        configuracao da rede neural
        - coco.names        classes (labels) reconhecidas pelo modelo
    """
    os.system(
        f"""
        tar -czf modelo_YOLO.tar.gz \
            -C {SHARED_PATH} yolov3.weights \
            -C {DARKNET_PATH}/cfg yolov3.cfg \
            -C {DARKNET_PATH}/data coco.names &&\
        mv modelo_YOLO.tar.gz {SHARED_PATH}
        """
    )


def descompactar_yolo() -> None:
    """
    Rotina para descompactar os arquivos do modelo YOLO
    """
    os.system(
        f"""
        mkdir -p {SHARED_PATH}/modelo_YOLO &&\
        tar -xf {SHARED_PATH}/modelo_YOLO.tar.gz -C {SHARED_PATH}/modelo_YOLO &&\
        rm -rf {SHARED_PATH}/modelo_YOLO.tar.gz
        """
    )


def carregar_yolo() -> YOLO:
    """
    Carregar o modelo YOLO
    """
    if not os.path.exists(f"{SHARED_PATH}/modelo_YOLO"):
        compactar_yolo()
        descompactar_yolo()

    labels_path = f"{SHARED_PATH}/modelo_YOLO/coco.names"
    labels = open(labels_path).read().strip().split("\n")

    return YOLO(
        labels_path=labels_path,
        weights_path=f"{SHARED_PATH}/modelo_YOLO/yolov3.weights",
        config_path=f"{SHARED_PATH}/modelo_YOLO/yolov3.cfg",
        labels=labels,
    )


def detectar_objetos(imagem: str, params: dict = {}) -> None:
    """
    Rotina para detectar objetos com a rede YOLO a partir do framework darknet

    Args:
        imagem: nome da imagem
        params: parametros adicionais para o algoritmo de deteccao

    Obs: a imagem sera salva em predictions.jpg
    """
    imagem = buscar_imagem(imagem)
    darknet_cmd = f"./darknet detect cfg/yolov3.cfg ../yolov3.weights {imagem}"

    if params.get("thresh"):
        darknet_cmd = f"{darknet_cmd} -thresh {params['thresh']}"

    if params.get("ext_output"):
        darknet_cmd = f"{darknet_cmd} -ext_output"

    # MAKE SURE TO SET OPENCV=0 ON darknet Makefile !!!
    os.system(
        f"cd {DARKNET_PATH} && {darknet_cmd}"
    )