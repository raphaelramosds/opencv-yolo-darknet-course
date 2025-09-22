from dataclasses import dataclass
import os
import cv2
import matplotlib.pyplot as plt

shared = os.path.abspath("../../shared")
img = f"{shared}/img"
darknet = f"{shared}/darknet"


def _compactar_yolov3() -> None:
    """
    Rotina para compactar os arquivos relevantes do modelo YOLOv3

    Apenas tres arquivos sao importantes para fazer deteccoes com o YOLOv3
    - yolov3.weights    pesos da rede neural
    - yolov3.cfg        configuracao da rede neural
    - coco.names        classes (labels) reconhecidas pelo modelo
    """
    os.system(
        f"""
        tar -czf modelo_YOLOv3.tar.gz \
            -C {shared} yolov3.weights \
            -C {darknet}/cfg yolov3.cfg \
            -C {darknet}/data coco.names &&\
        mv modelo_YOLOv3.tar.gz {shared}
        """
    )


def _descompactar_yolov3() -> None:
    """
    Rotina para descompactar os arquivos do modedlo YOLOv3
    """
    os.system(
        f"""
        mkdir -p {shared}/modelo_YOLOv3 &&\
        tar -xf {shared}/modelo_YOLOv3.tar.gz -C {shared}/modelo_YOLOv3 &&\
        rm -rf {shared}/modelo_YOLOv3.tar.gz
        """
    )


@dataclass
class yolov3:
    labels_path: str
    weights_path: str
    config_path: str
    labels: list


def carregar_yolov3() -> yolov3:
    _compactar_yolov3()
    _descompactar_yolov3()
    labels_path = f"{shared}/modelo_YOLOv3/coco.names"
    labels = open(labels_path).read().strip().split("\n")
    return yolov3(
        labels_path,
        f"{shared}/modelo_YOLOv3/yolov3.weights",
        f"{shared}/modelo_YOLOv3/yolov3.cfg",
        labels,
    )


def _mostrar_matriz(imagem: cv2.typing.MatLike) -> None:
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.show()


def mostrar_imagem(imagem: str) -> cv2.typing.MatLike:
    """
    Rotina para exibir uma imagem com o OpenCV

    Argumentos:
        imagem (str): nome da imagem

    Retorna:
        cv2.typing.MatLike : objeto matricial que representa a imagem
    """
    imagem = _buscar_imagem(imagem)
    imagem = cv2.imread(imagem)
    _mostrar_matriz(imagem)
    return imagem


def mostrar_imagem_cv2(imagem: cv2.typing.MatLike) -> None:
    """
    Rotina para exibir uma imagem a partir de um objeto MatLike do OpenCV

    Argumentos:
        imagem (cv2.typing.MatLike): objeto matricial que representa a imagem
    """
    _mostrar_matriz(imagem)


class ImageNotFound(Exception): ...


def _buscar_imagem(imagem: str) -> str:
    """
    Rotina para buscar imagem

    Esse metodo primeiro procura a imagem no diretorio de dados do repositorio darknet
    Caso nao encontre, ele tenta encontrá-la no diretório shared/img, presente na raiz
    desse projeto.

    Argumentos:
        imagem (str): nome da imagem

    Retorna:
        str : caminho absoluto para a imagem
    """
    if os.path.exists(f"{darknet}/data/{imagem}"):
        imagem = f"{darknet}/data/{imagem}"
    elif os.path.exists(f"{img}/{imagem}"):
        imagem = f"{img}/{imagem}"
    else:
        raise ImageNotFound(imagem)
    return imagem


def detectar_objetos(imagem: str, params: dict = {}) -> None:
    """
    Rotina para detectar objetos com a rede YOLO a partir do framework darknet

    Argumentos:
        imagem (str): nome da imagem
        params (dict): parametros adicionais para o algoritmo de deteccao
    """
    imagem = _buscar_imagem(imagem)
    darknet_cmd = f"./darknet detect cfg/yolov3.cfg ../yolov3.weights {imagem}"

    if params.get("thresh"):
        darknet_cmd = f"{darknet_cmd} -thresh {params['thresh']}"

    if params.get("ext_output"):
        darknet_cmd = f"{darknet_cmd} -ext_output"

    os.system(f"cd {darknet} && {darknet_cmd}")
    mostrar_imagem(f"{darknet}/predictions.jpg")
