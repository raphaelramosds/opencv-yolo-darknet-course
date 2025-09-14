import os
import cv2
import matplotlib.pyplot as plt

shared = os.path.abspath("../../shared")
img = f"{shared}/img"
darknet = f"{shared}/darknet"


def compactar_yolov3() -> None:
    """
    Rotina para compactar os arquivos relevantes do modelo YOLOv3

    Apenas tres arquivos sao importantes para fazer deteccoes com o YOLOv3
    - yolov3.weights    pesos da rede neural
    - yolov3.cfg        configuracao da rede neural
    - coco.names        conjunto de dados utilizado para treinar a rede neural
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


def mostrar_imagem(caminho_imagem: str) -> None:
    """
    Rotina para exibir uma imagem com o OpenCV

    Argumentos:
        caminho_imagem (str): caminho para imagem de interesse
    """
    imagem = cv2.imread(caminho_imagem)
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.show()


def buscar_imagem(caminho_imagem: str) -> str:
    """
    Rotina para buscar imagem

    Esse metodo primeiro procura a imagem no diretorio de dados do repositorio darknet
    Caso nao encontre, ele tenta encontrá-la no diretório shared/img, presente na raiz
    desse projeto.

    Argumentos:
        caminho_imagem (str): caminho para imagem de interesse

    """
    if os.path.exists(f"{darknet}/data/{caminho_imagem}"):
        caminho_imagem = f"{darknet}/data/{caminho_imagem}"
    elif os.path.exists(f"{img}/{caminho_imagem}"):
        caminho_imagem = f"{img}/{caminho_imagem}"
    else:
        raise Exception(f"Imagem {caminho_imagem} nao encontrada")
    return caminho_imagem


def detectar_objetos(caminho_imagem: str, params: dict = {}) -> None:
    """
    Rotina para detectar objetos com a rede YOLO a partir do framework darknet

    Argumentos:
        caminho_imagem (str): caminho para imagem de interesse
        params (dict): parametros adicionais para o algoritmo de deteccao
    """
    caminho_imagem = buscar_imagem(caminho_imagem)
    darknet_cmd = f"./darknet detect cfg/yolov3.cfg ../yolov3.weights {caminho_imagem}"

    if params.get("thresh"):
        darknet_cmd = f"{darknet_cmd} -thresh {params['thresh']}"

    if params.get("ext_output"):
        darknet_cmd = f"{darknet_cmd} -ext_output"

    os.system(f"cd {darknet} && {darknet_cmd}")
    mostrar_imagem(f"{darknet}/predictions.jpg")
