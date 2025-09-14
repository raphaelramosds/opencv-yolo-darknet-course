import os
import cv2
import matplotlib.pyplot as plt

IMG_PATH = "../../shared/img"
DARKNET_PATH = "../../shared/darknet"


def mostrar_imagem(caminho: str) -> None:
    imagem = cv2.imread(caminho)
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.show()


def detectar_objetos(caminho: str) -> None:
    if os.path.exists(f"{DARKNET_PATH}/data/{caminho}"):
        # Procurar no diretorio de dados do repositorio darknet
        caminho = f"{DARKNET_PATH}/data/{caminho}"
    elif os.path.exists(f"{IMG_PATH}/{caminho}"):
        # Procurar no diretorio de imagens local shared/img
        caminho = f"{IMG_PATH}/{caminho}"
    else:
        raise Exception(f"Imagem {caminho} nao encontrada")

    os.system(
        f"cd {DARKNET_PATH} && ./darknet detect cfg/yolov3.cfg ../yolov3.weights {caminho}"
    )
    mostrar_imagem(caminho=f"{DARKNET_PATH}/predictions.jpg")
