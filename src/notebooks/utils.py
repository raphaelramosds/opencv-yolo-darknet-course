import os
import cv2
import matplotlib.pyplot as plt

DARKNET_PATH = "../../shared/darknet"


def mostrar_imagem(caminho: str) -> None:
    imagem = cv2.imread(caminho)
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.show()


def detectar_objetos(caminho: str) -> None:
    os.system(
        f"cd {DARKNET_PATH} && ./darknet detect cfg/yolov3.cfg ../yolov3.weights {caminho}"
    )
    mostrar_imagem(caminho=f"{DARKNET_PATH}/predictions.jpg")
