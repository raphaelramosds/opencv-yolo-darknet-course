from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time

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
    Rotina para descompactar os arquivos do modelo YOLOv3
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
    if not os.path.exists(f"{shared}/modelo_YOLOv3"):
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

    Args:
        imagem: nome da imagem

    Returns:
        cv2.typing.MatLike : objeto matricial que representa a imagem
    """
    imagem = _buscar_imagem(imagem)
    imagem = cv2.imread(imagem)
    _mostrar_matriz(imagem)
    return imagem


def mostrar_imagem_cv2(imagem: cv2.typing.MatLike) -> None:
    """
    Rotina para exibir uma imagem a partir de um objeto MatLike do OpenCV

    Args:
        imagem: objeto matricial que representa a imagem
    """
    _mostrar_matriz(imagem)


@dataclass
class resultado:
    net: cv2.dnn.Net
    imagem: cv2.typing.MatLike
    layer_outputs: np.array


def blob_inferir_yolo_cv2(
    net: cv2.dnn.Net, imagem: cv2.typing.MatLike, mostrar_texto: bool = True
):
    """
    Executa a detecção de objetos em uma imagem utilizando YOLO via OpenCV.

    A função realiza duas etapas principais:
    1. Pré-processamento da imagem com `cv2.dnn.blobFromImage`,
    convertendo-a em um blob adequado para a rede neural.
    2. Inferência com a rede YOLO, obtendo as saídas das camadas de detecção.

    Args:
        net (cv2.dnn.Net): Rede YOLO já carregada no OpenCV.
        imagem (cv2.typing.MatLike): Imagem de entrada em formato OpenCV (BGR).
        mostrar_texto (bool, opcional): Se True, imprime o tempo gasto na inferência.
                                        Padrão é True.

    Returns:
        resultado: Objeto contendo a rede (`net`), a imagem original (`imagem`)
                e as saídas das camadas (`layer_outputs`).
    """
    ln = net.getLayerNames()
    ln_saida_indices = net.getUnconnectedOutLayers()
    ln_saida_ids = tuple(ln[i - 1] for i in ln_saida_indices)

    inicio = time.time()

    blob = cv2.dnn.blobFromImage(imagem, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln_saida_ids)

    termino = time.time()

    if mostrar_texto:
        print("YOLO levou {:.2f} segundos".format(termino - inicio))

    return resultado(net, imagem, layer_outputs)


def redimensionar_imagem_cv2(imagem: cv2.typing.MatLike, largura_maxima: int = 600):
    """
    Redimensiona uma imagem preservando a proporção, caso sua largura ultrapasse
    um limite máximo definido.

    Se a largura da imagem original for maior que `largura_maxima`, a função
    ajusta a largura para esse valor e recalcula a altura proporcionalmente.
    Caso contrário, a imagem é Returnsda sem alterações.

    Args:
        imagem (cv2.typing.MatLike): Imagem em formato OpenCV (BGR).
        largura_maxima (int, opcional): Valor máximo permitido para a largura.
            Padrão é 600.

    Returns:
        cv2.typing.MatLike: Imagem redimensionada mantendo a proporção original.
    """
    (largura, altura) = imagem.shape[:2]
    imagem_largura = largura
    imagem_altura = altura
    if largura > largura_maxima:
        proporcao = altura / largura
        imagem_largura = largura_maxima
        imagem_altura = int(imagem_largura / proporcao)
    imagem = cv2.resize(imagem, (imagem_largura, imagem_altura))
    return imagem


class ImageNotFound(Exception): ...


def _buscar_imagem(imagem: str) -> str:
    """
    Rotina para buscar imagem

    Esse metodo primeiro procura a imagem no diretorio de dados do repositorio darknet
    Caso nao encontre, ele tenta encontrá-la no diretório shared/img, presente na raiz
    desse projeto.

    Args:
        imagem: nome da imagem

    Returns:
        str: caminho absoluto para a imagem
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

    Args:
        imagem: nome da imagem
        params: parametros adicionais para o algoritmo de deteccao
    """
    imagem = _buscar_imagem(imagem)
    darknet_cmd = f"./darknet detect cfg/yolov3.cfg ../yolov3.weights {imagem}"

    if params.get("thresh"):
        darknet_cmd = f"{darknet_cmd} -thresh {params['thresh']}"

    if params.get("ext_output"):
        darknet_cmd = f"{darknet_cmd} -ext_output"

    os.system(f"cd {darknet} && {darknet_cmd}")
    mostrar_imagem(f"{darknet}/predictions.jpg")


def detectar_objetos_cv2(
    imagem: cv2.typing.MatLike,
    thre: float = 0.5,
    thre_nms: float = 0.3,
):
    """
    Detecta objetos em uma imagem utilizando YOLOv3 com OpenCV e desenha caixas delimitadoras.

    A função carrega a rede YOLOv3, executa a inferência sobre a imagem e aplica
    Non-Maxima Suppression (NMS) para eliminar caixas sobrepostas. Em seguida,
    desenha caixas delimitadoras e rótulos sobre os objetos detectados, salva o
    resultado em disco e exibe a imagem processada.

    Args:
        imagem (cv2.typing.MatLike): Imagem de entrada em formato OpenCV (BGR).
        thre (float, opcional): Limite mínimo de confiança para considerar uma detecção.
            Padrão é 0.5.
        thre_nms (float, opcional): Limite do Non-Maxima Suppression (NMS) para
            supressão de caixas sobrepostas. Padrão é 0.3.

    Returns:
        None
            A imagem resultante é salva em ``resultado.jpg`` e exibida no notebook
            (ou ambiente gráfico disponível).
    """
    yolov3 = carregar_yolov3()

    (H, W) = imagem.shape[:2]

    colors = np.random.randint(0, 255, size=(len(yolov3.labels), 3), dtype="uint8")
    net = cv2.dnn.readNet(yolov3.config_path, yolov3.weights_path)
    caixas = []
    confiancas = []
    IDclasses = []

    resultado = blob_inferir_yolo_cv2(net, imagem)

    for output in resultado.layer_outputs:
        for detection in output:
            scores = detection[5:]
            classeID = np.argmax(scores)
            confianca = scores[classeID]

            if confianca > thre:
                caixa = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = caixa.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                caixas.append([x, y, int(width), int(height)])
                confiancas.append(float(confianca))
                IDclasses.append(classeID)

    objetos = cv2.dnn.NMSBoxes(caixas, confiancas, thre, thre_nms)

    if len(objetos) > 0:
        for i in objetos.flatten():
            (x, y) = (caixas[i][0], caixas[i][1])
            (w, h) = (caixas[i][2], caixas[i][3])

            cor = [int(c) for c in colors[IDclasses[i]]]

            cv2.rectangle(imagem, (x, y), (x + w, y + h), cor, 2)
            texto = "{}: {:.4f}".format(yolov3.labels[IDclasses[i]], confiancas[i])
            cv2.putText(
                imagem, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2
            )

    cv2.imwrite("resultado.jpg", imagem)

    mostrar_imagem_cv2(imagem)
