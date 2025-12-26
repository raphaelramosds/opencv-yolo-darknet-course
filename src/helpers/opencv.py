import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import typing as t
import json
import os
from IPython.display import HTML
from base64 import b64encode

from .darknet import carregar_yolo
from .models import ResultadoDeteccao
from .media import buscar_imagem


def mostrar_imagem(imagem: cv2.typing.MatLike) -> None:
    """
    Rotina para exibir uma imagem a partir de um objeto MatLike do OpenCV

    Args:
        imagem: objeto matricial que representa a imagem
    """
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.show()


def carregar_imagem(
    imagem: str, exibir: bool = True
) -> t.Union[cv2.typing.MatLike, None]:
    """
    Rotina para carregar e exibir uma imagem com o OpenCV

    Args:
        imagem: nome da imagem
        exibir: imprimir ou nao a imagem no display

    Returns:
        cv2.typing.MatLike : objeto matricial que representa a imagem
    """
    imagem = buscar_imagem(imagem)
    imagem_matriz = cv2.imread(filename=imagem)

    if imagem_matriz is None:
        return None

    if exibir:
        mostrar_imagem(imagem_matriz)

    return imagem_matriz


def redimensionar_imagem(imagem: cv2.typing.MatLike, largura_maxima: int = 600):
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


def _executar_deteccao(
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

    return ResultadoDeteccao(net=net, imagem=imagem, layer_outputs=layer_outputs)


def _desenhar_detecoes_na_imagem(
    resultado: ResultadoDeteccao,
    imagem: cv2.typing.MatLike,
    thre,
    thre_nms,
    labels,
):
    (H, W) = imagem.shape[:2]

    # Definir caixas delimitadoras e confianças
    caixas = []
    confiancas = []
    IDclasses = []
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

    # Desenhar caixas
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    objetos = cv2.dnn.NMSBoxes(caixas, confiancas, thre, thre_nms)
    if len(objetos) > 0:
        for i in objetos.flatten():
            (x, y) = (caixas[i][0], caixas[i][1])
            (w, h) = (caixas[i][2], caixas[i][3])

            cor = [int(c) for c in colors[IDclasses[i]]]

            cv2.rectangle(imagem, (x, y), (x + w, y + h), cor, 2)
            texto = "{}: {:.4f}".format(labels[IDclasses[i]], confiancas[i])
            cv2.putText(
                imagem, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2
            )

    return imagem


def detectar_objetos(
    imagem: cv2.typing.MatLike,
    thre: float = 0.5,
    thre_nms: float = 0.3,
):
    """
    Detecta objetos em uma imagem utilizando YOLO com OpenCV

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

    modelo = carregar_yolo()

    resultado = _executar_deteccao(
        net=cv2.dnn.readNet(modelo.config_path, modelo.weights_path), imagem=imagem
    )

    imagem_matriz = _desenhar_detecoes_na_imagem(
        resultado=resultado,
        imagem=imagem,
        labels=modelo.labels,
        thre=thre,
        thre_nms=thre_nms,
    )

    if imagem_matriz is None:
        return None

    cv2.imwrite("resultado.jpg", imagem_matriz)
    mostrar_imagem(imagem_matriz)


def label_caixa(image, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
    """
    Desenha o label (rótulo) da predição, acima do objeto detectado
    """
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # espessura da fonte
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[
            0
        ]  # largura e altura do texto
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(
            image, p1, p2, color, -1, cv2.LINE_AA
        )  # retangulo preenchido (que ficará atrás do texto, para aumentar a legibilidade)
        cv2.putText(
            image,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            lw / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def desenha_caixas(image, boxes, labels=[], colors=[], score=True, conf=None):
    """
    Desenha as caixas delimitadoras (bounding boxes) sobre a imagem
    - Se conf=True então escreve a confiança ao lado do label
    """
    # Carrega os labels do arquivo JSON se não forem fornecidos
    if labels == []:
        json_path = os.path.join(os.path.dirname(__file__), "labels.json")
        with open(json_path, "r") as f:
            labels = json.load(f)
            # Converte chaves string para inteiros
            labels = {int(k): v for k, v in labels.items()}

    # Carrega as cores do arquivo JSON se não forem fornecidas
    if colors == []:
        json_path = os.path.join(os.path.dirname(__file__), "colors.json")
        with open(json_path, "r") as f:
            colors = json.load(f)
            # Converte listas em tuplas
            colors = [tuple(color) for color in colors]

    # desenha/plota cada caixa
    for box in boxes:
        # add score (confiança) no label se score=True
        if score:
            label = (
                labels[int(box[-1]) + 1]
                + " "
                + str(round(100 * float(box[-2]), 1))
                + "%"
            )
        else:
            label = labels[int(box[-1]) + 1]
        # filtra cada caixa, abaixo do limiar (threshold) conf, se conf=True
        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1])]
                label_caixa(image, box, label, color)
        else:
            color = colors[int(box[-1])]
            label_caixa(image, box, label, color)

    return image

    ## baseado no código de Tom Keldenich (inside-machinelearning.com)


def exibir_video(video_caminho: str):
    """
    Exibe um vídeo a partir do caminho fornecido usando OpenCV.

    Args:
        video_caminho (str): Caminho para o arquivo de vídeo.
    """
    mp4 = open(video_caminho, "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(
        """
  <video width=500 controls>
    <source src="%s" type="video/mp4"
  </video>
  """
        % data_url
    )
