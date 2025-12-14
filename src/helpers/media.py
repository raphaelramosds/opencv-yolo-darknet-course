import os

from .settings import DARKNET_PATH, IMG_PATH


def buscar_imagem(imagem: str) -> str:
    """
    Rotina para buscar imagem

    Esse metodo primeiro procura a imagem no diretorio de dados do repositorio darknet
    Caso nao encontre, ele tenta encontrá-la no diretório shared/img, presente na raiz
    desse projeto.

    Args:
        imagem: nome ou caminho para a imagem

    Returns:
        str: caminho absoluto para a imagem
    """
    if os.path.exists(imagem):
        pass

    elif os.path.exists(f"{DARKNET_PATH}/data/{imagem}"):
        imagem = f"{DARKNET_PATH}/data/{imagem}"

    elif os.path.exists(f"{DARKNET_PATH}/{imagem}"):
        imagem = f"{DARKNET_PATH}/{imagem}"

    elif os.path.exists(f"{IMG_PATH}/{imagem}"):
        imagem = f"{IMG_PATH}/{imagem}"

    else:
        raise Exception(f"Image not found: {imagem}")
    
    return imagem
