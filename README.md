# Detecção de Objetos com YOLO, Darknet, OpenCV e Python

## Dependências

Para executar com sucesso os exemplos desse repositório, é necessário instalar as seguintes dependências

```bash
# Instalação de pacotes do Linux
sudo apt-get install dos2unix wget git make python3

# Instalação do poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/$USER/.local/bin:$PATH:"
```

>💡**NOTE:** O último comando precisará ser adicionado ao .zhrc ou .bashrc para que o poetry possa ser chamado de qualquer diretório do terminal


## Configuração do ambiente

A preparação do ambiente foi automatizada. Com um único comando são instalados os pacotes necessários, feito o download do framework Darknet e dos pesos do modelo YOLOv3 pré-treinado utilizados nos exemplos deste repositório.

Para configurar e executar usando CPU:

```bash
make setup
```

## Reconhecimento

Os códigos e recursos utilizados nesse repositório são adaptações do material presente no curso **Detecção de Objetos com YOLO, Darknet, OpenCV e Python**, fornecido pela plataforma IA Expert Academy, na Udemy.

No link abaixo você pode fazer o download de todo o material do curso.

[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4.svg?style=for-the-badge&logo=Google-Drive&logoColor=white)](https://drive.google.com/drive/folders/1jcWIoIWlFJ2ocERjW0p2W1cZ4LRMEjM5)