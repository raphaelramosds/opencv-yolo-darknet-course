# Detecção de Objetos com YOLO, Darknet, OpenCV e Python

## Visão geral

Boa parte dos códigos e recursos utilizados nesse repositório são adaptações do material presente no curso **Detecção de Objetos com YOLO, Darknet, OpenCV e Python**, fornecido pela plataforma IA Expert Academy, na Udemy. Você pode fazer o download de todo o material do curso nesta pasta do drive [YOLO](https://drive.google.com/drive/folders/1jcWIoIWlFJ2ocERjW0p2W1cZ4LRMEjM5) 

**IMPORTANTE:** Antes de executar as implementações das sessões a seguir, por favor não deixe de ver [Configurando o ambiente](./docs/Configurando-Ambiente.md)

## Conteúdo

[![Jupyter Notebook](https://img.shields.io/badge/-Jupyter%20Notebook-05122A?style=flat&logo=jupyter&logoColor=F37626)](./src/notebooks/YOLOv4-Deteccao-de-objetos-com-Darknet.ipynb) **YOLOv4 - Detecção de objetos com Darknet**
- Detecção de objetos com o YOLO via linha de comando com o framework Darknet
- Explicação do parâmetro *threshold* (limiar)
- Como usar o modelo YOLO com outros pesos para detectar objetos de outros *datasets*

[![Jupyter Notebook](https://img.shields.io/badge/-Jupyter%20Notebook-05122A?style=flat&logo=jupyter&logoColor=F37626)](./src/notebooks/YOLOv4-Deteccao-de-objetos-com-OpenCV.ipynb) **YOLOv4 - Detecção de objetos com OpenCV**
- Detecção de objetos com o YOLO utilizando-se de abstrações de redes neurais convolucionais e densas presentes no OpenCV
- Processamento da imagem de entrada com as transformações *mean subtraction* e *resizing*
- Aplicação da Non-maxima Suppression (NMS)

[![Jupyter Notebook](https://img.shields.io/badge/-Jupyter%20Notebook-05122A?style=flat&logo=jupyter&logoColor=F37626)](./src/notebooks/YOLOv4-Deteccao-de-objetos-com-OpenCV-Explorando-Mais.ipynb) **YOLOv4 - Detecção de objetos com OpenCV - Explorando Mais**
- Redimensionamento da imagem de entrada
- Construção do blob da imagem
- Interpretação da saída da detecção para realizar contagem de objetos

[![Jupyter Notebook](https://img.shields.io/badge/-Jupyter%20Notebook-05122A?style=flat&logo=jupyter&logoColor=F37626)](./src/notebooks/YOLOv4-Criando-um-dataset.ipynb) **YOLOv4 - Criando um dataset**
- Download de imagens de classes específicas do Open Images Dataset utilizando a ferramenta OIDv4 Tookit
- Criação de um dataset de treino e de teste a partir das classes escolhidas para treinar a rede
- Tratamento dos rótulos dessas imagens para serem utilizadas no treinamento do YOLO dentro do framework Darknet
- Explicação dos parâmetros presentes nos arquivos .cfg e .data no contexto do treinamento e teste da rede YOLO

[![Jupyter Notebook](https://img.shields.io/badge/-Jupyter%20Notebook-05122A?style=flat&logo=jupyter&logoColor=F37626)](./src/notebooks/YOLOv8-Deteccao-de-objetos.ipynb) **YOLOv8 - Detecção de objetos com Ultralytics YOLO**
- Detecção de objetos com o YOLOv8 utilizando a biblioteca Ultralytics YOLO
- Instalação e configuração do ambiente
- Carregamento do modelo pré-treinado
- Realização de inferência em imagens, vídeos e webcam

[![Jupyter Notebook](https://img.shields.io/badge/-Jupyter%20Notebook-05122A?style=flat&logo=jupyter&logoColor=F37626)](./src/notebooks/YOLOv8-Treinamento-de-um-modelo-customizado.ipynb) **YOLOv8 - Treinamento de um modelo customizado**
- Treinamento de um modelo YOLOv8 customizado utilizando a biblioteca Ultralytics YOLO
- Preparação do dataset no formato YOLO
- Configuração dos hiperparâmetros de treinamento
- Monitoramento do processo de treinamento e avaliação do modelo treinado 

[![CXX](https://img.shields.io/badge/C++-00599C?style=flat-square&logo=C%2B%2B&logoColor=white)](./src/implementacoes-redes-neurais/) **Redes Neurais Artificiais**
- Implementações relativas a Redes Neurais Artificial (ANN) *feed-foward* foram feitas em C++ em formato de respostas a exercícios.
- Para compilar as implementações siga o seguinte passo a passo

```bash
$ cd src/implementacoes-redes-neurais

# Crie o diretorio de build
$ mkdir -p build

# Build dos arquivos fonte com CMake
$ cd build
$ cmake ..
$ cmake --build .

# Executar a Questao01
$ ./Questao01
```