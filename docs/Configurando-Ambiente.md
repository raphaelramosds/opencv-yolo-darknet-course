# Configurando o ambiente

## Python

Para executar com sucesso os exemplos desse repositório, é necessário instalar as seguintes dependências

```bash
# Instalação de pacotes do Linux
$ sudo apt-get install dos2unix wget git make python3

# Instalação do poetry
$ curl -sSL https://install.python-poetry.org | python3 -

# Definir binario do poetry
$ export PATH="/home/$USER/.local/bin:$PATH:"
```

**NOTE:** O último comando precisará ser adicionado ao .zhrc ou .bashrc para que o poetry possa ser chamado de qualquer diretório do terminal

A preparação do ambiente para executar os notebooks foi automatizada. Com um único comando são instalados os pacotes necessários, feito o download do framework Darknet e dos pesos do modelo YOLOv3 pré-treinado utilizados nos exemplos deste repositório.

```bash
# Para configurar e executar usando CPU
$ make setup
```

## C/C++

Instale o CMake e as libs BLAS e LAPACK para o uso de funcoes de Algebra Linear Numerica

```bash
$ sudo apt-get install cmake libblas-dev liblapack-dev gfortran
```