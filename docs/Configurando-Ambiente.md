# Configurando o ambiente

## Python

### Pyenv

```bash
curl -fsSL https://pyenv.run | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc\necho '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc\necho 'eval "$(pyenv init - zsh)"' >> ~/.zshrc

source ~/.zshrc

sudo apt update

sudo apt install make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Instale a versão do Python necessária para o projeto
pyenv install 3.11

```

### Poetry

```bash
sudo apt-get install dos2unix wget git make python3

curl -sSL https://install.python-poetry.org | python3 -

echo 'export PATH="/home/$USER/.local/bin:$PATH:"' >> ~/.zshrc

source ~/.zshrc
```

### Compilação do Darknet

```bash
# Darknet com suporte a CPU
make darknet-cpu

# Darknet com suporte apenas a GPU
make darknet-gpu
```

### Instalação das dependências Python

```bash
# Defina a versão do Python para o projeto
pyenv global 3.11

# Instale as dependências com Poetry
poetry install

# Ou use pip (sem ambiente virtual)
# OBS: crie um ambiente virtual antes de usar pip
pip install -r requirements.txt

# Ative o ambiente virtual do Poetry/Pip
source $(poetry env info --path)/bin/activate

# Instale o PyTorch (GPU)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

Caso queira uma instalação customizada do PyTorch, siga as instruções do site oficial [PyTorch - Get Started Locally](https://pytorch.org/get-started/locally/)

## C/C++

Instale o CMake e as libs BLAS e LAPACK para o uso de funcoes de Algebra Linear Numerica

```bash
sudo apt-get install cmake libblas-dev liblapack-dev gfortran
```