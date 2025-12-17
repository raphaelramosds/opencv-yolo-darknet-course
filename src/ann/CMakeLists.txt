cmake_minimum_required(VERSION 3.28)

project(Ann LANGUAGES CXX)  

# Definir C++ Standard para 20
set(CMAKE_CXX_STANDARD 20)

# CMake nao executara se C++ Standard nao for o que foi definido em CMAKE_CXX_STANDARD
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Este projeto nao vai usar extensoes
set(CMAKE_CXX_EXTENSIONS OFF)

# add_executable( executavel fontes )
# Sintaxe: o "executavel" vai ser construido a partir dos arquivos "fontes"

add_executable( PerceptronUmaCamada perceptron-uma-camada.cpp )

add_executable( OperadorAnd operador-and.cpp )