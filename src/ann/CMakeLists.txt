cmake_minimum_required(VERSION 3.28)

project(Ann LANGUAGES CXX)  

find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message(STATUS "Libraries BLAS encontradas: ${BLAS_LIBRARIES}")
else()
    message(WARNING "Nao foi possivel encontrar nenhuma library BLAS. Alguns codigos podem nao funcionar corretamente.")
endif()

# Definir C++ Standard para 20
set(CMAKE_CXX_STANDARD 20)

# CMake nao executara se C++ Standard nao for o que foi definido em CMAKE_CXX_STANDARD
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Este projeto nao vai usar extensoes
set(CMAKE_CXX_EXTENSIONS OFF)

# add_executable( executavel fontes )
# Sintaxe: o "executavel" vai ser construido a partir dos arquivos "fontes"

add_executable( Questao01 questao01.cpp )
target_link_libraries( Questao01 ${BLAS_LIBRARIES} )

add_executable( Questao02 questao02.cpp )
target_link_libraries( Questao02 ${BLAS_LIBRARIES} )

add_executable( Questao03 questao03.cpp )
target_link_libraries( Questao03 ${BLAS_LIBRARIES} )

add_executable( Questao04 questao04.cpp )
target_link_libraries( Questao04 ${BLAS_LIBRARIES} )