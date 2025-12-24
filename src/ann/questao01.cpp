#include <iostream>
#include <vector>
#include <cblas.h>

#include "activations.hpp"

// Instale as libs BLAS e LAPACK para o uso de funcoes de Algebra Linear Numerica
// sudo apt-get install libblas-dev liblapack-dev gfortran

int main()
{
    // Entradas
    std::vector<float> inputs = {1.0f, 7.0f, 5.0f};

    // Pesos
    std::vector<float> weights = {0.8f, 0.1f, 0.0f};

    // Camada de ativação
    float sum = cblas_sdot(3, inputs.data(), 1, weights.data(), 1);
    float output = stepfunc_activate(sum, 1.0f);

    std::cout << "Saída do perceptron: " << output << std::endl;

    return 0;
}