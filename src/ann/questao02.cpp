#include <iostream>
#include <iomanip>
#include <vector>
#include <cblas.h>

#include "activations.hpp"

// Instale as libs BLAS e LAPACK para o uso de funcoes de Algebra Linear Numerica
// sudo apt-get install libblas-dev liblapack-dev gfortran

struct result
{
    std::vector<std::vector<float>> inputs;
    std::vector<float> y_net;
    std::vector<int> y_real;
    std::vector<float> errors;
    int nrows;
};

void print_result(const result &r, const std::vector<float> &weights)
{
    std::cout << "Pesos: [" << std::fixed << std::setprecision(2)
              << weights[0] << ", " << weights[1] << "]\n";

    std::cout << std::setw(5) << "x1"
              << std::setw(5) << "x2"
              << std::setw(10) << "saida"
              << std::setw(10) << "real"
              << std::setw(10) << "erro"
              << std::endl;

    std::cout << std::string(40, '-') << std::endl;

    for (int i = 0; i < r.nrows; i++)
    {
        std::cout << std::setw(5) << r.inputs[i][0]
                  << std::setw(5) << r.inputs[i][1]
                  << std::setw(10) << r.y_net[i]
                  << std::setw(10) << r.y_real[i]
                  << std::setw(10) << r.errors[i]
                  << std::endl;
    }

    // calcular erro absoluto médio
    float mae = cblas_sasum(r.nrows, r.errors.data(), 1) / r.nrows;
    std::cout << "EMA = " << std::fixed << std::setprecision(2) << mae << "\n\n";
}

result and_perceptron(std::vector<float> &weights)
{
    unsigned int ncols = 2, nrows = 4;

    // Armazenar todas as combinações de entradas
    result r;
    r.nrows = nrows;
    r.inputs.resize(nrows, std::vector<float>(ncols));
    r.y_net.resize(nrows);
    r.y_real.resize(nrows);
    r.errors.resize(nrows);

    // Gerar tabela verdade 2x2 da tabela AND com saidas esperadas e calculadas pelo perceptron
    for (unsigned int i = 0; i < nrows; i++)
    {
        r.inputs[i][0] = (i >> 1) & 1;
        r.inputs[i][1] = i & 1;

        r.y_net[i] = stepfunc_activate(cblas_sdot(ncols, r.inputs[i].data(), 1, weights.data(), 1), 1.0f);
        r.y_real[i] = static_cast<uint8_t>(r.inputs[i][0]) & static_cast<uint8_t>(r.inputs[i][1]);
        
        r.errors[i] = std::abs(r.y_net[i] - r.y_real[i]);
    }

    return r;
}

int main()
{
    std::vector<float> weights;

    // a) w1 = w2 = 0.0
    weights = {0.0f, 0.0f};
    print_result(and_perceptron(weights), weights);

    // b) w1 = w2 = 0.1
    weights = {0.1f, 0.1f};
    print_result(and_perceptron(weights), weights);

    // c) w1 = w2 = 0.5
    weights = {0.5f, 0.5f};
    print_result(and_perceptron(weights), weights);

    return 0;
}