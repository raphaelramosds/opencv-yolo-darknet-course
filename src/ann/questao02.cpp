#include <iostream>
#include <vector>
#include <cblas.h>

#include "activations.hpp"
#include "helpers.hpp"

// Instale as libs BLAS e LAPACK para o uso de funcoes de Algebra Linear Numerica
// sudo apt-get install libblas-dev liblapack-dev gfortran

OutputTable and_network(std::vector<float> &weights)
{
    unsigned int ncols = 2, nrows = 4, i = 0;

    // Armazenar todas as combinações de entradas
    OutputTable output_table;
    output_table.nrows = nrows;
    output_table.inputs.resize(nrows, std::vector<float>(ncols));
    output_table.y_calc.resize(nrows);
    output_table.y_real.resize(nrows);
    output_table.errors.resize(nrows);

    // Gerar tabela verdade 2x2 da tabela AND com saidas esperadas e calculadas pelo perceptron
    for (; i < nrows; i++)
    {
        output_table.inputs[i][0] = (i >> 1) & 1;
        output_table.inputs[i][1] = i & 1;

        output_table.y_calc[i] = stepfunc_activate(
            cblas_sdot(ncols, output_table.inputs[i].data(), 1, weights.data(), 1),
            1.0f);
        output_table.y_real[i] = static_cast<uint8_t>(output_table.inputs[i][0]) & static_cast<uint8_t>(output_table.inputs[i][1]);

        output_table.errors[i] = std::abs(output_table.y_calc[i] - output_table.y_real[i]);
    }

    return output_table;
}

int main()
{
    std::vector<float> weights;

    // a) w1 = w2 = 0.0
    weights = {0.0f, 0.0f};
    print_output_table(and_network(weights), weights);

    // b) w1 = w2 = 0.1
    weights = {0.1f, 0.1f};
    print_output_table(and_network(weights), weights);

    // c) w1 = w2 = 0.5
    weights = {0.5f, 0.5f};
    print_output_table(and_network(weights), weights);

    return 0;
}