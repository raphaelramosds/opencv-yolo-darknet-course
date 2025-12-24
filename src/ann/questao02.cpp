#include <iostream>
#include <vector>
#include <cblas.h>

#include "activations.hpp"
#include "helpers.hpp"

// Instale as libs BLAS e LAPACK para o uso de funcoes de Algebra Linear Numerica
// sudo apt-get install libblas-dev liblapack-dev gfortran

float and_activate(std::vector<float> &inputs, std::vector<float> weights, int N_COLS)
{
    return stepfunc_activate(
        cblas_sdot(N_COLS, inputs.data(), 1, weights.data(), 1),
        1.0f);
}

OutputTable and_network(std::vector<float> &weights)
{
    unsigned int N_COLS = 2, N_ROWS = 4, i = 0;

    // Armazenar todas as combinações de entradas
    OutputTable output_table;
    output_table.N_ROWS = N_ROWS;
    output_table.inputs.resize(N_ROWS, std::vector<float>(N_COLS));
    output_table.y_calc.resize(N_ROWS);
    output_table.y_real.resize(N_ROWS);
    output_table.errors.resize(N_ROWS);

    // Gerar tabela verdade 2x2 da tabela AND com saidas esperadas e calculadas pelo perceptron
    for (; i < N_ROWS; i++)
    {
        output_table.inputs[i][0] = (i >> 1) & 1;
        output_table.inputs[i][1] = i & 1;

        output_table.y_calc[i] = and_activate(output_table.inputs[i], weights, N_COLS);
        output_table.y_real[i] = static_cast<uint8_t>(output_table.inputs[i][0]) & static_cast<uint8_t>(output_table.inputs[i][1]);

        output_table.errors[i] = std::abs(output_table.y_calc[i] - output_table.y_real[i]);
    }

    return output_table;
}

void print_results(std::vector<float> weights)
{
    std::cout << "Pesos: [" << std::fixed << std::setprecision(2)
              << weights[0] << ", " << weights[1] << "]\n";
    print_output_table(and_network(weights));
}

int main()
{
    std::vector<float> weights;

    // a) w1 = w2 = 0.0
    weights = {0.0f, 0.0f};
    print_results(weights);

    // b) w1 = w2 = 0.1
    weights = {0.1f, 0.1f};
    print_results(weights);

    // c) w1 = w2 = 0.5
    weights = {0.5f, 0.5f};
    print_results(weights);

    return 0;
}