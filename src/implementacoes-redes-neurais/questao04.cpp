#include "activations.hpp"
#include "helpers.hpp"

float xor_activate(std::vector<float> &inputs)
{
    // Pesos (2x3)
    std::vector<float> weights = {
        -0.424f,
        -0.740f,
        -0.961f,
        -0.358f,
        -0.577f,
        -0.469f,
    };
    const int WEIGHTS_ROWS = 2, WEIGHTS_COLS = 3;

    // Saída Camada Oculta (1x3)
    std::vector<float> h(3, 0.0f);

    mult_m<float>(inputs, 1, 2, weights, WEIGHTS_COLS, h);

    for (auto &val : h)
    {
        val = logistic_activate(val);
    }

    // Pesos (1x3)
    std::vector<float> v = {-0.017, -0.893, 0.148};
    const int V_ROWS = 1, V_COLS = 2;

    // Saida (1x1)
    return logistic_activate(
        cblas_sdot(WEIGHTS_COLS, h.data(), 1, v.data(), 1));
}

OutputTable xor_network()
{
    // Dimensoes da tabela verdade
    unsigned int N_COLS = 2, N_ROWS = 4, i = 0;

    // Armazenar todas as combinações de entradas
    OutputTable output_table(N_ROWS, N_COLS);

    // Gerar tabela verdade 2x2 da tabela XOR com saidas esperadas e calculadas pelo perceptron
    for (; i < N_ROWS; i++)
    {
        output_table.inputs[i][0] = (i >> 1) & 1;
        output_table.inputs[i][1] = i & 1;

        output_table.y_calc[i] = xor_activate(output_table.inputs[i]);
        output_table.y_real[i] = static_cast<uint8_t>(output_table.inputs[i][0]) ^ static_cast<uint8_t>(output_table.inputs[i][1]);

        output_table.errors[i] = std::abs(output_table.y_calc[i] - output_table.y_real[i]);
    }

    return output_table;
}

void print_results()
{
    OutputTable output_table = xor_network();
    print_output_table(output_table);
}

int main()
{
    print_results();

    return 0;
}