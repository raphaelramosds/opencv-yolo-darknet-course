#include <iostream>
#include <vector>
#include <cblas.h>

#include "activations.hpp"

int main()
{
    // Entradas (1x2)
    std::vector<float> inputs = {0.0f, 0.0f};
    const int INPUTS_ROWS = 1;
    const int INPUTS_COLS = 2;

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

    // Saída (1x3)
    std::vector<float> h(3, 0.0f);

    // Camada oculta
    cblas_sgemm(
        // Multiplicação: h = inputs * weights
        // Referencia: https://developer.apple.com/documentation/accelerate/cblas_sgemm(_:_:_:_:_:_:_:_:_:_:_:_:_:_:)
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        INPUTS_ROWS, WEIGHTS_COLS, INPUTS_COLS,
        1.0f,
        inputs.data(), INPUTS_COLS,
        weights.data(), WEIGHTS_COLS, // ROW MAJOR entao stride (passo entre uma linha e outra) eh o numero de colunas
        0.0f,
        h.data(), WEIGHTS_COLS);

    for (auto &val : h)
    {
        val = logistic_activate(val);
    }

    // Pesos (1x3)
    std::vector<float> v = {-0.017, -0.893, 0.148};
    const int V_ROWS = 1;
    const int V_COLS = 2;

    // Saida (1x1)
    float output = cblas_sdot(WEIGHTS_COLS, h.data(), 1, v.data(), 1);
    output = logistic_activate(output);
    std::cout << "Saida da rede neural: " << output << "\n";

    return 0;
}