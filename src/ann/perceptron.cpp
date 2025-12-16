#include <iostream>
#include <vector>

// Função Produto escalar
float dot_p(const std::vector<float> &x, const std::vector<float> &w)
{
    float result = 0.0f;
    for (size_t i = 0; i < x.size(); ++i)
    {
        result += x[i] * w[i];
    }
    return result;
}

// Função degrau (step function)
float step_func(float x, float offset)
{
    return (x >= offset) ? 1.0f : 0.0f;
}

int main()
{
    // Entradas
    std::vector<float> inputs = {1.0f, 7.0f, 5.0f};

    // Pesos
    std::vector<float> weights = {0.8f, 0.1f, 0.0f};

    // Camada de ativação
    float sum = dot_p(inputs, weights);
    float output = step_func(sum, 1.0f);

    std::cout << "Saída do perceptron: " << output << std::endl;

    return 0;
}