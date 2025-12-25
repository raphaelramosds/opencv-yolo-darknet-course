#include <iostream>
#include <iomanip>
#include <vector>
#include <cblas.h>

struct OutputTable
{
    size_t N_ROWS;
    size_t N_COLS;
    std::vector<std::vector<float>> inputs;
    std::vector<float> y_calc;
    std::vector<int> y_real;
    std::vector<float> errors;

    OutputTable(size_t rows, size_t cols)
        : N_ROWS(rows), N_COLS(cols),
          inputs(rows, std::vector<float>(cols)),
          y_calc(rows),
          y_real(rows),
          errors(rows) {}
};

static inline void print_output_table(const OutputTable &output_table)
{
    unsigned int i = 0;

    std::cout << std::setw(5) << "x1"
              << std::setw(5) << "x2"
              << std::setw(10) << "saida"
              << std::setw(10) << "real"
              << std::setw(10) << "erro"
              << std::endl;

    std::cout << std::string(40, '-') << std::endl;

    for (; i < output_table.N_ROWS; i++)
    {
        std::cout << std::setw(5) << output_table.inputs[i][0]
                  << std::setw(5) << output_table.inputs[i][1]
                  << std::setw(10) << output_table.y_calc[i]
                  << std::setw(10) << output_table.y_real[i]
                  << std::setw(10) << output_table.errors[i]
                  << std::endl;
    }

    // calcular erro absoluto médio
    float mae = cblas_sasum(output_table.N_ROWS, output_table.errors.data(), 1) / output_table.N_ROWS;
    std::cout << "EMA = " << std::fixed << std::setprecision(2) << mae << "\n\n";
}

// Transpor matriz
template <class T>
void trans_m(const std::vector<std::vector<T>> &matrix, std::vector<std::vector<T>> &output)
{
    int rows = matrix.size();
    if (rows == 0)
        return;

    int cols = matrix[0].size();
    output.resize(cols, std::vector<T>(rows));

    unsigned int i = 0;
    for (; i < rows; ++i)
    {
        unsigned int j = 0;
        for (; j < cols; ++j)
        {
            output[j][i] = matrix[i][j];
        }
    }
}

// Multiplicar matrizes ROW MAJOR
// NOTE 1. A_COLS = B_ROWS
// NOTE 2. NAO SE ESQUECA DE ALOCAR A MATRIZ Y!!
template <class T>
void mult_m(
    const std::vector<T> &A,
    const unsigned int A_ROWS, const unsigned int A_COLS,
    const std::vector<T> &B,
    const unsigned int B_COLS,
    std::vector<T> &Y)
{
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        A_ROWS, B_COLS, A_COLS,
        1.0f,
        A.data(), A_COLS,
        B.data(), B_COLS,
        0.0f,
        Y.data(), B_COLS);
}
