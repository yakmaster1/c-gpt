#include "matrix.h"
#include "../kernel/kernel.h"

#include <cstdlib>

void init_sfmatrix(float *matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        matrix[i] = ((float) rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    return;
}

float *init_fmatrix(int rows, int columns)
{
    float *matrix = (float*)malloc(sizeof(float) * rows * columns);
    if (matrix == NULL) exit(EXIT_FAILURE);
    for (int i = 0; i < (rows * columns); i++)
    {
        matrix[i] = ((float) rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    return matrix;
}

void dispose_fmatrix(float *matrix)
{
    if (matrix == NULL) return;
    free(matrix);
    return;
}

float *matrix_get_at(float* matrix, int row, int col, int col_size) 
{
    return &matrix[col * col_size + row];
}

void matrix_set_at(float* matrix, int row, int col, int col_size, float value) 
{
    matrix[col * col_size + row] = value;
    return;
}

float *create_input_matrix(int tokens[], int token_len, float *cudaW_e, int embed_dim, int num_tokens)
{
    int size = embed_dim * num_tokens;
    float *W_e = (float*)malloc(sizeof(float) * size);
    if (W_e == NULL) exit(EXIT_FAILURE);
    kernel_memcpydevicetohost(W_e, cudaW_e, size * sizeof(float));
    
    float *input_matrix = (float*)malloc(sizeof(float) * token_len * embed_dim);
    if (input_matrix == NULL) exit(EXIT_FAILURE);
    for (int i = 0; i < token_len; i++)
    {
        int token = tokens[i];
        for (int j = 0; j < embed_dim; j++)
        {
            float value = *matrix_get_at(W_e, j, token, embed_dim);
            matrix_set_at(input_matrix, j, i, embed_dim, value);
        }
    }
    free(W_e);
    return input_matrix;
}