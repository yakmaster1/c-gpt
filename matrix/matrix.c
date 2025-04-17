#include "matrix.h"

#include <stdlib.h>

float *init_fmatrix(int rows, int columns)
{
    float *matrix = malloc(sizeof(float) * rows * columns);
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
}

float *matrix_get_at(float* matrix, int row, int col, int col_size) 
{
    return &matrix[col * col_size + row];
}

void matrix_set_at(float* matrix, int row, int col, int col_size, float value) 
{
    matrix[col * col_size + row] = value;
}

float *create_input_matrix(int tokens[], int token_len, float *embed_matrix, int embed_dim)
{
    float *input_matrix = malloc(sizeof(float) * token_len * embed_dim);
    if (input_matrix == NULL) exit(EXIT_FAILURE);
    for (int i = 0; i < token_len; i++)
    {
        int token = tokens[i];
        for (int j = 0; j < embed_dim; j++)
        {
            float value = *matrix_get_at(embed_matrix, j, token, embed_dim);
            matrix_set_at(input_matrix, j, i, embed_dim, value);
        }
    }
    return input_matrix;
}