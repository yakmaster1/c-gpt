#include <stdlib.h>

#include "embedding.h"
#include "../tokenizer/char_tokenizer.h"

float *init_embed_matrix(int rows, int columns)
{
    float *embed_matrix = malloc(sizeof(float) * EMBED_DIM * NUM_TOKENS);
    if (embed_matrix == NULL) exit(EXIT_FAILURE);
    for (int i = 0; i < (rows * columns); i++)
    {
        embed_matrix[i] = ((float) rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    return embed_matrix;
}

float *embed_matrix_at(float* matrix, int row, int col, int elements_in_col) 
{
    return &matrix[row * elements_in_col + col];
}

void dispose_embed_matrix(float *matrix)
{
    if (matrix == NULL) return;
    free(matrix);
}