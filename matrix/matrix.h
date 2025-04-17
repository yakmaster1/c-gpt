#ifndef MATRIX_H
#define MATRIX_H

#define EMBED_DIM 32

float *init_fmatrix(int rows, int columns);
void dispose_fmatrix(float *matrix);
float *matrix_get_at(float* matrix, int row, int col, int col_size);
void matrix_set_at(float* matrix, int row, int col, int col_size, float value);

float *create_input_matrix(int tokens[], int token_len, float *embed_matrix, int embed_dim);

#endif
