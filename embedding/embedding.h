#ifndef EMBEDDING_H
#define EMBEDDING_H

#define EMBED_DIM 32

float *init_embed_matrix(int rows, int columns);
float *embed_matrix_at(float* matrix, int row, int col, int cols);
void dispose_embed_matrix(float *matrix);

#endif
