#include "tokenizer/char_tokenizer.h"
#include "matrix/matrix.h"
#include "kernel/kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_INPUT_LEN 256

void get_input(char input[], int len)
{
    printf("User: ");
    if (fgets(input, sizeof(char) * len, stdin)) 
    {
        input[strcspn(input, "\n")] = '\0';
    }
}

void add_position_embedding(float *input_matrix, int token_len, float *position_matrix, int embed_dim)
{
    for (int i = 0; i < token_len; i++)
    {
        for (int j = 0; j < embed_dim; j++)
        {
            float token = *matrix_get_at(input_matrix, j, i, embed_dim);
            float pos_encoding = *matrix_get_at(position_matrix, j, i, embed_dim);
            matrix_set_at(input_matrix, j, i, embed_dim, token+pos_encoding);
        }
    }
}

int main() {
    float *embed_matrix = init_fmatrix(EMBED_DIM, NUM_TOKENS);
    float *position_matrix = init_fmatrix(EMBED_DIM, MAX_INPUT_LEN);
    
    char input[MAX_INPUT_LEN];
    get_input(input, MAX_INPUT_LEN);
    
    int tokens[MAX_INPUT_LEN];
    int token_len = char_tokenize(input, tokens, MAX_INPUT_LEN);

    float *input_matrix = create_input_matrix(tokens, token_len, embed_matrix, EMBED_DIM);
    
    gpu_addvectors(input_matrix, position_matrix, token_len * EMBED_DIM);
    
    dispose_fmatrix(embed_matrix);
    dispose_fmatrix(position_matrix);

    dispose_fmatrix(input_matrix);
    return 0;
}