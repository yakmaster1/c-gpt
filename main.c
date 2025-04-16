#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer/char_tokenizer.h"
#include "embedding/embedding.h"

#define MAX_INPUT_LEN 256

void get_input(char input[], int len)
{
    printf("User: ");
    if (fgets(input, sizeof(char) * len, stdin)) 
    {
        input[strcspn(input, "\n")] = '\0';
    }
}

float *create_input_matrix(int tokens[], int token_len, float *embed_matrix, int embed_dim)
{
    float *input_matrix = malloc(sizeof(float) * token_len * embed_dim);
    if (input_matrix == NULL) exit(EXIT_FAILURE);
    int k = 0;
    for (int i = 0; i < token_len; i++)
    {
        int token = tokens[i];
        for (int j = 0; j < embed_dim; j++)
        {
            float value = *embed_matrix_at(embed_matrix, j, token, embed_dim);
            input_matrix[k] = value;
            k++;
        }
    }
    return input_matrix;
}

int main() {
    char input[MAX_INPUT_LEN];
    get_input(input, MAX_INPUT_LEN);

    int tokens[MAX_INPUT_LEN];
    int token_len = char_tokenize(input, tokens, MAX_INPUT_LEN);

    float *embed_matrix = init_embed_matrix(EMBED_DIM, NUM_TOKENS);

    float *input_matrix = create_input_matrix(tokens, token_len, embed_matrix, EMBED_DIM);
    
    dispose_embed_matrix(embed_matrix);
    return 0;
}