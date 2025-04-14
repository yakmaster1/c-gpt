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

int main() {
    char input[MAX_INPUT_LEN];
    get_input(input, MAX_INPUT_LEN);

    int tokens[MAX_INPUT_LEN];
    int token_len = char_tokenize(input, tokens, MAX_INPUT_LEN);

    float *embed_matrix = init_embed_matrix(EMBED_DIM, NUM_TOKENS);
    
    dispose_embed_matrix(embed_matrix);
    return 0;
}