#include <stdio.h>
#include <string.h>

#include "interface.h"

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
    int token_len = tokenize(input, tokens, MAX_INPUT_LEN);

    for (int i = 0; i < token_len; i++)
    {
        printf("%d ", tokens[i]);
    }

    return 0;
}