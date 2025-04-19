#include "input.h"

#include <cstdio>

void get_input(char input[], int len)
{
    printf("User: ");
    if (fgets(input, sizeof(char) * len, stdin)) 
    {
        input[strcspn(input, "\n")] = '\0';
    }
    return;
}