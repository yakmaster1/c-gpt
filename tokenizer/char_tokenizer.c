#include "char_tokenizer.h"

int char_to_id(char c) 
{
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c == ' ') return 26;
    if (c == '.') return 27;
    if (c == ',') return 28;
    if (c == '?') return 29;
    if (c == '!') return 30;
    if (c == '\n') return 31;
    return -1;
}

char id_to_char(int id) 
{
    if (id >= 0 && id <= 25) return 'a' + id;
    if (id == 26) return ' ';
    if (id == 27) return '.';
    if (id == 28) return ',';
    if (id == 29) return '?';
    if (id == 30) return '!';
    if (id == 31) return '\n';
    return '#';
}

int char_tokenize(char *text, int *tokens, int max_len) 
{
    int i = 0;
    while (text[i] != '\0' && i < max_len) 
    {
        tokens[i] = char_to_id(text[i]);
        i++;
    }
    return i;
}