#include "interface.h"
#include "tokenizer/char_tokenizer.h"

int tokenize(char *text, int *tokens, int max_len) 
{
    return char_tokenize(text, tokens, max_len);
}