#ifndef CHAR_TOKENIZER_H
#define CHAR_TOKENIZER_H

#define NUM_TOKENS 32

int char_to_id(char c);
char id_to_char(int id);
int char_tokenize(char text[], int tokens[], int max_len);

#endif