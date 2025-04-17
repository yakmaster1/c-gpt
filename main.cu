#include "tokenizer/char_tokenizer.h"
#include "matrix/matrix.h"
#include "kernel/kernel.h"
#include "attention/attention.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define MAX_INPUT_LEN 256

void get_input(char input[], int len)
{
    printf("User: ");
    if (fgets(input, sizeof(char) * len, stdin)) 
    {
        input[strcspn(input, "\n")] = '\0';
    }
    return;
}

void init_gpu_pointers(float **cudaW_e, float **cudaW_p, float **cudaW_q, float **cudaW_k)
{
    float W_e[EMBED_DIM * NUM_TOKENS];
    float W_p[EMBED_DIM * MAX_INPUT_LEN];
    float W_q[QUERY_KEY_DIM * EMBED_DIM];
    float W_k[QUERY_KEY_DIM * EMBED_DIM];
    init_sfmatrix(W_e, EMBED_DIM * NUM_TOKENS);
    init_sfmatrix(W_p, EMBED_DIM * MAX_INPUT_LEN);
    init_sfmatrix(W_q, QUERY_KEY_DIM * EMBED_DIM);
    init_sfmatrix(W_k, QUERY_KEY_DIM * EMBED_DIM);

    gpu_init(cudaW_e, W_e, EMBED_DIM * NUM_TOKENS);
    gpu_init(cudaW_p, W_p, EMBED_DIM * MAX_INPUT_LEN);
    gpu_init(cudaW_q, W_q, QUERY_KEY_DIM * EMBED_DIM);
    gpu_init(cudaW_k, W_k, QUERY_KEY_DIM * EMBED_DIM);    
    return;
}

int main() {
    float *cudaW_e = 0;
    float *cudaW_p = 0;
    float *cudaW_q = 0;
    float *cudaW_k = 0;
    init_gpu_pointers(&cudaW_e, &cudaW_p, &cudaW_q, &cudaW_k);
    
    char input[MAX_INPUT_LEN];
    get_input(input, MAX_INPUT_LEN);
    
    int tokens[MAX_INPUT_LEN];
    int token_len = char_tokenize(input, tokens, MAX_INPUT_LEN);

    float *embed_matrix = create_input_matrix(tokens, token_len, cudaW_e, EMBED_DIM, NUM_TOKENS);
    
    gpu_addDeviceToHostMatrix(embed_matrix, token_len * EMBED_DIM, cudaW_p);

    dispose_fmatrix(embed_matrix);
    kernel_cudafree(cudaW_e);
    kernel_cudafree(cudaW_p);
    kernel_cudafree(cudaW_q);
    kernel_cudafree(cudaW_k);
    return 0;
}