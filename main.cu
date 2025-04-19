#include "tokenizer/char_tokenizer.h"
#include "matrix/matrix.h"
#include "kernel/kernel.h"
#include "attention/attention.h"
#include "input/input.h"

#define MAX_INPUT_LEN 256

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

    float *input_matrix = create_input_matrix(tokens, token_len, cudaW_e, EMBED_DIM, NUM_TOKENS);
    float *cuda_embed_matrix = 0;
    gpu_init(&cuda_embed_matrix, input_matrix, EMBED_DIM * token_len);
    dispose_fmatrix(input_matrix);
    
    gpu_addPositionMatrix(cuda_embed_matrix, cudaW_p, EMBED_DIM * token_len);

    float *cuda_logit = 0;
    gpu_init_zero(&cuda_logit, token_len * token_len);

    float *cuda_query = 0;
    gpu_init_zero(&cuda_query, QUERY_KEY_DIM * token_len);

    float *cuda_key = 0;
    gpu_init_zero(&cuda_key, QUERY_KEY_DIM * token_len);

    // Compute Queries
    gpu_cublas_matmul(cudaW_q, cuda_embed_matrix, cuda_query, QUERY_KEY_DIM, EMBED_DIM, token_len);

    // Compute Keys
    gpu_cublas_matmul(cudaW_k, cuda_embed_matrix, cuda_key, QUERY_KEY_DIM, EMBED_DIM, token_len);

    kernel_cudafree(cuda_logit);

    kernel_cudafree(cuda_query);
    kernel_cudafree(cuda_key);
    
    kernel_cudafree(cudaW_e);
    kernel_cudafree(cudaW_p);
    kernel_cudafree(cudaW_q);
    kernel_cudafree(cudaW_k);

    kernel_cudafree(cuda_embed_matrix);
    return 0;
}