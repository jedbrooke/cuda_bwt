#include "bwt.hpp"
#include <iostream>
#include <list>

const int blockSize = 256;

/* 
    generates a list of ints where the value of each item in the list is it's index,
    unless it is greater than the sequence length, in which case it is set to -1.
*/
__global__ void generate_table(int* table, int table_size, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < table_size; i+=stride) {
        if( i < n) {
            table[i] = i;
        } else {
            table[i] = -1;
        }
    }
}

/* 
    compare two rotations of the input sequence lexicographically
    a, b are index pointers to the index of the start of each rotation 
*/

__device__ bool compare_rotations(const int& a, const int& b, char* genome, int n) {
    if (a < 0) {
        return false;
    }
    if (b < 0) {
        return true;
    }
    for(size_t i = 0; i < n; i++) {
        if (genome[(a + i) % n] != genome[(b + i) % n]) {
            return genome[(a + i) % n] < genome[(b + i) % n];
        }
    }
    return false;
}


__global__ void bitonic_sort_step(int* table, int table_size, int j, int k, char* genome, int n) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;
    if(i < table_size) {
        if(ixj > i) {
            if ((i & k) == 0) {
                if (compare_rotations(table[ixj], table[i], genome, n)) {
                    int temp = table[i];
                    table[i] = table[ixj];
                    table[ixj] = temp;
                }
            }
            if ((i & k) != 0) {
                if (compare_rotations(table[i], table[ixj], genome, n)) {
                    int temp = table[i];
                    table[i] = table[ixj];
                    table[ixj] = temp;
                }
            }
        }
    }
}

/* 
    returns a std::pair object
    the first item is the burrows wheeler transform of the input sequence in a std::string,
    the second item is the suffix array of the input sequence, represented as indicies of the given suffix, as an int*
    
    assumes input sequence already has ETX appended to it.
*/

std::pair<std::string,int*> bwt_with_suffix_array(const std::string sequence) {
    
    const int n = sequence.size();
    int table_size = sequence.size();
    // round the table size up to a power of 2 for bitonic sort
    table_size--;
    table_size |= table_size >> 1;
    table_size |= table_size >> 2;
    table_size |= table_size >> 4;
    table_size |= table_size >> 8;
    table_size |= table_size >> 16;
    table_size++;

    int* table_cu;
    cudaMalloc(&table_cu, table_size * sizeof(int));
    int* table = (int*) malloc(table_size * sizeof(int));
    
   
    int numBlocks = (table_size + blockSize - 1) / blockSize;

    generate_table<<<numBlocks,blockSize>>>(table_cu, table_size, n);
    // wait for cuda kernel to finish
    cudaDeviceSynchronize();

    
    char* sequence_cu;
    cudaMalloc(&sequence_cu, n * sizeof(char));
    cudaMemcpy(sequence_cu, sequence.c_str(), n * sizeof(char), cudaMemcpyHostToDevice);
    
    int j,k;
    for (k = 2; k <= table_size; k <<= 1) {
        for (j = k >> 1; j > 0; j = j >> 1) {
            bitonic_sort_step<<<numBlocks,blockSize>>>(table_cu, table_size, j, k, sequence_cu, n);
        }
    }
    cudaDeviceSynchronize();

    cudaMemcpy(table, table_cu, table_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(table_cu);
    cudaFree(sequence_cu);

    // reconstruct transformed sequence from suffix array
    std::string transformed_sequence;
    for(int i = 0; i < n; i++) {
        transformed_sequence += sequence[(n + table[i] - 1) % n];
    }
    return std::make_pair(transformed_sequence,table);
}

std::string bwt(const std::string sequence) {
    auto data = bwt_with_suffix_array(sequence);
    free(data.second);
    return data.first;

}