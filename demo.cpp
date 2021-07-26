#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <chrono>
#include <cstdlib>

#include "bwt.hpp"

#define NOW() std::chrono::high_resolution_clock::now()

std::string bwt_cpu(const std::string sequence) {
    const size_t n = sequence.size();
    const char* c_sequence = sequence.c_str();

    std::vector<int> table(n);

    for (size_t i = 0; i < n; i++){
        table[i] = i;
    }

    std::list<int> sorted_table(table.begin(), table.end());
    sorted_table.sort([c_sequence,n](const int& a, const int& b) -> bool {
        for(size_t i = 0; i < n; i++) {
            if(c_sequence[(a + i) % n] != c_sequence[(b + i) % n]) {
                return c_sequence[(a + i) % n] < c_sequence[(b + i) % n];
            }
        }
        return false;
    });
    

    std::string transformed_sequence;


    for(auto r = sorted_table.begin(); r != sorted_table.end(); ++r){
        transformed_sequence += c_sequence[(n + *r - 1) % n];
    }
    return transformed_sequence;
}




int main(int argc, char const *argv[])
{
    std::string alphabet("ATCG");
    const int N = 1E5;
    std::cout << "running sample of " << N << std::endl;
    char* sequence = (char*) malloc((N+1) * sizeof(char));
    for (size_t i = 0; i < N; i++) {
        sequence[i] = alphabet[rand() % alphabet.size()];
    }
    sequence[N] = ETX;
    
    // TODO: make time dynamic so we don't end up with 0
    std::cout << "running cpu version..." << std::endl;
    auto start = NOW();
    auto cpu_seq = bwt_cpu(sequence);
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(NOW() - start);

    std::cout << "running gpu version..." << std::endl;
    start = NOW();
    auto gpu_seq = bwt(sequence);
    auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(NOW() - start);

    std::cout << "cpu version: " << cpu_time.count() << "ms" << std::endl;
    std::cout << "gpu version: " << gpu_time.count() << "ms" << std::endl;

    // TODO: make this optional
    if(cpu_seq.compare(gpu_seq) == 0) {
        std::cout << "outputs match!" << std::endl;
    } else {
        std::cout << "uh oh, outputs mismatch, something went wrong!" << std::endl;
    }



    return 0;
}
