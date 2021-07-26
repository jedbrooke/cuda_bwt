# CUDA accelerated burrows-wheeler transform
The burrows wheeler transform is an algorithm commonly used in bioinformatics and data compression applications.
It involves sorting large amounts of data, which generally doesn't paralellize well.
However, using bitonic sort, which generally has a runtime of **O(*n* log<sup>2</sup>(*n*))** if not parallelized, worse than traditional **O(*n* log(*n*))** algos. However, it can be greatly parallelized.

## Implementation
Requires CUDA and c++11

This implementation uses `'$'` as the end marker character, as is commonly used in bioinformatics, since our data only contains A,C,T,G. if you need a different ETX character you can change it in "bwt.hpp"

Normally the bwt algorithm has **O(*n<sup>2</sup>*)** Memory requirements, but we can get around that by just storing each "rotation number" of each string and using that to generate the entire string only when we need it for sorting so we don't need to store it, bringin the memory requirement to just **O(*n*)**

## Running
the included Makefile has rules for building the object file and the demo/benchmark executable

`make` will generate `bwt.o`, to use in other projects just include `"bwt.hpp"` and link the object file.

`make demo` will generate the `demo` executable which will run a comparison between the gpu version, and a cpu version based around `std::list.sort()` for the bwt.


## So how fast is it?
testing platform:
* **OS**: Ubuntu 20.04
* **CPU**: Intel i7 4930k @ 4.0Ghz
* **GPU**: Nvidia Tesla M40 @ 1.1Ghz

| Size:  | Cpu time:  |Gpu time:|
|---    |---        |---        |
|100Kbp |0.132s     | 0.114s    |
|1Mbp   |1.956s     | 0.237s    |
|10Mbp  |29.386s    | 2.985s    |
|100Mbp |6m53.08s   | 0m58.167s |

for small sequences below 1Mbp, it probably isn't worth the extra overhead and latency from sending the data to the GPU, but for larger datasets, it offers a 7-10x speedup.
