# CUDA accelerated burrows-wheeler transform
The burrows wheeler transform is an algorithm commonly used in bioinformatics and data compression applications.
It involves sorting large amounts of data, which generally doesn't paralellize well.
We can bitonic sort, which generally has a runtime of **O(*n* log<sup>2</sup>(*n*))** if not parallelized, worse than traditional **O(*n* log(*n*))** algos, but it can be greatly parallelized.

## Implementation
Requires CUDA and c++11

This implementation uses `'$'` as the end marker character, as is commonly used in bioinformatics, since our data only contains A,C,T,G. if you need a different ETX character you can change it in "bwt.hpp"

Normally the bwt algorithm has **O(*n<sup>2</sup>*)** Memory requirements, but we can get around that by just storing each "rotation number" of each string and using that to generate the entire string only when we need it for sorting so we don't need to store it, bringing the memory requirement to just **O(*n*)**

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
|100Kbp |0.129s     | 0.99s    |
|1Mbp   |1.977s     | 0.228s    |
|10Mbp  |26.996s    | 2.862s    |
|100Mbp |6m21.266s   | 0m56.382s* |

*When I run the gpu version by itself without running the cpu version like in demo.cpp, the time is 39.528s. not sure what is causing the difference


Some tests of just the gpu, I did not run the cpu version for these tests because I thought it would take too long

|Size:  | GPU time: |
|---    |---        |
| 500Mbp|5m23.737s   |
| 800Mpb|11m0.889s   |
| 1Bbp  |14m9.651s   |


for small sequences below 1Mbp, it probably isn't worth the extra overhead and latency from sending the data to the GPU, but for larger datasets, it offers a 7-10x speedup.
