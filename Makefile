NVCC=nvcc
NVFLAGS=-O3
CXX=g++

default: bwt.o

demo: bwt.o demo.cpp
	$(CXX) -o demo -g demo.cpp bwt.o --std=c++11 -lcudart -O3

bwt.o: bwt.cu
	$(NVCC) -c bwt.cu $(NVFLAGS)

clean:
	rm -rf bwt.o demo

