IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17
#CUDA_FLAGS=-lineinfo -gencode arch=compute_50,code=sm_50 # Adjust compute capability as needed
CUDA_FLAGS=
RELEASE_FLAGS=-O3
DEBUG_FLAGS=-g -G

.PHONY: clean build run debug release

build: COMPILER_FLAGS += $(CUDA_FLAGS)
build: simple

debug: COMPILER_FLAGS += $(DEBUG_FLAGS)
debug: build

release: COMPILER_FLAGS += $(RELEASE_FLAGS)
release: build

simple: simple.cu simple.h
	$(COMPILER) $(COMPILER_FLAGS) simple.cu -o $@

clean:
	rm -f simple

run:
	./simple

all: clean build run
