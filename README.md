# miraculix-BA
This repository contains open source code for the bachelor thesis 'GPU-Berechnung von genetischen Verwandtschaftsmatrizen durch Warp-Level-Matrixmultiplikation' submitted on 27th of June 2022. 

## Requirements
* R 4.0 or higher
* gcc 9.0 or higher
* CUDA 11.0 or higher
* NVIDIA GPU of compute capability 8.0 or higher (Ampere architecture or newer)
* A local copy of cutlass (https://github.com/NVIDIA/cutlass)

## Installation
* R CMD INSTALL RandomFieldsUtils --configure-args="USE_GPU=yes USE_AVX=yes"
* R CMD INSTALL miraculix --configure-args="CXX_FLAGS='-mavx2 -DGPU_DEV' USE_GPU='yes'"

## Structure
* miraculix contains all functions for the calculation of the genomic relationship matrix. GPU code can be found in files starting with the 'mma' prefix.
* R files in the main folder provide benchmarks, use-cases and syntax hints. 

## Known bugs:
The custom CUDA kernel does not work properly yet.

## Helpful links
https://github.com/NVIDIA/cutlass/blob/master/media/docs/functionality.md
https://github.com/NVIDIA/cutlass/blob/master/test/unit/gemm/device/gemm_b1t_b1n_s32n_tensor_op_s32_sm75.cu

 
