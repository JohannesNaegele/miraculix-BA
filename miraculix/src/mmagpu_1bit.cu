
/*
 Authors
 Martin Schlather, schlather@math.uni-mannheim.de


 Copyright (C) 2020 -- 2022  Martin Schlather, Alexander Freudenberg, Johannes Naegele

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
*/

#define MY_METHOD MMA1Bit
#define BitsPerCode 1

/////////////////////////////////////////////////////////
//  DO NOT MOVE OR DELETE INCLUDES OR CHANGE ORDER     //
//  very nasty compile errors caused by redefinitions  //
/////////////////////////////////////////////////////////

#include "mma1bitIntern.h"
#include "intrinsics.h"
#include "IntrinsicsBase.h"
#include "error.h"
#include "MX.h"


// helper function to join decomposed matrices
__global__ void join(
  int32_t *x_1, int32_t *x_2, int32_t *x_3, int32_t *x_4
)
{
  int i = threadIdx.x;
  x_1[i] += 2*(x_2[i] + x_3[i]) + 4*(x_4[i]);
}


// CGM contains two consecutive matrices in 1-bit
static void gpuCrossprodIntern(
  unsigned int *CGM, size_t snps, size_t individuals, double *ans,
  bool warp, bool naive, 
  unsigned int shape, size_t TileSize, size_t n_streams
)
{
  printf("warp: %i shape: %i tilesize: %i streams: %i\n",
    warp, shape, TileSize, n_streams
  );
  // force multiples of 32 byte
  const size_t BytesPerRow = 
    (1 + ((1 + (snps - 1) / CodesPerByte) - 1) / 32) * 32;
  const size_t IntsPerRow = 1 + (BytesPerRow - 1) / sizeof(unsigned int);

  // sanity checks
  // limit TileSize to individuals
  TileSize = TileSize > individuals ? individuals : TileSize;

  // get total GPU memory
  size_t free_mem;
  size_t total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  PRINTF("Using device %s\n", prop.name);

  size_t factor = naive ? 1 : 4;
  // calculates total memory requirements (two inputs with two matrices each)
  size_t req_mem = n_streams * (
    2 * 2 * BytesPerRow * TileSize +
    factor * TileSize * TileSize * sizeof(unsigned int)
  );
  printf("Total memory: %zu\n", total_mem);
  printf("Available memory: %zu\n", free_mem);
  printf("Required memory: %zu\n", req_mem);
  if (req_mem > free_mem) {
    printf("Tilesize: %u\n", TileSize);
    ERR("Not enough global memory available.");
  }

  // Input data
  data_type *d_x;
  data_type *d_y;
  // Buffer for output  
  int32_t *d_val;
  // Buffer for copying back results from device to host
  int32_t *h_val;

  const int size_of_input = 
    BytesPerRow * TileSize * CodesPerByte / MEMORY_FACTOR;
  const int size_of_output = sizeof(int32_t) * TileSize * TileSize;
  // Initialization of buffers:
  // We calculate n_streams of tile matrix multiplications in parallel
  // and allocate the corresponding amount of memory
  cudaMalloc((void **)&d_x, 2 * n_streams * size_of_input);
  cudaMalloc((void **)&d_y, 2 * n_streams * size_of_input);
  cudaMalloc((void **)&d_val, factor * n_streams * size_of_output);
  cudaMallocHost((void **)&h_val, n_streams * size_of_output);
  err_check("Memory allocation: ");

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  #define KERNEL(i, j, k, m, n, o, number) \
  /* Define a CUTLASS GEMM type (wmma) */ \
  using cutlass_gemm_warp_##number = cutlass::gemm::device::Gemm< \
    cutlass::uint1b_t, \
    cutlass::layout::RowMajor, \
    cutlass::uint1b_t, \
    cutlass::layout::ColumnMajor, \
    ElementOutput, \
    cutlass::layout::RowMajor, \
    ElementAccumulator, \
    cutlass::arch::OpClassWmmaTensorOp, \
    cutlass::arch::Sm80, \
    cutlass::gemm::GemmShape<i, j, k>,    /* ThreadblockShape */ \
    cutlass::gemm::GemmShape<m, n, o>,    /* WarpShape        */ \
    cutlass::gemm::GemmShape<8, 8, 128>,  /* InstructionShape */ \
    cutlass::epilogue::thread::LinearCombination< \
        ElementOutput, \
        128 / cutlass::sizeof_bits<ElementOutput>::value, \
        ElementAccumulator, \
        ElementCompute>, \
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, \
    2, 128, 128, false, \
    cutlass::arch::bmmaBitOpAndWMMA>; \
  /* Define a CUTLASS GEMM type (mma) */ \
  using cutlass_gemm_##number = cutlass::gemm::device::Gemm< \
    cutlass::uint1b_t, \
    cutlass::layout::RowMajor, \
    cutlass::uint1b_t, \
    cutlass::layout::ColumnMajor, \
    ElementOutput, \
    cutlass::layout::RowMajor, \
    ElementAccumulator, \
    cutlass::arch::OpClassTensorOp, \
    cutlass::arch::Sm80, \
    cutlass::gemm::GemmShape<i, j, k>,    /* ThreadblockShape */ \
    cutlass::gemm::GemmShape<m, n, o>,    /* WarpShape        */ \
    cutlass::gemm::GemmShape<8, 8, 128>,  /* InstructionShape */ \
    cutlass::epilogue::thread::LinearCombination< \
        ElementOutput, \
        128 / cutlass::sizeof_bits<ElementOutput>::value, \
        ElementAccumulator, \
        ElementCompute>, \
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, \
    2, 128, 128, false, \
    cutlass::arch::bmmaBitOpAnd>;

// problem: 256-bit alignment seems not to be implemented yet
#define KERNEL_NEW(i, j, k, m, n, o, number) \
  /* Define a CUTLASS GEMM type (mma) */ \
  using cutlass_gemm_##number = cutlass::gemm::device::Gemm< \
    cutlass::uint1b_t, \
    cutlass::layout::RowMajor, \
    cutlass::uint1b_t, \
    cutlass::layout::ColumnMajor, \
    ElementOutput, \
    cutlass::layout::RowMajor, \
    ElementAccumulator, \
    cutlass::arch::OpClassTensorOp, \
    cutlass::arch::Sm80, \
    cutlass::gemm::GemmShape<i, j, k>,    /* ThreadblockShape */ \
    cutlass::gemm::GemmShape<m, n, o>,    /* WarpShape        */ \
    cutlass::gemm::GemmShape<16, 8, 256>, /* InstructionShape */ \
    cutlass::epilogue::thread::LinearCombination< \
        ElementOutput, \
        128 / cutlass::sizeof_bits<ElementOutput>::value, \
        ElementAccumulator, \
        ElementCompute>, \
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, \
    3, 128, 128, false, /* this is different! */ \
    cutlass::arch::OpMultiplyAdd>; /* this is different! */

  /*
    ///< Possible specifications:
    For 8x8x128:
      64x64x512_32x32x512
      64x128x512_32x64x512
      128x64x512_64x32x512
      64x256x512_64x64x512
      256x64x512_64x64x512
      128x128x512_64x64x512
      128x256x512_64x64x512
      256x128x512_64x64x512
    For 16x8x256:
      64x256x1024_64x64x1024
      64x128x1024_32x64x1024
      128x64x1024_64x32x1024
      64x64x1024_32x32x1024
      64x64x512_32x32x512
      64x128x512_32x64x512
      128x64x512_64x32x512
      64x256x512_64x64x512
      256x64x512_64x64x512
      128x128x512_64x64x512
      128x256x512_64x64x512
      256x128x512_64x64x512
  */

  KERNEL(         64,  64,  512, 32, 32,  512,  0)
  KERNEL(         64, 128,  512, 32, 64,  512,  1)
  KERNEL(        128,  64,  512, 64, 32,  512,  2)
  KERNEL(         64, 256,  512, 64, 64,  512,  3)
  KERNEL(        256,  64,  512, 64, 64,  512,  4)
  KERNEL(        128, 128,  512, 64, 64,  512,  5)
  KERNEL(        128, 256,  512, 64, 64,  512,  6)
  KERNEL(        256,  64,  512, 64, 64,  512,  7)
  KERNEL_NEW(     64, 256, 1024, 64, 64, 1024,  8)
  KERNEL_NEW(     64, 128, 1024, 32, 64, 1024,  9)
  KERNEL_NEW(    128,  64, 1024, 64, 32, 1024, 10)
  KERNEL_NEW(     64,  64, 1024, 32, 32, 1024, 11)
  KERNEL_NEW(     64,  64,  512, 32, 32,  512, 12)
  KERNEL_NEW(     64, 128,  512, 32, 64,  512, 13)
  KERNEL_NEW(    128,  64,  512, 64, 32,  512, 14)
  KERNEL_NEW(     64, 256,  512, 64, 64,  512, 15)
  KERNEL_NEW(    256,  64,  512, 64, 64,  512, 16)
  KERNEL_NEW(    128, 128,  512, 64, 64,  512, 17)
  KERNEL_NEW(    128, 256,  512, 64, 64,  512, 18)
  KERNEL_NEW(    256,  64,  512, 64, 64,  512, 19)

  #define GEMM_VERSION(type, number) \
    {cutlass_gemm_##type##number CutlassGemm; \
    cutlass_gemm_##type##number::Arguments args( \
      {x_tile_size, y_tile_size, k_gemm}, \
      {x_dev, k_gemm}, \
      {y_dev, k_gemm}, \
      {d_val, y_tile_size}, \
      {d_val, y_tile_size}, \
      {alpha, beta}); \
      return CutlassGemm(args, nullptr, stream);}

  #define LOOP(type, number) case #number : GEMM_VERSION(#type, #number);

  // anonymous function for CUTLASS GEMM choice
  auto gemm_operator = [](
        int x_tile_size,
        int y_tile_size,
        int k_gemm,
        data_type* x_dev,
        data_type* y_dev,
        int32_t* d_val,
        int alpha,
        int beta,
        int warp,
        unsigned int shape,
        cudaStream_t stream
      ) {
    if (warp) {
      switch (shape) {
        case 1 : GEMM_VERSION(warp_, 1);
        case 2 : GEMM_VERSION(warp_, 2);
        case 3 : GEMM_VERSION(warp_, 3);
        case 4 : GEMM_VERSION(warp_, 4);
        case 5 : GEMM_VERSION(warp_, 5);
        case 6 : GEMM_VERSION(warp_, 6);
        case 7 : GEMM_VERSION(warp_, 7);
        default : GEMM_VERSION(warp_, 0);
      }
    } else {
      switch (shape) {
        case 1 : GEMM_VERSION(, 1);
        case 2 : GEMM_VERSION(, 2);
        case 3 : GEMM_VERSION(, 3);
        case 4 : GEMM_VERSION(, 4);
        case 5 : GEMM_VERSION(, 5);
        case 6 : GEMM_VERSION(, 6);
        case 7 : GEMM_VERSION(, 7);
        case 8 : GEMM_VERSION(, 8); // this one gives calculation failures
        case 9 : GEMM_VERSION(, 9);
        case 10 : GEMM_VERSION(, 10);
        case 11 : GEMM_VERSION(, 11);
        case 12 : GEMM_VERSION(, 12);
        case 13 : GEMM_VERSION(, 13);
        case 14 : GEMM_VERSION(, 14);
        case 15 : GEMM_VERSION(, 15);
        case 16 : GEMM_VERSION(, 16);
        case 17 : GEMM_VERSION(, 17);
        case 18 : GEMM_VERSION(, 18);
        case 19 : GEMM_VERSION(, 19);
        default : GEMM_VERSION(, 0);
      }
    }
  };
  if (warp) {
    printf("Using wmma version\n\n");
  } else {
    printf("Using mma version\n\n");
  }

  size_t n_threads = n_streams < 1 + (individuals - 1) / TileSize ? 
    n_streams : 1 + (individuals - 1) / TileSize;

  // Main loop
  #ifdef DO_PARALLEL
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
  #endif
  for (int64_t i = 0; i < individuals; i += TileSize)
  {
    int threadidx = omp_get_thread_num();
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    cudaStreamSynchronize(stream);

    if (err != cudaSuccess)
      ERR("Stream couldn't be created");

    // Pointer to the first element of current rows
    unsigned int *x_1 = (CGM + i * IntsPerRow);
    unsigned int *x_2 = (CGM + i * IntsPerRow + individuals * IntsPerRow);
    data_type *x_dev_1 = d_x + 2 * threadidx * TileSize * BytesPerRow;
    data_type *x_dev_2 = x_dev_1 + TileSize * BytesPerRow;
    data_type *y_dev_1 = d_y + 2 * threadidx * TileSize * BytesPerRow;
    data_type *y_dev_2 = y_dev_1 + TileSize * BytesPerRow;

    // Number of rows in matrix
    size_t const rows_left = individuals - i;
    // Size x of current tile
    size_t const x_tile_size = TileSize < rows_left ? TileSize : rows_left;

    cudaMemcpyAsync(
      x_dev_1, x_1, x_tile_size * BytesPerRow, cudaMemcpyHostToDevice, stream
    );
    cudaMemcpyAsync(
      x_dev_2, x_2, x_tile_size * BytesPerRow, cudaMemcpyHostToDevice, stream
    );
    std::string copy_1 = ((std::string) "Copy 1: ") + std::to_string(i) + 
      " x_tile_size: " + std::to_string(x_tile_size);
    err_check(copy_1.c_str());
    cudaStreamSynchronize(stream);
    unsigned int *y_1 = CGM + i * IntsPerRow;
    unsigned int *y_2 = CGM + i * IntsPerRow + individuals * IntsPerRow;

    for (int64_t j = i; j < individuals; j += TileSize)
    {
      // Same as above with y
      size_t const columns_left = individuals - j;
      size_t const y_tile_size = 
        TileSize < columns_left ? TileSize : columns_left;

      std::string copy_2 = ((std::string) "Copy 2: ") + std::to_string(j) +
        " y_tile_size: " + std::to_string(y_tile_size);
      y_1 = (CGM + j * IntsPerRow);
      y_2 = (CGM + j * IntsPerRow + individuals * IntsPerRow);        
      cudaMemcpyAsync(
        y_dev_1, y_1, y_tile_size * BytesPerRow, cudaMemcpyHostToDevice, stream
      );
      // One can perform this copy later after the first two gemm calls and 
      // reuse y_dev_1 in order to save global memory. This however is 
      // (depending on the omp criticals) slower or equally fast.
      cudaMemcpyAsync(
        y_dev_2, y_2, y_tile_size * BytesPerRow, cudaMemcpyHostToDevice, stream
      );
      err_check(copy_2.c_str());
      cudaStreamSynchronize(stream);
      
      /// initialize gemm arguments 
      // COMPRESSION_GPU from MMAGPU was removed for convenience 
      // since everything is in 1-bit
      int k_gemm;
      k_gemm = int(BytesPerRow * CodesPerByte);
      // compute Multiplication
      if (naive) {
        // omp critical is necessary bc host kernel is not thread safe
        #ifdef DO_PARALLEL
        #pragma omp critical
        #endif
        // one can omit these brackets and leave the cudaStreamSynchronize
        // call out of omp critical, this however turns out to be slower;
        // probably because of worse memory movement
        {cutlass::Status status_1 = gemm_operator(
          int(x_tile_size), int(y_tile_size), k_gemm,
          x_dev_1,
          y_dev_1,
          d_val + threadidx * TileSize * TileSize,
          1, 0,
          warp,
          shape,
          stream
        );
        err_check("Calculation 1:");
        cudaStreamSynchronize(stream);}
        #ifdef DO_PARALLEL
        #pragma omp critical
        #endif
        {cutlass::Status status_3 = gemm_operator(
          int(x_tile_size), int(y_tile_size), k_gemm,
          x_dev_2,
          y_dev_1,
          d_val + threadidx * TileSize * TileSize,
          2, 1,
          warp,
          shape,
          stream
        );
        err_check("Calculation 3:");
        cudaStreamSynchronize(stream);}
        #ifdef DO_PARALLEL
        #pragma omp critical
        #endif
        {cutlass::Status status_2 = gemm_operator(
          int(x_tile_size), int(y_tile_size), k_gemm,
          x_dev_1,
          y_dev_2,
          d_val + threadidx * TileSize * TileSize,
          2, 1,
          warp,
          shape,
          stream
        );
        err_check("Calculation 2:");
        cudaStreamSynchronize(stream);}
        #ifdef DO_PARALLEL
        #pragma omp critical
        #endif
        {cutlass::Status status_4 = gemm_operator(
          int(x_tile_size), int(y_tile_size), k_gemm,
          x_dev_2,
          y_dev_2,
          d_val + threadidx * TileSize * TileSize,
          4, 1,
          warp,
          shape,
          stream
        );
        cudaStreamSynchronize(stream);
        err_check("Calculation 4:");} 

        // copy results back to host
        cudaMemcpyAsync(h_val + threadidx * TileSize * TileSize,
          d_val + factor * threadidx * TileSize * TileSize,
          TileSize * TileSize * sizeof(int32_t),
          cudaMemcpyDeviceToHost, stream
        );
      } else {
        size_t offset = factor * threadidx * TileSize * TileSize;
        #ifdef DO_PARALLEL
        #pragma omp critical
        #endif
        cutlass::Status status = gemm_operator(
          int(2*TileSize), int(2*TileSize), k_gemm,
          x_dev_1,
          y_dev_1,
          d_val + offset,
          1, 0,
          warp,
          shape,
          stream
        );
        cudaStreamSynchronize(stream);
        err_check("Calculation (CUTLASS):");
        // this is inefficient but even without it
        // there is no sufficient improvement
        for (size_t array = 0; array < TileSize; array++) {
          join<<<1, TileSize>>>(
            d_val + offset + 2*array*TileSize,
            d_val + offset + (TileSize + array)*2*TileSize,
            d_val + offset + 2*array*TileSize + TileSize,
            d_val + offset + (TileSize + array)*2*TileSize + TileSize
          );
        }
        cudaStreamSynchronize(stream);
        err_check("Calculation (join):");
        
        // copy results back to host
        for (size_t array = 0; array < x_tile_size; array++) {
          cudaMemcpyAsync(
            h_val + threadidx * TileSize * TileSize + array*y_tile_size,
            d_val + offset + 2*array*TileSize,
            y_tile_size * sizeof(int32_t),
            cudaMemcpyDeviceToHost, stream
          );
        }
      }
        
      err_check("Copy back:");
      cudaStreamSynchronize(stream);

      // in some cases CUDA does not return errors on check;
      // therefore we check heuristically
      if (*(h_val + threadidx * TileSize * TileSize) == 0)
      {
        printf("Computation failed at thread %d, (%d,%d)\n", threadidx, i, j);
        print_kernel<<<1, 1>>>(
          (int32_t *) d_val + threadidx * TileSize * TileSize);
        PRINTF("x_1: %u ", *x_1);
        PRINTF("y_1: %u\n", *y_1);
        print_kernel<<<1, 1>>>((int32_t *) x_dev_1);
        print_kernel<<<1, 1>>>((int32_t *) y_dev_1);
      }
      
      // Loop over tile and store values in output matrix
      #ifdef DO_PARALLEL
      #pragma omp parallel for num_threads(n_streams) schedule(static)
      #endif
      for (int64_t di = 0; di < x_tile_size; ++di)
      {
        for (int64_t dj = 0; dj < y_tile_size; ++dj)
        {
          // Get result
          const auto Mij = 
            *(h_val + threadidx * TileSize * TileSize + dj + di * y_tile_size);
          // Create pointers to the (symmetric) output matrix
          double *ans0 = ans + (i + di),
                *ans1 = ans + (i + di) * individuals;
          // Store result in ouput matrix
          ans0[(j + dj) * individuals] = ans1[j + dj] = (double)Mij;
        }
      }
    }
    cudaStreamDestroy(stream);
  }

  // Free memory
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_val);
  cudaFreeHost(h_val);
}

static void crossprodIntern(Uint *CM, Uint snps, Uint individuals, double *ans, bool warp, bool naive, unsigned int shape, size_t tilesize, size_t n_streams) {
  // Initialize host pointers and copy input data cuda managed memory
  Uint *h_CM;
  const size_t BytesPerIndiv = UnitsPerIndiv256_1(snps) * BytesPerUnit;
  cudaMallocHost((void **)&h_CM, 2 * individuals * BytesPerIndiv);
  MEMCOPY(h_CM, CM, 2 * individuals * BytesPerIndiv);

  gpuCrossprodIntern(h_CM, snps, individuals, ans, warp, naive, shape, tilesize, n_streams);
  cudaFreeHost(h_CM);
}

bool useMMA1Bit(snpcoding method) {
  return method == MMA1Bit ? true : false;
}

void crossprod_mma1Bit(Uint *CGM, Uint snps, Uint individuals, double *ans, bool warp, bool naive, unsigned int shape, size_t tilesize, size_t n_streams) {
  crossprodIntern(CGM, snps, individuals, ans, warp, naive, shape, tilesize, n_streams);
}

static bool check_7_5() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major < 7) | (deviceProp.major == 7 & deviceProp.minor < 5))
  {
    ERR("No GPU kernel for compute capability 7.0 or lower yet.");
    return false;
  }
  return true;
  // helpful values
}

static SEXP matrix_start_Intern(Uint snps, Uint individuals, SEXP VARIABLE_IS_NOT_USED file) {
  SEXP Code;
  PROTECT(Code = CreateEmptyCodeVector(snps, individuals, MY_METHOD));
  UNPROTECT(1);
  return Code;
}

SEXP matrix_start_mma1Bit(Uint snps, Uint individuals, SEXP file) {
  check_7_5();
  return matrix_start_Intern(snps, individuals, file);
}