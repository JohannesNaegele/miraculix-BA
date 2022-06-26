// R CMD INSTALL /home/johannes/Nextcloud/Documents/GitHub/miraculix-1/miraculix --configure-args="USE_GPU=yes USE_AVX=yes"
// scp johannes@capsicum.math.uni-mannheim.de:~/miraculix_1.2.3.tar.gz ~/Downloads


// kernel call
// matmul1Bit<<<dimGrid, dimBlock>>>(
//     deviceA, deviceB, deviceC, individuals, snps, snps,
//     individuals, individuals, individuals);

// Beispiel fuer custom kernel (warp-level-gemm)
// https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
// siehe aehnliches Beispiel:
// https://github.com/NVIDIA/cutlass/blob/master/media/docs/gemm_api.md
// siehe außerdem
// https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
// leichter verstaendlich, aber mit wmma:
// https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu
// https://github.com/BigNerd95/CUDASamples/blob/master/samples/0_Simple/cudaTensorCoreGemm/cudaTensorCoreGemm.cu
// high-performance example: cudaTensorCoreGemm
// void foo() {
//   using Mma = cutlass::gemm::warp::DefaultMmaTensorOp<
//   GemmShape<64, 64, 16>,
//   half_t, LayoutA, // GEMM A operand
//   half_t, LayoutB, // GEMM B operand
//   float, RowMajor // GEMM C operand
//   >;
//   // problem dimensions
//   int lda = Mma::Shape::kM;
//   int ldb = Mma::Shape::kN;
//   // smem tiles
//   __shared__ ElementA smem_buffer_A[Mma::Shape::kM * GemmK];
//   __shared__ ElementB smem_buffer_B[Mma::Shape::kN * GemmK];
//   // Construct iterators into SMEM tiles
//   Mma::IteratorA iter_A({smem_buffer_A, lda}, thread_id); // I guess this is pointer offset
//   Mma::IteratorB iter_B({smem_buffer_B, ldb}, thread_id);
//   Mma::FragmentA frag_A;
//   Mma::FragmentB frag_B;
//   Mma::FragmentC accum;
//   Mma mma;
//   accum.clear();
//   #pragma unroll 1
//   for (int k = 0; k < GemmK; k += Mma::Shape::kK) {
//     iter_A.load(frag_A); // Load fragments from A and B matrices
//     iter_B.load(frag_B);
//     ++iter_A; ++iter_B; // Advance along GEMM K to next tile in A
//     // and B matrices
//     // Compute matrix product
//     mma(accum, frag_A, frag_B, accum);
//   }
// }

// Anfang custom kernel
// calculates the upper triangel of the 1-bit matrix multiplication for A and B where dim(A^T) = (snps, individuals) = dim(B)
// i.e. numAColumns = numBRows = snps & numCColumns = numCRows = numARows = numBColumns = individuals
// __global__ void matmul1Bit(unsigned int* A, unsigned int* B, size_t snps, size_t individuals,
//   double* ans, size_t TileSize) {

//   int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y,
//     Row = by * TileSize + ty, Col = bx * TileSize + tx;
//   // Tile using a 2D grid
//   int warpM = (bx * blockDim.x + tx) / WARP_SIZE; // tx should be divisible by 32
//   int warpN = (by * blockDim.y + ty);
//   // define unique thread_id for iterators
//   int thread_id = tx + ty*blockDim.y
//   // check because of symmetry: last entry in tile should be in triangel
//   if (by * TileSize <= bx * TileSize + blockDim.y) {
//     // TensorOp 8-by-8-by-128
//     using Mma = cutlass::gemm::warp::DefaultMmaTensorOp<
//     GemmShape<8, 8, 128>,
//     bin1_t, LayoutA, // GEMM A operand
//     bin1_t, LayoutB, // GEMM B operand
//     int32_t, RowMajor // GEMM C operand
//     >;
//     // problem dimensions (tensor core)
//     int lda = Mma::Shape::kM;
//     int ldb = Mma::Shape::kN;
//     // shared memory for tiles
//     __shared__ float ds_A[TileSize][TileSize];
//     __shared__ float ds_B[TileSize][TileSize];
//     // Construct iterators into SMEM tiles
//     Mma::IteratorA iter_A({ds_A, lda}, thread_id);
//     Mma::IteratorB iter_B({ds_B, ldb}, thread_id);
//     Mma::FragmentA frag_A;
//     Mma::FragmentB frag_B;
//     Mma::FragmentC accum;
//     Mma mma;

//     for (int m = 0; m < (snps - 1) / TileSize + 1; ++m) {
//       if (Row < individuals && m * TileSize + tx < snps)
//         ds_A[ty][tx] = A[Row * snps + m * TileSize + tx];
//       else
//         ds_A[ty][tx] = 0;
//       if (Col < individuals && m * TileSize + ty < snps)
//         ds_B[ty][tx] = B[(m * TileSize + ty) * individuals + Col];
//       else
//         ds_B[ty][tx] = 0;

//       __syncthreads();
//       // accum.clear();
//       #pragma unroll 1
//       for (int k = 0; k < TileSize; ++k) {
//         // hier tensor core operation
//         iter_A.load(frag_A); // Load fragments from A and B matrices
//         iter_B.load(frag_B);
//         ++iter_A; ++iter_B; // Advance along GEMM K to next tile in A
//         // and B matrices
//         // Compute matrix product
//         mma(accum, frag_A, frag_B, accum);
//       }
//       __syncthreads();
//     }
//     if (Row < individuals && Col < individuals) {
//         C[Row * individuals + Col] = accum[0];
//     }
//   }
// }

// #define WARP_SIZE 32
// const int WMMA_M = 8;
// const int WMMA_N = 8;
// const int WMMA_K = 128;

// {
//   using namespace nvcuda;
//   using uint1b_t = cutlass::uint1b_t;
//   __global__ void wmma1Bit_no_tiling(uint1b_t *x_1, uint1b_t *x_2, uint1b_t *y_1, uint1b_t *y_2, uint1b_t *dest, int M, int N, int K) {
//     // Leading dimensions. Packed with no transpositions.
//     int lda = M;
//     int ldb = K;
//     int ldc = N;

//     // Tile using a 2D grid
//     int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
//     int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

//     // Declare the fragments
//     wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint1b_t, wmma::row_major> a_frag;
//     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint1b_t, wmma::col_major> b_frag;
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag_1;
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag_2;
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag_4;

//     wmma::fill_fragment(acc_frag_1, 0.0f);
//     wmma::fill_fragment(acc_frag_2, 0.0f);
//     wmma::fill_fragment(acc_frag_4, 0.0f);

//     // Loop over k
//     for (int i = 0; i < K; i += WMMA_K) {
//       int aRow = warpM * WMMA_M;
//       int aCol = i;

//       int bRow = i;
//       int bCol = warpN * WMMA_N;

//       // Bounds checking
//       if (aRow < M && aCol < K && bRow < K && bCol < N) {
//         /// 1
//         // Load the inputs
//         wmma::load_matrix_sync(a_frag, x_1 + aRow + aCol * lda, lda);
//         wmma::load_matrix_sync(b_frag, y_1 + bRow + bCol * ldb, ldb);
//         // Perform the matrix multiplication
//         wmma::bmma_sync(acc_frag_1, a_frag, b_frag, acc_frag_1, nvcuda::wmma::experimental::bmmaBitOpAND, 
//           nvcuda::wmma::experimental::bmmaAccumulateOpPOPC);
//         /// 2.1
//         // Load the inputs
//         wmma::load_matrix_sync(b_frag, y_2 + bRow + bCol * ldb, ldb);
//         // Perform the matrix multiplication
//         wmma::bmma_sync(acc_frag_2, a_frag, b_frag, acc_frag_2, nvcuda::wmma::experimental::bmmaBitOpAND, 
//           nvcuda::wmma::experimental::bmmaAccumulateOpPOPC);
//         /// 4
//         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
//         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
//         // Perform the matrix multiplication
//         wmma::bmma_sync(acc_frag_4, a_frag, b_frag, acc_frag_4, nvcuda::wmma::experimental::bmmaBitOpAND, 
//           nvcuda::wmma::experimental::bmmaAccumulateOpPOPC);
//         /// 2.2
//         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
//         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
//         // Perform the matrix multiplication
//         wmma::bmma_sync(acc_frag_2, a_frag, b_frag, acc_frag_2, nvcuda::wmma::experimental::bmmaBitOpAND, 
//           nvcuda::wmma::experimental::bmmaAccumulateOpPOPC);
//       }
//     }

//     // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
//     int cRow = warpM * WMMA_M;
//     int cCol = warpN * WMMA_N;

//     if (cRow < M && cCol < N) {
//       for(int i=0; i < acc_frag_1.num_elements; i++) {
//           acc_frag_1.x[i] = acc_frag_1.x[i] + 2 * acc_frag_2.x[i] + 4 * acc_frag_4.x[i];
//       }
//       // Store the output
//       wmma::store_matrix_sync(dest + cRow + cCol * ldc, acc_frag_1, ldc, wmma::mem_row_major);
//     }
//   }
// } // namespace nvcuda