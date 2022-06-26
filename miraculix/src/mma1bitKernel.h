
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
along with this program; if not, writne to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.  
*/

// This software contains source code provided by NVIDIA Corporation.

// GPU configuration.
#define WARP_SIZE 32

// MMA matrix tile dimensions interpreted as int (MMA_K = K/32)
#define MMA_M 8
#define MMA_N 8
#define MMA_K 4

// GEMM configuration.
#define M_TILES 128 // 1024/8
#define N_TILES 128 // 1024/8
#define K_TILES 3908 // 1000448/((8 Bit) *4)

// sizes of whole matrix in global mem (should be changed to variables later)
#define M_GLOBAL (MMA_M * M_TILES)
#define N_GLOBAL (MMA_N * N_TILES)
#define K_GLOBAL (MMA_K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

// can be multiple of 2
// purpose: ensure 256-bit loads and load efficient amount depending on shmem size
#define CHUNK_K 6

// Distribute warps among block dimension
#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

// mma tiles per warp (we want 64x64 tile)
#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

// (global) memory tile: number of subtiles (wmma)
#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

// (global) memory row length
#define GLOBAL_MEM_STRIDE N_GLOBAL

// (shared) memory stride
#define SHMEM_STRIDE (MMA_N * BLOCK_ROW_TILES)

// (shared) memory row offset from one warp to another
#define SHMEM_OFFSET (MMA_N * WARP_ROW_TILES)

/*
  The macro below is used to shift rows of the A matrix and columns of the B matrix
  in shared memory to minimize possible bank conflicts.
  Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
  data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
  is not specified for that function, each lane in the warp can read one or multiple matrix
  elements from different matrix rows or columns.
  For shared memory, such access can result in bank conflicts if different rows / columns
  of the matrix map to the same bank. By shifting each row and column by a few bytes, we
  make sure that they map to different banks, thus reducing the number of possible bank
  conflicts. 
  The number of 8 4-byte int elements is chosen as the minimum possible shift because we 
  must keep each row and column 256-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
*/
#define SKEW_INT 0
// helper value to divide warps
#define WARP_PART (WARPS_PER_BLOCK/4)

// using uint1b_t = cutlass::uint1b_t;
using namespace nvcuda;

__global__ void compute_gemm(int *x_1, int *x_2, int *y_1, int *y_2, int *dest)
{
  // !!!
  extern __shared__ int shmem[][CHUNK_K * MMA_K + SKEW_INT];

  // !!!
  // Warp and lane identification.
  const size_t warpId = threadIdx.x / WARP_SIZE;
  const size_t laneId = threadIdx.x % WARP_SIZE;

  // !!!
  // Offset in shared memory from one input matrix to another.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * MMA_M;

  // !!!
  // This pointer is used to access the dest matrix tiles this warp computes.
  int *shmem_warp_tile_ptr = (int*)&shmem[0][0] + 
    (warpId/BLOCK_ROW_WARPS) * SHMEM_STRIDE * MMA_K * BLOCK_ROW_WARPS + (warpId%BLOCK_ROW_WARPS) * SHMEM_OFFSET;

  // !!!
  // one warp 32 (in m) * 16 (in n) = 512; SHMEM_STRIDE * MMA_K = 64 * 4 = 256
  // old case: WARP_ROW_TILES*WARP_COL_TILES 512*4 = 8*16*16
  // This pointer is used to stream the dest matrices block-wide tile to and from shared memory.
  int *shmem_warp_stream_ptr = (int*)&shmem[0][0] + warpId * SHMEM_STRIDE * MMA_K;

  // Each CTA slides along the 64 x 64 tiles from the top left corner of the matrix to the
  // right and down, and selects the next tile to compute. Once there's no such tile,
  // all warps in this CTA exit.
  for(unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {

    // !!!
    // be careful: has to be adjusted if m != n
    const unsigned int block_tile_i = ((block_pos * BLOCK_COL_TILES) / N_TILES) * BLOCK_COL_TILES;
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;  
    // Stop when there are no more dest matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }
    // !!!
    // This warp's pointer to the dest matrix data to copy memory from and to shared memory.
    // gmem_idx moves down M rows per warp
    const size_t gmem_idx = (block_tile_i + warpId) * MMA_M * GLOBAL_MEM_STRIDE + block_tile_j * MMA_N;
    
    // These fragments will accumulate the result of A and B matrix fragment multiplications
    // along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, 128, int> c_1[WARP_COL_TILES][WARP_ROW_TILES];
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, 128, int> c_2[WARP_COL_TILES][WARP_ROW_TILES];
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, 128, int> c_4[WARP_COL_TILES][WARP_ROW_TILES];

    // !!!
    // Select what warp copies what matrix from global to shared memory.
    // Warps 0-1 copy the x_1 matrix, warps 2-3 copy x_2, warps 4-5 copy y_1, warps 6-7 copy y_2
    // -> % 2
    // Every warp takes care of 32 rows -> 8 = WARP_ROW_TILES * WARP_COL_TILES
    //
    int *warp_ptr = 
      (warpId < 1*WARP_PART) ? 
      (&x_1[block_tile_i * MMA_K * K_GLOBAL] + MMA_K * K_GLOBAL * (warpId % 2) * 8) :
      (warpId < 2*WARP_PART) ? 
      (&x_2[block_tile_i * MMA_K * K_GLOBAL] + MMA_K * K_GLOBAL * (warpId % 2) * 8) :
      (warpId < 3*WARP_PART) ? 
      (&y_1[block_tile_j * MMA_K * K_GLOBAL] + MMA_K * K_GLOBAL * (warpId % 2) * 8) :
      (&y_2[block_tile_j * MMA_K * K_GLOBAL] + MMA_K * K_GLOBAL * (warpId % 2) * 8);

    // Go through the global K dimension by a fixed step (CHUNK_K) at a time.
    #pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // ???
      // Copy slices of the A and B matrices to shared memory.
      // The warps in the CTA are divided in quarters to copy x_1, x_2, y_1 and y_2      
      size_t shmem_idx = 
        (warpId < 1*WARP_PART) ?
        ((MMA_M * (warpId % (WARPS_PER_BLOCK/4)) * 4)) :
        (warpId < 2*WARP_PART) ?
        ((MMA_M * (warpId % (WARPS_PER_BLOCK/4)) * 4) + 1 * shmem_idx_b_off) :
        (warpId < 3*WARP_PART) ?
        ((MMA_N * (warpId % (WARPS_PER_BLOCK/4)) * 4) + 2 * shmem_idx_b_off) :
        ((MMA_N * (warpId % (WARPS_PER_BLOCK/4)) * 4) + 3 * shmem_idx_b_off);

      // ???
      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      // (laneId % (WARP_SIZE/2)) is in {0, ..., 15} which gives us 16 * (4 ints) = 64 ints
      int4 *lane_ptr = (int4*)(warp_ptr + tile_k * MMA_K + (laneId / (WARP_SIZE/2)) * K_GLOBAL) + (laneId % (WARP_SIZE/2));
      // Shift the second half of the warp to the next row / column in the shared memory.
      shmem_idx += laneId / (WARP_SIZE/2);

      // // load from global memory, (WARP_SIZE/2)* (2 Warps) * (2 rows each) = (64 Byte)
      // #pragma unroll
      // for(int i = 0; i < (WARP_SIZE/2); i++) {
      //   // Copy 16 bytes at once in each lane.
      //   *((int4*) &shmem[shmem_idx][0] + (laneId % (WARP_SIZE/2))) = *lane_ptr;
      //   // Advance the global memory pointer and the shared memory index.
      //   lane_ptr = (int4*) ((int*) lane_ptr + K_GLOBAL * 2);
      //   shmem_idx += 2;
      // }
      // __syncthreads();
      #pragma unroll
      // (8 warps) * (MMA_N) = 64 = (4 Byte * 16 threads)
      for (int i = 0; i < MMA_M; i++) {
        *((int*)((int4*) (shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) + 0) = 2147483647;
        *((int*)((int4*) (shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) + 1) = 2147483647;
        *((int*)((int4*) (shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) + 2) = 2147483647;
        *((int*)((int4*) (shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) + 3) = 2147483647;
      }
      __syncthreads();

      // !? (WARP_ROW_TILES * MMA_N) ist mir nicht klar
      // if (threadIdx.x == 0 && blockIdx.x == 0) {
      //   printf("tile_ptr_2 on GPU: %i\n", *tile_ptr_2);
      // }
      // Compute a grid of dest matrix tiles in each warp.
      #pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, 128, wmma::experimental::precision::b1, wmma::row_major> a_1[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, 128, wmma::experimental::precision::b1, wmma::row_major> a_2[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, 128, wmma::experimental::precision::b1, wmma::col_major> b_1[WARP_ROW_TILES];
        wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, 128, wmma::experimental::precision::b1, wmma::col_major> b_2[WARP_ROW_TILES];
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a_1 = 0*shmem_idx_b_off + (warpId/2) * MMA_M * 2 + (i * MMA_M);
          size_t shmem_idx_a_2 = 1*shmem_idx_b_off + (warpId/2) * MMA_M * 2 + (i * MMA_M);
          const int *tile_ptr_1 = &shmem[0][0];
          const int *tile_ptr_2 = &shmem[0][0];
          // stride has to be multiple of 128, we load MMA_M*MMA_K elements each
          wmma::load_matrix_sync(a_1[i], tile_ptr_1, MMA_K * CHUNK_K + SKEW_INT);
          wmma::load_matrix_sync(a_2[i], tile_ptr_2, MMA_K * CHUNK_K + SKEW_INT);
          #pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be reused
              // against the other A matrix fragments.
              size_t shmem_idx_b_1 = 2*shmem_idx_b_off + (WARP_ROW_TILES * MMA_N) * (warpId%2) + (j * MMA_N);
              size_t shmem_idx_b_2 = 3*shmem_idx_b_off + (WARP_ROW_TILES * MMA_N) * (warpId%2) + (j * MMA_N);
              // const int *tile_ptr_1 = &shmem[shmem_idx_b_1][k_step * MMA_K];
              // const int *tile_ptr_2 = &shmem[shmem_idx_b_2][k_step * MMA_K];
              // load has to be multiple of 128, we load MMA_M*MMA_K elements each
              wmma::load_matrix_sync(b_1[j], tile_ptr_1, MMA_K * CHUNK_K + SKEW_INT);
              wmma::load_matrix_sync(b_2[j], tile_ptr_2, MMA_K * CHUNK_K + SKEW_INT);
            }
            // Some kind of restructuring with the loads might increase performance
            wmma::bmma_sync(
              c_1[i][j], a_1[i], b_1[j], c_1[i][j], 
              wmma::experimental::bmmaBitOpAND, wmma::experimental::bmmaAccumulateOpPOPC
            );
            wmma::bmma_sync(
              c_2[i][j], a_2[i], b_1[j], c_2[i][j], 
              wmma::experimental::bmmaBitOpAND, wmma::experimental::bmmaAccumulateOpPOPC
            );
            wmma::bmma_sync(
              c_2[i][j], a_1[i], b_2[j], c_2[i][j], 
              wmma::experimental::bmmaBitOpAND, wmma::experimental::bmmaAccumulateOpPOPC
            );
            wmma::bmma_sync(
              c_4[i][j], a_2[i], b_2[j], c_4[i][j], 
              wmma::experimental::bmmaBitOpAND, wmma::experimental::bmmaAccumulateOpPOPC
            );
          }
        }
      }
      __syncthreads();
    }

    // Store the c fragments to shared memory.
    #pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
      #pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {          
        // Calculate the c matrix, c_1[i][j].num_elements should be MMA_M*MMA_N
        #pragma unroll
        for (int t = 0; t < c_1[i][j].num_elements; t++) {
          c_1[i][j].x[t] += 2*c_2[i][j].x[t] + 4*c_4[i][j].x[t];
          // c_1[i][j].x[t] = 1;
        }
        /// ???
        int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * MMA_M + j * MMA_N;
        wmma::store_matrix_sync(tile_ptr, c_1[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }
    __syncthreads();

    // Now that shared memory contains all the dest tiles, stream them to global memory.
    int *dst_gmem_warp_stream_ptr = &dest[gmem_idx];
    #pragma unroll
    // (8 warps) * (MMA_N) = 64 = (4 Byte * 16 threads)
    for (int i = 0; i < MMA_M; i++) {
      *((int4*) (dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
          *((int4*) (shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }
    __syncthreads();
  }
}

// A call of compute_gemm would look like this
  // /*
  //   Compute the right amount of shared memory to request. We need shared
  //   memory to hold per-CTA dest matrix tiles, and to cache per-CTA 
  //   chunks of the A and B matrices. Therefore, the right amount to 
  //   request is the maximum of those two numbers.
  // */
  // enum {
  //   SHMEM_SZ = (size_t) MAX(
  //     4 * sizeof(int) * (BLOCK_COL_TILES * MMA_M) * 
  //     (CHUNK_K * MMA_K + SKEW_INT),
  //     sizeof(int) * MMA_M * BLOCK_ROW_TILES * MMA_N * BLOCK_COL_TILES
  //   )
  // };
  // compute_gemm<<<(1024*1024)/(64*64), THREADS_PER_BLOCK, SHMEM_SZ>>>(
  //   (int*) ((void*) x_dev_1), (int*) ((void*) x_dev_2),
  //   (int*) ((void*) y_dev_1), (int*) ((void*) y_dev_2),
  //   (int*) ((d_val + threadidx * TileSize * TileSize)));        
  // err_check("Calculation gemm:");
  // cudaStreamSynchronize(stream);