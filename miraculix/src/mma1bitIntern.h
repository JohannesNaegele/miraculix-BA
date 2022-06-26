
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

// This file supplies the includes for 1-bit matrix multiplication

// for CUDA custom kernel
#include <cuda.h>

// CUTLASS includes
#if defined(CUTLASS_ARCH_WMMA_ENABLED)
// includes for nvcuda::wmma needed for binarized matrix multiply.
#include "cutlass/wmma_array.h"
#endif
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

namespace cutlass{
namespace arch{
struct bmmaBitOpAnd;
template <>
struct Mma<
  gemm::GemmShape<8, 8, 128>,      ///< Size of the matrix product
  32,                                       ///< Size per output
  uint1b_t,                                 ///< ElementA
  layout::RowMajor,                         ///< LayoutA
  uint1b_t,                                 ///< ElementB
  layout::ColumnMajor,                      ///< LayoutB
  int32_t,                                  ///< ElementC
  layout::RowMajor,                         ///< LayoutC
  bmmaBitOpAnd                              ///< Operator
> {

  using Shape = gemm::GemmShape<8, 8, 128>;

  using ElementA = uint1b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint1b_t, 32>;

  using ElementB = uint1b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint1b_t, 32>;

  using ElementC = int32_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int32_t, 2>;

  using Operator = bmmaBitOpAnd;
  using ArchTag = Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int32_t const *C = reinterpret_cast<int const *>(&c);
  int32_t *D = reinterpret_cast<int *>(&d);

  asm volatile(
    "{\n\t"
    "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
    "{%0,%1}, {%2}, {%3}, {%4,%5};\n\t"
    "}\n"
    : "=r"(D[0]), "=r"(D[1])
    : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#else
  assert(0);
#endif
  }
};

struct bmmaBitOpAndWMMA;
template <
typename Shape_, 
typename LayoutA_, 
typename LayoutB_,
typename LayoutC_>
struct Wmma<
  Shape_,                                   ///< Size of the matrix product
  uint1b_t,                                 ///< ElementA
  LayoutA_,                                 ///< LayoutA
  uint1b_t,                                 ///< ElementB
  LayoutB_,                                 ///< LayoutB
  int32_t,                                  ///< ElementC
  LayoutC_,                                 ///< LayoutC
  bmmaBitOpAndWMMA                          ///< Operator
> {
// for some reason I can't specify SM80 here
#if defined(CUTLASS_ARCH_WMMA_SM75_ENABLED)
  using Shape = Shape_;

  using ElementA = uint1b_t;
  using LayoutA = LayoutA_;

  using ElementB = uint1b_t;
  using LayoutB = LayoutB_;

  using ElementC = int32_t;
  using LayoutC = LayoutC_;
  
  using Operator = bmmaBitOpAndWMMA;
  using ArchTag = Sm80;

  // check supported wmma shape for the given multiplicand data types
  static_assert(
    platform::is_same<gemm::GemmShape<8, 8, 128>, Shape>::value,
    "Supported list of wmma operator shape for b1 multiplicands is: 8x8x128"
  );


  // Wmma Fragment
  using FragmentA = nvcuda::wmma::fragment<
    nvcuda::wmma::matrix_a,
    Shape::kM,
    Shape::kN,
    Shape::kK,
    typename CutlassToWmmaDataType<ElementA>::Type,
    typename CutlassToWmmaLayout<LayoutA>::Layout>;

  using FragmentB = nvcuda::wmma::fragment<
    nvcuda::wmma::matrix_b,
    Shape::kM,
    Shape::kN,
    Shape::kK,
    typename CutlassToWmmaDataType<ElementB>::Type,
    typename CutlassToWmmaLayout<LayoutB>::Layout>;

  using FragmentC = nvcuda::wmma::fragment<
    nvcuda::wmma::accumulator,
    Shape::kM,
    Shape::kN,
    Shape::kK,
    typename CutlassToWmmaDataType<ElementC>::Type>;
  
  /// Performs a nvcuda::wmma matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    FragmentA const &A, 
    FragmentB const &B, 
    FragmentC const &C) const {
      nvcuda::wmma::bmma_sync(D, A, B, C,
        nvcuda::wmma::experimental::bmmaBitOpAND, 
        nvcuda::wmma::experimental::bmmaAccumulateOpPOPC
      );
  }

#else
  static_assert(false, "SM80 or higher required");
#endif

};
} // namespace arch
} // namespace cutlass

// this provides the custom CUDA kernel
// #include "mma1bitKernel.h"

/////////////////////////////////////////////////////////
//  DO NOT MOVE OR DELETE INCLUDES OR CHANGE ORDER     //
//  very nasty compile errors caused by redefinitions  //
/////////////////////////////////////////////////////////

#include <omp.h>
#include "error.h"
#include "MX.h"
#include "intrinsics.h"
#include "IntrinsicsBase.h"
#include "xport_import.h"
#include "align.h"
#include "haplogeno.h"
#include "Haplo.h"
#include <inttypes.h>
#include <string>

#define MEMORY_FACTOR 8L // this is 1/sizeof(1 Bit)

using data_type = cutlass::uint1b_t;

static void err_check(const char* string){
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) printf("%s %s\n", string, cudaGetErrorString(err)); 
}
__global__ static void print_kernel(int32_t* ptr){
  printf("Value on GPU is %d \n", *ptr);
}