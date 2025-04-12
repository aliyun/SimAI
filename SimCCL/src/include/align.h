/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALIGN_H_
#define NCCL_ALIGN_H_

/**
 * @brief Macro DIVUP is used to compute the ceiling result of a division.
 *
 * This macro achieves the upward rounding of the result of x divided by y by performing integer division after 
 * adding half of the divisor y. It is mainly used in situations where you need to divide x by y and want 
 * the result to always be rounded up. For example, in resource allocation or calculating batch sizes, 
 * you may need to round up to the next integer when division is not exact.
 *
 * @param x Dividend, the number to be divided.
 * @param y Divisor, the denominator in the division operation.
 * @return The result of (x + y - 1) / y, which is the ceiling of the division result.
 */
#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

#define ALIGN_POWER(x, y) \
    ((x) > (y) ? ROUNDUP(x, y) : ((y)/((y)/(x))))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

#if !__CUDA_ARCH__
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
#endif

template<typename X, typename Y, typename Z = decltype(X()+Y())>
__host__ __device__ constexpr Z divUp(X x, Y y) {
  return (x+y-1)/y;
}

template<typename X, typename Y, typename Z = decltype(X()+Y())>
__host__ __device__ constexpr Z roundUp(X x, Y y) {
  return (x+y-1) - (x+y-1)%y;
}

// assumes second argument is a power of 2
template<typename X, typename Z = decltype(X()+int())>
__host__ __device__ constexpr Z alignUp(X x, int a) {
  return (x+a-1) & Z(-a);
}

#endif
