#pragma once

#include <cuda.h>

__device__ __forceinline__ unsigned int ACTIVE_MASK()
{
#if !defined(USE_ROCM)
    return __activemask();
#else
// will be ignored anyway
    return 0xffffffff;
#endif
}

#if defined(USE_ROCM)
__device__ __forceinline__ unsigned long long int WARP_BALLOT(int predicate)
{
return __ballot(predicate);
}
#else
__device__ __forceinline__ unsigned int WARP_BALLOT(int predicate, unsigned int mask = 0xffffffff)
{
#if !defined(USE_ROCM)
    return __ballot_sync(mask, predicate);
#else
    return __ballot(predicate);
#endif
}
#endif

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if !defined(USE_ROCM)
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if !defined(USE_ROCM)
    return __shfl_sync(mask, value, srcLane, width);
#else
    return __shfl(value, srcLane, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_UP(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if !defined(USE_ROCM)
    return __shfl_up_sync(mask, value, delta, width);
#else
    return __shfl_up(value, delta, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if !defined(USE_ROCM)
    return __shfl_down_sync(mask, value, delta, width);
#else
    return __shfl_down(value, delta, width);
#endif
}

#if defined(USE_ROCM)
template<>
__device__ __forceinline__ int64_t WARP_SHFL_DOWN<int64_t>(int64_t value, unsigned int delta, int width , unsigned int mask)
{
  //(HIP doesn't support int64_t). Trick from https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
  int2 a = *reinterpret_cast<int2*>(&value);
  a.x = __shfl_down(a.x, delta);
  a.y = __shfl_down(a.y, delta);
  return *reinterpret_cast<int64_t*>(&a);
}
#endif

/**
 * For CC 3.5+, perform a load using __ldg
 */
template <typename T>
__device__ __forceinline__ T doLdg(const T* p) {
#if __CUDA_ARCH__ >= 350 && !defined(USE_ROCM)
  return __ldg(p);
#else
  return *p;
#endif
}
