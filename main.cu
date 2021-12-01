#include <iostream>
#include "from_pytorch.cuh"

template<typename bitwise_t>
__device__ __forceinline__ bitwise_t highBitsFrom(int start) {
  bitwise_t ones = ~(bitwise_t(0));
  bitwise_t lowBits = (bitwise_t(1) << bitwise_t(start)) - 1;
  return ones & ~lowBits;
}

// This kernel must be launched with 1024 threads
template<typename scalar_t, typename bitwise_t, typename index_t, int radix_bits, bool order, typename start_offset_t>
__launch_bounds__(1024)
__global__ void countRadixMultipleBlock(
  scalar_t* data,
  start_offset_t startOffset,
  index_t sliceSize,
  index_t withinSliceStride,
  index_t numSlices,
  int itemsPerBlock,
  int startBit,
  bitwise_t *desired,
  index_t *counts  // counts for each slice for each radix, needs to be initialized 0 before this kernel
) {
  constexpr int radix_size = (1 << radix_bits);
  constexpr int radix_mask = radix_size - 1;
  static_assert(radix_size <= 32, "This implementation assumes radix_size <= 32");

  int blocksPerSlice = (sliceSize + itemsPerBlock - 1) / itemsPerBlock;
  int slice = blockIdx.x / blocksPerSlice;
  if (slice >= numSlices) return;
  int blockInSlice = blockIdx.x % blocksPerSlice;
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  scalar_t *sliceData = data + startOffset(slice);

  // Each thread count number of items locally.
  // There are 32 * 32 thread-local counts in each block.
  bitwise_t desiredMask = highBitsFrom<bitwise_t>(startBit + radix_bits);
  index_t local_count = 0;  // The lane i's thread-local variable stores the count of radix i for each warp.
  for (index_t i = blockInSlice * 1024 + threadIdx.x; i < sliceSize; i += blocksPerSlice * 1024) {
    scalar_t raw_value = doLdg(sliceData + withinSliceStride * i);
    bitwise_t value = TopKTypeConfig<scalar_t>::convert(raw_value);
    bool hasVal = ((value & desiredMask) == desired[slice]);
    bitwise_t digitInRadix = at::cuda::Bitfield<bitwise_t>::getBitfield(value, startBit, radix_bits);
    #pragma unroll
    for (uint32_t j = 0; j < radix_size; ++j) {
      bool vote = hasVal && (digitInRadix == j);
      int count = __popc(WARP_BALLOT(vote, ACTIVE_MASK()));
      local_count += (j == lane ? count : 0);
    }
  }

  // Each warp write its values to shared memory.
  __shared__ index_t shared_counts[32][radix_size];
  if (lane < radix_size) {
    shared_counts[warp][lane] = local_count;
  }
  __syncthreads();

  // Each warp computes the block level count, and uses atomic add to update global counts.
  if (warp < radix_size) {
    index_t *global_count = counts + radix_size * slice + warp;
    local_count = shared_counts[lane][warp];
    // TODO: warp shuffle down
    gpuAtomicAddNoReturn(global_count, local_count);
  }
}


int main() {
  std::cout << "Hello World!!!" << std::endl;
}
