#include <iostream>
#include <cstdlib>
#include <random>
#include <algorithm>

template<typename T>
T *generateInput(int64_t numSlices, int64_t sliceSize, std::vector<int64_t> ks) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<int64_t> dist(0, sliceSize);

  T *h = new T[numSlices * sliceSize];
  for (int64_t i = 0; i < numSlices; i++) {
    T *hi = h + i * sliceSize;
    int64_t k = ks[i];
    hi[k] = k;
    for (int64_t j = 0; j < sliceSize; j++) {
      if (j < k) {
        T r;
        do {
          r = static_cast<T>(dist(rng));
        } while (r > k);
        hi[j] = r;
      } else if (j > k) {
        T r;
        do {
          r = static_cast<T>(dist(rng));
        } while (r < k);
        hi[j] = r;
      }
    }
    for (int64_t l = 0; l < sliceSize; l++) {
      std::cout << hi[l] << ", ";
    }
    std::cout << std::endl;
    std::shuffle(hi, hi + sliceSize, rng);
  }
  delete [] h;
  return nullptr;
}

void testCountRadixMultipleBlock() {}

int main() {
  generateInput<int>(5, 10, {0, 3, 2, 1, 5});
}
