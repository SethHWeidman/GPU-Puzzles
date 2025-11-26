#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t err = (call);                                                                      \
    if (err != cudaSuccess) {                                                                      \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " -> "                          \
                << cudaGetErrorString(err) << std::endl;                                           \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

constexpr int kVectorSize = 1024 * 1024; // 1,048,576 elements
constexpr int kMatrixDim = 1024;         // 1,024 x 1,024 matrices
constexpr float kTolerance = 1e-3f;
constexpr int kBlockSize1D = 256;
constexpr int kSharedBlockSize = 256;
constexpr int kDotBlockSize = 256;
constexpr int kPoolWindow = 3;
constexpr int kConvKernelSize = 7;
constexpr int kPrefixBlockSize = 256;
constexpr int kAxisBlockSize = 256;
constexpr int kMatmulTile = 32;

namespace {
std::mt19937 g_rng(251125);
std::uniform_real_distribution<float> g_dist(-1.0f, 1.0f);

inline float randf() { return g_dist(g_rng); }
} // namespace

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t n) : size_(n) {
    if (n > 0) {
      CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      if (ptr_) {
        cudaFree(ptr_);
      }
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  T *get() { return ptr_; }
  const T *get() const { return ptr_; }
  std::size_t size() const { return size_; }

private:
  T *ptr_ = nullptr;
  std::size_t size_ = 0;
};

float max_abs_diff(const std::vector<float> &ref, const std::vector<float> &got) {
  if (ref.size() != got.size()) {
    std::cerr << "Mismatched vector sizes: " << ref.size() << " vs " << got.size() << std::endl;
    std::exit(EXIT_FAILURE);
  }
  float diff = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    diff = std::max(diff, std::fabs(ref[i] - got[i]));
  }
  return diff;
}

void expect_close(const std::string &name, float diff, float tol = kTolerance) {
  if (diff > tol || std::isnan(diff)) {
    std::cerr << name << " failed (max |diff| = " << diff << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "[OK] " << name << " | max |diff| = " << diff << std::endl;
}

// ---------------------------------------------------------------------------
// Kernels

__global__ void puzzle1_map(const float *__restrict__ a, float *__restrict__ out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < size; i += stride) {
    out[i] = a[i] + 10.0f;
  }
}

__global__ void puzzle2_zip(const float *__restrict__ a, const float *__restrict__ b,
                            float *__restrict__ out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] + b[idx];
  }
}

__global__ void puzzle3_guard(const float *__restrict__ a, float *__restrict__ out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] + 10.0f;
  }
}

__global__ void puzzle4_map2d(const float *__restrict__ a, float *__restrict__ out, int width,
                              int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y * width + x;
    out[idx] = a[idx] + 10.0f;
  }
}

__global__ void puzzle5_broadcast(const float *__restrict__ col_vec,
                                  const float *__restrict__ row_vec, float *__restrict__ out,
                                  int rows, int cols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < cols && y < rows) {
    out[y * cols + x] = col_vec[y] + row_vec[x];
  }
}

__global__ void puzzle6_blocks(const float *__restrict__ a, float *__restrict__ out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] + 10.0f;
  }
}

__global__ void puzzle7_blocks2d(const float *__restrict__ a, float *__restrict__ out, int width,
                                 int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y * width + x;
    out[idx] = a[idx] + 10.0f;
  }
}

__global__ void puzzle8_shared(const float *__restrict__ a, float *__restrict__ out, int size) {
  __shared__ float tile[kSharedBlockSize];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;

  float value = 0.0f;
  if (idx < size) {
    value = a[idx];
  }
  tile[local] = value;
  __syncthreads();

  if (idx < size) {
    out[idx] = tile[local] + 10.0f;
  }
}

__global__ void puzzle9_pool3(const float *__restrict__ inp, float *__restrict__ out, int size) {
  extern __shared__ float tile[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;
  int block_start = blockIdx.x * blockDim.x;
  int shared_idx = local + (kPoolWindow - 1);

  float cur = 0.0f;
  if (idx < size) {
    cur = inp[idx];
  }
  tile[shared_idx] = cur;

  if (local < kPoolWindow - 1) {
    int halo_idx = block_start + local - (kPoolWindow - 1);
    tile[local] = (halo_idx >= 0) ? inp[halo_idx] : 0.0f;
  }
  __syncthreads();

  if (idx < size) {
    float sum = 0.0f;
    sum += tile[shared_idx];
    sum += tile[shared_idx - 1];
    sum += tile[shared_idx - 2];
    out[idx] = sum;
  }
}

__global__ void puzzle10_dot(const float *__restrict__ a, const float *__restrict__ b,
                             double *__restrict__ out, int size) {
  __shared__ double cache[kDotBlockSize];
  int tid = threadIdx.x;

  double sum = 0.0;
  for (int i = tid; i < size; i += blockDim.x) {
    sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  cache[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      cache[tid] += cache[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[0] = cache[0];
  }
}

__global__ void puzzle11_conv1d(const float *__restrict__ inp, const float *__restrict__ kernel,
                                float *__restrict__ out, int size, int kernel_size) {
  extern __shared__ float tile[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;
  int block_start = blockIdx.x * blockDim.x;

  if (idx < size) {
    tile[local] = inp[idx];
  } else {
    tile[local] = 0.0f;
  }

  if (local < kernel_size - 1) {
    int halo_idx = block_start + blockDim.x + local;
    tile[blockDim.x + local] = (halo_idx < size) ? inp[halo_idx] : 0.0f;
  }
  __syncthreads();

  if (idx < size) {
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
      sum += tile[local + k] * kernel[k];
    }
    out[idx] = sum;
  }
}

__global__ void puzzle12_prefix_sum(const float *__restrict__ inp, float *__restrict__ out,
                                    int size) {
  __shared__ float cache[kPrefixBlockSize];
  int block_start = blockIdx.x * blockDim.x;
  int local = threadIdx.x;
  int global = block_start + local;
  int remaining = size - block_start;
  int elements = remaining > blockDim.x ? blockDim.x : (remaining > 0 ? remaining : 0);

  if (local < elements) {
    cache[local] = inp[global];
  } else {
    cache[local] = 0.0f;
  }
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride <<= 1) {
    float val = 0.0f;
    if (local >= stride && local < elements) {
      val = cache[local - stride];
    }
    __syncthreads();
    if (local < elements) {
      cache[local] += val;
    }
    __syncthreads();
  }

  if (elements > 0 && local == elements - 1) {
    out[blockIdx.x] = cache[local];
  }
}

__global__ void puzzle13_axis_sum(const float *__restrict__ inp, float *__restrict__ out, int rows,
                                  int cols, int chunks) {
  __shared__ float cache[kAxisBlockSize];
  int chunk = blockIdx.x;
  int row = blockIdx.y;
  int local = threadIdx.x;
  int col = chunk * blockDim.x + local;

  float value = 0.0f;
  if (row < rows && col < cols) {
    value = inp[row * cols + col];
  }
  cache[local] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local < stride) {
      cache[local] += cache[local + stride];
    }
    __syncthreads();
  }

  if (local == 0 && row < rows && chunk < chunks) {
    out[row * chunks + chunk] = cache[0];
  }
}

__global__ void puzzle14_matmul(const float *__restrict__ a, const float *__restrict__ b,
                                float *__restrict__ out, int dim) {
  __shared__ float tile_a[kMatmulTile][kMatmulTile];
  __shared__ float tile_b[kMatmulTile][kMatmulTile];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.0f;

  int tiles = (dim + kMatmulTile - 1) / kMatmulTile;
  for (int t = 0; t < tiles; ++t) {
    int tiled_col = t * kMatmulTile + threadIdx.x;
    int tiled_row = t * kMatmulTile + threadIdx.y;

    if (row < dim && tiled_col < dim) {
      tile_a[threadIdx.y][threadIdx.x] = a[row * dim + tiled_col];
    } else {
      tile_a[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (col < dim && tiled_row < dim) {
      tile_b[threadIdx.y][threadIdx.x] = b[tiled_row * dim + col];
    } else {
      tile_b[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < kMatmulTile; ++k) {
      acc += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < dim && col < dim) {
    out[row * dim + col] = acc;
  }
}

// ---------------------------------------------------------------------------
// Host verification helpers

void run_puzzle_map() {
  const int n = kVectorSize;
  std::vector<float> input(n);
  std::vector<float> ref(n);
  std::vector<float> output(n, 0.0f);

  for (int i = 0; i < n; ++i) {
    input[i] = randf();
    ref[i] = input[i] + 10.0f;
  }

  DeviceBuffer<float> d_in(n);
  DeviceBuffer<float> d_out(n);
  CUDA_CHECK(cudaMemcpy(d_in.get(), input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, n * sizeof(float)));

  const dim3 block(kBlockSize1D);
  const dim3 grid((n + block.x - 1) / block.x);
  puzzle1_map<<<grid, block>>>(d_in.get(), d_out.get(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_out.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
  expect_close("Puzzle 1 (Map)", max_abs_diff(ref, output));
}

void run_puzzle_zip() {
  const int n = kVectorSize;
  std::vector<float> a(n);
  std::vector<float> b(n);
  std::vector<float> ref(n);
  std::vector<float> output(n, 0.0f);

  for (int i = 0; i < n; ++i) {
    a[i] = randf();
    b[i] = randf();
    ref[i] = a[i] + b[i];
  }

  DeviceBuffer<float> d_a(n);
  DeviceBuffer<float> d_b(n);
  DeviceBuffer<float> d_out(n);
  CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, n * sizeof(float)));

  const dim3 block(kBlockSize1D);
  const dim3 grid((n + block.x - 1) / block.x);
  puzzle2_zip<<<grid, block>>>(d_a.get(), d_b.get(), d_out.get(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_out.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
  expect_close("Puzzle 2 (Zip)", max_abs_diff(ref, output));
}

void run_puzzle_guard() {
  const int n = kVectorSize;
  std::vector<float> input(n);
  std::vector<float> ref(n);
  std::vector<float> output(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    input[i] = randf();
    ref[i] = input[i] + 10.0f;
  }

  DeviceBuffer<float> d_in(n);
  DeviceBuffer<float> d_out(n);
  CUDA_CHECK(cudaMemcpy(d_in.get(), input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, n * sizeof(float)));

  const int threads = 512;
  const int blocks = (n + threads - 1) / threads;
  puzzle3_guard<<<blocks, threads>>>(d_in.get(), d_out.get(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_out.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
  expect_close("Puzzle 3 (Guard)", max_abs_diff(ref, output));
}

void run_puzzle_map2d() {
  const int dim = kMatrixDim;
  const int total = dim * dim;
  std::vector<float> input(total);
  std::vector<float> ref(total);
  std::vector<float> output(total, 0.0f);

  for (int idx = 0; idx < total; ++idx) {
    input[idx] = randf();
    ref[idx] = input[idx] + 10.0f;
  }

  DeviceBuffer<float> d_in(total);
  DeviceBuffer<float> d_out(total);
  CUDA_CHECK(cudaMemcpy(d_in.get(), input.data(), total * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, total * sizeof(float)));

  dim3 block(32, 32);
  dim3 grid((dim + block.x - 1) / block.x, (dim + block.y - 1) / block.y);
  puzzle4_map2d<<<grid, block>>>(d_in.get(), d_out.get(), dim, dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_out.get(), total * sizeof(float), cudaMemcpyDeviceToHost));
  expect_close("Puzzle 4 (Map 2D)", max_abs_diff(ref, output));
}

void run_puzzle_broadcast() {
  const int dim = kMatrixDim;
  const int total = dim * dim;
  std::vector<float> col(dim);
  std::vector<float> row(dim);
  std::vector<float> ref(total);
  std::vector<float> output(total, 0.0f);

  for (int i = 0; i < dim; ++i) {
    col[i] = randf();
    row[i] = randf();
  }

  for (int y = 0; y < dim; ++y) {
    for (int x = 0; x < dim; ++x) {
      ref[y * dim + x] = col[y] + row[x];
    }
  }

  DeviceBuffer<float> d_col(dim);
  DeviceBuffer<float> d_row(dim);
  DeviceBuffer<float> d_out(total);
  CUDA_CHECK(cudaMemcpy(d_col.get(), col.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_row.get(), row.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, total * sizeof(float)));

  dim3 block(32, 32);
  dim3 grid((dim + block.x - 1) / block.x, (dim + block.y - 1) / block.y);
  puzzle5_broadcast<<<grid, block>>>(d_col.get(), d_row.get(), d_out.get(), dim, dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_out.get(), total * sizeof(float), cudaMemcpyDeviceToHost));
  expect_close("Puzzle 5 (Broadcast)", max_abs_diff(ref, output));
}

void run_puzzle_blocks() {
  const int n = kVectorSize;
  std::vector<float> input(n);
  std::vector<float> ref(n);
  std::vector<float> output(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    input[i] = randf();
    ref[i] = input[i] + 10.0f;
  }

  DeviceBuffer<float> d_in(n);
  DeviceBuffer<float> d_out(n);
  CUDA_CHECK(cudaMemcpy(d_in.get(), input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, n * sizeof(float)));

  const int threads = 64;
  const int blocks = (n + threads - 1) / threads;
  puzzle6_blocks<<<blocks, threads>>>(d_in.get(), d_out.get(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_out.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
  expect_close("Puzzle 6 (Blocks)", max_abs_diff(ref, output));
}

void run_puzzle_blocks2d() {
  const int dim = kMatrixDim;
  const int total = dim * dim;
  std::vector<float> input(total);
  std::vector<float> ref(total);
  std::vector<float> output(total, 0.0f);
  for (int idx = 0; idx < total; ++idx) {
    input[idx] = randf();
    ref[idx] = input[idx] + 10.0f;
  }

  DeviceBuffer<float> d_in(total);
  DeviceBuffer<float> d_out(total);
  CUDA_CHECK(cudaMemcpy(d_in.get(), input.data(), total * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, total * sizeof(float)));

  dim3 block(16, 16);
  dim3 grid((dim + block.x - 1) / block.x, (dim + block.y - 1) / block.y);
  puzzle7_blocks2d<<<grid, block>>>(d_in.get(), d_out.get(), dim, dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_out.get(), total * sizeof(float), cudaMemcpyDeviceToHost));
  expect_close("Puzzle 7 (Blocks 2D)", max_abs_diff(ref, output));
}

void run_puzzle_shared() {
  const int n = kVectorSize;
  std::vector<float> input(n);
  std::vector<float> ref(n);
  std::vector<float> output(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    input[i] = randf();
    ref[i] = input[i] + 10.0f;
  }

  DeviceBuffer<float> d_in(n);
  DeviceBuffer<float> d_out(n);
  CUDA_CHECK(cudaMemcpy(d_in.get(), input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, n * sizeof(float)));

  const int threads = kSharedBlockSize;
  const int blocks = (n + threads - 1) / threads;
  puzzle8_shared<<<blocks, threads>>>(d_in.get(), d_out.get(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_out.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
  expect_close("Puzzle 8 (Shared)", max_abs_diff(ref, output));
}

void run_puzzle_pool() {
  const int n = kVectorSize;
  std::vector<float> input(n);
  std::vector<float> ref(n);
  std::vector<float> output(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    input[i] = randf();
    float sum = 0.0f;
    for (int k = 0; k < kPoolWindow; ++k) {
      int idx = i - k;
      if (idx >= 0) {
        sum += input[idx];
      }
    }
    ref[i] = sum;
  }

  DeviceBuffer<float> d_in(n);
  DeviceBuffer<float> d_out(n);
  CUDA_CHECK(cudaMemcpy(d_in.get(), input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, n * sizeof(float)));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const size_t shared_bytes = (threads + kPoolWindow - 1) * sizeof(float);
  puzzle9_pool3<<<blocks, threads, shared_bytes>>>(d_in.get(), d_out.get(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_out.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
  expect_close("Puzzle 9 (Pooling)", max_abs_diff(ref, output));
}

void run_puzzle_dot() {
  const int n = kVectorSize;
  std::vector<float> a(n);
  std::vector<float> b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = randf();
    b[i] = randf();
  }
  double ref = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);

  DeviceBuffer<float> d_a(n);
  DeviceBuffer<float> d_b(n);
  const int threads = kDotBlockSize;
  const int blocks = 1;

  DeviceBuffer<double> d_out(1);
  CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, sizeof(double)));

  puzzle10_dot<<<blocks, threads>>>(d_a.get(), d_b.get(), d_out.get(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  double result = 0.0;
  CUDA_CHECK(cudaMemcpy(&result, d_out.get(), sizeof(double), cudaMemcpyDeviceToHost));
  expect_close("Puzzle 10 (Dot)", static_cast<float>(std::fabs(ref - result)));
}

void run_puzzle_conv() {
  const int n = kVectorSize;
  std::vector<float> input(n);
  std::vector<float> kernel(kConvKernelSize);
  std::vector<float> ref(n, 0.0f);
  std::vector<float> output(n, 0.0f);

  for (int i = 0; i < n; ++i) {
    input[i] = randf();
  }
  for (int k = 0; k < kConvKernelSize; ++k) {
    kernel[k] = randf();
  }

  for (int i = 0; i < n; ++i) {
    float sum = 0.0f;
    for (int k = 0; k < kConvKernelSize; ++k) {
      int idx = i + k;
      if (idx < n) {
        sum += input[idx] * kernel[k];
      }
    }
    ref[i] = sum;
  }

  DeviceBuffer<float> d_in(n);
  DeviceBuffer<float> d_kernel(kConvKernelSize);
  DeviceBuffer<float> d_out(n);
  CUDA_CHECK(cudaMemcpy(d_in.get(), input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kernel.get(), kernel.data(), kConvKernelSize * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, n * sizeof(float)));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const size_t shared_bytes = (threads + kConvKernelSize - 1) * sizeof(float);
  puzzle11_conv1d<<<blocks, threads, shared_bytes>>>(d_in.get(), d_kernel.get(), d_out.get(), n,
                                                     kConvKernelSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_out.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
  expect_close("Puzzle 11 (1D Conv)", max_abs_diff(ref, output));
}

void run_puzzle_prefix_sum() {
  const int n = kVectorSize;
  std::vector<float> input(n);
  std::vector<float> output((n + kPrefixBlockSize - 1) / kPrefixBlockSize, 0.0f);
  std::vector<float> ref(output.size(), 0.0f);

  for (int i = 0; i < n; ++i) {
    input[i] = randf();
  }

  for (std::size_t block = 0; block < ref.size(); ++block) {
    std::size_t start = block * kPrefixBlockSize;
    std::size_t end =
        std::min(start + static_cast<std::size_t>(kPrefixBlockSize), static_cast<std::size_t>(n));
    float sum = 0.0f;
    for (std::size_t idx = start; idx < end; ++idx) {
      sum += input[idx];
    }
    ref[block] = sum;
  }

  DeviceBuffer<float> d_in(n);
  DeviceBuffer<float> d_out(ref.size());
  CUDA_CHECK(cudaMemcpy(d_in.get(), input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, ref.size() * sizeof(float)));

  const int blocks = (n + kPrefixBlockSize - 1) / kPrefixBlockSize;
  puzzle12_prefix_sum<<<blocks, kPrefixBlockSize>>>(d_in.get(), d_out.get(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(output.data(), d_out.get(), ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
  expect_close("Puzzle 12 (Prefix Sum)", max_abs_diff(ref, output));
}

void run_puzzle_axis_sum() {
  const int rows = kMatrixDim;
  const int cols = kMatrixDim;
  const int chunks = (cols + kAxisBlockSize - 1) / kAxisBlockSize;
  std::vector<float> input(rows * cols);
  std::vector<float> output(rows * chunks, 0.0f);
  std::vector<float> ref(rows * chunks, 0.0f);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      input[r * cols + c] = randf();
    }
  }

  for (int r = 0; r < rows; ++r) {
    for (int chunk = 0; chunk < chunks; ++chunk) {
      int start = chunk * kAxisBlockSize;
      int end = std::min(start + kAxisBlockSize, cols);
      float sum = 0.0f;
      for (int c = start; c < end; ++c) {
        sum += input[r * cols + c];
      }
      ref[r * chunks + chunk] = sum;
    }
  }

  DeviceBuffer<float> d_in(rows * cols);
  DeviceBuffer<float> d_out(rows * chunks);
  CUDA_CHECK(
      cudaMemcpy(d_in.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, output.size() * sizeof(float)));

  dim3 block(kAxisBlockSize);
  dim3 grid(chunks, rows);
  puzzle13_axis_sum<<<grid, block>>>(d_in.get(), d_out.get(), rows, cols, chunks);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_out.get(), output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  expect_close("Puzzle 13 (Axis Sum)", max_abs_diff(ref, output));
}

void run_puzzle_matmul() {
  const int dim = kMatrixDim;
  const int total = dim * dim;
  std::vector<float> a(total);
  std::vector<float> b(total);
  std::vector<float> output(total, 0.0f);
  std::vector<float> ref(total, 0.0f);

  for (int r = 0; r < dim; ++r) {
    for (int c = 0; c < dim; ++c) {
      a[r * dim + c] = randf();
      b[r * dim + c] = randf();
    }
  }

  for (int i0 = 0; i0 < dim; ++i0) {
    for (int k = 0; k < dim; ++k) {
      float a_val = a[i0 * dim + k];
      for (int j = 0; j < dim; ++j) {
        ref[i0 * dim + j] += a_val * b[k * dim + j];
      }
    }
  }

  DeviceBuffer<float> d_a(total);
  DeviceBuffer<float> d_b(total);
  DeviceBuffer<float> d_out(total);
  CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), total * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), total * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, total * sizeof(float)));

  dim3 block(kMatmulTile, kMatmulTile);
  dim3 grid((dim + block.x - 1) / block.x, (dim + block.y - 1) / block.y);
  puzzle14_matmul<<<grid, block>>>(d_a.get(), d_b.get(), d_out.get(), dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_out.get(), total * sizeof(float), cudaMemcpyDeviceToHost));
  expect_close("Puzzle 14 (Matmul)", max_abs_diff(ref, output));
}

int main() {
  CUDA_CHECK(cudaSetDevice(0));

  run_puzzle_map();
  run_puzzle_zip();
  run_puzzle_guard();
  run_puzzle_map2d();
  run_puzzle_broadcast();
  run_puzzle_blocks();
  run_puzzle_blocks2d();
  run_puzzle_shared();
  run_puzzle_pool();
  run_puzzle_dot();
  run_puzzle_conv();
  run_puzzle_prefix_sum();
  run_puzzle_axis_sum();
  run_puzzle_matmul();

  std::cout << "All GPU puzzle kernels validated on ~1M-sized problems." << std::endl;
  return 0;
}
