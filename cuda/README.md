# GPU Puzzles – Native CUDA Solutions

`gpu_puzzles_kernels.cu` contains CUDA C++ implementations of the 14 kernels from the original
GPU-Puzzles notebook plus a standalone verification harness; see the [GPU-Puzzles
notebook](../GPU_puzzlers.ipynb) for the source puzzles and context.

## Building

```bash
cd GPU-Puzzles/cuda
make
```

The Makefile mirrors the other submodules of the repository: it compiles the CUDA translation unit
with `nvcc` (C++17, `-gencode arch=compute_89,code=sm_89`) and links with `g++`. Any 40-series GPU
or newer CUDA driver should run this binary directly.

## Running the verification suite

```bash
./gpu_puzzles_kernels
```

The program launches every kernel on large problem sizes to ensure they scale past the toy workloads
in the notebook:

- 1D puzzles use vectors with 1,024 × 1,024 (= 1,048,576) elements.
- 2D puzzles use 1,024 × 1,024 matrices.

Results are copied back to the host and compared against CPU reference implementations. The
matrix-multiplication check computes a full CPU reference, so expect that particular test to take a
bit longer than the others.

### Output

Running this on a RunPod instance with an L4 GPU:

```bash
make clean && make
```

shows:

```
rm -rf objs *.ppm *~ gpu_puzzles_kernels
mkdir -p objs/
nvcc gpu_puzzles_kernels.cu -O3 -m64 -std=c++17 -gencode arch=compute_89,code=sm_89 -c -o objs/gpu_puzzles_kernels.o
g++ -m64 -O3 -Wall -o gpu_puzzles_kernels objs/gpu_puzzles_kernels.o -L/usr/local/cuda/lib64/ -lcudart
```

and then running `./gpu_puzzles_kernels` prints:

```
[OK] Puzzle 1 (Map) | max |diff| = 0
[OK] Puzzle 2 (Zip) | max |diff| = 0
[OK] Puzzle 3 (Guard) | max |diff| = 0
[OK] Puzzle 4 (Map 2D) | max |diff| = 0
[OK] Puzzle 5 (Broadcast) | max |diff| = 0
[OK] Puzzle 6 (Blocks) | max |diff| = 0
[OK] Puzzle 7 (Blocks 2D) | max |diff| = 0
[OK] Puzzle 8 (Shared) | max |diff| = 0
[OK] Puzzle 9 (Pooling) | max |diff| = 0
[OK] Puzzle 10 (Dot) | max |diff| = 6.42006e-06
[OK] Puzzle 11 (1D Conv) | max |diff| = 7.15256e-07
[OK] Puzzle 12 (Prefix Sum) | max |diff| = 1.90735e-05
[OK] Puzzle 13 (Axis Sum) | max |diff| = 1.90735e-05
[OK] Puzzle 14 (Matmul) | max |diff| = 1.90735e-05
All GPU puzzle kernels validated on ~1M-sized problems.
```

in under ten seconds.