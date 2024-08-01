# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

- Docs: https://minitorch.github.io/

- Overview: https://minitorch.github.io/module3/module3/

You will need to modify `tensor_functions.py` slightly in this assignment.

- Tests:

```
python run_tests.py
```

- Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py

## Notes

### Misc

- `# *` marks important implementation details
- `# !` marks code to fix bugs, outdated code, or outdated code itself that requires updating

### [Parallelism with numba on CPU](https://minitorch.github.io/module3/parallel/)

> Parellism = distribute the work across multiple threads or processes

Main optimizations on CPU -> [fast_ops.py](minitorch/fast_ops.py):

- The decorator `@numba.njit(parallel=True)` can be used to parallelize a function
- Use `numba.prange` instead of `range` for outer loops, so that the iterations can be parallelized
  - `for i in numba.prange(...)`
  - Avoid using `numba.prange` for inner loops _extensively_ since it can create excessive threads and context switches, which can slow down the code.
- Use numpy functions and arrays, which are already parallelized

### Matrix Multiplication ([guide](https://minitorch.github.io/module3/matrixmult/))

In matrix multiplication involving tensors, the **batch dimension** is an additional dimension used to group multiple matrices together in a single tensor. This is useful when performing operations like matrix multiplication on a set of matrices that share the same dimensional properties.

Consider three-dimensional tensors where the first dimension represents the batch size:
The content you provided can be rewritten in proper LaTeX format as follows:

- **Tensor A**: Shape $(B, M, K)$
- **Tensor B**: Shape $(B, K, N)$

The result tensor $C$ will have the shape $(B, M, N)$, where each matrix $C[b]$ is the result of multiplying $A[b]$ and $B[b]$ for the $b$-th batch.

#### Role of Batch Dimension in [`_tensor_matrix_multiply`](minitorch/fast_ops.py)

The batch dimension allows the function to **handle multiple matrix multiplications in one go**. For each matrix pair in the batch, the function performs the matrix multiplication independently.

```python
for i in prange(out_shape[0]):  # Iterate over batch dimension
```

In this loop, `i` represents the index of the current matrix pair in the batch. Each iteration processes a different pair of matrices from tensors A and B.

The calculation for a single element in the output tensor is:

```python
a_inner = i * a_batch_stride + j * a_strides[1]
b_inner = i * b_batch_stride + k * b_strides[2]

num = 0.0
for _ in range(a_shape[-1]):  # Sum over the inner dimension
    num += a_storage[a_inner] * b_storage[b_inner]
    a_inner += a_strides[2]
    b_inner += b_strides[1]
```

Here, `a_inner` and `b_inner` are adjusted for the current batch and element positions. The inner loop performs the dot product calculation for the matrices at index `i` in the batch.

### [Parallelism with CUDA on GPU](https://minitorch.github.io/module3/cuda/)
