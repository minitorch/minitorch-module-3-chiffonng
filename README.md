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

### [Parallelism with CUDA on GPU](https://minitorch.github.io/module3/cuda/)
