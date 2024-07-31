from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba  # type: ignore
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

# https://minitorch.github.io/module2/tensordata/
MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in storage based on strides.
    Formula: `position = index[0] * stride[0] + index[1] * stride[1] + ... + index[n] * stride[n]`. Reference: https://minitorch.github.io/module2/tensordata/

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage

    Raises:
        ValueError : if the length of `index` and `strides` are not equal
    """
    if len(index) != len(strides):
        raise ValueError(
            "Index and strides must have the same length. Got %d and %d."
            % (len(index), len(strides))
        )
    return sum(i * stride for i, stride in zip(index, strides))


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` in the tensor's storage to an index in the
    tensor's `shape`. Should ensure that enumerating position 0 ... size
    of a tensor produces every index  exactly once. It may not be the
    inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert. It refers to the position in tensor's storage; it is the position in single-dimensional, contiguous blocks of memory.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # Start from the last dimension and work backwards
    # dim_id is the dimension index
    for dim_id in range(len(shape) - 1, -1, -1):
        # Use modulo operation to find the index in the current dimension
        out_index[dim_id] = ordinal % shape[dim_id]
        # Update the ordinal for the next iteration (moving to the next dimension)
        ordinal //= shape[dim_id]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    """
    offset = len(big_shape) - len(shape)
    for i in range(len(shape)):
        if shape[i] == 1:
            out_index[i] = 0
        else:
            out_index[i] = big_index[i + offset]


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast. When the dimensions (of shape1 and shape2) are not equal and neither is 1 (i.e. the considered dimension is not missing in either shape)

    Examples:
    ```python
    out = minitorch.zeros((2, 3, 1)) + minitorch.zeros((7, 2, 1, 5))
    out.shape # (7, 3, 2, 5)
    ```
    """
    # Get the maximum length of the two shapes
    max_len = max(len(shape1), len(shape2))
    # Initialize the output shape to zeros
    out_shape = [0] * max_len

    # Iterate over the dimensions of the two shapes in reverse order
    for i in range(1, max_len + 1):
        # Get the dimension from the end of the shape
        # If the dimension is missing, set it to 1
        dim1 = shape1[-i] if i <= len(shape1) else 1
        dim2 = shape2[-i] if i <= len(shape2) else 1

        # Check if dimensions can be broadcast
        # If the dimensions are not equal and neither is 1, broadcasting is not possible
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            raise IndexingError(f"Shapes {shape1} and {shape2} cannot be broadcast")
        else:
            # Set the output shape to the maximum of the two dimensions
            out_shape[-i] = max(dim1, dim2)

    # Return the broadcasted shape
    return tuple(out_shape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Example:
        Given a tensor with dimensions (2, 3, 4) and an order (2, 0, 1),
        the permute function will return a new tensor with dimensions
        (4, 2, 3). The new first dimension (index 0) should be index 2,
        which is old third dimension (size 4).

        Args:
            order (list): a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        # The length of the order must match the number of dimensions
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # Create a new shape and strides based on the order. They must be tuples as declared above
        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self.strides[i] for i in order)
        return TensorData(self._storage, new_shape, new_strides)

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            line = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    line = "\n%s[" % ("\t" * i) + line
                else:
                    break
            s += line
            v = self.get(index)
            s += f"{v:3.2f}"
            line = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    line += "]"
                else:
                    break
            if line:
                s += line
            else:
                s += " "
        return s
