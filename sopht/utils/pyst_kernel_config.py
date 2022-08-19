import numpy as np

import pystencils as ps

from typing import Tuple, Union


def get_pyst_dtype(real_t: Union[np.float32, np.float64]) -> str:
    """Return the pystencils data type based on real dtype."""
    if real_t == np.float32:
        return "float32"
    elif real_t == np.float64:
        return "float64"
    else:
        raise ValueError("Invalid real type")


def get_pyst_kernel_config(
    pyst_dtype: str, num_threads: int, iteration_slice: Tuple = None
):
    """Returns the pystencils kernel config based on the data
    dtype and number of threads"""
    # TODO check out more options here!
    kernel_config = ps.CreateKernelConfig(
        data_type=pyst_dtype,
        default_number_float=pyst_dtype,
        cpu_openmp=num_threads,
        iteration_slice=iteration_slice,
    )
    return kernel_config
