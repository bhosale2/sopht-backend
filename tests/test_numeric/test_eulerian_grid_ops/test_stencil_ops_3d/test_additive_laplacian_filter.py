import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_additive_laplacian_filter_kernel_3d,
)
from sopht.utils.precision import get_real_t, get_test_tol
from numpy.testing import assert_allclose


# constant scalar field
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16, 32])
@pytest.mark.parametrize("filter_order", [1, 2, 4])
def test_additive_laplacian_filter_constant_scalar_field(
    n_values, precision, filter_order
):
    real_t = get_real_t(precision)
    additive_laplacian_filter = gen_additive_laplacian_filter_kernel_3d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="scalar",
    )

    test_field = 2 * np.ones((n_values, n_values, n_values), dtype=real_t)
    post_filtered_field = 2 * np.ones((n_values, n_values, n_values), dtype=real_t)
    scalar_field_buffer = np.zeros((n_values, n_values, n_values), dtype=real_t)
    filtered_field_buffer = np.zeros((n_values, n_values, n_values), dtype=real_t)

    additive_laplacian_filter(
        test_field, scalar_field_buffer, filtered_field_buffer, filter_order
    )

    assert_allclose(post_filtered_field, test_field, atol=get_test_tol(precision))


# constant scalar field
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16, 32])
@pytest.mark.parametrize("filter_order", [1, 2, 4])
def test_additive_laplacian_filter_constant_vector_field(
    n_values, precision, filter_order
):
    real_t = get_real_t(precision)
    additive_laplacian_filter = gen_additive_laplacian_filter_kernel_3d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="vector",
    )

    test_field = 2 * np.ones((3, n_values, n_values, n_values), dtype=real_t)
    post_filtered_field = 2 * np.ones((3, n_values, n_values, n_values), dtype=real_t)
    vector_field_buffer = np.zeros((3, n_values, n_values, n_values), dtype=real_t)
    filtered_field_buffer = np.zeros((n_values, n_values, n_values), dtype=real_t)

    additive_laplacian_filter(
        test_field, vector_field_buffer, filtered_field_buffer, filter_order
    )

    assert_allclose(post_filtered_field, test_field, atol=get_test_tol(precision))
