import numpy as np
import psutil
import pytest
from sopht.numeric.eulerian_grid_ops import (
    gen_multiplicative_laplacian_filter_kernel_3d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def scalar_laplacian_filter(scalar_field: np.ndarray, filter_order: int) -> None:
    """
    Numpy implementation of multiplicative filter for scalar field
    """
    field_buffer = np.zeros_like(scalar_field)
    filter_flux_buffer = np.zeros_like(scalar_field)

    # Laplacian filter in x direction
    field_buffer[...] = scalar_field
    for _ in range(filter_order):
        filter_flux_buffer[1:-1, 1:-1, 1:-1] = (
            -field_buffer[1:-1, 1:-1, 2:]
            - field_buffer[1:-1, 1:-1, :-2]
            + 2.0 * field_buffer[1:-1, 1:-1, 1:-1]
        ) / 4
        field_buffer[...] = filter_flux_buffer
    scalar_field -= filter_flux_buffer

    # Laplacian filter in y direction
    field_buffer[...] = scalar_field
    for _ in range(filter_order):
        filter_flux_buffer[1:-1, 1:-1, 1:-1] = (
            -field_buffer[1:-1, 2:, 1:-1]
            - field_buffer[1:-1, :-2, 1:-1]
            + 2.0 * field_buffer[1:-1, 1:-1, 1:-1]
        ) * 0.25
        field_buffer[...] = filter_flux_buffer
    scalar_field -= filter_flux_buffer

    # Laplacian filter in z direction
    field_buffer[...] = scalar_field
    for _ in range(filter_order):
        filter_flux_buffer[1:-1, 1:-1, 1:-1] = (
            -field_buffer[2:, 1:-1, 1:-1]
            - field_buffer[:-2, 1:-1, 1:-1]
            + 2.0 * field_buffer[1:-1, 1:-1, 1:-1]
        ) * 0.25
        field_buffer[...] = filter_flux_buffer
    scalar_field -= filter_flux_buffer


def vector_laplacian_filter(vector_field: np.ndarray, filter_order: int) -> None:
    """
    Numpy implementation of multiplicative filter for vector field
    """
    dim = 3
    for axis in range(dim):
        scalar_laplacian_filter(
            scalar_field=vector_field[axis], filter_order=filter_order
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
@pytest.mark.parametrize("filter_order", [1, 2])
@pytest.mark.parametrize("field_type", ["scalar", "vector"])
def test_multiplicative_laplacian_filter_constant_field(
    n_values,
    precision,
    filter_order,
    field_type,
):
    real_t = get_real_t(precision)
    dim = 3
    test_field = (
        2 * np.ones((n_values, n_values, n_values)).astype(real_t)
        if field_type == "scalar"
        else 2 * np.ones((dim, n_values, n_values, n_values)).astype(real_t)
    )

    post_filtered_field = test_field.copy()
    field_buffer = (
        np.zeros_like(test_field)
        if field_type == "scalar"
        else np.zeros_like(test_field[0])
    )
    filter_flux_buffer = np.zeros_like(field_buffer)
    multiplicative_laplacian_filter = gen_multiplicative_laplacian_filter_kernel_3d(
        filter_order=filter_order,
        field_buffer=field_buffer,
        filter_flux_buffer=filter_flux_buffer,
        real_t=real_t,
        fixed_grid_size=(n_values, n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type=field_type,
    )
    multiplicative_laplacian_filter(test_field)
    np.testing.assert_allclose(
        post_filtered_field, test_field, atol=get_test_tol(precision)
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
@pytest.mark.parametrize("filter_order", [1, 2])
@pytest.mark.parametrize("field_type", ["scalar", "vector"])
def test_multiplicative_laplacian_filter_random_field(
    n_values,
    precision,
    filter_order,
    field_type,
):
    real_t = get_real_t(precision)
    dim = 3
    test_field = (
        np.random.rand(n_values, n_values, n_values).astype(real_t)
        if field_type == "scalar"
        else np.random.rand(dim, n_values, n_values, n_values).astype(real_t)
    )
    numpy_ref_field = test_field.copy()
    field_buffer = (
        np.zeros_like(test_field)
        if field_type == "scalar"
        else np.zeros_like(test_field[0])
    )
    filter_flux_buffer = np.zeros_like(field_buffer)
    multiplicative_laplacian_filter = gen_multiplicative_laplacian_filter_kernel_3d(
        filter_order=filter_order,
        field_buffer=field_buffer,
        filter_flux_buffer=filter_flux_buffer,
        real_t=real_t,
        fixed_grid_size=(n_values, n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type=field_type,
    )

    multiplicative_laplacian_filter(test_field)
    if field_type == "scalar":
        scalar_laplacian_filter(numpy_ref_field, filter_order)
    elif field_type == "vector":
        vector_laplacian_filter(numpy_ref_field, filter_order)
    np.testing.assert_allclose(
        numpy_ref_field, test_field, atol=get_test_tol(precision)
    )
