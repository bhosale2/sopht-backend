import numpy as np
import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_multiplicative_laplacian_filter_kernel_3d,
)
from sopht.utils.precision import get_real_t, get_test_tol
from numpy.testing import assert_allclose


def scalar_laplacian_filter(rate_collection: np.ndarray, filter_order: int) -> None:
    """
    Numpy implementation of multiplicative filter
    """

    filter_term = rate_collection.copy()  # 3, Zgrid, Ygrid, Xgrid

    for i in range(filter_order):

        # Do in Z direction first
        filter_term[1:-1, :, :] = (
            -filter_term[2:, :, :]
            - filter_term[:-2, :, :]
            + 2.0 * filter_term[1:-1, :, :]
        ) / 4.0

        # Do in Y direction
        filter_term[:, 1:-1, :] = (
            -filter_term[:, 2:, :]
            - filter_term[:, :-2, :]
            + 2.0 * filter_term[:, 1:-1, :]
        ) / 4.0

        # Do in X direction
        filter_term[:, :, 1:-1] = (
            -filter_term[:, :, 2:]
            - filter_term[:, :, :-2]
            + 2.0 * filter_term[:, :, 1:-1]
        ) / 4.0

        # dont touch boundary values
        # Z boundaries
        filter_term[0, :, :] = 0.0
        filter_term[-1, :, :] = 0.0
        # Y boundaries
        filter_term[:, 0, :] = 0.0
        filter_term[:, -1, :] = 0.0
        # X boundaries
        filter_term[:, :, 0] = 0.0
        filter_term[:, :, -1] = 0.0

    rate_collection[...] = rate_collection - filter_term


def vectorial_laplacian_filter(rate_collection: np.ndarray, filter_order: int) -> None:
    """
    Numpy implementation of multiplicative filter
    """

    filter_term = rate_collection.copy()  # 3, Zgrid, Ygrid, Xgrid

    for i in range(filter_order):

        # Do in Z direction first
        filter_term[:, 1:-1, :, :] = (
            -filter_term[:, 2:, :, :]
            - filter_term[:, :-2, :, :]
            + 2.0 * filter_term[:, 1:-1, :, :]
        ) / 4.0

        # Do in Y direction
        filter_term[:, :, 1:-1, :] = (
            -filter_term[:, :, 2:, :]
            - filter_term[:, :, :-2, :]
            + 2.0 * filter_term[:, :, 1:-1, :]
        ) / 4.0

        # Do in X direction
        filter_term[:, :, :, 1:-1] = (
            -filter_term[:, :, :, 2:]
            - filter_term[:, :, :, :-2]
            + 2.0 * filter_term[:, :, :, 1:-1]
        ) / 4.0

        # dont touch boundary values
        # Z boundaries
        filter_term[:, 0, :, :] = 0.0
        filter_term[:, -1, :, :] = 0.0
        # Y boundaries
        filter_term[:, :, 0, :] = 0.0
        filter_term[:, :, -1, :] = 0.0
        # X boundaries
        filter_term[:, :, :, 0] = 0.0
        filter_term[:, :, :, -1] = 0.0

    rate_collection[...] = rate_collection - filter_term


# constant scalar field
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16, 32])
@pytest.mark.parametrize("filter_order", [1, 2, 4])
def test_multiplicative_laplacian_filter_constant_scalar_field(
    n_values, precision, filter_order
):
    real_t = get_real_t(precision)
    multiplicative_laplacian_filter = gen_multiplicative_laplacian_filter_kernel_3d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="scalar",
    )

    test_field = 2 * np.ones((n_values, n_values, n_values), dtype=real_t)
    post_filtered_field = 2 * np.ones((n_values, n_values, n_values), dtype=real_t)
    scalar_field_buffer = np.zeros((n_values, n_values, n_values), dtype=real_t)
    filtered_field_buffer = np.zeros((n_values, n_values, n_values), dtype=real_t)

    multiplicative_laplacian_filter(
        test_field, scalar_field_buffer, filtered_field_buffer, filter_order
    )

    assert_allclose(post_filtered_field, test_field, atol=get_test_tol(precision))


# # random scalar field
# # FIXME NOT WORKING numpy version and pystencil gives different results
# @pytest.mark.parametrize("precision", ["double"])
# @pytest.mark.parametrize("n_values", [4])
# @pytest.mark.parametrize("filter_order", [1])
# def test_multiplicative_laplacian_filter_random_scalar_field(
#     n_values, precision, filter_order
# ):
#     real_t = get_real_t(precision)
#     multiplicative_laplacian_filter = gen_multiplicative_laplacian_filter_kernel_3d(
#         real_t=real_t,
#         fixed_grid_size=(n_values, n_values, n_values),
#         num_threads=psutil.cpu_count(logical=False),
#         field_type="scalar",
#     )
#
#     test_field = np.zeros((n_values, n_values, n_values), dtype=real_t)
#     test_field[1::2, : :] = 2.0
#     compared_field = test_field.copy()
#     # post_filtered_field = 2 * np.ones((n_values, n_values, n_values), dtype=real_t)
#     scalar_field_buffer = np.zeros((n_values, n_values, n_values), dtype=real_t)
#     filtered_field_buffer = np.zeros((n_values, n_values, n_values), dtype=real_t)
#
#     multiplicative_laplacian_filter(
#         test_field, scalar_field_buffer, filtered_field_buffer, filter_order
#     )
#
#     scalar_laplacian_filter(compared_field, filter_order)
#
#     assert_allclose(compared_field, test_field, atol=get_test_tol(precision))


# constant scalar field
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16, 32])
@pytest.mark.parametrize("filter_order", [1, 2, 4])
def test_multiplicative_laplacian_filter_constant_vector_field(
    n_values, precision, filter_order
):
    real_t = get_real_t(precision)
    multiplicative_laplacian_filter = gen_multiplicative_laplacian_filter_kernel_3d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="vector",
    )

    test_field = 2 * np.ones((3, n_values, n_values, n_values), dtype=real_t)
    post_filtered_field = 2 * np.ones((3, n_values, n_values, n_values), dtype=real_t)
    vector_field_buffer = np.zeros((3, n_values, n_values, n_values), dtype=real_t)
    filtered_field_buffer = np.zeros((n_values, n_values, n_values), dtype=real_t)

    multiplicative_laplacian_filter(
        test_field, vector_field_buffer, filtered_field_buffer, filter_order
    )

    assert_allclose(post_filtered_field, test_field, atol=get_test_tol(precision))
