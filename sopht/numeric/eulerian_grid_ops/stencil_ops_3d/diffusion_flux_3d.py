"""Kernels for computing diffusion flux in 3D."""
import numpy as np

import pystencils as ps

import sympy as sp


def gen_diffusion_flux_pyst_kernel_3d(
    real_t,
    num_threads=False,
    fixed_grid_size=False,
    field_type="scalar",
):
    # TODO expand docs
    """3D Diffusion flux kernel generator."""
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    pyst_dtype = "float32" if real_t == np.float32 else "float64"
    kernel_config = ps.CreateKernelConfig(
        data_type=pyst_dtype, default_number_float=pyst_dtype, cpu_openmp=num_threads
    )
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if fixed_grid_size
        else "3D"
    )

    @ps.kernel
    def _diffusion_stencil_3d():
        diffusion_flux, field = ps.fields(
            f"diffusion_flux, field : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        diffusion_flux[0, 0, 0] @= prefactor * (
            field[1, 0, 0]
            + field[-1, 0, 0]
            + field[0, 1, 0]
            + field[0, -1, 0]
            + field[0, 0, 1]
            + field[0, 0, -1]
            - 6 * field[0, 0, 0]
        )

    diffusion_flux_pyst_kernel_3d = ps.create_kernel(
        _diffusion_stencil_3d, config=kernel_config
    ).compile()
    if field_type == "scalar":
        return diffusion_flux_pyst_kernel_3d
    elif field_type == "vector":

        def vector_field_diffusion_flux_pyst_kernel_3d(
            vector_field_diffusion_flux, vector_field, prefactor
        ):
            """Vector field diffusion flux in 3D.

            Computes diffusion flux (3D vector field) essentially vector
            Laplacian for a 3D vector field
            assumes shape of fields (3, n, n, n)
            """
            diffusion_flux_pyst_kernel_3d(
                diffusion_flux=vector_field_diffusion_flux[0],
                field=vector_field[0],
                prefactor=prefactor,
            )
            diffusion_flux_pyst_kernel_3d(
                diffusion_flux=vector_field_diffusion_flux[1],
                field=vector_field[1],
                prefactor=prefactor,
            )
            diffusion_flux_pyst_kernel_3d(
                diffusion_flux=vector_field_diffusion_flux[2],
                field=vector_field[2],
                prefactor=prefactor,
            )

        return vector_field_diffusion_flux_pyst_kernel_3d
