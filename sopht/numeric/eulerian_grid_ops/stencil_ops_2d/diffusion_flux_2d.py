"""Kernels for computing diffusion flux in 2D."""
import numpy as np

import pystencils as ps

import sympy as sp


def gen_diffusion_flux_pyst_kernel_2d(real_t, num_threads=False, fixed_grid_size=False):
    # TODO expand docs
    """2D Diffusion flux kernel generator."""
    pyst_dtype = "float32" if real_t == np.float32 else "float64"
    kernel_config = ps.CreateKernelConfig(data_type=pyst_dtype, cpu_openmp=num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
    )

    @ps.kernel
    def _diffusion_stencil_2d():
        diffusion_flux, field = ps.fields(
            f"diffusion_flux, field : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        diffusion_flux[0, 0] @= diffusion_flux[0, 0] + prefactor * (
            field[1, 0] + field[-1, 0] + field[0, 1] + field[0, -1] - 4 * field[0, 0]
        )

    diffusion_flux_kernel_2d = ps.create_kernel(
        _diffusion_stencil_2d, config=kernel_config
    ).compile()
    return diffusion_flux_kernel_2d
