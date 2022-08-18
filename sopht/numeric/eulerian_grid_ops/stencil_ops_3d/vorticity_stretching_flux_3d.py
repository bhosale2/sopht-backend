"""Kernels for computing vorticity stretching flux in 3D."""
import numpy as np

import pystencils as ps

import sympy as sp


def gen_vorticity_stretching_flux_pyst_kernel_3d(
    real_t, num_threads=False, fixed_grid_size=False
):
    # TODO expand docs
    """3D Vorticity stretching flux kernel generator."""
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
    def _vorticity_stretching_flux_single_comp_stencil_3d():
        vorticity_stretching_flux_field_comp, velocity_field_comp = ps.fields(
            f"vorticity_stretching_flux_field_comp, "
            f"velocity_field_comp : {pyst_dtype}[{grid_info}]"
        )
        vorticity_field_x, vorticity_field_y, vorticity_field_z = ps.fields(
            f"vorticity_field_x, vorticity_field_y, "
            f"vorticity_field_z : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        vorticity_stretching_flux = prefactor * (
            vorticity_field_x[0, 0, 0]
            * (velocity_field_comp[0, 0, 1] - velocity_field_comp[0, 0, -1])
            + vorticity_field_y[0, 0, 0]
            * (velocity_field_comp[0, 1, 0] - velocity_field_comp[0, -1, 0])
            + vorticity_field_z[0, 0, 0]
            * (velocity_field_comp[1, 0, 0] - velocity_field_comp[-1, 0, 0])
        )
        vorticity_stretching_flux_field_comp[0, 0, 0] @= (
            vorticity_stretching_flux_field_comp[0, 0, 0] + vorticity_stretching_flux
        )

    _vorticity_stretching_flux_single_comp_kernel_3d = ps.create_kernel(
        _vorticity_stretching_flux_single_comp_stencil_3d,
        config=kernel_config,
    ).compile()

    def vorticity_stretching_flux_pyst_kernel_3d(
        vorticity_stretching_flux_field, vorticity_field, velocity_field, prefactor
    ):
        """Vorticity stretching flux kernel in 3D.

        Computes the vorticity stretching flux in 3D, for a 3D
        vorticity_field (3, n, n, n) and velocity_field (3, n, n, n), and
        stores result in vorticity_stretching_flux_field (3, n, n, n).
        """
        _vorticity_stretching_flux_single_comp_kernel_3d(
            vorticity_stretching_flux_field_comp=vorticity_stretching_flux_field[0],
            velocity_field_comp=velocity_field[0],
            vorticity_field_z=vorticity_field[0],
            vorticity_field_y=vorticity_field[1],
            vorticity_field_x=vorticity_field[2],
            prefactor=prefactor,
        )
        _vorticity_stretching_flux_single_comp_kernel_3d(
            vorticity_stretching_flux_field_comp=vorticity_stretching_flux_field[1],
            velocity_field_comp=velocity_field[1],
            vorticity_field_z=vorticity_field[0],
            vorticity_field_y=vorticity_field[1],
            vorticity_field_x=vorticity_field[2],
            prefactor=prefactor,
        )
        _vorticity_stretching_flux_single_comp_kernel_3d(
            vorticity_stretching_flux_field_comp=vorticity_stretching_flux_field[2],
            velocity_field_comp=velocity_field[2],
            vorticity_field_z=vorticity_field[0],
            vorticity_field_y=vorticity_field[1],
            vorticity_field_x=vorticity_field[2],
            prefactor=prefactor,
        )

    return vorticity_stretching_flux_pyst_kernel_3d
