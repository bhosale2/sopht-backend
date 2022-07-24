import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_vorticity_stretching_timestep_euler_forward_pyst_kernel_3d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def vorticity_stretching_timestep_euler_forward_reference(
    vorticity_field, velocity_field, dt_by_2_dx
):
    new_vorticity_field = vorticity_field.copy()
    new_vorticity_field[0, 1:-1, 1:-1, 1:-1] = vorticity_field[
        0, 1:-1, 1:-1, 1:-1
    ] + dt_by_2_dx * (
        vorticity_field[0, 1:-1, 1:-1, 1:-1]
        * (velocity_field[0, 2:, 1:-1, 1:-1] - velocity_field[0, :-2, 1:-1, 1:-1])
        + vorticity_field[1, 1:-1, 1:-1, 1:-1]
        * (velocity_field[0, 1:-1, 2:, 1:-1] - velocity_field[0, 1:-1, :-2, 1:-1])
        + vorticity_field[2, 1:-1, 1:-1, 1:-1]
        * (velocity_field[0, 1:-1, 1:-1, 2:] - velocity_field[0, 1:-1, 1:-1, :-2])
    )
    new_vorticity_field[1, 1:-1, 1:-1, 1:-1] = vorticity_field[
        1, 1:-1, 1:-1, 1:-1
    ] + dt_by_2_dx * (
        vorticity_field[0, 1:-1, 1:-1, 1:-1]
        * (velocity_field[1, 2:, 1:-1, 1:-1] - velocity_field[1, :-2, 1:-1, 1:-1])
        + vorticity_field[1, 1:-1, 1:-1, 1:-1]
        * (velocity_field[1, 1:-1, 2:, 1:-1] - velocity_field[1, 1:-1, :-2, 1:-1])
        + vorticity_field[2, 1:-1, 1:-1, 1:-1]
        * (velocity_field[1, 1:-1, 1:-1, 2:] - velocity_field[1, 1:-1, 1:-1, :-2])
    )
    new_vorticity_field[2, 1:-1, 1:-1, 1:-1] = vorticity_field[
        2, 1:-1, 1:-1, 1:-1
    ] + dt_by_2_dx * (
        vorticity_field[0, 1:-1, 1:-1, 1:-1]
        * (velocity_field[2, 2:, 1:-1, 1:-1] - velocity_field[2, :-2, 1:-1, 1:-1])
        + vorticity_field[1, 1:-1, 1:-1, 1:-1]
        * (velocity_field[2, 1:-1, 2:, 1:-1] - velocity_field[2, 1:-1, :-2, 1:-1])
        + vorticity_field[2, 1:-1, 1:-1, 1:-1]
        * (velocity_field[2, 1:-1, 1:-1, 2:] - velocity_field[2, 1:-1, 1:-1, :-2])
    )
    return new_vorticity_field


class VorticityStretchingTimestepEulerForwardSolution:
    def __init__(self, n_samples, precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_vorticity_field = np.random.randn(
            3, n_samples, n_samples, n_samples
        ).astype(real_t)
        self.ref_velocity_field = np.random.randn(
            3, n_samples, n_samples, n_samples
        ).astype(real_t)
        self.dt_by_2_dx = real_t(0.1)
        self.ref_new_vorticity_field = (
            vorticity_stretching_timestep_euler_forward_reference(
                self.ref_vorticity_field, self.ref_velocity_field, self.dt_by_2_dx
            )
        )

    def check_equals(self, new_vorticity_field):
        np.testing.assert_allclose(
            self.ref_new_vorticity_field[:, 1:-1, 1:-1, 1:-1],
            new_vorticity_field[:, 1:-1, 1:-1, 1:-1],
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_vort_stretching_timestep_euler_forward_3d(n_values, precision):
    real_t = get_real_t(precision)
    solution = VorticityStretchingTimestepEulerForwardSolution(n_values, precision)
    vorticity_field = solution.ref_vorticity_field.copy()
    vorticity_stretching_flux_field = np.ones_like(vorticity_field)
    vorticity_stretching_timestep_euler_forward_pyst_kernel_3d = (
        gen_vorticity_stretching_timestep_euler_forward_pyst_kernel_3d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
        )
    )
    vorticity_stretching_timestep_euler_forward_pyst_kernel_3d(
        vorticity_field=vorticity_field,
        velocity_field=solution.ref_velocity_field,
        vorticity_stretching_flux_field=vorticity_stretching_flux_field,
        dt_by_2_dx=solution.dt_by_2_dx,
    )
    solution.check_equals(vorticity_field)
