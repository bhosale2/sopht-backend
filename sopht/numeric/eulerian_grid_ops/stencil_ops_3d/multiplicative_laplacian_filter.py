"""Kernels applying laplacian filter on 3d grid for scalar and vectorial fields"""
import pystencils as ps
from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config


def gen_multiplicative_laplacian_filter_kernel_3d(
    real_t,
    num_threads=False,
    fixed_grid_size=False,
    field_type="scalar",
):
    """
    Multiplicative laplacian filter kernel generator. Based on the field type filter kernels for both scalar and
    vectorial field can be constructed. One dimensional laplacian filter applied on the field in 3d.

    Notes
    -----
    For details regarding the numerics behind the filtering, refer to [1]_, [2]_.
    .. [1] Jeanmart, H., & Winckelmans, G. (2007). Investigation of eddy-viscosity
       models modified using discrete filters: a simplified “regularized variational
       multiscale model” and an “enhanced field model”. Physics of fluids, 19(5), 055110.
    .. [2] Lorieul, G. (2018). Development and validation of a 2D Vortex Particle-Mesh
       method for incompressible multiphase flows (Doctoral dissertation,
       Université Catholique de Louvain).
    """

    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if fixed_grid_size
        else "3D"
    )

    @ps.kernel
    def _laplacian_filter_3d_x():
        filtered_field, field = ps.fields(
            f"filtered_field, field : {pyst_dtype}[{grid_info}]"
        )
        filtered_field[0, 0, 0] @= 0.25 * (
            -field[1, 0, 0] - field[-1, 0, 0] + 2 * field[0, 0, 0]
        )

    laplacian_filter_3d_x = ps.create_kernel(
        _laplacian_filter_3d_x, config=kernel_config
    ).compile()

    @ps.kernel
    def _laplacian_filter_3d_y():
        filtered_field, field = ps.fields(
            f"filtered_field, field : {pyst_dtype}[{grid_info}]"
        )
        filtered_field[0, 0, 0] @= 0.25 * (
            -field[0, 1, 0] - field[0, -1, 0] + 2 * field[0, 0, 0]
        )

    laplacian_filter_3d_y = ps.create_kernel(
        _laplacian_filter_3d_y, config=kernel_config
    ).compile()

    @ps.kernel
    def _laplacian_filter_3d_z():
        filtered_field, field = ps.fields(
            f"filtered_field, field : {pyst_dtype}[{grid_info}]"
        )
        filtered_field[0, 0, 0] @= 0.25 * (
            -field[0, 0, 1] - field[0, 0, -1] + 2 * field[0, 0, 0]
        )

    laplacian_filter_3d_z = ps.create_kernel(
        _laplacian_filter_3d_z, config=kernel_config
    ).compile()

    def laplacian_filter_3d(filtered_field, field):
        """
        Apply laplacian filter on a scalar field
        """

        # Laplacian filter in x direction
        laplacian_filter_3d_x(filtered_field=filtered_field, field=field)

        # # Laplacian filter in y direction,
        # # swap the field and filtered field arrays. Thus, we don't need a second buffer array.
        field *= 0
        laplacian_filter_3d_y(filtered_field=field, field=filtered_field)

        # Laplacian filter in z direction
        laplacian_filter_3d_z(filtered_field=filtered_field, field=field)

        # X boundaries
        filtered_field[0, :, :] = 0.0
        filtered_field[-1, :, :] = 0.0
        # Y boundaries
        filtered_field[:, 0, :] = 0.0
        filtered_field[:, -1, :] = 0.0
        # Z boundaries
        filtered_field[:, :, 0] = 0.0
        filtered_field[:, :, -1] = 0.0

    def scalar_field_filter_kernel_3d(
        scalar_field,
        scalar_field_buffer,
        filtered_field_buffer,
        filter_order: int,
    ):
        """
        Applies laplacian filter on any scalar field.
        """
        scalar_field_buffer[:] = scalar_field[:]

        for _ in range(filter_order):

            laplacian_filter_3d(
                filtered_field=filtered_field_buffer, field=scalar_field_buffer
            )
            scalar_field_buffer[:] = filtered_field_buffer

        scalar_field -= scalar_field_buffer

    def vector_filed_filter_kernel_3d(
        vector_field,
        vector_field_buffer,
        filtered_field_buffer,
        filter_order: int,
    ):
        """
        Applies laplacian filter on any vectorial field.
        """
        vector_field_buffer[:] = vector_field[:]
        # Slice vector field first
        for _ in range(filter_order):

            # X dimension of vector field
            laplacian_filter_3d(
                filtered_field=filtered_field_buffer, field=vector_field_buffer[0]
            )
            vector_field_buffer[0, :, :, :] = filtered_field_buffer

            # Y dimension of vector field
            laplacian_filter_3d(
                filtered_field=filtered_field_buffer, field=vector_field_buffer[1]
            )
            vector_field_buffer[1, :, :, :] = filtered_field_buffer

            # Z dimension of vector field
            laplacian_filter_3d(
                filtered_field=filtered_field_buffer, field=vector_field_buffer[2]
            )
            vector_field_buffer[2, :, :, :] = filtered_field_buffer

        vector_field -= vector_field_buffer

    # Depending on the field type return the relevant filter implementation
    if field_type == "scalar":
        return scalar_field_filter_kernel_3d
    elif field_type == "vector":
        return vector_filed_filter_kernel_3d
