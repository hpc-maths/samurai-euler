import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import argparse


def pred_coeff(pred_s, sign):
    if pred_s == 0:
        return np.array([1])
    elif pred_s == 1:
        return np.array([sign / 8, 1, -sign / 8])
    elif pred_s == 2:
        return np.array(
            [
                -sign * 3.0 / 128.0,
                sign * 22.0 / 128.0,
                1,
                -sign * 22 / 128.0,
                sign * 3.0 / 128.0,
            ]
        )


def read_frame_euler_2d(filename, box=None):
    mesh = h5py.File(filename + ".h5", "r")["mesh"]
    points = mesh["points"]
    connectivity = mesh["connectivity"]

    xyz = points[:][connectivity[:]][:, :, :]
    centers = 0.25 * (xyz[:, 0, :] + xyz[:, 1, :] + xyz[:, 2, :] + xyz[:, 3, :])

    sol = {}
    sol["rho"] = mesh["fields"]["rho"][:]
    sol["pressure"] = mesh["fields"]["pressure"][:]
    sol["ux"] = mesh["fields"]["velocity_0"][:]
    sol["uy"] = mesh["fields"]["velocity_1"][:]

    if "levels" in mesh["fields"]:
        levels = mesh["fields"]["levels"][:]
    else:
        # "levels" is only written with --save-debug-fields; the level of a cell
        # is otherwise recovered from its size, dx = ref * 2**-level.
        if box is None:
            raise ValueError("no 'levels' field in the file: pass the domain box")
        dx = xyz[:, :, 0].max(axis=1) - xyz[:, :, 0].min(axis=1)
        ref = min(box[1][0] - box[0][0], box[1][1] - box[0][1])
        levels = np.round(np.log2(ref / dx)).astype(int)

    return centers[:, 0], centers[:, 1], sol, levels


def grid_shape(box, level):
    """Number of leaf cells in each direction at `level`, and the cell size.

    samurai's cell length at a given level is ref * 2**-level, where ref is the
    smallest side of the domain, so a non-square box holds size/ref times more
    cells along its long side.
    """
    (xmin, ymin), (xmax, ymax) = box
    ref = min(xmax - xmin, ymax - ymin)
    dx = ref * 2.0**-level
    nx = int(round((xmax - xmin) / dx))
    ny = int(round((ymax - ymin) / dx))
    return nx, ny, dx


def cell_centers(box, level, pred_s):
    """Centers of the leaf cells at `level`, ghost layers included."""
    nx, ny, dx = grid_shape(box, level)
    xmin, ymin = box[0]
    i = np.arange(-pred_s, nx + pred_s)
    j = np.arange(-pred_s, ny + pred_s)
    return np.meshgrid(xmin + (i + 0.5) * dx, ymin + (j + 0.5) * dx, indexing="ij")


def ghost_mask(shape, pred_s):
    mask = np.ones(shape, dtype=bool)
    if pred_s > 0:
        mask[pred_s:-pred_s, pred_s:-pred_s] = False
    return mask


def make_double_mach_bc(Tf):
    """Boundary condition of the double Mach reflection test case."""
    alpha = np.pi / 3.0
    x0 = 2.0 / 3

    left_state = {
        "rho": 8.0,
        "pressure": 116.5,
        "ux": 8.25 * np.sin(alpha),
        "uy": -8.25 * np.cos(alpha),
    }

    right_state = {"rho": 1.4, "pressure": 1.0, "ux": 0.0, "uy": 0.0}

    def bc(u, pred_s, level, box, var_name):
        if pred_s == 0:
            return
        nx, _, dx = grid_shape(box, level)
        xmin = box[0][0]

        # left / right boundaries
        u[:pred_s, :] = left_state[var_name]
        u[-pred_s:, :] = right_state[var_name]

        x = xmin + (np.arange(nx) + 0.5) * dx
        # the incident shock has moved to x1 at time Tf
        x1 = x0 + 10 * Tf / np.sin(alpha) + 1 / np.tan(alpha)
        top = np.where(x < x1, left_state[var_name], right_state[var_name])
        bottom = np.where(x < x0, left_state[var_name], right_state[var_name])

        for g in range(pred_s):
            u[pred_s:-pred_s, -(g + 1)] = top
            u[pred_s:-pred_s, g] = bottom

    return bc


def make_exact_bc(exact_fn):
    """Fill the ghost layers with an analytic solution `exact_fn(x, y)`.

    Used by the smooth test cases (isentropic vortex, free stream), where the
    exact solution is imposed on the boundaries by the solver as well.
    """

    def bc(u, pred_s, level, box, var_name):
        if pred_s == 0:
            return
        x, y = cell_centers(box, level, pred_s)
        mask = ghost_mask(u.shape, pred_s)
        u[mask] = exact_fn(x, y)[var_name][mask]

    return bc


def recons_2d(box, x, y, pred_s, sol, level, bc, var_name="rho"):
    """Reconstruct the solution on the uniform grid at the finest level present.

    `bc` fills the ghost layers, see make_double_mach_bc / make_exact_bc.
    """
    xmin, ymin = box[0]

    min_level = int(np.min(level))
    max_level = int(np.max(level))

    u = sol[var_name]

    # read leaf cells
    ul = {}
    for ilevel in range(min_level, max_level + 1):
        nx, ny, dx = grid_shape(box, ilevel)
        ul[ilevel] = np.empty((nx + 2 * pred_s, ny + 2 * pred_s))
        ul[ilevel][:] = np.nan
        (index,) = np.where(level == ilevel)

        index_x = ((x[index] - xmin - 0.5 * dx) / dx).astype(int)
        index_y = ((y[index] - ymin - 0.5 * dx) / dx).astype(int)
        ul[ilevel][index_x + pred_s, index_y + pred_s] = u[index]

    for ilevel in range(min_level, max_level + 1):
        bc(ul[ilevel], pred_s, ilevel, box, var_name)

    # projection of leaves (vectorized, stride 2)
    for ilevel in range(max_level - 1, min_level - 1, -1):
        parent = ul[ilevel]
        child = ul[ilevel + 1]
        child_interior = child[pred_s:-pred_s, pred_s:-pred_s]
        # Extract all 2x2 non-overlapping patches from child (stride 2)
        patches = sliding_window_view(child_interior, (2, 2))[::2, ::2]
        patch_sums = np.sum(patches, axis=(2, 3))
        # Assign directly to parent interior
        parent_interior = parent[pred_s:-pred_s, pred_s:-pred_s]
        mask = ~np.isnan(child_interior[::2, ::2])
        parent_interior[mask] = 0.25 * patch_sums[mask]

    # Compute the four stencils
    st00 = np.outer(pred_coeff(pred_s, 1), pred_coeff(pred_s, 1))
    st10 = np.outer(pred_coeff(pred_s, -1), pred_coeff(pred_s, 1))
    st01 = np.outer(pred_coeff(pred_s, 1), pred_coeff(pred_s, -1))
    st11 = np.outer(pred_coeff(pred_s, -1), pred_coeff(pred_s, -1))

    # prediction (vectorized)
    for ilevel in range(min_level, max_level):
        parent = ul[ilevel]
        child = ul[ilevel + 1]

        # Extract all (2*pred_s+1, 2*pred_s+1) patches
        patches = sliding_window_view(parent, (2 * pred_s + 1, 2 * pred_s + 1))

        # Compute the four child values (vectorized sum over last two dims)
        vals00 = np.tensordot(patches, st00, axes=([2, 3], [0, 1]))
        vals10 = np.tensordot(patches, st10, axes=([2, 3], [0, 1]))
        vals01 = np.tensordot(patches, st01, axes=([2, 3], [0, 1]))
        vals11 = np.tensordot(patches, st11, axes=([2, 3], [0, 1]))

        child_interior = child[pred_s:-pred_s, pred_s:-pred_s]
        mask = np.isnan(child_interior[::2, ::2])

        child_interior[::2, ::2][mask] = vals00[mask]
        child_interior[1::2, ::2][mask] = vals00[mask]
        child_interior[::2, 1::2][mask] = vals10[mask]
        child_interior[1::2, 1::2][mask] = vals11[mask]

    if pred_s > 0:
        return ul[max_level][pred_s:-pred_s, pred_s:-pred_s]
    return ul[max_level][:]


def main():
    parser = argparse.ArgumentParser(description="Samurai Euler Reconstruction")
    parser.add_argument(
        "--pred_s", type=int, default=1, help="Prediction stencil order (0, 1, 2)"
    )
    parser.add_argument("--Tf", type=float, default=0.2, help="Final time Tf")
    parser.add_argument(
        "--filename",
        type=str,
        default="../build/results/double_mach_reflection_hllc_ite_9",
        help="Input filename (without .h5)",
    )
    args = parser.parse_args()

    filename = args.filename
    box = [[0.0, 0.0], [4.0, 1.0]]

    x, y, sol, level = read_frame_euler_2d(filename, box)
    bc = make_double_mach_bc(args.Tf)
    r = recons_2d(box, x, y, args.pred_s, sol, level, bc, var_name="rho")

    import matplotlib.pyplot as plt

    plt.imshow(r.T, origin="lower")
    plt.show()


if __name__ == "__main__":
    main()
