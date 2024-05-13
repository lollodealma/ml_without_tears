import numpy as np
from scipy.interpolate import BSpline


def get_bsplines(x_bounds, n_basis, degree=3, **kwargs):
    grid_len = n_basis - degree + 1
    step = (x_bounds[1] - x_bounds[0]) / (grid_len - 1)
    basis, basis_der = {}, {}

    # SiLU bias function
    basis[0] = lambda x: x / (1 + np.exp(-x))
    basis_der[0] = lambda x: (1 + np.exp(-x) + x * np.exp(-x)) / np.power((1 + np.exp(-x)), 2)

    # B-splines
    t = np.linspace(x_bounds[0] - degree * step, x_bounds[1] + degree * step, grid_len + 2 * degree)
    t[degree], t[-degree - 1] = x_bounds[0], x_bounds[1]
    for ind_spline in range(n_basis - 1):
        basis[ind_spline + 1] = BSpline.basis_element(t[ind_spline:ind_spline + degree + 2], extrapolate=False)
        basis_der[ind_spline + 1] = basis[ind_spline + 1].derivative()
    return basis, basis_der


def get_chebishev(x_bounds, n_basis, **kwargs):
    basis, basis_der = {}, {}
    for deg in range(n_basis):
        basis[deg] = np.polynomial.chebyshev.Chebyshev.basis(deg=deg, domain=x_bounds)
        basis_der[deg] = basis[deg].deriv(1)
    return basis, basis_der

