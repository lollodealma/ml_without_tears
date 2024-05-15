import numpy as np
from scipy.interpolate import BSpline


def get_bsplines(x_bounds, n_fun, degree=3, **kwargs):
    grid_len = n_fun - degree + 1
    step = (x_bounds[1] - x_bounds[0]) / (grid_len - 1)
    edge_fun, edge_fun_der = {}, {}

    # SiLU bias function
    edge_fun[0] = lambda x: x / (1 + np.exp(-x))
    edge_fun_der[0] = lambda x: (1 + np.exp(-x) + x * np.exp(-x)) / np.power((1 + np.exp(-x)), 2)

    # B-splines
    t = np.linspace(x_bounds[0] - degree * step, x_bounds[1] + degree * step, grid_len + 2 * degree)
    t[degree], t[-degree - 1] = x_bounds[0], x_bounds[1]
    for ind_spline in range(n_fun - 1):
        edge_fun[ind_spline + 1] = BSpline.basis_element(t[ind_spline:ind_spline + degree + 2], extrapolate=False)
        edge_fun_der[ind_spline + 1] = edge_fun[ind_spline + 1].derivative()
    return edge_fun, edge_fun_der


def get_chebyshev(x_bounds, n_fun, **kwargs):
    edge_fun, edge_fun_der = {}, {}
    for deg in range(n_fun):
        edge_fun[deg] = np.polynomial.chebyshev.Chebyshev.basis(deg=deg, domain=x_bounds)
        edge_fun_der[deg] = edge_fun[deg].deriv(1)
    return edge_fun, edge_fun_der

