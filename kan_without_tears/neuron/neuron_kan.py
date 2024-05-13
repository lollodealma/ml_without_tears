import numpy as np
from neuron.neuron_template import Neuron
from utils.activations import tanh_act
from utils.basis import get_bsplines


class NeuronKAN(Neuron):

    def __init__(self, n_in, n_par_per_edge, x_bounds, params_init=None, params_range=None, get_basis=get_bsplines, **kwargs):
        self.x_bounds = x_bounds
        super().__init__(n_in, n_par_per_edge=n_par_per_edge, params_init=params_init, params_range=params_range)
        self.basis, self.basis_der = get_basis(self.x_bounds, self.n_par_per_edge, **kwargs)

    def get_xmid(self):
        self.phi_x_mat = np.array([self.basis[b](self.xin) for b in self.basis]).T  # shape (n_in, n_par_per_edge)
        self.phi_x_mat[np.isnan(self.phi_x_mat)] = 0
        self.xmid = (self.params * self.phi_x_mat).sum(axis=1)

    def get_xout(self):
        self.xout = tanh_act(sum(self.xmid.flatten()), get_derivative=False)

    def get_dxout_dxmid(self):
        self.dxout_dxmid = tanh_act(sum(self.xmid.flatten()), get_derivative=True) * np.ones(self.n_in)

    def get_dxmid_dpar(self):
        self.dxmid_dpar = self.phi_x_mat

    def get_dxmid_dxin(self):
        phi_x_der_mat = np.array([self.basis_der[b](self.xin) if self.basis[b](self.xin) is not None else 0
                                  for b in self.basis_der]).T  # shape (n_in, n_par_per_edge)
        phi_x_der_mat[np.isnan(phi_x_der_mat)] = 0
        self.dxmid_dxin = (self.params * phi_x_der_mat).sum(axis=1)
