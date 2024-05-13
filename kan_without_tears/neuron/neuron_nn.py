import numpy as np
from neuron.neuron_template import Neuron
from utils.activations import relu


class NeuronNN(Neuron):

    def __init__(self, n_in, params_init=None, params_range=None, activation=relu):
        super().__init__(n_in, n_par_per_edge=2, params_init=params_init, params_range=params_range)
        self.activation = activation

    def get_xmid(self):
        self.xmid = self.params[:, 0] + self.params[:, 1] * self.xin

    def get_xout(self):
        self.xout = self.activation(sum(self.xmid.flatten()), get_derivative=False)

    def get_dxout_dxmid(self):
        self.dxout_dxmid = self.activation(sum(self.xmid.flatten()), get_derivative=True) * np.ones(self.n_in)

    def get_dxmid_dpar(self):
        self.dxmid_dpar = np.concatenate((np.ones((self.n_in, 1)), np.reshape(self.xin, (-1, 1))), axis=1)

    def get_dxmid_dxin(self):
        self.dxmid_dxin = self.params[:, 1]
