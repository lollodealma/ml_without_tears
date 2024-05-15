import numpy as np
from neuron.neuron_template import Neuron
from utils.activations import relu


class NeuronNN(Neuron):

    def __init__(self, n_in, weights_range=None, activation=relu):
        super().__init__(n_in, n_weights_per_edge=1, weights_range=weights_range)
        self.activation = activation
        self.activation_input = None

    def get_xmid(self):
        self.xmid = self.weights[:, 0] * self.xin

    def get_xout(self):
        self.activation_input = sum(self.xmid.flatten()) + self.bias
        self.xout = self.activation(self.activation_input, get_derivative=False)

    def get_dxout_dxmid(self):
        self.dxout_dxmid = self.activation(self.activation_input, get_derivative=True) * np.ones(self.n_in)

    def get_dxout_dbias(self):
        self.dxout_dbias = self.activation(self.activation_input, get_derivative=True)

    def get_dxmid_dw(self):
        self.dxmid_dw = np.reshape(self.xin, (-1, 1))

    def get_dxmid_dxin(self):
        self.dxmid_dxin = self.weights.flatten()
