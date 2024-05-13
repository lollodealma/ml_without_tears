import numpy as np
from neuron.neuron_nn import NeuronNN


class FullyConnectedLayer:

    def __init__(self, n_in, n_out, neuron_class=NeuronNN, **kwargs):
        self.n_in, self.n_out = n_in, n_out
        self.neurons = [neuron_class(n_in) if (kwargs == {}) else neuron_class(n_in, **kwargs) for _ in range(n_out)]
        self.xin = None  # input, shape (n_in,)
        self.xout = None  # output, shape (n_out,)
        self.dloss_dxin = None  # derivative loss wrt xin, shape (n_in,)
        self.dloss_dpar = None  # derivative loss wrt par, shape (n_in, n_par_per_edge)
        self.zero_grad()

    def __call__(self, xin):
        # forward pass
        self.xin = xin
        self.xout = np.array([nn(self.xin) for nn in self.neurons])
        return self.xout

    def zero_grad(self, which=None):
        # reset gradients to zero
        if which is None:
            which = ['xin', 'par']
        for w in which:
            match w:
                case 'xin':
                    self.dloss_dxin = np.zeros(self.n_in)
                case 'par':
                    for nn in self.neurons:
                        nn.dloss_dpar = np.zeros((self.n_in, self.neurons[0].n_par_per_edge))
                case _:
                    raise ValueError('input \'which\' value not recognized')

    def update_grad(self, dloss_dxout):
        # update gradients
        for ii, dloss_dxout_tmp in enumerate(dloss_dxout):
            self.dloss_dxin += self.neurons[ii].dxout_dxin * dloss_dxout_tmp
            self.neurons[ii].dloss_dpar += self.neurons[ii].dxout_dpar * dloss_dxout_tmp
        return self.dloss_dxin
