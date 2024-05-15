import numpy as np
from neuron.neuron_nn import NeuronNN


class FullyConnectedLayer:

    def __init__(self, n_in, n_out, neuron_class=NeuronNN, **kwargs):
        self.n_in, self.n_out = n_in, n_out
        self.neurons = [neuron_class(n_in) if (kwargs == {}) else neuron_class(n_in, **kwargs) for _ in range(n_out)]
        self.xin = None  # input, shape (n_in,)
        self.xout = None  # output, shape (n_out,)
        self.dloss_dxin = None  # d loss / d xin, shape (n_in,)
        self.zero_grad()

    def __call__(self, xin):
        # forward pass
        self.xin = xin
        self.xout = np.array([nn(self.xin) for nn in self.neurons])
        return self.xout

    def zero_grad(self, which=None):
        # reset gradients to zero
        if which is None:
            which = ['xin', 'weights', 'bias']
        for w in which:
            if w == 'xin':  # reset layer's d loss / d xin
                self.dloss_dxin = np.zeros(self.n_in)
            elif w == 'weights':  # reset d loss / dw to zero for every neuron
                for nn in self.neurons:
                    nn.dloss_dw = np.zeros((self.n_in, self.neurons[0].n_weights_per_edge))
            elif w == 'bias':  # reset d loss / db to zero for every neuron
                for nn in self.neurons:
                    nn.dloss_dbias = 0
            else:
                raise ValueError('input \'which\' value not recognized')

    def update_grad(self, dloss_dxout):
        # update gradients by chain rule
        for ii, dloss_dxout_tmp in enumerate(dloss_dxout):
            # update layer's d loss / d xin via chain rule
            # note: account for all possible xin -> xout -> loss paths!
            self.dloss_dxin += self.neurons[ii].dxout_dxin * dloss_dxout_tmp
            # update neuron's d loss / dw and d loss / d bias
            self.neurons[ii].update_dloss_dw_dbias(dloss_dxout_tmp)
        return self.dloss_dxin
