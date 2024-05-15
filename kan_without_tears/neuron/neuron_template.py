import numpy as np


class Neuron:

    def __init__(self, n_in, n_weights_per_edge, weights_range=None):
        self.n_in = n_in  # n. inputs
        self.n_weights_per_edge = n_weights_per_edge
        weights_range = [-1, 1] if weights_range is None else weights_range
        self.weights = np.random.uniform(weights_range[0], weights_range[-1], size=(self.n_in, self.n_weights_per_edge))
        self.bias = 0
        self.xin = None  # input variable
        self.xmid = None  # edge variables
        self.xout = None  # output variable
        self.dxout_dxmid = None  # derivative d xout / d xmid: (n_in, )
        self.dxout_dbias = None  # derivative d xout / d bias
        self.dxmid_dw = None  # derivative d xmid / d w: (n_in, n_par_per_edge)
        self.dxmid_dxin = None  # derivative d xmid / d xin
        self.dxout_dxin = None  # (composite) derivative d xout / d xin
        self.dxout_dw = None  # (composite) derivative d xout / d w
        self.dloss_dw = np.zeros((self.n_in, self.n_weights_per_edge))  # (composite) derivative d loss / d w
        self.dloss_dbias = 0  # (composite) derivative d loss / d bias

    def __call__(self, xin):
        # forward pass: compute neuron's output
        self.xin = np.array(xin)
        self.get_xmid()
        self.get_xout()

        # compute internal derivatives
        self.get_dxout_dxmid()
        self.get_dxout_dbias()
        self.get_dxmid_dw()
        self.get_dxmid_dxin()

        assert self.dxout_dxmid.shape == (self.n_in, )
        assert self.dxmid_dxin.shape == (self.n_in, )
        assert self.dxmid_dw.shape == (self.n_in, self.n_weights_per_edge)

        # compute external derivatives
        self.get_dxout_dxin()
        self.get_dxout_dw()

        return self.xout

    def get_xmid(self):
        # compute self.xmid
        pass

    def get_xout(self):
        # compute self.xout
        pass

    def get_dxout_dxmid(self):
        # compute self.dxout_dxmid
        pass

    def get_dxout_dbias(self):
        # compute self.dxout_dbias
        pass  #self.dxout_dbias = 0  # by default

    def get_dxmid_dw(self):
        # compute self.dxmid_dw
        pass

    def get_dxmid_dxin(self):
        # compute self.dxmid_dxin
        pass

    def get_dxout_dxin(self):
        self.dxout_dxin = self.dxout_dxmid * self.dxmid_dxin

    def get_dxout_dw(self):
        self.dxout_dw = np.diag(self.dxout_dxmid) @ self.dxmid_dw

    def update_dloss_dw_dbias(self, dloss_dxout):
        self.dloss_dw += self.dxout_dw * dloss_dxout
        self.dloss_dbias += self.dxout_dbias * dloss_dxout

    def gradient_descent(self, eps):
        self.weights -= eps * self.dloss_dw
        self.bias -= eps * self.dloss_dbias
