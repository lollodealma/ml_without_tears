import numpy as np


class Neuron:

    def __init__(self, n_in, n_par_per_edge, params_init=None, params_range=None):
        self.n_in = n_in
        self.n_par_per_edge = n_par_per_edge
        if params_init is not None:
            self.params = np.array(params_init)
            assert self.params.shape == (self.n_in, self.n_par_per_edge)
        else:
            params_range = [-1, 1] if params_range is None else params_range
            self.params = np.random.uniform(params_range[0], params_range[-1], size=(self.n_in, self.n_par_per_edge))
        self.xin = None  # input
        self.xmid = None  # edge variables
        self.xout = None  # output float
        self.dxout_dxmid = None  # derivative d xout / d xmid: (n_in, )
        self.dxmid_dpar = None  # derivative d xmid / d par: (n_in, n_par_per_edge)
        self.dxmid_dxin = None  # derivative d xmid / d xin
        self.dxout_dxin = None  # (composite) derivative d xout / d xin
        self.dxout_dpar = None  # (composite) derivative d xout / d par
        self.dloss_dpar = None  # (composite) derivative d loss / d par

    def __call__(self, xin):
        # forward pass: compute neuron's output
        self.xin = np.array(xin)
        self.get_xmid()
        self.get_xout()

        # compute internal derivatives
        self.get_dxout_dxmid()
        self.get_dxmid_dpar()
        self.get_dxmid_dxin()

        assert self.dxout_dxmid.shape == (self.n_in, )
        assert self.dxmid_dxin.shape == (self.n_in, )
        assert self.dxmid_dpar.shape == (self.n_in, self.n_par_per_edge)

        self.get_dxout_dxin()
        self.get_dxout_dpar()

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

    def get_dxmid_dpar(self):
        # compute self.dxmid_dpar
        pass

    def get_dxmid_dxin(self):
        # compute self.dxmid_dxin
        pass

    def get_dxout_dxin(self):
        self.dxout_dxin = self.dxout_dxmid * self.dxmid_dxin

    def get_dxout_dpar(self):
        self.dxout_dpar = np.diag(self.dxout_dxmid) @ self.dxmid_dpar

    def gradient_descent_par(self, eps):
        self.params -= eps * self.dloss_dpar
