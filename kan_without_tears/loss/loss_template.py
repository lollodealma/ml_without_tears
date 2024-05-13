import numpy as np


class Loss:

    def __init__(self, n_in):
        self.n_in = n_in
        self.xin, self.dloss_dxin, self.loss, self.y = None, None, None, None

    def __call__(self, xin, y):
        # xin: output of network
        # y: ground truth
        self.xin, self.y = np.array(xin), y
        self.get_loss()
        self.get_dloss_dxin()
        return self.loss

    def get_loss(self):
        # compute loss l(xin, y)
        pass

    def get_dloss_dxin(self):
        # compute gradient of loss wrt xin
        pass
