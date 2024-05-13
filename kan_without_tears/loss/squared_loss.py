import numpy as np
from loss.loss_template import Loss


class SquaredLoss(Loss):

    def get_loss(self):
        # compute loss l(xin, y)
        self.loss = np.mean(np.power(self.xin - self.y, 2))

    def get_dloss_dxin(self):
        # compute gradient of loss wrt xin
        self.dloss_dxin = 2 * (self.xin - self.y) / self.n_in
