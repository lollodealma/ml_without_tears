import numpy as np
from loss.loss_template import Loss


class CrossEntropyLoss(Loss):

    def get_loss(self):
        # compute loss l(xin, y)
        self.loss = - np.log(np.exp(self.xin[self.y[0]]) / sum(np.exp(self.xin)))

    def get_dloss_dxin(self):
        # compute gradient of loss wrt xin
        self.dloss_dxin = np.exp(self.xin) / sum(np.exp(self.xin))
        self.dloss_dxin[self.y] -= 1

