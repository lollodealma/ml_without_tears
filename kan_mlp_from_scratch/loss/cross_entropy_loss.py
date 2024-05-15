import numpy as np
from loss.loss_template import Loss


class CrossEntropyLoss(Loss):

    def get_loss(self):
        # compute loss l(xin, y)
        self.loss = - np.log(np.exp(self.y[self.y_train[0]]) / sum(np.exp(self.y)))

    def get_dloss_dy(self):
        # compute gradient of loss wrt xin
        self.dloss_dy = np.exp(self.y) / sum(np.exp(self.y))
        self.dloss_dy[self.y_train] -= 1
