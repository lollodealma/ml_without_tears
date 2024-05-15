import numpy as np
from loss.loss_template import Loss


class SquaredLoss(Loss):

    def get_loss(self):
        # compute loss l(xin, y)
        self.loss = np.mean(np.power(self.y - self.y_train, 2))

    def get_dloss_dy(self):
        # compute gradient of loss wrt xin
        self.dloss_dy = 2 * (self.y - self.y_train) / self.n_in
