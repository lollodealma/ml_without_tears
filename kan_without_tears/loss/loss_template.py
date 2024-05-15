import numpy as np


class Loss:

    def __init__(self, n_in):
        self.n_in = n_in
        self.y, self.dloss_dy, self.loss, self.y_train = None, None, None, None

    def __call__(self, y, y_train):
        # y: output of network
        # y_train: ground truth
        self.y, self.y_train = np.array(y), y_train
        self.get_loss()
        self.get_dloss_dy()
        return self.loss

    def get_loss(self):
        # compute loss l(y, y_train)
        pass

    def get_dloss_dy(self):
        # compute gradient of loss wrt y
        pass
