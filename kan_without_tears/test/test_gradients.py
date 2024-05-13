import unittest
from feed_forward_network.feedforward import FeedForward
from loss.cross_entropy_loss import CrossEntropyLoss
from utils.activations import relu, tanh_act, sigmoid_act
from neuron.neuron_kan import NeuronKAN
import numpy as np


class TestMLP(unittest.TestCase):
    def test_gradients_MLP_NN(self):
        h = .0001
        for _ in [relu, tanh_act, sigmoid_act]:
            mlp = FeedForward([1, 3, 2, 1], eps=.0001, activation=sigmoid_act)
            x_train = [[np.random.rand()]]
            y_train = [[np.random.rand()]]

            # forward pass
            x_out = mlp(x_train[0])

            # compute loss
            loss = mlp.loss(x_out, y_train[0])

            # backward propagation
            mlp.backprop()

            for ll in mlp.layers:
                for nn in ll.neurons:
                    for i_row in range(nn.params.shape[0]):
                        for i_col in range(nn.params.shape[1]):
                            nn.params[i_row, i_col] += h
                            x_out_dh = mlp(x_train[0])
                            loss_dh = mlp.loss(x_out_dh, y_train[0])
                            dloss_dpar_est = (loss_dh - loss) / h
                            dloss_dpar_calc = nn.dloss_dpar[i_row, i_col]
                            # reset parameters
                            nn.params[i_row, i_col] -= h
                            error = abs(dloss_dpar_est - dloss_dpar_calc)
                            self.assertTrue(error < .001, f'error={error}')  # add assertion here

    def test_gradients_MLP_KAN(self):
        h = .0001

        mlp = FeedForward([1, 3, 2, 2],
                          eps=.0001,
                          loss=CrossEntropyLoss,
                          n_par_per_edge=5,
                          neuron_class=NeuronKAN,
                          x_bounds=[-1, 1])
        x_train = [[np.random.rand()]]
        y_train = [[1]]  #[[np.random.rand()]]

        # forward pass
        x_out = mlp(x_train[0])

        # compute loss
        loss = mlp.loss(x_out, y_train[0])

        # backward propagation
        mlp.backprop()

        for ll in mlp.layers:
            for nn in ll.neurons:
                for i_row in range(nn.params.shape[0]):
                    for i_col in range(nn.params.shape[1]):
                        nn.params[i_row, i_col] += h
                        x_out_dh = mlp(x_train[0])
                        loss_dh = mlp.loss(x_out_dh, y_train[0])
                        dloss_dpar_est = (loss_dh - loss) / h
                        dloss_dpar_calc = nn.dloss_dpar[i_row, i_col]
                        # reset parameters
                        nn.params[i_row, i_col] -= h
                        error = abs(dloss_dpar_est - dloss_dpar_calc)
                        self.assertTrue(error < .001, f'error={error}')  # add assertion here


if __name__ == '__main__':
    unittest.main()
