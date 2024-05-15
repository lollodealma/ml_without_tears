import unittest
from feed_forward_network.feedforward import FeedForward
from loss.cross_entropy_loss import CrossEntropyLoss
from utils.activations import relu, tanh_act, sigmoid_act
from neuron.neuron_kan import NeuronKAN
import numpy as np


class TestGradients(unittest.TestCase):

    def test_gradients_MLP_KAN(self):
        # validate gradient of loss wrt weights via finite differences
        h = 1e-7
        seed = np.random.randint(1000)
        np.random.seed(seed)
        print(f'KAN seed={seed}')
        mlp = FeedForward([1, 3, 2, 2],
                          eps=.0001,
                          loss=CrossEntropyLoss,
                          n_weights_per_edge=5,
                          neuron_class=NeuronKAN,
                          x_bounds=[-1, 1])
        x_train = [[np.random.rand()]]
        y_train = [[1]]  # [[np.random.rand()]]

        # forward pass
        x_out = mlp(x_train[0])

        # compute loss
        loss = mlp.loss(x_out, y_train[0])

        # backward propagation
        mlp.backprop()

        for ll in mlp.layers:
            for nn in ll.neurons:
                for i_row in range(nn.weights.shape[0]):
                    for i_col in range(nn.weights.shape[1]):
                        dloss_dpar_calc = nn.dloss_dw[i_row, i_col]
                        nn.weights[i_row, i_col] += h
                        x_out_dh = mlp(x_train[0])
                        loss_dh = mlp.loss(x_out_dh, y_train[0])
                        dloss_dpar_est = (loss_dh - loss) / h
                        # reset parameters
                        nn.weights[i_row, i_col] -= h
                        error = abs(dloss_dpar_est - dloss_dpar_calc)
                        self.assertTrue(error < .001, f'error={error}')  # add assertion here

    def test_gradients_MLP_NN(self):
        # validate gradient of loss wrt weights and biases via finite differences
        h = 1e-7
        seed = np.random.randint(1000)
        np.random.seed(seed)
        print(f'MLP seed={seed}')
        for aa in [relu, tanh_act, sigmoid_act]: # relu, tanh_act, sigmoid_act]:
            mlp = FeedForward([1, 4, 3, 1], eps=.0001, activation=aa)
            x_train = [[np.random.rand()]]
            y_train = [[np.random.rand()]]

            # forward pass
            x_out = mlp(x_train[0])

            # compute loss
            loss = mlp.loss(x_out, y_train[0])

            # backward propagation
            mlp.backprop()

            for ll in mlp.layers[::-1]:
                for nn in ll.neurons:

                    # gradient wrt weights
                    for i_row in range(nn.weights.shape[0]):
                        for i_col in range(nn.weights.shape[1]):
                            dloss_dw_calc = nn.dloss_dw[i_row, i_col]
                            nn.weights[i_row, i_col] += h
                            x_out_dh = mlp(x_train[0])
                            loss_dh = mlp.loss(x_out_dh, y_train[0])
                            dloss_dw_est = (loss_dh - loss) / h
                            # reset parameters
                            nn.weights[i_row, i_col] -= h
                            error = abs(dloss_dw_est - dloss_dw_calc)
                            self.assertTrue(error < .001, f'error={error}')  # add assertion here

                    # gradient wrt bias
                    dloss_dbias_calc = nn.dloss_dbias
                    nn.bias += h
                    x_out_dh = mlp(x_train[0])
                    loss_dh = mlp.loss(x_out_dh, y_train[0])
                    dloss_dbias_est = (loss_dh - loss) / h
                    # reset parameters
                    nn.bias -= h
                    error = abs(dloss_dbias_est - dloss_dbias_calc)
                    self.assertTrue(error < .001, f'error={error}')


if __name__ == '__main__':
    unittest.main()
