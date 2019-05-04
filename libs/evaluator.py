from keras import backend as K
import numpy as np


"""
Compute loss and gradients in one pass (faster and more efficient than doing
it twice), and retrieve each value with its corresponding function
"""


class Evaluator(object):

    def __init__(self, generated_image, loss, img_dims):
        # Get gradient tensor of loss with respect to generated_image
        grads = K.gradients(loss, generated_image)[0]
        # Outputs loss and grads from generated_image
        self.fetch_loss_and_grads = K.function([generated_image],
                                               [loss, grads])
        self.img_height, self.img_width = img_dims
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, self.img_height, self.img_width, 3))
        outs = self.fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
