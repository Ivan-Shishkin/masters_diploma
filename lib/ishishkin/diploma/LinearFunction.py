import numpy as np


class LinearFunction:

    def __init__(self, *, a_0=0, a_1=1):
        self.a_0 = a_0
        self.a_1 = a_1

    def get_y(self, x):
        return self.a_0 + self.a_1*x

    def make_x(self, start, stop, step):
        return np.array(np.arange(start, stop, step))

    def make_true_y(self, start, stop, step):

        x_arr = self.make_x(start, stop, step)
        y_arr = self.get_y(x_arr)

        return y_arr