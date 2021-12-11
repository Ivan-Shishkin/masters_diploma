import numpy as np


class LinearFunction:

    def __init__(self, *, a_0=0, a_1=1):
        self.a_0 = a_0
        self.a_1 = a_1

    def get_y(self, x):
        return self.a_0 + self.a_1*x

    def make_x(self, start=0, stop=10, step=0.5):
        return np.array(np.arange(start, stop, step))

    def make_true_y(self, start=0, stop=10, step=0.5):

        x_arr = self.make_x(start, stop, step)
        y_arr = self.get_y(x_arr)

        return y_arr

    def uniform_noise(self, start=0, stop=10, step=0.5, low=-1, high=1):
        x_arr = self.make_x(start, stop, step)
        noise = np.random.uniform(low, high, x_arr.shape)
        y_arr_noise = self.get_y(x_arr) + noise
        intervals = np.ones(x_arr.shape) * (low+high)/2

        return y_arr_noise, intervals

    def uni_normal_noise(self, start=0, stop=10, step=0.5, loc=0, scale=1):

        x_arr = self.make_x(start, stop, step)
        noise = np.random.normal(loc, scale, x_arr.shape)
        y_arr_noise = self.get_y(x_arr) + noise
        intervals = np.ones(x_arr.shape)*scale*3.1

        return y_arr_noise, intervals

    def normal_noise(self, start=0, stop=10, step=0.5, strategy='random', loc_list=np.empty(0),scales_list=np.empty(0)):

        x_arr = self.make_x(start, stop, step)

        if strategy == 'random':

            scales_list = np.random.uniform(0, stop/10, x_arr.shape)
            loc_list = np.zeros(x_arr.shape)

        elif strategy == 'fixed':

            if scales_list.shape != x_arr.shape:
                raise AttributeError("scales_list don't match the shape of x_arr")
            if loc_list.shape != x_arr.shape:
                raise AttributeError("loc_list don't match the shape of x_arr")

        noise = np.empty(0)
        intervals = np.empty(0)

        for loc,scale in zip(loc_list, scales_list):

            noise = np.append(noise, np.random.normal(loc, scale))
            intervals = np.append(intervals, 3.1*scale)


        return x_arr, intervals