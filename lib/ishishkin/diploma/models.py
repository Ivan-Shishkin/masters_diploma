import numpy as np


# y = a_1*x + a_0 + noise

class LinearFunctionOneDimensionalNoise:

    def __init__(self, *, a_0=0, a_1=1):
        self.a_0 = a_0
        self.a_1 = a_1

    def get_y(self, x):
        return self.a_0 + self.a_1 * x

    def make_true_x(self, start=0, stop=10, step=0.5):
        x_arr_true = np.array(np.arange(start, stop, step))
        return x_arr_true

    def make_true_y(self, start=0, stop=10, step=0.5):

        x_arr = self.make_true_x(start, stop, step)
        y_arr = self.get_y(x_arr)

        return y_arr

    def uni_normal_noise(self, start=0, stop=10, step=0.5, loc=0, scale=1):

        x_arr = self.make_true_x(start, stop, step)
        noise = np.random.normal(loc, scale, x_arr.shape)
        y_arr_noise = self.get_y(x_arr) + noise
        intervals = np.ones(x_arr.shape) * scale * 3.1

        return y_arr_noise, intervals

    def normal_noise(self, start=0, stop=10, step=0.5, strategy='random', loc_list=np.empty(0),
                     scales_list=np.empty(0)):

        x_arr = self.make_true_x(start, stop, step)

        if strategy == 'random':

            scales_list = np.random.uniform(0, stop / 10, x_arr.shape)
            loc_list = np.zeros(x_arr.shape)

        elif strategy == 'fixed':

            if scales_list.shape != x_arr.shape:
                raise AttributeError("scales_list don't match the shape of y_arr")
            if loc_list.shape != x_arr.shape:
                raise AttributeError("loc_list don't match the shape of y_arr")

        noise = np.empty(0)
        intervals = np.empty(0)

        for loc, scale in zip(loc_list, scales_list):
            noise = np.append(noise, np.random.normal(loc, scale))
            intervals = np.append(intervals, 3.1 * scale)

        y_arr_noise = self.get_y(x_arr) + noise

        return y_arr_noise, intervals


# y = a_1*(x + noise_x) + a_0 + noise_y

class LinearFunctionTwoDimensionalNoise:

    def __init__(self, *, a_0=0, a_1=1):
        self.a_0 = a_0
        self.a_1 = a_1

    def get_y(self, x):
        return self.a_0 + self.a_1 * x

    def make_true_x(self, start=0, stop=10, step=0.5):
        return np.array(np.arange(start, stop, step))

    def make_true_y(self, start=0, stop=10, step=0.5):

        x_arr = self.make_true_x(start, stop, step)
        y_arr = self.get_y(x_arr)

        return y_arr

    def uni_normal_noise(self, start=0, stop=10, step=0.5, loc_x=0, scale_x=1, loc_y=0, scale_y=1):

        x_arr_true = self.make_true_x(start, stop, step)
        noise_x = np.random.normal(loc_x, scale_x, x_arr_true.shape)
        x_arr_noise = x_arr_true + noise_x

        noise_y = np.random.normal(loc_y, scale_y, x_arr_true.shape)
        y_arr_noise = self.get_y(x_arr_noise) + noise_y

        intervals_x = np.ones(x_arr_noise.shape) * scale_x * 3.1
        intervals_y = np.ones(x_arr_noise.shape) * scale_y * 3.1

        return (x_arr_noise, intervals_x), (y_arr_noise, intervals_y)

    def normal_noise(self, start=0, stop=10, step=0.5,
                     strategy='random',
                     loc_x_list=np.empty(0), scales_x_list=np.empty(0),
                     loc_y_list=np.empty(0), scales_y_list=np.empty(0)):

        x_arr_true = self.make_true_x(start, stop, step)

        if strategy == 'random':

            scales_x_list = np.random.uniform(0, stop / 10, x_arr_true.shape)
            loc_x_list = np.zeros(x_arr_true.shape)

            scales_y_list = np.random.uniform(0, stop / 10, x_arr_true.shape)
            loc_y_list = np.zeros(x_arr_true.shape)

        elif strategy == 'fixed':

            if scales_x_list.shape != x_arr_true.shape:
                raise AttributeError("scales_x_list don't match the shape of x_arr")
            if loc_x_list.shape != x_arr_true.shape:
                raise AttributeError("loc_x_list don't match the shape of x_arr")
            if scales_y_list.shape != x_arr_true.shape:
                raise AttributeError("scales_y_list don't match the shape of y_arr")
            if loc_y_list.shape != x_arr_true.shape:
                raise AttributeError("loc_y_list don't match the shape of y_arr")

        noise_x = np.empty(0)
        intervals_x = np.empty(0)

        noise_y = np.empty(0)
        intervals_y = np.empty(0)

        for loc_x, scale_x, loc_y, scale_y in zip(loc_x_list, scales_x_list, loc_y_list, scales_y_list):
            noise_x = np.append(noise_x, np.random.normal(loc_x, scale_x))
            intervals_x = np.append(intervals_x, 3.1 * scale_x)

            noise_y = np.append(noise_y, np.random.normal(loc_y, scale_y))
            intervals_y = np.append(intervals_y, 3.1 * scale_y)

        x_arr_noise = x_arr_true + noise_x
        y_arr_noise = self.get_y(x_arr_noise) + noise_y

        return (x_arr_noise, intervals_x), (y_arr_noise, intervals_y)