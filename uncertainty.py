import argparse

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras.backend as K


class UncertaintyDropout(Dropout):
    def call(self, x, mask=None):
        if 0. < self.p < 1.:
            noise_shape = self._get_noise_shape(x)
            x = K.dropout(x, self.p, noise_shape)  # always on
        return x


parser = argparse.ArgumentParser()
parser.add_argument('--noise-mean', type=float, default=0.)
parser.add_argument('--noise-std', type=float, default=1.)
parser.add_argument('--nb-samples', type=int, default=10)
parser.add_argument('--x-range', nargs=2, type=float, default=(-5., 5.))
parser.add_argument('--x-valid-range', nargs=2, type=float, default=(-30., 30.))
parser.add_argument('--nb-mc-samples', type=int, default=100)
parser.add_argument('--ab', nargs=2, type=float, default=None)
args = parser.parse_args()


# Create dataset. We randomly sample a function f(x) = ax + b and then add additive Gaussian
# noise.
if args.ab is None:
    a, b = np.random.uniform(-1., 1., size=2)
else:
    a, b = args.ab
print('Using f(x) = {:.3f}x + {:.3f} with noise from N({}, {}) to sample data points ...'.format(a, b, args.noise_mean, args.noise_std))
xs = np.random.normal(np.mean(args.x_range), np.std(args.x_range), size=(args.nb_samples, 1))
ys = a * xs + b + np.random.normal(args.noise_mean, args.noise_std, size=xs.shape)
print('done, sampled data points: xs = {} - ys = {}'.format(xs.shape, ys.shape))
print('')

# Create model.
model = Sequential()
model.add(Dense(50, input_shape=xs.shape[1:]))
model.add(UncertaintyDropout(p=.5))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
print(model.summary())

# Train model.
model.fit(xs, ys, nb_epoch=100)

# Predict data over grid.
xs_valid = np.arange(*args.x_valid_range, step=.1)[:, np.newaxis]
ys_valid = []
for _ in range(args.nb_mc_samples):
    ys_valid.append(model.predict(xs_valid))
ys_valid_mean = np.mean(ys_valid, axis=0)
ys_valid_std = np.std(ys_valid, axis=0)
assert ys_valid_mean.shape == xs_valid.shape
assert ys_valid_std.shape == xs_valid.shape

# Now plot the data.
plt.scatter(xs, ys)
plt.plot(args.x_valid_range, a * np.array(args.x_valid_range) + b)
plt.xlim(*args.x_valid_range)
plt.plot(xs_valid, ys_valid_mean, color='green')
plt.fill_between(xs_valid.flatten(), ys_valid_mean.flatten() + ys_valid_std.flatten(), ys_valid_mean.flatten() - ys_valid_std.flatten(), facecolor='green', alpha=0.5)
plt.show()
