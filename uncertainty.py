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
parser.add_argument('--dropout', type=float, default=.5)
parser.add_argument('--nb-bootstrap-models', type=int, default=10)
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


def train_and_test_dropout_model(xs, ys, args):
    # Create model.
    model = Sequential()
    model.add(Dense(50, input_shape=xs.shape[1:]))
    model.add(UncertaintyDropout(p=args.dropout))
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

    return xs_valid, ys_valid_mean, ys_valid_std


def train_and_test_bootstrap(xs, ys, args):
    models = []
    for _ in range(args.nb_bootstrap_models):
        # Create model.
        model = Sequential()
        model.add(Dense(50, input_shape=xs.shape[1:]))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        bootstrapped_idxs = np.random.random_integers(xs.shape[0], size=xs.shape[0]) - 1
        print bootstrapped_idxs.shape, xs.shape
        bootstrapped_xs = xs[bootstrapped_idxs, ...]
        bootstrapped_ys = ys[bootstrapped_idxs, ...]
        assert bootstrapped_xs.shape == xs.shape
        assert bootstrapped_xs.shape == xs.shape

        model.fit(bootstrapped_xs, bootstrapped_ys, nb_epoch=100)
        models.append(model)

    # Predict data over grid.
    xs_valid = np.arange(*args.x_valid_range, step=.1)[:, np.newaxis]
    ys_valid = []
    for model in models:
        ys_valid.append(model.predict(xs_valid))
    ys_valid_mean = np.mean(ys_valid, axis=0)
    ys_valid_std = np.std(ys_valid, axis=0)
    assert ys_valid_mean.shape == xs_valid.shape
    assert ys_valid_std.shape == xs_valid.shape

    return xs_valid, ys_valid_mean, ys_valid_std


# xs_valid, ys_valid_mean, ys_valid_std = train_and_test_dropout(xs, ys, args)
xs_valid, ys_valid_mean, ys_valid_std = train_and_test_bootstrap(xs, ys, args)


# Now plot the data.
plt.scatter(xs, ys)
plt.plot(args.x_valid_range, a * np.array(args.x_valid_range) + b)
plt.xlim(*args.x_valid_range)
plt.plot(xs_valid, ys_valid_mean, color='green')
plt.fill_between(xs_valid.flatten(), ys_valid_mean.flatten() + ys_valid_std.flatten(), ys_valid_mean.flatten() - ys_valid_std.flatten(), facecolor='green', alpha=0.5)
plt.show()
