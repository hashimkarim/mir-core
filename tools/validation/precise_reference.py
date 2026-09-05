"""Independent NumPy implementation of the fixed-operation deployment contract.

This consumes the source model's checkpoint tensors, never an exported graph or
the production lowering/math helpers. NumPy elementwise operations implement
the specified order; BLAS and NumPy exp/tanh are deliberately not used.
"""
import math
import numpy as np

COEFFICIENTS = [1 / math.factorial(n) for n in range(15)]
POWERS = np.array([2.0**n for n in range(-128, 129)], dtype=np.float64)


def total(values):
    while values.shape[-1] > 1:
        width = values.shape[-1]
        result = values[..., :width - width % 2:2] + values[..., 1:width:2]
        if width % 2:
            result = np.concatenate((result, values[..., -1:] + np.zeros_like(values[..., -1:])), axis=-1)
        values = result
    return values[..., 0]


def linear(values, weight, bias):
    return total(values[..., None, :] * weight) + bias


def exp(values):
    limited = np.clip(values, -80, 80)
    exponent = np.rint(limited * np.float64(1.4426950408889634))
    remainder = (limited - exponent * np.float64(0.6931471803691238)) - exponent * np.float64(1.9082149292705877e-10)
    result = np.full_like(remainder, COEFFICIENTS[-1])
    for index in range(13, -1, -1):
        result = result * remainder + COEFFICIENTS[index]
    return POWERS[exponent.astype(np.int64) + 128] * result


def sigmoid(values):
    return np.float64(1) / (np.float64(1) + exp(-values))


def tanh(values):
    magnitude = np.abs(values)
    power = exp(-np.float64(2) * magnitude)
    result = (np.float64(1) - power) / (np.float64(1) + power)
    reduced = np.clip(-np.float64(2) * magnitude, -0.5, 0)
    polynomial = np.full_like(reduced, COEFFICIENTS[-1])
    for index in range(13, 0, -1):
        polynomial = polynomial * reduced + COEFFICIENTS[index]
    minus_one = reduced * polynomial
    small = -minus_one / (np.float64(2) + minus_one)
    result = np.where(magnitude < 0.25, small, result)
    return np.where(values < 0, -result, result)


class NumpyPreciseStep:
    def __init__(self, source):
        self.parameters = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in source.state_dict().items()}
        self.layers = source.lstm.num_layers
        self.hidden = np.zeros((self.layers, 1, source.lstm.hidden_size), dtype=np.float32)
        self.cell = np.zeros_like(self.hidden)
        kernel = source.conv1.kernel_size[0]
        self.indices = np.arange(source.input_dim-kernel+1)[:, None] + np.arange(kernel)[None, :]

    def __call__(self, frame):
        p = self.parameters
        patches = np.asarray(frame, dtype=np.float64)[self.indices]
        values = linear(patches, p['conv1.weight'][:, 0, :], p['conv1.bias']).T
        values = np.maximum(values, np.float64(0))
        width = values.shape[-1] // 2 * 2
        values = np.maximum(values[:, :width:2], values[:, 1:width:2]).reshape(1, -1)
        values = linear(values, p['linear0.weight'], p['linear0.bias'])
        next_hidden, next_cell = [], []
        for layer in range(self.layers):
            inputs = linear(values, p[f'lstm.weight_ih_l{layer}'], p[f'lstm.bias_ih_l{layer}'])
            recurrent = linear(self.hidden[layer].astype(np.float64), p[f'lstm.weight_hh_l{layer}'], p[f'lstm.bias_hh_l{layer}'])
            i, f, g, o = np.split(inputs + recurrent, 4, axis=-1)
            current = sigmoid(f) * self.cell[layer].astype(np.float64) + sigmoid(i) * tanh(g)
            values = sigmoid(o) * tanh(current)
            next_hidden.append(values)
            next_cell.append(current)
        logits = linear(values, p['linear.weight'], p['linear.bias'])
        probabilities = exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities /= total(probabilities)[..., None]
        activations = np.array([probabilities[0, 0] + probabilities[0, 1], probabilities[0, 1]], dtype=np.float32)
        self.hidden = np.asarray(next_hidden, dtype=np.float32)
        self.cell = np.asarray(next_cell, dtype=np.float32)
        return activations, self.hidden.copy(), self.cell.copy()
