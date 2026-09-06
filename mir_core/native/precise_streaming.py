"""Fixed-operation recurrent deployment math with the original float32 ABI.

Weights are unchanged. Float64 products reduce through an explicit binary tree.
Range-reduced polynomial nonlinearities avoid platform libm/vector kernels.
Only the returned activations and recurrent states round to float32. Training
and evaluation models are not mutated.
"""
from __future__ import annotations

from copy import deepcopy
import math
import torch
from torch import nn


def ordered_sum(values):
    """An explicit pairwise tree; no provider-selected reduction order."""
    while values.shape[-1] > 1:
        if values.shape[-1] % 2:
            values = torch.cat((values, torch.zeros_like(values[..., :1])), dim=-1)
        values = values[..., 0::2] + values[..., 1::2]
    return values.squeeze(-1)


def ordered_linear(values, weight, bias=None):
    result = ordered_sum(values.unsqueeze(-2) * weight)
    return result if bias is None else result + bias


class PortableMath(nn.Module):
    """IEEE binary64 operations with a shared exp table/polynomial.

    Range reduction leaves |r| <= log(2)/2. A degree-14 Taylor polynomial has
    truncation error below 2e-19 there; floating arithmetic dominates that
    bound. Exp arguments saturate at +/-80, outside the meaningful float32
    probability range for this model's parity budget. Tanh uses expm1 near
    zero to avoid subtracting almost equal numbers.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("powers", torch.tensor([2.0**n for n in range(-128, 129)], dtype=torch.float64))
        self.coefficients = tuple(1.0 / math.factorial(n) for n in range(15))

    def exp(self, values):
        x = torch.clamp(values, -80.0, 80.0)
        exponent = torch.round(x * 1.4426950408889634)
        # The high part has trailing zero bits, making exponent*high exact.
        r = (x - exponent * 0.6931471803691238) - exponent * 1.9082149292705877e-10
        p = torch.ones_like(r) * self.coefficients[14]
        for coefficient in reversed(self.coefficients[:14]):
            p = r * p + coefficient
        return p * self.powers[(exponent + 128).to(torch.int64)]

    def sigmoid(self, values):
        return 1.0 / (1.0 + self.exp(-values))

    def tanh(self, values):
        magnitude = torch.abs(values)
        e = self.exp(-2.0 * magnitude)
        large = (1.0 - e) / (1.0 + e)
        x = torch.clamp(-2.0 * magnitude, -0.5, 0.0)
        p = torch.ones_like(x) * self.coefficients[14]
        for coefficient in reversed(self.coefficients[1:14]):
            p = x * p + coefficient
        expm1 = x * p
        small = -expm1 / (2.0 + expm1)
        result = torch.where(magnitude < 0.25, small, large)
        return torch.where(values < 0.0, -result, result)

    def softmax(self, logits):
        values = self.exp(logits - logits.max(dim=-1, keepdim=True).values)
        return values / ordered_sum(values).unsqueeze(-1)


class PreciseRecurrentStep(nn.Module):
    """One BeatNet-family step, lowered without GEMM or libm nonlinearities."""
    def __init__(self, model: nn.Module, output_layer: nn.Module):
        super().__init__()
        conv, lstm = model.conv1, model.lstm
        if (conv.in_channels != 1 or conv.groups != 1 or conv.stride != (1,)
                or conv.padding != (0,) or conv.dilation != (1,)
                or lstm.bidirectional or lstm.proj_size != 0):
            raise ValueError("Unsupported recurrent geometry for the precise step")
        self.input_dim = int(model.input_dim)
        self.layers = int(lstm.num_layers)
        self.channels = int(conv.out_channels)
        kernel = int(conv.kernel_size[0])
        self.pool_frames = (self.input_dim - kernel + 1) // 2
        self.conv = deepcopy(conv).double().eval()
        self.projection = deepcopy(model.linear0).double().eval()
        self.lstm = deepcopy(lstm).double().eval()
        self.output = deepcopy(output_layer).double().eval()
        self.math = PortableMath()
        indices = torch.arange(self.input_dim - kernel + 1)[:, None] + torch.arange(kernel)[None, :]
        self.register_buffer("patch_indices", indices)

    def forward(self, features, hidden, cell):
        features = features.to(torch.float64).reshape(1, self.input_dim)
        hidden, cell = hidden.to(torch.float64), cell.to(torch.float64)
        patches = features[:, self.patch_indices]
        values = ordered_linear(patches, self.conv.weight[:, 0, :], self.conv.bias)
        values = torch.relu(values).transpose(1, 2)[:, :, :self.pool_frames * 2]
        values = values.reshape(1, self.channels, self.pool_frames, 2).max(dim=-1).values
        values = ordered_linear(values.reshape(1, self.channels * self.pool_frames), self.projection.weight, self.projection.bias)
        next_hidden, next_cell = [], []
        for layer in range(self.layers):
            input_gates = ordered_linear(values, getattr(self.lstm, f"weight_ih_l{layer}"),
                                         getattr(self.lstm, f"bias_ih_l{layer}"))
            recurrent_gates = ordered_linear(hidden[layer], getattr(self.lstm, f"weight_hh_l{layer}"),
                                             getattr(self.lstm, f"bias_hh_l{layer}"))
            ingate, forgetgate, candidate, outgate = (input_gates + recurrent_gates).chunk(4, dim=-1)
            current_cell = self.math.sigmoid(forgetgate) * cell[layer] + self.math.sigmoid(ingate) * self.math.tanh(candidate)
            values = self.math.sigmoid(outgate) * self.math.tanh(current_cell)
            next_hidden.append(values)
            next_cell.append(current_cell)
        logits = ordered_linear(values, self.output.weight, self.output.bias).reshape(1, 1, -1)
        return logits, torch.stack(next_hidden).float(), torch.stack(next_cell).float()
