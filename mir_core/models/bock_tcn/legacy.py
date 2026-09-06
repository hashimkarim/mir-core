"""Inference port of the eight packaged madmom 2019 TCN networks.

This preserves the original beat/tempo heads and the ensemble's ordered mean.
It is separate from the project's trainable hybrid BockTCN architecture.
Only checksum-verified packaged resources are deserialized.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from mir_core.checkpoints.bocktcn import (
    BASELINE_CHECKPOINT_SHA256,
    baseline_checkpoint_names,
    baseline_checkpoint_path,
)


def _activation(name, values):
    if name is None:
        return values
    if name == "elu":
        return functional.elu(values)
    if name == "sigmoid":
        return torch.sigmoid(values)
    if name == "softmax":
        return torch.softmax(values, dim=-1)
    raise ValueError(f"unsupported historical activation: {name}")


def _parameter(values):
    array = np.asarray(values)
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("historical weights must be finite float32")
    return nn.Parameter(torch.from_numpy(array.copy()), requires_grad=False)


class _Convolution(nn.Module):
    def __init__(self, layer):
        super().__init__()
        if (
            type(layer).__name__ != "ConvolutionalLayer"
            or layer.pad != "valid"
            or layer.stride not in (None, 1, (1, 1))
        ):
            raise ValueError("unsupported historical convolution")
        self.weight = _parameter(layer.weights)
        self.bias = _parameter(layer.bias)
        self.activation = getattr(layer.activation_fn, "__name__", None)

    def forward(self, values):
        # SciPy accumulates each kernel in double and returns float32. madmom
        # then sums input channels in order, rounding after each channel.
        time_size, frequency_size = self.weight.shape[-2:]
        time = values.shape[-2] - time_size + 1
        frequency = values.shape[-1] - frequency_size + 1
        weights = self.weight.to(torch.float64)
        product = None
        for t in range(time_size):
            for f in range(frequency_size):
                data = values[:, :, t : t + time, f : f + frequency].to(torch.float64)
                term = (
                    data.unsqueeze(2)
                    * weights[:, :, time_size - 1 - t, frequency_size - 1 - f][
                        None, :, :, None, None
                    ]
                )
                product = term if product is None else product + term
        result = product[:, 0].to(torch.float32)
        for channel in range(1, self.weight.shape[0]):
            result = result + product[:, channel].to(torch.float32)
        return _activation(self.activation, result + self.bias[None, :, None, None])


class _Dense(nn.Module):
    def __init__(self, layer):
        super().__init__()
        if type(layer).__name__ != "FeedForwardLayer":
            raise ValueError("unsupported historical dense layer")
        self.weight = _parameter(layer.weights)
        self.bias = _parameter(layer.bias)
        self.activation = getattr(layer.activation_fn, "__name__", None)

    def forward(self, values):
        return _activation(
            self.activation, torch.matmul(values, self.weight) + self.bias
        )


class _Block(nn.Module):
    def __init__(self, layer):
        super().__init__()
        if type(layer).__name__ != "TCNBlock" or layer.activation_fn is not None:
            raise ValueError("unsupported historical TCN block")
        self.convolution = _Convolution(layer.dilated_conv)
        self.skip = _Dense(layer.skip_conv)
        self.residual = _Dense(layer.residual_conv)
        self.dilation = int(layer.dilation_rate)
        if tuple(self.convolution.weight.shape) != (16, 16, 1, 5) or self.dilation < 1:
            raise ValueError("unsupported historical dilated convolution")

    def forward(self, values):
        time = values.shape[-2]
        padded = functional.pad(values, (0, 0, 2 * self.dilation, 2 * self.dilation))
        windows = torch.cat(
            [
                padded[:, :, offset * self.dilation : offset * self.dilation + time, :]
                for offset in range(5)
            ],
            dim=-1,
        )
        skip = self.skip(self.convolution(windows).permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        residual = self.residual(values.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return residual + skip, skip


class _Member(nn.Module):
    def __init__(self, network):
        super().__init__()
        layers = network.layers
        expected = [
            "ConvolutionalLayer",
            "MaxPoolLayer",
            "ConvolutionalLayer",
            "MaxPoolLayer",
            "ConvolutionalLayer",
            "ReshapeLayer",
            "TCNLayer",
            "MultiTaskLayer",
        ]
        if [type(layer).__name__ for layer in layers] != expected:
            raise ValueError("unknown historical Bock architecture")
        for pool in (layers[1], layers[3]):
            if pool.size != (1, 3) or pool.stride != (1, 3) or pool.axis is not None:
                raise ValueError("unknown historical pool layout")
        if layers[5].newshape != (-1, 1, 16) or layers[5].order != "C":
            raise ValueError("unknown historical feature layout")
        tcn, heads = layers[6:]
        if (
            not tcn.skip_connections
            or tcn.activation_fn.__name__ != "elu"
            or heads.mapping != {0: 0, 1: 1}
        ):
            raise ValueError("unknown historical TCN/head configuration")
        self.convolutions = nn.ModuleList(
            [_Convolution(layers[index]) for index in [0, 2, 4]]
        )
        self.blocks = nn.ModuleList([_Block(block) for block in tcn.tcn_blocks])
        if [block.dilation for block in self.blocks] != [2**i for i in range(11)]:
            raise ValueError("unknown historical dilation schedule")
        self.beats = _Dense(heads.layers[0])
        tempo_layers = heads.layers[1].layers
        average = tempo_layers[0]
        if (
            type(average).__name__ != "AverageLayer"
            or average.axis != 0
            or average.keepdims
            or average.dtype is not None
        ):
            raise ValueError("unknown historical tempo averaging")
        self.tempo = _Dense(tempo_layers[1])

    def forward(self, features):
        values = functional.pad(features, (0, 0, 2, 2), mode="replicate")
        for index, convolution in enumerate(self.convolutions):
            values = convolution(values)
            if index < 2:
                values = functional.max_pool2d(values, (1, 3), stride=(1, 3))
        skips = None
        for block in self.blocks:
            values, skip = block(values)
            skips = skip if skips is None else skips + skip
        beats = self.beats(functional.elu(values).squeeze(-1).transpose(1, 2))
        tempo = self.tempo(skips.squeeze(-1).mean(dim=-1))
        return beats, tempo


class LegacyBockTCN(nn.Module):
    """One or all eight original Bock TCN members, on [batch,1,time,81]."""

    def __init__(self, selectors: Sequence[str] | None = None):
        super().__init__()
        selected = tuple(
            baseline_checkpoint_names() if selectors is None else selectors
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("select at least one distinct historical member")
        members, sources = [], []
        for selector in selected:
            path = baseline_checkpoint_path(selector)
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != BASELINE_CHECKPOINT_SHA256[selector]:
                raise ValueError(
                    f"historical Bock checkpoint checksum mismatch: {selector}"
                )
            network = pickle.loads(data, encoding="latin1")
            members.append(_Member(network))
            sources.append({"selector": selector, "sha256": digest})
        self.members = nn.ModuleList(members)
        self.port_config = {
            "architecture": "madmom-tcn-2019",
            "members": sources,
            "reduction": "ordered_float32_sum_then_divide",
            "input_edge_padding": 2,
            "convolution": "scipy_double_kernel_float32_channel_sum",
        }
        self.checkpoint_sha256 = hashlib.sha256(
            json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._original_state_digest = self.state_digest()
        self._original_config = json.dumps(self.port_config, sort_keys=True)
        self._original_checkpoint_digest = self.checkpoint_sha256

    def state_digest(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def verify_original_weights(self):
        if self.state_digest() != self._original_state_digest:
            raise ValueError("historical Bock weights were modified after loading")
        if (
            json.dumps(self.port_config, sort_keys=True) != self._original_config
            or self.checkpoint_sha256 != self._original_checkpoint_digest
        ):
            raise ValueError(
                "historical Bock source metadata was modified after loading"
            )

    def forward(self, features):
        beats, tempo = self.members[0](features)
        for member in self.members[1:]:
            next_beats, next_tempo = member(features)
            beats, tempo = beats + next_beats, tempo + next_tempo
        return beats / len(self.members), tempo / len(self.members)
