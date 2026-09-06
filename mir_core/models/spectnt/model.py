"""
SpecTNT model for frame-wise beat and downbeat activation.

Based on Hung et al. "Modeling Beats and Downbeats with a
Time-Frequency Transformer" (ICASSP 2022), implemented against the unofficial
MWM-io reproduction at commit 5fa5fd1964a92c4eb02e7b1ab6a83ca0c50b934f:
https://github.com/MWM-io/SpecTNT-pytorch

Full-geometry numerical parity with that reproduction is verified. Fidelity
to an original-author implementation/checkpoint and the paper's reported
accuracy has not been established.

The upstream beat-tracking checkpoint order is [beat, downbeat, non-beat].
This module exposes the same order so decoder adapters can treat BeatNet and
SpecTNT frame outputs consistently.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F

from mir_core.beats.schema import (
    EVENT_ACTIVATION_DEFINITION,
    FRAME_CLASS_DEFINITION,
    FRAME_CLASS_NAMES,
    EventChannel,
    FrameClassActivations,
)
from mir_core.beats.tensor_converters import to_event_activation_data


class Res2DMaxPoolModule(nn.Module):
    """Residual 2D convolution block used by the SpecTNT front end."""

    def __init__(self, in_channels: int, out_channels: int, pooling=(2, 2)):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(tuple(pooling))

        self.diff = in_channels != out_channels
        if self.diff:
            self.conv_3 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn_3 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        return self.mp(self.relu(out))


class ResFrontEnd(nn.Module):
    """Three-block ResNet front end from the MWM-io reproduction's beat config."""

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 256,
        freq_pooling=(2, 2, 2),
        time_pooling=(2, 2, 1),
    ):
        super().__init__()
        self.input_bn = nn.BatchNorm2d(in_channels)
        self.layer1 = Res2DMaxPoolModule(
            in_channels,
            out_channels,
            pooling=(freq_pooling[0], time_pooling[0]),
        )
        self.layer2 = Res2DMaxPoolModule(
            out_channels,
            out_channels,
            pooling=(freq_pooling[1], time_pooling[1]),
        )
        self.layer3 = Res2DMaxPoolModule(
            out_channels,
            out_channels,
            pooling=(freq_pooling[2], time_pooling[2]),
        )

    def forward(self, hcqt: torch.Tensor) -> torch.Tensor:
        out = self.input_bn(hcqt)
        out = self.layer1(out)
        out = self.layer2(out)
        return self.layer3(out)


class SpecTNTBlock(nn.Module):
    """One spectral-temporal Transformer-in-Transformer block."""

    def __init__(
        self,
        n_channels: int,
        n_frequencies: int,
        n_times: int,
        spectral_dmodel: int,
        spectral_nheads: int,
        spectral_dimff: int,
        temporal_dmodel: int,
        temporal_nheads: int,
        temporal_dimff: int,
        embed_dim: int,
        dropout: float,
        use_tct: bool,
    ):
        super().__init__()

        self.D = embed_dim
        self.F = n_frequencies
        self.K = n_channels
        self.T = n_times + (1 if use_tct else 0)

        self.D_to_K = nn.Linear(self.D, self.K)
        self.K_to_D = nn.Linear(self.K, self.D)

        self.spectral_linear_in = nn.Linear(self.F + 1, spectral_dmodel)
        self.spectral_encoder_layer = nn.TransformerEncoderLayer(
            d_model=spectral_dmodel,
            nhead=spectral_nheads,
            dim_feedforward=spectral_dimff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.spectral_linear_out = nn.Linear(spectral_dmodel, self.F + 1)

        self.temporal_linear_in = nn.Linear(self.T, temporal_dmodel)
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dmodel,
            nhead=temporal_nheads,
            dim_feedforward=temporal_dimff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_linear_out = nn.Linear(temporal_dmodel, self.T)

    def forward(
        self,
        spec_in: torch.Tensor,
        temp_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spec_in = spec_in + F.pad(self.D_to_K(temp_in), (0, 0, 0, self.F))

        spec_in = spec_in.flatten(0, 1).transpose(1, 2)
        spec_emb = self.spectral_linear_in(spec_in)
        spec_out = self.spectral_encoder_layer(spec_emb)
        spec_out = self.spectral_linear_out(spec_out)
        spec_out = spec_out.view(-1, self.T, self.K, self.F + 1).transpose(2, 3)

        temp_in = temp_in + self.K_to_D(spec_out[:, :, :1, :])
        temp_in = temp_in.permute(0, 2, 3, 1).flatten(0, 1)
        temp_emb = self.temporal_linear_in(temp_in)
        temp_out = self.temporal_encoder_layer(temp_emb)
        temp_out = self.temporal_linear_out(temp_out)
        temp_out = temp_out.unsqueeze(1).permute(0, 3, 1, 2)

        return spec_out, temp_out


class SpecTNTModule(nn.Module):
    """Stack of SpecTNT blocks with learned FCT/FPE/TE parameters."""

    def __init__(
        self,
        n_channels: int,
        n_frequencies: int,
        n_times: int,
        spectral_dmodel: int,
        spectral_nheads: int,
        spectral_dimff: int,
        temporal_dmodel: int,
        temporal_nheads: int,
        temporal_dimff: int,
        embed_dim: int,
        n_blocks: int,
        dropout: float,
        use_tct: bool,
    ):
        super().__init__()

        self.fct = nn.Parameter(torch.zeros(1, n_times, 1, n_channels))
        self.fpe = nn.Parameter(torch.zeros(1, 1, n_frequencies + 1, n_channels))
        self.tct = nn.Parameter(torch.zeros(1, 1, 1, embed_dim)) if use_tct else None
        self.te = nn.Parameter(torch.rand(1, n_times, 1, embed_dim))

        self.spectnt_blocks = nn.ModuleList(
            [
                SpecTNTBlock(
                    n_channels,
                    n_frequencies,
                    n_times,
                    spectral_dmodel,
                    spectral_nheads,
                    spectral_dimff,
                    temporal_dmodel,
                    temporal_nheads,
                    temporal_dimff,
                    embed_dim,
                    dropout,
                    use_tct,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        fct = self.fct.expand(batch_size, -1, -1, -1)
        spec_emb = torch.cat([fct, x], dim=2) + self.fpe
        if self.tct is not None:
            spec_emb = F.pad(spec_emb, (0, 0, 0, 0, 1, 0))

        temp_emb = self.te.expand(batch_size, -1, -1, -1)
        if self.tct is not None:
            tct = self.tct.expand(batch_size, -1, -1, -1)
            temp_emb = torch.cat([tct, temp_emb], dim=1)

        for block in self.spectnt_blocks:
            spec_emb, temp_emb = block(spec_emb, temp_emb)

        return spec_emb, temp_emb


class SpecTNT(nn.Module):
    """
    SpecTNT beat/downbeat model with BeatNet-style public outputs.

Args match the beat-tracking config in
``configs/beats.yaml`` from the MWM-io reproduction. The model expects
harmonic representation features shaped ``(batch, 6, freq, time)``.
The reproduction's config/code instantiate 5,664,042 trainable parameters; this
differs from the ICASSP paper's reported 4,637,392, so this port prioritizes
reproduction-code parity. Original-author checkpoint fidelity is unverified.

Returns a dictionary with:
    - ``logits``: frame logits, ``[beat, downbeat, non-beat]``
    - ``activations``: softmax probabilities in the same order
    - ``one_hot``: argmax one-hot frame decisions in the same order
    - ``beats`` / ``downbeats``: single-channel frame activations
    """

    class_order = FRAME_CLASS_NAMES
    output_definition = FRAME_CLASS_DEFINITION
    data_definition = FRAME_CLASS_DEFINITION
    frame_class_definition = FRAME_CLASS_DEFINITION
    event_activation_definition = EVENT_ACTIVATION_DEFINITION

    def __init__(
        self,
        fe_model: nn.Module | None = None,
        n_channels: int = 256,
        n_frequencies: int = 16,
        n_times: int = 78,
        spectral_dmodel: int = 64,
        spectral_nheads: int = 4,
        spectral_dimff: int = 64,
        temporal_dmodel: int = 256,
        temporal_nheads: int = 8,
        temporal_dimff: int = 256,
        embed_dim: int = 128,
        n_blocks: int = 5,
        dropout: float = 0.15,
        use_tct: bool = False,
        n_classes: int = 3,
    ):
        super().__init__()
        if n_classes != 3:
            raise ValueError("SpecTNT beat tracking uses exactly 3 output classes.")

        self.use_tct = use_tct
        self.n_times = n_times
        self.n_frequencies = n_frequencies
        self.n_classes = n_classes

        self.fe_model = fe_model or ResFrontEnd()
        self.main_model = SpecTNTModule(
            n_channels,
            n_frequencies,
            n_times,
            spectral_dmodel,
            spectral_nheads,
            spectral_dimff,
            temporal_dmodel,
            temporal_nheads,
            temporal_dimff,
            embed_dim,
            n_blocks,
            dropout,
            use_tct,
        )
        self.linear_out = nn.Linear(embed_dim, n_classes)

    def forward(self, features: torch.Tensor) -> Dict[str, Any]:
        if features.ndim == 3:
            features = features.unsqueeze(1)

        fe_out = self.fe_model(features)
        fe_out = fe_out.permute(0, 3, 2, 1)
        if fe_out.shape[1] != self.n_times or fe_out.shape[2] != self.n_frequencies:
            raise ValueError(
                "SpecTNT front-end output shape does not match configured "
                f"(n_times={self.n_times}, n_frequencies={self.n_frequencies}); "
                f"got time={fe_out.shape[1]}, frequency={fe_out.shape[2]}."
            )

        _, temp_emb = self.main_model(fe_out)
        if self.use_tct:
            upstream_logits = self.linear_out(temp_emb[:, 0, 0, :]).unsqueeze(1)
        else:
            upstream_logits = self.linear_out(temp_emb[:, :, 0, :])

        return self._format_outputs(upstream_logits)

    def _format_outputs(self, logits: torch.Tensor) -> Dict[str, Any]:
        activations = F.softmax(logits, dim=-1)
        classes = activations.argmax(dim=-1)
        one_hot = F.one_hot(classes, num_classes=3).to(activations.dtype)
        frame_class_activation_data = FrameClassActivations(
            activations,
            definition=self.frame_class_definition,
        )
        event_activation_data = to_event_activation_data(
            frame_class_activation_data
        )
        event_activations = event_activation_data.values

        return {
            "logits": logits,
            "beats": event_activations[:, :, int(EventChannel.beat)].unsqueeze(-1),
            "downbeats": event_activations[:, :, int(EventChannel.downbeat)].unsqueeze(-1),
            "activations": activations,
            "frame_class_activations": activations,
            "frame_classes": classes,
            "event_activations": event_activations,
            "activation_data": frame_class_activation_data,
            "frame_class_activation_data": frame_class_activation_data,
            "event_activation_data": event_activation_data,
            "one_hot": one_hot,
            "class_order": self.class_order,
            "data_definition": self.output_definition,
        }
