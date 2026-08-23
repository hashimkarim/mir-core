"""Causal BeatNet variants with nested beat/downbeat/DanceBeat heads."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mir_core.beats.dance import (
    DANCE_BEAT_CHANNEL,
    DANCE_DOWNBEAT_CHANNEL,
    DANCEBEAT_CHANNEL,
    DanceEventActivations,
    TrackingTarget,
)
from mir_core.beats.schema import EVENT_ACTIVATION_DEFINITION


class _DanceBeatNetBase(nn.Module):
    """Shared causal BeatNet frontend with an independently named dance head."""

    event_activation_definition = EVENT_ACTIVATION_DEFINITION

    def __init__(
        self,
        input_dim: int = 272,
        hidden_dim: int = 150,
        num_layers: int = 2,
        dropout: float = 0.0,
        tracking_target: TrackingTarget | str = TrackingTarget.dance,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.conv_out = 150
        self.kernel_size = 10
        self.tracking_target = TrackingTarget(tracking_target)

        self.conv1 = nn.Conv1d(1, 2, kernel_size=self.kernel_size, padding=0)
        conv_out_dim = 2 * int((self.input_dim - self.kernel_size + 1) / 2)
        self.linear0 = nn.Linear(conv_out_dim, self.conv_out)
        self.lstm = nn.LSTM(
            input_size=self.conv_out,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        # A distinct key prevents an old mutually-exclusive BeatNet classifier
        # from being mistaken for this nested multi-label head.
        self.dance_head = nn.Linear(self.hidden_dim, 3)

    def _frontend(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, frequency_bins = x.shape
        if frequency_bins != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} BeatNet features, got {frequency_bins}."
            )
        flat = x.reshape(batch_size * time_steps, 1, frequency_bins)
        convolved = F.max_pool1d(F.relu(self.conv1(flat)), 2)
        projected = self.linear0(convolved.reshape(batch_size * time_steps, -1))
        return projected.reshape(batch_size, time_steps, self.conv_out)

    def _outputs(self, logits: torch.Tensor) -> dict[str, Any]:
        probabilities = torch.sigmoid(logits)
        dance_data = DanceEventActivations(probabilities)
        tracking_data = dance_data.for_tracking(self.tracking_target)
        return {
            "logits": logits,
            "beats": probabilities[..., DANCE_BEAT_CHANNEL].unsqueeze(-1),
            "downbeats": probabilities[..., DANCE_DOWNBEAT_CHANNEL].unsqueeze(-1),
            "dancebeats": probabilities[..., DANCEBEAT_CHANNEL].unsqueeze(-1),
            "dance_event_activations": probabilities,
            "dance_activation_data": dance_data,
            "event_activations": tracking_data.values,
            "event_activation_data": tracking_data,
            "activation_data": tracking_data,
            "data_definition": EVENT_ACTIVATION_DEFINITION,
            "tracking_target": self.tracking_target.value,
        }


class DanceBeatNetBatch(_DanceBeatNetBase):
    """Batch-optimized three-head BeatNet used for training."""

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        latent, _ = self.lstm(self._frontend(x))
        return self._outputs(self.dance_head(latent))


class DanceBeatNetCRNN(_DanceBeatNetBase):
    """Stateful online three-head BeatNet used for causal evaluation."""

    def __init__(
        self,
        input_dim: int = 272,
        hidden_dim: int = 150,
        num_layers: int = 2,
        device: str = "cpu",
        tracking_target: TrackingTarget | str = TrackingTarget.dance,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            tracking_target=tracking_target,
        )
        self.device_str = str(device)
        self.register_buffer("hidden", torch.zeros(num_layers, 1, hidden_dim))
        self.register_buffer("cell", torch.zeros(num_layers, 1, hidden_dim))

    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = next(self.parameters()).device
        self.hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device,
        )
        self.cell = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        batch_size = x.shape[0]
        if self.hidden.shape[1] != batch_size or self.hidden.device != x.device:
            self.reset_hidden(batch_size, x.device)
        latent, (self.hidden, self.cell) = self.lstm(
            self._frontend(x),
            (self.hidden.detach(), self.cell.detach()),
        )
        return self._outputs(self.dance_head(latent))
