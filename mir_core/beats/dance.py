"""Hierarchical beat/downbeat/DanceBeat activation contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Generic, TypeVar

from .schema import EVENT_ACTIVATION_DEFINITION, EventActivations

ArrayT = TypeVar("ArrayT")


class DanceEventChannel(IntEnum):
    """Nested event channels; each DanceBeat is also a downbeat and beat."""

    all_beats = 0
    beat = all_beats
    downbeat = 1
    dancebeat = 2


class TrackingTarget(str, Enum):
    """Accent stream exposed to conventional beat/downbeat postprocessors."""

    beat = "beat"
    dance = "dance"


DANCE_BEAT_CHANNEL = int(DanceEventChannel.all_beats)
DANCE_DOWNBEAT_CHANNEL = int(DanceEventChannel.downbeat)
DANCEBEAT_CHANNEL = int(DanceEventChannel.dancebeat)
DANCE_EVENT_CHANNEL_NAMES = ("beat", "downbeat", "dancebeat")
NUM_DANCE_EVENT_CHANNELS = len(DANCE_EVENT_CHANNEL_NAMES)


@dataclass(frozen=True)
class DanceEventActivations(Generic[ArrayT]):
    """Tagged overlapping ``[beat, downbeat, dancebeat]`` probabilities."""

    values: ArrayT

    def __post_init__(self) -> None:
        shape = getattr(self.values, "shape", None)
        if shape is None:
            raise TypeError("Dance event activation values must expose a shape.")
        if len(shape) == 0 or int(shape[-1]) != NUM_DANCE_EVENT_CHANNELS:
            raise ValueError(
                "Dance event activations must have exactly three channels in "
                f"[beat, downbeat, dancebeat] order; got shape {tuple(shape)}."
            )

    def channel(self, channel: DanceEventChannel) -> Any:
        return self.values[..., int(channel)]

    @property
    def all_beats(self) -> Any:
        return self.channel(DanceEventChannel.all_beats)

    @property
    def beats(self) -> Any:
        return self.all_beats

    @property
    def downbeats(self) -> Any:
        return self.channel(DanceEventChannel.downbeat)

    @property
    def dancebeats(self) -> Any:
        return self.channel(DanceEventChannel.dancebeat)

    def for_tracking(
        self,
        target: TrackingTarget | str,
    ) -> EventActivations[ArrayT]:
        """Project to canonical ``[beat, accent]`` decoder input semantics."""
        resolved = TrackingTarget(target)
        accent_index = (
            DANCE_DOWNBEAT_CHANNEL
            if resolved is TrackingTarget.beat
            else DANCEBEAT_CHANNEL
        )
        # Both NumPy arrays and torch tensors support a list on the final axis.
        values = self.values[..., [DANCE_BEAT_CHANNEL, accent_index]]
        return EventActivations(values, definition=EVENT_ACTIVATION_DEFINITION)
