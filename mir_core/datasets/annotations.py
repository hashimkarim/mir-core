"""
Read and write .beats annotation files, and convert from legacy formats.

.beats format (tab-separated, right-optional columns):
    time                    — beats only
    time\\tbeat_position     — beats with metrical position (1 = downbeat)
    time\\tbeat_position\\tbar — full musical annotation with bar number
    time\\tbeat_position\\tbar\\tdance_position\\tdancebar
                            — optional aligned dance-count structure

All times are in seconds.

Converters:
    from_candombe_csv     — Candombe CSV with bar.beat encoding
    from_beats_tsv        — .beats tab-separated (BRID, Candombe w/o bar)
    from_salsa_dataset    — Salsa Dataset millisecond timestamps
    from_salsaset_csv     — SalsaSet comma-separated CSV
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass
class BeatAnnotation:
    """Parsed .beat annotation.

    Attributes:
        times: Beat onset times in seconds, shape (N,).
        positions: Beat position within bar (1-indexed), shape (N,) or None.
        bars: Bar number (1-indexed), shape (N,) or None.
        dance_positions: Beat position within the style-appropriate dance count
            (1-indexed), shape (N,) or None.
        dance_bars: Dance-count cycle number (1-indexed), shape (N,) or None.

    ``positions``/``bars`` always describe the musical meter.  The optional
    dance fields form a second aligned axis over the same event times.  Keeping
    the axes separate is important for styles such as Salsa, where musical
    downbeats occur on dance counts 1 and 5 but only count 1 is a dancebeat.
    """

    times: np.ndarray
    positions: Optional[np.ndarray] = None
    bars: Optional[np.ndarray] = None
    dance_positions: Optional[np.ndarray] = None
    dance_bars: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times)
        if times.ndim != 1:
            raise ValueError("Beat annotation times must be one-dimensional.")
        for name in ("positions", "bars", "dance_positions", "dance_bars"):
            values = getattr(self, name)
            if values is None:
                continue
            array = np.asarray(values)
            if array.ndim != 1 or len(array) != len(times):
                raise ValueError(
                    f"Beat annotation {name} must be a one-dimensional array "
                    "aligned with times."
                )
        if self.bars is not None and self.positions is None:
            raise ValueError("Musical bar numbers require musical beat positions.")
        if (self.dance_positions is None) != (self.dance_bars is None):
            raise ValueError(
                "Dance positions and dancebar numbers must be present together."
            )
        if self.dance_positions is not None and self.positions is None:
            raise ValueError(
                "Dancebeat positions require the aligned musical beat positions."
            )

    @property
    def beat_times(self) -> np.ndarray:
        return self.times

    @property
    def downbeat_times(self) -> Optional[np.ndarray]:
        if self.positions is None:
            return None
        return self.times[self.positions == 1]

    @property
    def downbeat_mask(self) -> Optional[np.ndarray]:
        """Return the musical count-one mask, when meter is annotated."""
        if self.positions is None:
            return None
        return self.positions == 1

    @property
    def dancebeat_times(self) -> Optional[np.ndarray]:
        """Return count-one events on the dance axis, when annotated."""
        if self.dance_positions is None:
            return None
        return self.times[self.dance_positions == 1]

    @property
    def dancebeat_mask(self) -> Optional[np.ndarray]:
        """Return the dance count-one mask, when dance structure is annotated."""
        if self.dance_positions is None:
            return None
        return self.dance_positions == 1

    @property
    def dancebeats_available(self) -> bool:
        return self.dance_positions is not None


def with_dance_structure(
    annotation: BeatAnnotation,
    *,
    beats_per_bar: int,
    dance_cycle_beats: int,
    first_bar_dance_position: int = 1,
) -> BeatAnnotation:
    """Append an aligned dance-count axis without changing event timestamps.

    ``first_bar_dance_position`` gives the dance position of musical position 1
    in the first annotated musical bar.  For Salsa it is either 1 or 5; for a
    dance whose basic cycle is one musical bar it is 1.  The dance cycle must
    contain an integer number of musical bars.
    """
    beats_per_bar = int(beats_per_bar)
    dance_cycle_beats = int(dance_cycle_beats)
    first_bar_dance_position = int(first_bar_dance_position)
    if beats_per_bar <= 0:
        raise ValueError("beats_per_bar must be positive.")
    if dance_cycle_beats <= 0 or dance_cycle_beats % beats_per_bar:
        raise ValueError(
            "dance_cycle_beats must be a positive multiple of beats_per_bar."
        )
    valid_starts = tuple(range(1, dance_cycle_beats + 1, beats_per_bar))
    if first_bar_dance_position not in valid_starts:
        raise ValueError(
            "first_bar_dance_position must begin a musical bar within the "
            f"dance cycle; expected one of {valid_starts}."
        )
    if annotation.positions is None or annotation.bars is None:
        raise ValueError(
            "Dance structure requires musical beat positions and bar numbers."
        )

    positions = np.asarray(annotation.positions, dtype=np.int32)
    bars = np.asarray(annotation.bars, dtype=np.int32)
    if np.any((positions < 1) | (positions > beats_per_bar)):
        invalid = sorted(
            int(value)
            for value in np.unique(positions)
            if int(value) < 1 or int(value) > beats_per_bar
        )
        raise ValueError(
            f"Musical beat positions must be within 1..{beats_per_bar}; "
            f"found {invalid}."
        )
    if len(bars) and np.any(np.diff(bars) < 0):
        raise ValueError("Musical bar numbers must be non-decreasing.")

    if len(positions) == 0:
        dance_positions = np.asarray([], dtype=np.int32)
        dance_bars = np.asarray([], dtype=np.int32)
    else:
        first_bar = int(bars[0])
        absolute_offsets = (
            first_bar_dance_position
            - 1
            + (bars.astype(np.int64) - first_bar) * beats_per_bar
            + positions.astype(np.int64)
            - 1
        )
        dance_positions = (absolute_offsets % dance_cycle_beats + 1).astype(np.int32)
        raw_dance_bars = absolute_offsets // dance_cycle_beats
        dance_bars = (raw_dance_bars - raw_dance_bars.min() + 1).astype(np.int32)

    return BeatAnnotation(
        times=np.asarray(annotation.times, dtype=np.float64),
        positions=positions,
        bars=bars,
        dance_positions=dance_positions,
        dance_bars=dance_bars,
    )


# ------------------------------------------------------------------
# Reader / Writer
# ------------------------------------------------------------------


def read_beat(path: Union[str, Path]) -> BeatAnnotation:
    """Read a .beats annotation file."""
    path = Path(path)
    times, positions, bars, dance_positions, dance_bars = [], [], [], [], []
    has_positions = None
    has_bars = None
    has_dance_positions = None
    has_dance_bars = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            times.append(float(parts[0]))

            if len(parts) >= 2:
                if has_positions is None:
                    has_positions = True
                positions.append(int(parts[1]))
            else:
                if has_positions is None:
                    has_positions = False

            if len(parts) >= 3:
                if has_bars is None:
                    has_bars = True
                bars.append(int(parts[2]))
            else:
                if has_bars is None:
                    has_bars = False

            if len(parts) >= 4:
                if has_dance_positions is None:
                    has_dance_positions = True
                dance_positions.append(int(parts[3]))
            else:
                if has_dance_positions is None:
                    has_dance_positions = False

            if len(parts) >= 5:
                if has_dance_bars is None:
                    has_dance_bars = True
                dance_bars.append(int(parts[4]))
            else:
                if has_dance_bars is None:
                    has_dance_bars = False

    return BeatAnnotation(
        times=np.array(times, dtype=np.float64),
        positions=np.array(positions, dtype=np.int32) if has_positions else None,
        bars=np.array(bars, dtype=np.int32) if has_bars else None,
        dance_positions=(
            np.array(dance_positions, dtype=np.int32) if has_dance_positions else None
        ),
        dance_bars=(np.array(dance_bars, dtype=np.int32) if has_dance_bars else None),
    )


def write_beat(path: Union[str, Path], ann: BeatAnnotation) -> None:
    """Write a BeatAnnotation to a .beats file."""
    path = Path(path)
    with open(path, "w") as f:
        for i, t in enumerate(ann.times):
            parts = [f"{t:.9f}".rstrip("0").rstrip(".")]
            if ann.positions is not None:
                parts.append(str(ann.positions[i]))
            if ann.bars is not None:
                parts.append(str(ann.bars[i]))
            if ann.dance_positions is not None:
                parts.append(str(ann.dance_positions[i]))
            if ann.dance_bars is not None:
                parts.append(str(ann.dance_bars[i]))
            f.write("\t".join(parts) + "\n")


# ------------------------------------------------------------------
# Converters
# ------------------------------------------------------------------


def from_candombe_csv(path: Union[str, Path]) -> BeatAnnotation:
    """Convert Candombe CSV (time,bar.beat) to BeatAnnotation."""
    times, positions, bars = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            times.append(float(parts[0]))
            bar_beat = parts[1]  # e.g. "1.1" -> bar=1, beat=1
            bar, beat = bar_beat.split(".")
            bars.append(int(bar))
            positions.append(int(beat))

    return BeatAnnotation(
        times=np.array(times, dtype=np.float64),
        positions=np.array(positions, dtype=np.int32),
        bars=np.array(bars, dtype=np.int32),
    )


def from_beats_tsv(path: Union[str, Path]) -> BeatAnnotation:
    """Convert .beats TSV (time\\tposition) to BeatAnnotation."""
    return read_beat(path)  # same format


def from_salsa_dataset(path: Union[str, Path]) -> BeatAnnotation:
    """Convert Salsa Dataset TXT (millisecond timestamps) to BeatAnnotation."""
    times = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            times.append(int(line) / 1000.0)

    return BeatAnnotation(times=np.array(times, dtype=np.float64))


def from_salsaset_csv(path: Union[str, Path]) -> BeatAnnotation:
    """Convert SalsaSet CSV (time,position) to BeatAnnotation."""
    times, positions = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            times.append(float(parts[0]))
            positions.append(int(parts[1]))

    return BeatAnnotation(
        times=np.array(times, dtype=np.float64),
        positions=np.array(positions, dtype=np.int32),
    )
