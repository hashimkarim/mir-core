import numpy as np
import pytest

from mir_core.beats.schema import (
    ActivationFormatMismatchError,
    BeatActivationFormat,
    BeatDataDefinition,
    ExclusiveBeatDownbeatActivations,
    ExclusiveBeatDownbeatChannel,
)
from mir_core.postprocessing import ParticleFilterTracker
from mir_core.postprocessing.particle_filter import (
    _beat_densities,
    _down_densities,
)


def test_sparse_particle_likelihoods_match_full_state_densities() -> None:
    tracker = ParticleFilterTracker(
        particle_size=100,
        down_particle_size=20,
        num_tempi=30,
    )
    observation = 0.73
    down_observations = np.asarray([0.42, 0.31])

    beat_full = _beat_densities(
        observation,
        tracker.om,
        tracker.st,
        tracker.background_weight,
    )
    down_full = _down_densities(
        down_observations,
        tracker.om2,
        tracker.st2,
        tracker.background_weight,
    )

    np.testing.assert_array_equal(
        tracker._beat_particle_weights(observation),
        beat_full[tracker.particles],
    )
    np.testing.assert_array_equal(
        tracker._down_particle_weights(down_observations),
        down_full[tracker.down_particles],
    )


def test_particle_filter_accepts_two_channel_activations_at_fractional_fps() -> None:
    fps = 44100 / 1024
    tracker = ParticleFilterTracker(
        fps=fps,
        particle_size=100,
        down_particle_size=20,
        num_tempi=30,
    )

    decoded = tracker.process(
        ExclusiveBeatDownbeatActivations(np.zeros((4, 2), dtype=np.float32))
    )

    assert tracker.fps == pytest.approx(fps)
    assert tracker.T == pytest.approx(1 / fps)
    assert decoded.shape == (0, 2)


def test_particle_filter_rejects_untagged_two_channel_arrays() -> None:
    tracker = ParticleFilterTracker(
        particle_size=100,
        down_particle_size=20,
        num_tempi=30,
    )

    with pytest.raises(ActivationFormatMismatchError, match="untagged"):
        tracker.process(np.zeros((4, 2), dtype=np.float32))


def test_particle_filter_rejects_nonexclusive_probability_rows() -> None:
    tracker = ParticleFilterTracker(
        particle_size=100,
        down_particle_size=20,
        num_tempi=30,
    )

    with pytest.raises(ValueError, match="sum to at most 1"):
        tracker.process(
            ExclusiveBeatDownbeatActivations(np.asarray([[0.9, 0.2]], dtype=np.float32))
        )


def test_particle_filter_normalizes_declared_exclusive_channel_order() -> None:
    tracker = ParticleFilterTracker(
        particle_size=100,
        down_particle_size=20,
        num_tempi=30,
    )
    downbeat_first = BeatDataDefinition(
        representation=BeatActivationFormat.exclusive_beat_downbeat,
        order=(
            ExclusiveBeatDownbeatChannel.downbeat,
            ExclusiveBeatDownbeatChannel.beat_only,
        ),
        names=("downbeat", "beat_only"),
    )

    tracker.process(
        ExclusiveBeatDownbeatActivations(
            np.asarray([[0.2, 0.7]], dtype=np.float32),
            definition=downbeat_first,
        )
    )

    assert np.allclose(tracker.both_activations, [[0.7, 0.2]])


def test_particle_filter_enforces_last_emitted_event_refractory() -> None:
    random_state = np.random.get_state()
    try:
        np.random.seed(7)
        fps = 50
        values = np.zeros((500, 2), dtype=np.float32)
        values[:, 0] = 0.9
        values[:, 1] = 0.05
        tracker = ParticleFilterTracker(
            fps=fps,
            min_bpm=80,
            max_bpm=240,
            particle_size=300,
            down_particle_size=60,
            num_tempi=60,
            offset=0,
        )

        decoded = tracker.process(ExclusiveBeatDownbeatActivations(values))
    finally:
        np.random.set_state(random_state)

    minimum_separation = 0.4 * tracker.T * np.min(tracker.st.state_intervals)
    assert len(decoded) > 1
    assert np.all(np.diff(decoded[:, 0]) > minimum_separation)
    assert len(np.unique(decoded[:, 0])) == len(decoded)
    assert set(np.unique(decoded[:, 1])).issubset({1.0, 2.0})


def test_particle_filter_keeps_bounded_populations_after_injection() -> None:
    random_state = np.random.get_state()
    try:
        np.random.seed(13)
        tracker = ParticleFilterTracker(
            fps=50,
            min_bpm=80,
            max_bpm=180,
            particle_size=120,
            down_particle_size=24,
            num_tempi=30,
            beat_injection_threshold=0.5,
            downbeat_injection_threshold=0.5,
        )
        # Exercise both paths: the combined activation injects beat particles,
        # while the downbeat channel injects downbeat particles.
        values = np.tile(np.asarray([[0.35, 0.6]], dtype=np.float32), (300, 1))

        decoded = tracker.process(ExclusiveBeatDownbeatActivations(values))
    finally:
        np.random.set_state(random_state)

    assert len(decoded) > 0
    assert len(tracker.particles) == tracker.particle_size
    assert len(tracker.down_particles) == tracker.down_particle_size


def test_particle_filter_supports_a_single_locked_eight_count_meter() -> None:
    random_state = np.random.get_state()
    try:
        np.random.seed(0)
        tracker = ParticleFilterTracker(
            beats_per_bar=[8],
            min_beats_per_bar=8,
            max_beats_per_bar=8,
            min_bpm=80,
            max_bpm=200,
            particle_size=100,
            down_particle_size=20,
            num_tempi=20,
        )
        values = np.tile(
            np.asarray([[0.35, 0.6]], dtype=np.float32),
            (100, 1),
        )

        decoded = tracker.process(ExclusiveBeatDownbeatActivations(values))
    finally:
        np.random.set_state(random_state)

    assert decoded.ndim == 2
    assert decoded.shape[1] == 2
    assert len(tracker.particles) == tracker.particle_size
    assert len(tracker.down_particles) == tracker.down_particle_size


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("min_bpm", 220, "min_bpm"),
        ("lambda_d", 1.1, "lambda_d"),
        ("beat_activation_threshold", -0.1, "beat_activation_threshold"),
        ("observation_lambda_b", "broken", "observation_lambda_b"),
    ],
)
def test_particle_filter_rejects_invalid_tunable_parameters(
    parameter: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "max_bpm": 215,
        "particle_size": 100,
        "down_particle_size": 20,
        "num_tempi": 30,
        parameter: value,
    }

    with pytest.raises(ValueError, match=message):
        ParticleFilterTracker(**kwargs)
