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
    PARTICLE_FILTER_RNG_LEGACY_GLOBAL_V1,
    PARTICLE_FILTER_RNG_PORTABLE_V1,
    PortableParticleFilterRNG,
    _beat_densities,
    _down_densities,
    normalize_particle_filter_rng_contract,
)


def _numpy_random_states_equal(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_portable_rng_v1_has_frozen_splitmix64_vectors() -> None:
    rng = PortableParticleFilterRNG(42)

    assert [rng.next_u64() for _ in range(6)] == [
        0xBDD732262FEB6E95,
        0x28EFE333B266F103,
        0x47526757130F9F52,
        0x581CE1FF0E4AE394,
        0x9BC585A244823F2,
        0xDE4431FA3C80DB06,
    ]
    rng.reset()
    assert [rng.randbelow(bound) for bound in (3, 10, 1000, 2**32 + 1)] == [
        1,
        1,
        858,
        3056468374,
    ]
    rng.reset()
    np.testing.assert_array_equal(
        rng.sample_without_replacement(12, 5),
        [1, 6, 10, 3, 0],
    )


@pytest.mark.parametrize("alias", ["portable", "splitmix64-v1"])
def test_portable_rng_contract_aliases_are_explicit(alias: str) -> None:
    assert (
        normalize_particle_filter_rng_contract(alias) == PARTICLE_FILTER_RNG_PORTABLE_V1
    )


def test_legacy_global_rng_remains_the_default_contract() -> None:
    tracker = ParticleFilterTracker(
        particle_size=20,
        down_particle_size=5,
        num_tempi=30,
    )

    assert tracker.rng_contract == PARTICLE_FILTER_RNG_LEGACY_GLOBAL_V1
    assert tracker.random_seed is None


def test_legacy_global_rng_preserves_frozen_historical_replay() -> None:
    values = np.zeros((360, 2), dtype=np.float32)
    values[:, 0] = 0.9
    values[:, 1] = 0.05
    random_state = np.random.get_state()
    try:
        np.random.seed(12345)
        tracker = ParticleFilterTracker(
            fps=50,
            min_bpm=80,
            max_bpm=180,
            particle_size=96,
            down_particle_size=24,
            num_tempi=64,
        )
        decoded = tracker.process(ExclusiveBeatDownbeatActivations(values))
    finally:
        np.random.set_state(random_state)

    np.testing.assert_array_equal(
        np.rint(decoded[:, 0] * tracker.fps).astype(np.int64),
        [
            18,
            27,
            39,
            48,
            60,
            69,
            81,
            90,
            102,
            111,
            123,
            132,
            144,
            153,
            165,
            174,
            186,
            195,
            207,
            216,
            228,
            237,
            254,
            269,
            278,
            288,
            296,
            309,
            317,
            329,
            337,
            349,
            357,
        ],
    )
    np.testing.assert_array_equal(decoded[:, 1], np.full(33, 2.0))
    assert int(tracker.particles.sum()) == 6124
    assert int(tracker.down_particles.sum()) == 134


def test_portable_rng_requires_an_owned_seed() -> None:
    with pytest.raises(ValueError, match="requires an explicit random_seed"):
        ParticleFilterTracker(
            particle_size=20,
            down_particle_size=5,
            num_tempi=30,
            rng_contract=PARTICLE_FILTER_RNG_PORTABLE_V1,
        )


def test_legacy_rng_rejects_a_misleading_owned_seed() -> None:
    with pytest.raises(ValueError, match="does not own a random_seed"):
        ParticleFilterTracker(
            particle_size=20,
            down_particle_size=5,
            num_tempi=30,
            rng_contract=PARTICLE_FILTER_RNG_LEGACY_GLOBAL_V1,
            random_seed=42,
        )


def test_portable_particle_filter_reset_replays_exactly_without_global_rng() -> None:
    values = np.zeros((360, 2), dtype=np.float32)
    values[:, 0] = 0.9
    values[:, 1] = 0.05
    global_before = np.random.get_state()
    tracker = ParticleFilterTracker(
        fps=50,
        min_bpm=80,
        max_bpm=180,
        particle_size=120,
        down_particle_size=24,
        num_tempi=30,
        rng_contract=PARTICLE_FILTER_RNG_PORTABLE_V1,
        random_seed=42,
    )
    initial_particles = tracker.particles.copy()
    initial_down_particles = tracker.down_particles.copy()

    first = tracker.process(ExclusiveBeatDownbeatActivations(values))
    first_particles = tracker.particles.copy()
    first_down_particles = tracker.down_particles.copy()
    tracker.reset()
    np.testing.assert_array_equal(tracker.particles, initial_particles)
    np.testing.assert_array_equal(tracker.down_particles, initial_down_particles)
    second = tracker.process(ExclusiveBeatDownbeatActivations(values))

    np.testing.assert_array_equal(second, first)
    np.testing.assert_array_equal(tracker.particles, first_particles)
    np.testing.assert_array_equal(tracker.down_particles, first_down_particles)
    assert _numpy_random_states_equal(global_before, np.random.get_state())


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
