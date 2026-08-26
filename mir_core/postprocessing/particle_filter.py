"""
Particle filter cascade for joint beat and downbeat tracking.

Ported from BeatNet (Heydari et al.):
https://github.com/mjhydri/BeatNet/blob/main/src/BeatNet/particle_filtering_cascade.py

Classes:
    ParticleFilterTracker — Clean interface around the particle filter cascade.
"""

# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
# Adapted for mir_core by removing plotting/pyaudio dependencies.

import math

import numpy as np
from madmom.features.beats_hmm import BarStateSpace, BarTransitionModel
from madmom.ml.hmm import TransitionModel, ObservationModel

from mir_core.beats.schema import (
    ExclusiveBeatDownbeatActivations,
    require_exclusive_beat_downbeat_activations,
    to_exclusive_beat_downbeat_activation_data,
)


class BDObservationModel(ObservationModel):
    """
    Observation model for beat and downbeat tracking with particle filtering.

    Parameters
    ----------
    state_space : :class:`BarStateSpace` instance
        BarStateSpace instance.
    observation_lambda : str
        Based on the first character of this parameter, each (down-)beat period gets split into (down-)beat states
        "B" stands for border model which classifies 1/(observation lambda) fraction of states as downbeat states and
        the rest as the beat states (if it is used for downbeat tracking state space) or the same fraction of states
        as beat states and the rest as the none beat states (if it is used for beat tracking state space).
        "N" model assigns a constant number of the beginning states as downbeat states and the rest as beat states
         or beginning states as beat and the rest as none-beat states
        "G" model is a smooth Gaussian transition (soft border) between downbeat/beat or beat/none-beat states
    """

    def __init__(self, state_space, observation_lambda):

        if observation_lambda[0] == 'B':
            observation_lambda = int(observation_lambda[1:])
            # compute observation pointers
            # always point to the non-beat densities
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            # unless they are in the beat range of the state space
            border = 1. / observation_lambda
            pointers[state_space.state_positions % 1 < border] = 1
            # the downbeat (i.e. the first beat range) points to density column 2
            pointers[state_space.state_positions < border] = 2
            # instantiate a ObservationModel with the pointers
            super(BDObservationModel, self).__init__(pointers)

        elif observation_lambda[0] == 'N':
            observation_lambda = int(observation_lambda[1:])
            # compute observation pointers
            # always point to the non-beat densities
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            # unless they are in the beat range of the state space
            for i in range(observation_lambda):
                border = np.asarray(state_space.first_states) + i
                pointers[border[1:]] = 1
                # the downbeat (i.e. the first beat range) points to density column 2
                pointers[border[0]] = 2
                # instantiate a ObservationModel with the pointers
            super(BDObservationModel, self).__init__(pointers)

        elif observation_lambda[0] == 'G':
            observation_lambda = float(observation_lambda[1:])
            pointers = np.zeros((state_space.num_beats + 1, state_space.num_states))
            for i in range(state_space.num_beats + 1):
                pointers[i] = _gaussian(state_space.state_positions, i, observation_lambda)
            pointers[0] = pointers[0] + pointers[-1]
            pointers[1] = np.sum(pointers[1:-1], axis=0)
            pointers = pointers[:2]
            super(BDObservationModel, self).__init__(pointers)


def _gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu) / sig, 2.) / 2)


#   assigning beat vs non-beat weights
def _beat_densities(
    observations,
    observation_model,
    state_model,
    background_weight,
):
    new_obs = np.zeros(state_model.num_states, float)
    if len(np.shape(observation_model.pointers)) != 2:  # B or N
        new_obs[np.argwhere(observation_model.pointers == 2)] = observations
        new_obs[observation_model.pointers == 0] = background_weight
    elif len(np.shape(observation_model.pointers)) == 2:  # G
        new_obs = observation_model.pointers[0] * observations
        new_obs[new_obs < 0.005] = background_weight
    return new_obs


#   assigning downbeat vs beat weights
def _down_densities(
    observations,
    observation_model,
    state_model,
    background_weight,
):
    new_obs = np.zeros(state_model.num_states, float)
    if len(np.shape(observation_model.pointers)) != 2:  # B or N
        new_obs[observation_model.pointers == 2] = observations[1]
        new_obs[observation_model.pointers == 0] = observations[0]
    elif len(np.shape(observation_model.pointers)) == 2:  # G
        new_obs = (
            observation_model.pointers[0] * observations[1]
            + observation_model.pointers[1] * observations[0]
        )
        new_obs[new_obs < 0.005] = background_weight
    return new_obs


def _universal_resample(particles, weights):
    J = len(particles)
    if J == 0:
        raise ValueError("Cannot resample an empty particle population.")
    total_weight = float(np.sum(weights))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        weights = np.full(J, 1.0 / J, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float) / total_weight
    cumsum_weights = np.cumsum(weights)
    cumsum_weights[-1] = 1.0
    r = np.random.uniform(0, 1 / J, J)
    U = r + np.arange(J) * (1 / J)
    new_particles = particles[np.searchsorted(cumsum_weights, U)]
    return new_particles


class ParticleFilterTracker:
    """
    Particle filter cascade for joint beat and downbeat tracking.

    Implements the two-stage particle filter from BeatNet:
    1. Beat-level particle filter tracks beat positions
    2. Downbeat-level particle filter tracks bar positions

    Args:
        fps: Frames per second of input activations (default 50 for BeatNet)
        min_bpm: Minimum tempo in BPM
        max_bpm: Maximum tempo in BPM
        beats_per_bar: List of possible beats per bar (e.g. [2,3,4]).
                       If empty, uses min/max_beats_per_bar range.
        min_beats_per_bar: Minimum beats per bar (used if beats_per_bar is empty)
        max_beats_per_bar: Maximum beats per bar (used if beats_per_bar is empty)
        particle_size: Number of beat particles
        down_particle_size: Number of downbeat particles
        num_tempi: Number of tempo states
        lambda_b: Beat transition lambda
        lambda_d: Downbeat transition lambda
        observation_lambda_b: Beat observation lambda string (e.g. "B56")
        observation_lambda_d: Downbeat observation lambda string (e.g. "B56")
        offset: Time offset (seconds) before inference starts
        ig_threshold: Information gate threshold
        background_weight: Observation floor below the information gate
        state_tolerance_seconds: Beat-state boundary tolerance
        min_separation_fraction: Minimum event separation as a tempo-period fraction
        downbeat_injection_threshold: Downbeat activation needed to inject particles
        downbeat_activation_threshold: Downbeat activation needed to emit a downbeat
        beat_activation_threshold: Combined activation needed to emit a beat
        resampling_threshold: Combined activation needed to resample beat particles
        beat_injection_threshold: Combined activation needed to inject beat particles
        beat_callback: Optional callback on each detected beat; receives bool (is_downbeat)
    """

    # Default constants
    PARTICLE_SIZE = 1500
    DOWN_PARTICLE_SIZE = 250
    MIN_BPM = 55.
    MAX_BPM = 215.
    NUM_TEMPI = 300
    LAMBDA_B = 60
    LAMBDA_D = 0.1
    OBSERVATION_LAMBDA_B = "B56"
    OBSERVATION_LAMBDA_D = "B56"
    MIN_BEAT_PER_BAR = 2
    MAX_BEAT_PER_BAR = 4
    OFFSET = 0
    IG_THRESHOLD = 0.4
    BACKGROUND_WEIGHT = 0.03
    STATE_TOLERANCE_SECONDS = 0.07
    MIN_SEPARATION_FRACTION = 0.4
    DOWNBEAT_INJECTION_THRESHOLD = 0.7
    DOWNBEAT_ACTIVATION_THRESHOLD = 0.4
    BEAT_ACTIVATION_THRESHOLD = 0.4
    RESAMPLING_THRESHOLD = 0.1
    BEAT_INJECTION_THRESHOLD = 0.8

    def __init__(
        self,
        fps: float = 50.0,
        min_bpm: float = MIN_BPM,
        max_bpm: float = MAX_BPM,
        beats_per_bar=None,
        min_beats_per_bar: int = MIN_BEAT_PER_BAR,
        max_beats_per_bar: int = MAX_BEAT_PER_BAR,
        particle_size: int = PARTICLE_SIZE,
        down_particle_size: int = DOWN_PARTICLE_SIZE,
        num_tempi: int = NUM_TEMPI,
        lambda_b: float = LAMBDA_B,
        lambda_d: float = LAMBDA_D,
        observation_lambda_b: str = OBSERVATION_LAMBDA_B,
        observation_lambda_d: str = OBSERVATION_LAMBDA_D,
        offset: float = OFFSET,
        ig_threshold: float = IG_THRESHOLD,
        background_weight: float = BACKGROUND_WEIGHT,
        state_tolerance_seconds: float = STATE_TOLERANCE_SECONDS,
        min_separation_fraction: float = MIN_SEPARATION_FRACTION,
        downbeat_injection_threshold: float = DOWNBEAT_INJECTION_THRESHOLD,
        downbeat_activation_threshold: float = DOWNBEAT_ACTIVATION_THRESHOLD,
        beat_activation_threshold: float = BEAT_ACTIVATION_THRESHOLD,
        resampling_threshold: float = RESAMPLING_THRESHOLD,
        beat_injection_threshold: float = BEAT_INJECTION_THRESHOLD,
        beat_callback=None,
    ):
        if beats_per_bar is None:
            beats_per_bar = []

        fps = float(fps)
        min_bpm = float(min_bpm)
        max_bpm = float(max_bpm)
        particle_size = int(particle_size)
        down_particle_size = int(down_particle_size)
        num_tempi = int(num_tempi)
        lambda_b = float(lambda_b)
        lambda_d = float(lambda_d)
        min_beats_per_bar = int(min_beats_per_bar)
        max_beats_per_bar = int(max_beats_per_bar)
        beats_per_bar = [int(value) for value in beats_per_bar]
        offset = float(offset)
        observation_lambda_b = str(observation_lambda_b)
        observation_lambda_d = str(observation_lambda_d)
        ig_threshold = float(ig_threshold)

        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("Particle-filter fps must be positive and finite.")
        if (
            not math.isfinite(min_bpm)
            or not math.isfinite(max_bpm)
            or min_bpm <= 0.0
            or min_bpm >= max_bpm
        ):
            raise ValueError(
                "Particle-filter min_bpm and max_bpm must be positive and "
                "min_bpm must be lower than max_bpm."
            )
        if particle_size < 1 or down_particle_size < 1 or num_tempi < 1:
            raise ValueError(
                "Particle-filter particle sizes and num_tempi must be at least 1."
            )
        if not math.isfinite(lambda_b) or lambda_b <= 0.0:
            raise ValueError("Particle-filter lambda_b must be positive and finite.")
        if not math.isfinite(lambda_d) or not 0.0 <= lambda_d <= 1.0:
            raise ValueError("Particle-filter lambda_d must be between 0 and 1.")
        if min_beats_per_bar < 1 or min_beats_per_bar > max_beats_per_bar:
            raise ValueError(
                "Particle-filter meter bounds must be positive and ordered."
            )
        if beats_per_bar and (
            any(value < 1 for value in beats_per_bar)
            or len(set(beats_per_bar)) != len(beats_per_bar)
        ):
            raise ValueError(
                "Particle-filter beats_per_bar must contain unique positive integers."
            )
        if not math.isfinite(offset) or offset < 0.0:
            raise ValueError("Particle-filter offset must be finite and non-negative.")
        for name, value in (
            ("ig_threshold", ig_threshold),
            ("background_weight", background_weight),
            ("downbeat_injection_threshold", downbeat_injection_threshold),
            ("downbeat_activation_threshold", downbeat_activation_threshold),
            ("beat_activation_threshold", beat_activation_threshold),
            ("resampling_threshold", resampling_threshold),
            ("beat_injection_threshold", beat_injection_threshold),
        ):
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Particle-filter {name} must be finite and between 0 and 1."
                )
        state_tolerance_seconds = float(state_tolerance_seconds)
        min_separation_fraction = float(min_separation_fraction)
        if (
            not math.isfinite(state_tolerance_seconds)
            or state_tolerance_seconds < 0.0
        ):
            raise ValueError(
                "Particle-filter state_tolerance_seconds must be finite and "
                "non-negative."
            )
        if (
            not math.isfinite(min_separation_fraction)
            or min_separation_fraction < 0.0
        ):
            raise ValueError(
                "Particle-filter min_separation_fraction must be finite and "
                "non-negative."
            )
        for name, value in (
            ("observation_lambda_b", observation_lambda_b),
            ("observation_lambda_d", observation_lambda_d),
        ):
            token = str(value)
            if len(token) < 2 or token[0] not in {"B", "N", "G"}:
                raise ValueError(
                    f"Particle-filter {name} must use B, N, or G followed by "
                    "a positive value."
                )
            try:
                suffix = float(token[1:])
            except ValueError as exc:
                raise ValueError(
                    f"Particle-filter {name} must end in a positive number."
                ) from exc
            if not math.isfinite(suffix) or suffix <= 0.0:
                raise ValueError(
                    f"Particle-filter {name} must end in a positive number."
                )
            if token[0] in {"B", "N"} and not suffix.is_integer():
                raise ValueError(
                    f"Particle-filter {name} requires an integer B/N value."
                )

        self.particle_size = particle_size
        self.down_particle_size = down_particle_size
        self.beats_per_bar = beats_per_bar
        self.fps = fps
        self.Lambda_b = lambda_b
        self.Lambda_d = lambda_d
        self.observation_lambda_b = observation_lambda_b
        self.observation_lambda_d = observation_lambda_d
        self.min_beats_per_bar = min_beats_per_bar
        self.max_beats_per_bar = max_beats_per_bar
        self.offset = offset
        self.ig_threshold = ig_threshold
        self.background_weight = float(background_weight)
        self.state_tolerance_seconds = state_tolerance_seconds
        self.min_separation_fraction = min_separation_fraction
        self.downbeat_injection_threshold = float(
            downbeat_injection_threshold
        )
        self.downbeat_activation_threshold = float(
            downbeat_activation_threshold
        )
        self.beat_activation_threshold = float(beat_activation_threshold)
        self.resampling_threshold = float(resampling_threshold)
        self.beat_injection_threshold = float(beat_injection_threshold)
        self.beat_callback = beat_callback

        # Convert timing information to construct a beat state space
        min_interval = 60. * fps / max_bpm
        max_interval = 60. * fps / min_bpm
        self.st = BarStateSpace(1, min_interval, max_interval, num_tempi)  # beat tracking state space

        if beats_per_bar:  # if the number of beats per bar is given
            self.st2 = BarStateSpace(
                1, min(self.beats_per_bar), max(self.beats_per_bar),
                max(self.beats_per_bar) - min(self.beats_per_bar) + 1
            )  # downbeat tracking state space
        else:  # if the number of beats per bar is not given
            self.st2 = BarStateSpace(
                1, self.min_beats_per_bar, self.max_beats_per_bar,
                self.max_beats_per_bar - self.min_beats_per_bar + 1
            )  # downbeat tracking state space

        tm = BarTransitionModel(self.st, self.Lambda_b)
        self.tm = list(TransitionModel.make_dense(tm.states, tm.pointers, tm.probabilities))  # beat transition model
        self.om = BDObservationModel(self.st, self.observation_lambda_b)  # beat observation model
        self.st.last_states = list(np.concatenate(self.st.last_states).flat)  # beat last states
        self.om2 = BDObservationModel(self.st2, self.observation_lambda_d)  # downbeat observation model
        downbeat_states = len(self.st2.first_states[0])
        self.tm2 = np.zeros((downbeat_states, downbeat_states))
        if downbeat_states == 1:
            self.tm2[0, 0] = 1.0
        else:
            for i in range(downbeat_states):
                for j in range(downbeat_states):
                    if i == j:
                        self.tm2[i, j] = 1 - self.Lambda_d
                    else:
                        self.tm2[i, j] = self.Lambda_d / (downbeat_states - 1)

        self.T = 1 / self.fps
        self.counter = -1
        self.path = np.zeros((1, 2), dtype=float)

        # Particles initialization
        self.particles = np.sort(np.random.choice(
            np.arange(0, self.st.num_states - 1), self.particle_size, replace=True
        ))
        self.down_particles = np.sort(np.random.choice(
            np.arange(0, self.st2.num_states - 1), self.down_particle_size, replace=True
        ))
        self.beat = np.squeeze(self.st.first_states)
        self._beat_last_mask = np.zeros(self.st.num_states, dtype=bool)
        self._beat_last_mask[np.asarray(self.st.last_states, dtype=np.int64)] = True
        down_last_states = np.asarray(self.st2.last_states[0], dtype=np.int64)
        self._down_last_mask = np.zeros(self.st2.num_states, dtype=bool)
        self._down_last_mask[down_last_states] = True
        self._down_last_row = {
            int(state): index for index, state in enumerate(down_last_states)
        }
        self._down_transition_targets = np.asarray(
            self.st2.first_states[0],
            dtype=self.down_particles.dtype,
        ).reshape(-1)
        transition_to = np.asarray(self.tm[0])
        transition_from = np.asarray(self.tm[1])
        transition_probability = np.asarray(self.tm[2])
        self._beat_transitions = {
            int(state): (
                transition_to[transition_from == state],
                transition_probability[transition_from == state],
            )
            for state in np.unique(np.asarray(self.st.last_states, dtype=np.int64))
        }

    def _beat_particle_weights(self, observation: float) -> np.ndarray:
        """Evaluate only occupied states instead of the full tempo state space."""

        pointers = np.asarray(self.om.pointers)
        if pointers.ndim != 2:
            occupied = pointers[self.particles]
            weights = np.zeros(len(self.particles), dtype=float)
            weights[occupied == 2] = observation
            weights[occupied == 0] = self.background_weight
            return weights
        weights = pointers[0, self.particles] * observation
        weights[weights < 0.005] = self.background_weight
        return weights

    def _down_particle_weights(self, observations: np.ndarray) -> np.ndarray:
        """Evaluate downbeat likelihoods only at occupied particle states."""

        pointers = np.asarray(self.om2.pointers)
        if pointers.ndim != 2:
            occupied = pointers[self.down_particles]
            weights = np.zeros(len(self.down_particles), dtype=float)
            weights[occupied == 2] = observations[1]
            weights[occupied == 0] = observations[0]
            return weights
        weights = (
            pointers[0, self.down_particles] * observations[1]
            + pointers[1, self.down_particles] * observations[0]
        )
        weights[weights < 0.005] = self.background_weight
        return weights

    def process(
        self,
        activations: ExclusiveBeatDownbeatActivations[np.ndarray],
    ) -> np.ndarray:
        """
        Run particle filtering over the given activation function to infer beats/downbeats.

        Args:
            activations: Tagged mutually-exclusive beat-only/downbeat
                probabilities. Bare two-channel arrays are rejected because
                they are ambiguous with canonical all-beat/downbeat data.

        Returns:
            numpy array, shape (num_beats, 2)
                Detected (down-)beat positions [seconds] and beat numbers.
        """
        supplied = require_exclusive_beat_downbeat_activations(activations)
        exclusive = to_exclusive_beat_downbeat_activation_data(
            supplied,
            dtype=np.float64,
        )
        values = exclusive.values
        if values.ndim != 2:
            raise ValueError(
                "Particle-filter activations must be a 2-dimensional array."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Particle-filter activations must be finite.")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(
                "Particle-filter activations must be probabilities in [0, 1]."
            )
        if np.any(np.sum(values, axis=1) > 1.0 + 1e-6):
            raise ValueError(
                "Exclusive beat-only and downbeat probabilities must sum to "
                "at most 1."
            )

        # Applying the offset and information gate thresholds
        values = values[int(self.offset / self.T):]
        both_activations = values.copy()
        combined_activations = np.max(values, axis=1)
        combined_activations[
            combined_activations < self.ig_threshold
        ] = self.background_weight
        self.activations = combined_activations
        self.both_activations = both_activations

        for i in range(len(combined_activations)):  # loop through the provided frame/s to infer beats/downbeats
            self.counter += 1
            gathering = int(np.median(self.particles))  # calculating beat particles clutter
            # checking if the clutter is within the beat interval
            if ((gathering - self.beat[self.st.state_intervals[self.beat] == self.st.state_intervals[gathering]]) < (
                    int(self.state_tolerance_seconds / self.T)) + 1).any() and (self.offset + self.counter * self.T) - self.path[-1][0] > self.min_separation_fraction * self.T * \
                    self.st.state_intervals[gathering]:

                # downbeat particles motion
                down_last = self._down_last_mask[self.down_particles]
                last1 = self.down_particles[down_last]
                state1 = self.down_particles[~down_last] + 1
                transitioned = np.empty(len(last1), dtype=self.down_particles.dtype)
                for j, state in enumerate(last1):
                    arg1 = self._down_last_row[int(state)]
                    transitioned[j] = np.random.choice(
                        self._down_transition_targets,
                        1,
                        p=self.tm2[arg1],
                    )[0]
                if len(transitioned):
                    state1 = np.concatenate((state1, transitioned))
                self.down_particles = state1

                # downbeat particles correction
                injected_downbeat_count = 0
                if both_activations[i][1] > self.downbeat_injection_threshold:
                    injected = np.concatenate(self.st2.first_states).reshape(-1)
                    injected_downbeat_count = len(injected)
                    self.down_particles = np.append(self.down_particles, injected)
                down_weights = self._down_particle_weights(
                    both_activations[i]
                )
                self.down_particles = _universal_resample(
                    self.down_particles,
                    down_weights,
                )
                if injected_downbeat_count:
                    # np.delete is not in-place.  Assign its result and remove
                    # the actual number injected so the population stays fixed.
                    remove = np.random.choice(
                        len(self.down_particles),
                        injected_downbeat_count,
                        replace=False,
                    )
                    self.down_particles = np.delete(self.down_particles, remove)
                m = np.bincount(self.down_particles)
                self.down_max = np.argmax(m)  # calculating downbeat particles clutter

                # beat vs downbeat distinguishment
                if self.down_max in self.st2.first_states[0] and self.path[-1][1] != 1 and both_activations[i][1] > self.downbeat_activation_threshold:
                    self.path = np.append(self.path, [[self.offset + self.counter * self.T, 1]], axis=0)
                    if self.beat_callback is not None:
                        self.beat_callback(True)
                elif combined_activations[i] > self.beat_activation_threshold:
                    self.path = np.append(self.path, [[self.offset + self.counter * self.T, 2]], axis=0)
                    if self.beat_callback is not None:
                        self.beat_callback(False)

            # beat particles motion
            beat_last = self._beat_last_mask[self.particles]
            last = self.particles[beat_last]
            state = self.particles[~beat_last] + 1
            transitioned = np.empty(len(last), dtype=self.particles.dtype)
            for j, last_state in enumerate(last):
                choices, probabilities = self._beat_transitions[int(last_state)]
                transitioned[j] = np.random.choice(
                    np.squeeze(choices),
                    1,
                    p=np.squeeze(probabilities),
                )[0]
            if len(transitioned):
                state = np.concatenate((state, transitioned))
            self.particles = state

            # beat particles correction
            beat_weights = self._beat_particle_weights(
                combined_activations[i]
            )
            if combined_activations[i] > self.resampling_threshold:
                injected_beat_count = 0
                if combined_activations[i] > self.beat_injection_threshold:
                    injected = np.asarray(
                        self.st.first_states[0][
                            np.arange(
                                np.random.randint(4),
                                len(self.st.first_states[0]),
                                6,
                            )
                        ]
                    ).reshape(-1)
                    injected_beat_count = len(injected)
                    self.particles = np.append(self.particles, injected)
                self.particles = _universal_resample(
                    self.particles,
                    beat_weights,
                )  # beat correction
                if injected_beat_count:
                    # BeatNet discarded np.delete's return value and used the
                    # first-state container length rather than the injected count.
                    remove = np.random.choice(
                        len(self.particles),
                        injected_beat_count,
                        replace=False,
                    )
                    self.particles = np.delete(self.particles, remove)

        return self.path[1:]
