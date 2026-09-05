"""Both RNGs share immutable draw/state digests and regression detection."""

import json
import numpy as np
import pytest


@pytest.fixture
def contract():
    from mir_core.testing import particle_filter_contract

    return particle_filter_contract


@pytest.mark.parametrize("mode", [0, 1], ids=["legacy", "portable"])
def test_changing_either_real_rng_breaks_the_frozen_gate(contract, monkeypatch, mode):
    from mir_core.postprocessing import particle_filter as pf

    kind = pf._LegacyNumpyGlobalRandom if mode == 0 else pf.PortableParticleFilterRNG
    original = kind.uniform_choices

    def changed(self, upper, count):
        result = original(self, upper, count).copy()
        result[0] = (result[0] + 1) % upper
        return result

    monkeypatch.setattr(kind, "uniform_choices", changed)
    _, seed, frames, config = contract.PROFILES[0]
    _, metadata = contract.make_fixture(contract.MODES[mode], seed, frames, config)
    name = f"small-{'legacy' if mode == 0 else 'portable'}.bin"
    with pytest.raises(AssertionError, match="Frozen RNG contract changed"):
        contract.assert_case_frozen(
            name, metadata, json.loads(contract.DEFAULT_MANIFEST.read_text())["cases"]
        )


@pytest.mark.parametrize("mode", [0, 1], ids=["legacy", "portable"])
def test_shared_state_update_regression_breaks_both_modes(contract, monkeypatch, mode):
    original = contract.ParticleFilterTracker.process

    def changed(self, activations):
        result = original(self, activations)
        self.particles[0] = (self.particles[0] + 1) % self.st.num_states
        return result

    monkeypatch.setattr(contract.ParticleFilterTracker, "process", changed)
    _, seed, frames, config = contract.PROFILES[0]
    _, metadata = contract.make_fixture(contract.MODES[mode], seed, frames, config)
    name = f"small-{'legacy' if mode == 0 else 'portable'}.bin"
    with pytest.raises(AssertionError, match="Frozen RNG contract changed"):
        contract.assert_case_frozen(
            name, metadata, json.loads(contract.DEFAULT_MANIFEST.read_text())["cases"]
        )


def test_recording_preserves_the_ambient_numpy_stream(contract):
    state = np.random.get_state()
    _, seed, frames, config = contract.PROFILES[0]
    contract.make_fixture(contract.MODES[0], seed, frames, config)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])


def test_frozen_dual_rng_manifest(contract):
    result = contract.check()
    assert len(result["cases"]) == 8
    assert not result["native_reference_checked"]
    assert not result["native_production_checked"]


def test_native_manifest_cannot_drift_from_the_shared_contract(contract, tmp_path):
    manifest = json.loads(contract.DEFAULT_MANIFEST.read_text())
    manifest["cases"]["small-portable.bin"]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(
        AssertionError, match="Native and shared RNG contract manifests differ"
    ):
        contract.check(tmp_path)
