from __future__ import annotations

import torch

from mir_core.beats.dance import DanceEventActivations
from mir_core.beats.schema import EventActivations
from mir_core.models.beatnet.crnn import BeatNetBatch
from mir_core.models.beatnet.dance import DanceBeatNetBatch, DanceBeatNetCRNN


def test_dance_beatnet_exposes_three_heads_and_dance_tracking_projection() -> None:
    model = DanceBeatNetBatch(input_dim=32, hidden_dim=8, num_layers=1)
    output = model(torch.randn(2, 7, 32))

    assert output["logits"].shape == (2, 7, 3)
    assert output["beats"].shape == (2, 7, 1)
    assert output["downbeats"].shape == (2, 7, 1)
    assert output["dancebeats"].shape == (2, 7, 1)
    assert isinstance(output["dance_activation_data"], DanceEventActivations)
    assert isinstance(output["event_activation_data"], EventActivations)
    assert (
        output["event_activation_data"].values.data_ptr()
        != output["dance_event_activations"].data_ptr()
    )
    torch.testing.assert_close(
        output["event_activation_data"].downbeats,
        output["dancebeats"].squeeze(-1),
    )


def test_beat_tracking_projection_uses_musical_downbeat_head() -> None:
    model = DanceBeatNetBatch(
        input_dim=32,
        hidden_dim=8,
        num_layers=1,
        tracking_target="beat",
    )
    output = model(torch.randn(1, 5, 32))

    torch.testing.assert_close(
        output["event_activation_data"].downbeats,
        output["downbeats"].squeeze(-1),
    )


def test_old_beatnet_checkpoint_only_initializes_shared_layers() -> None:
    old = BeatNetBatch(input_dim=32, hidden_dim=8, num_layers=1)
    dance = DanceBeatNetBatch(input_dim=32, hidden_dim=8, num_layers=1)

    incompatible = dance.load_state_dict(old.state_dict(), strict=False)

    assert set(incompatible.missing_keys) == {"dance_head.weight", "dance_head.bias"}
    assert set(incompatible.unexpected_keys) == {"linear.weight", "linear.bias"}
    torch.testing.assert_close(dance.conv1.weight, old.conv1.weight)
    torch.testing.assert_close(dance.linear0.weight, old.linear0.weight)
    torch.testing.assert_close(dance.lstm.weight_ih_l0, old.lstm.weight_ih_l0)


def test_batch_checkpoint_maps_to_stateful_online_model() -> None:
    batch = DanceBeatNetBatch(input_dim=32, hidden_dim=8, num_layers=1)
    online = DanceBeatNetCRNN(input_dim=32, hidden_dim=8, num_layers=1)

    incompatible = online.load_state_dict(batch.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {"hidden", "cell"}
    assert not incompatible.unexpected_keys
    output = online(torch.randn(1, 4, 32))
    assert output["event_activation_data"].values.shape == (1, 4, 2)
