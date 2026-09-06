"""Authenticity and output-shape tests for SpecTNT."""

from __future__ import annotations

import torch

from mir_core.beats.schema import (
    EVENT_ACTIVATION_DEFINITION,
    FRAME_CLASS_DEFINITION,
    EventActivations,
    EventChannel,
    FrameClassActivations,
)
from mir_core.models import SpecTNT
from mir_core.models.spectnt import ResFrontEnd


def test_spectnt_defaults_match_mwm_reproduction_beat_config() -> None:
    model = SpecTNT()

    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 5_664_042
    assert isinstance(model.fe_model, ResFrontEnd)
    assert model.n_times == 78
    assert model.n_frequencies == 16
    assert len(model.main_model.spectnt_blocks) == 5
    block = model.main_model.spectnt_blocks[0]
    assert block.spectral_linear_in.out_features == 64
    assert block.spectral_encoder_layer.self_attn.num_heads == 4
    assert block.temporal_linear_in.out_features == 256
    assert block.temporal_encoder_layer.self_attn.num_heads == 8
    assert model.linear_out.out_features == 3


def test_spectnt_preserves_upstream_checkpoint_key_names() -> None:
    model = SpecTNT()
    keys = set(model.state_dict())

    assert "fe_model.layer1.conv_1.weight" in keys
    assert "main_model.fct" in keys
    assert "main_model.spectnt_blocks.0.spectral_linear_in.weight" in keys
    assert "main_model.spectnt_blocks.0.temporal_linear_in.weight" in keys
    assert "linear_out.weight" in keys


def test_spectnt_public_output_matches_beatnet_class_order() -> None:
    model = SpecTNT(n_times=2, n_frequencies=2, n_channels=1, embed_dim=4, n_blocks=0)
    logits = torch.tensor([[[10.0, 0.0, -10.0], [-10.0, 0.0, 10.0]]])

    output = model._format_outputs(logits)

    assert output["class_order"] == ("beat", "downbeat", "non_beat")
    assert model.output_definition is FRAME_CLASS_DEFINITION
    assert model.event_activation_definition is EVENT_ACTIVATION_DEFINITION
    assert output["data_definition"] is FRAME_CLASS_DEFINITION
    assert isinstance(output["activation_data"], FrameClassActivations)
    assert isinstance(output["event_activation_data"], EventActivations)
    assert torch.equal(output["one_hot"][0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(output["one_hot"][0, 1], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(
        output["beats"].squeeze(-1),
        output["event_activations"][:, :, int(EventChannel.beat)],
    )
    assert torch.allclose(
        output["downbeats"].squeeze(-1),
        output["event_activations"][:, :, int(EventChannel.downbeat)],
    )


def test_spectnt_forward_returns_framewise_three_class_output() -> None:
    torch.manual_seed(0)
    model = SpecTNT(
        n_channels=8,
        n_frequencies=2,
        n_times=4,
        spectral_dmodel=4,
        spectral_nheads=1,
        spectral_dimff=4,
        temporal_dmodel=4,
        temporal_nheads=1,
        temporal_dimff=4,
        embed_dim=4,
        n_blocks=1,
        fe_model=ResFrontEnd(
            in_channels=6,
            out_channels=8,
            freq_pooling=(2, 2, 2),
            time_pooling=(2, 2, 1),
        ),
    )
    model.eval()

    with torch.no_grad():
        output = model(torch.zeros(2, 6, 16, 16))

    assert output["logits"].shape == (2, 4, 3)
    assert output["activations"].shape == (2, 4, 3)
    assert output["frame_class_activations"].shape == (2, 4, 3)
    assert output["frame_classes"].shape == (2, 4)
    assert output["event_activations"].shape == (2, 4, 2)
    assert output["one_hot"].shape == (2, 4, 3)
    assert output["beats"].shape == (2, 4, 1)
    assert output["downbeats"].shape == (2, 4, 1)
    assert torch.allclose(output["activations"].sum(dim=-1), torch.ones(2, 4))
    assert torch.equal(output["one_hot"].sum(dim=-1), torch.ones(2, 4))
