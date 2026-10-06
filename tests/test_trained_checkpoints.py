from __future__ import annotations

import json
from pathlib import Path

import pytest

from mir_core.checkpoints import (
    TRAINED_MODEL_BUNDLE_SCHEMA,
    beatnet_stock_postprocessor_selection_path,
    list_trained_model_bundles,
    load_trained_model_bundle,
    trained_checkpoint_path,
    trained_postprocessor_path,
)


EXPECTED_BUNDLES = {
    "beatnet/brid/finetune_latin_general",
    "beatnet/brid/scratch",
    "beatnet/candombe/finetune_latin_general",
    "beatnet/candombe/scratch",
    "beatnet/latin_general/scratch",
    "beatnet/salsa/finetune_latin_general",
    "beatnet/salsa/scratch",
    "classifier/latin_router/efficientat",
    "classifier/latin_router/yamnet",
}
STOCK_POSTPROCESSORS = {
    "stock-1d",
    "stock-dbn",
    "stock-pf",
}
REQUIRED_TUNED_POSTPROCESSORS = {
    "tuned-1d",
    "tuned-dbn",
    "tuned-pf",
}
BEAT_BUNDLES = (
    ("latin_general", "scratch"),
    ("brid", "scratch"),
    ("brid", "finetune_latin_general"),
    ("candombe", "scratch"),
    ("candombe", "finetune_latin_general"),
    ("salsa", "scratch"),
    ("salsa", "finetune_latin_general"),
)
TUNED_METHODS = {
    "tuned-1d": "heydari_1d_state_space",
    "tuned-dbn": "dbn_downbeat",
    "tuned-pf": "particle_filter",
}


def test_all_completed_training_bundles_are_packaged_and_hash_valid() -> None:
    bundles = list_trained_model_bundles(verify_files=True)

    assert {bundle.bundle_id for bundle in bundles} == EXPECTED_BUNDLES
    assert all(bundle.fold_count == 5 for bundle in bundles)
    assert all(
        bundle.split_contract["contract_hash"] == "e2e-537f350dbaf7e925"
        for bundle in bundles
    )
    assert all(
        json.loads(bundle.manifest_path.read_text(encoding="utf-8"))["schema"]
        == TRAINED_MODEL_BUNDLE_SCHEMA
        for bundle in bundles
    )
    for bundle in bundles:
        if bundle.task != "beat_tracking":
            continue
        assert set(bundle.stock_postprocessors) == STOCK_POSTPROCESSORS
        assert REQUIRED_TUNED_POSTPROCESSORS <= set(bundle.tuned_postprocessors)
        assert set(bundle.postprocessors) == (
            set(bundle.stock_postprocessors) | set(bundle.tuned_postprocessors)
        )
        assert bundle.default_postprocessor in bundle.tuned_postprocessors
        assert all(
            name.startswith(f"{record.metadata['kind']}-")
            for name, record in bundle.postprocessors.items()
        )


def test_checkpoint_and_postprocessor_can_be_selected_independently() -> None:
    bundle = load_trained_model_bundle("beatnet", "candombe", "scratch")

    assert bundle.lifecycle == "candidate"
    assert bundle.checkpoint_path(3).name == "seed_42_fold_3.pt"
    assert set(bundle.tuned_postprocessors) == set(TUNED_METHODS)
    assert set(bundle.stock_postprocessors) == STOCK_POSTPROCESSORS
    assert bundle.postprocessor_path("tuned-dbn", fold_index=3).name == "fold_3.json"
    assert bundle.tuned_postprocessors["tuned-dbn"].metadata["source_id"] == (
        "dbn-live-validation-rerank-per-fold"
    )
    assert trained_checkpoint_path(
        "beatnet", "candombe", "scratch", 0
    ).is_file()
    assert trained_postprocessor_path(
        "beatnet", "candombe", "scratch", "tuned-pf", fold_index=0
    ).is_file()
    assert trained_postprocessor_path(
        "beatnet", "candombe", "scratch", "stock-dbn"
    ).is_file()


@pytest.mark.parametrize(("target", "condition"), BEAT_BUNDLES)
def test_every_bundle_has_tuned_settings_per_fold_and_shared_stock_settings(
    target: str, condition: str
) -> None:
    bundle = load_trained_model_bundle("beatnet", target, condition)

    assert bundle.default_postprocessor == "tuned-dbn"
    assert set(bundle.tuned_postprocessors) == set(TUNED_METHODS)
    assert set(bundle.stock_postprocessors) == STOCK_POSTPROCESSORS
    for name in STOCK_POSTPROCESSORS:
        assert not bundle.postprocessor_is_per_fold(name)
        assert bundle.postprocessor_path(name, fold_index=3) == bundle.postprocessor_path(name)
    with pytest.raises(KeyError, match="no fold 9"):
        bundle.postprocessor_path("stock-dbn", fold_index=9)

    for name, method in TUNED_METHODS.items():
        record = bundle.tuned_postprocessors[name]
        assert bundle.postprocessor_is_per_fold(name)
        assert record.metadata["selection_scope"] == "per_fold"
        assert record.metadata["selection_split"] == "validation"
        assert record.metadata["test_used_for_selection"] is False
        with pytest.raises(ValueError, match="pass fold_index"):
            bundle.postprocessor_path(name)
        with pytest.raises(KeyError, match="no fold 5"):
            bundle.postprocessor_path(name, fold_index=5)
        selection = json.loads(
            (bundle.root / record.relative_path).read_text(encoding="utf-8")
        )
        assert selection["selection_policy"] == "one_parameter_set_per_fold"
        selected = {
            row["fold_index"]: row["parameters"] for row in selection["fold_parameters"]
        }
        for fold_index in range(bundle.fold_count):
            path = bundle.postprocessor_path(name, fold_index=fold_index)
            assert path.name == f"fold_{fold_index}.json"
            parameters = json.loads(path.read_text(encoding="utf-8"))
            assert parameters == selected[fold_index]
            assert parameters["method"] == method
            assert path == trained_postprocessor_path(
                "beatnet", target, condition, name, fold_index=fold_index
            )
    assert bundle.postprocessor_path(fold_index=0) == bundle.postprocessor_path(
        "tuned-dbn", fold_index=0
    )


def test_tuned_dbn_no_longer_uses_the_late_salsa_setting() -> None:
    bundle = load_trained_model_bundle("beatnet", "salsa", "finetune_latin_general")

    for fold_index in range(bundle.fold_count):
        parameters = json.loads(
            bundle.postprocessor_path(fold_index=fold_index).read_text(encoding="utf-8")
        )
        assert parameters["online"] is True and parameters["correct"] is False
        assert parameters["observation_lambda"] != 16


@pytest.mark.parametrize(("target", "condition"), BEAT_BUNDLES)
def test_tuned_1d_settings_announce_immediately(target: str, condition: str) -> None:
    bundle = load_trained_model_bundle("beatnet", target, condition)

    for fold_index in range(bundle.fold_count):
        parameters = json.loads(
            bundle.postprocessor_path("tuned-1d", fold_index=fold_index).read_text(
                encoding="utf-8"
            )
        )
        assert parameters["mode"] == "at"
        assert parameters["1d_ss_type"] == "1d-ss-at"
        assert int(parameters.get("peak_snap_window_frames", 0)) == 0


def test_classifier_bundles_have_no_beat_postprocessor() -> None:
    bundle = load_trained_model_bundle(
        "classifier", "latin_router", "efficientat"
    )

    assert bundle.task == "classification"
    assert bundle.postprocessors == {}
    assert bundle.stock_postprocessors == {}
    assert bundle.tuned_postprocessors == {}
    assert bundle.default_postprocessor is None
    with pytest.raises(ValueError, match="no postprocessors"):
        bundle.postprocessor_path()


def test_stock_postprocessor_file_contains_only_online_choices() -> None:
    path = beatnet_stock_postprocessor_selection_path()
    payload = json.loads(path.read_text(encoding="utf-8"))

    choices = payload["evaluation_postprocessors"]
    assert {choice["id"] for choice in choices} == STOCK_POSTPROCESSORS
    assert next(choice for choice in choices if choice["id"] == "stock-dbn")[
        "online"
    ] is True
    stock_1d = next(choice for choice in choices if choice["id"] == "stock-1d")
    assert stock_1d["mode"] == "at"
    assert stock_1d["1d_ss_type"] == "1d-ss-at"
    assert stock_1d["offset"] == pytest.approx(0.0)
    assert stock_1d["event_activation_threshold"] == pytest.approx(0.5)
    assert stock_1d["downbeat_activation_threshold"] == pytest.approx(0.4)

    bundle = load_trained_model_bundle("beatnet", "latin_general", "scratch")
    by_id = {choice["id"]: choice for choice in choices}
    for name in STOCK_POSTPROCESSORS:
        attached = json.loads(
            bundle.postprocessor_path(name).read_text(encoding="utf-8")
        )
        assert attached == by_id[name]
        assert bundle.stock_postprocessors[name].metadata["kind"] == "stock"


def test_trained_bundle_rejects_path_traversal_identifier() -> None:
    with pytest.raises(ValueError, match="lowercase letters"):
        load_trained_model_bundle("beatnet", "../salsa", "scratch")


def test_unknown_fold_fails_clearly() -> None:
    bundle = load_trained_model_bundle(
        "classifier", "latin_router", "yamnet", verify_files=False
    )

    with pytest.raises(KeyError, match="no fold 5"):
        bundle.checkpoint_path(5, verify=False)
