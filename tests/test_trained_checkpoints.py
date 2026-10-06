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
    "stock-particle-filter",
}
REQUIRED_TUNED_POSTPROCESSORS = {
    "tuned-dbn",
    "tuned-particle-filter",
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
    assert set(bundle.tuned_postprocessors) == {
        "tuned-1d",
        "tuned-dbn",
        "tuned-dbn-per-fold",
        "tuned-particle-filter",
    }
    assert set(bundle.stock_postprocessors) == STOCK_POSTPROCESSORS
    assert bundle.postprocessor_path("tuned-dbn").name == "params.json"
    assert bundle.tuned_postprocessors["tuned-dbn"].metadata["source_id"] == (
        "dbn-hybrid-joint"
    )
    assert trained_checkpoint_path(
        "beatnet", "candombe", "scratch", 0
    ).is_file()
    assert trained_postprocessor_path(
        "beatnet", "candombe", "scratch", "tuned-particle-filter"
    ).is_file()
    assert trained_postprocessor_path(
        "beatnet", "candombe", "scratch", "stock-dbn"
    ).is_file()


PER_FOLD_DBN_BUNDLES = (
    ("latin_general", "scratch"),
    ("brid", "finetune_latin_general"),
    ("candombe", "scratch"),
    ("salsa", "finetune_latin_general"),
)


@pytest.mark.parametrize(("target", "condition"), PER_FOLD_DBN_BUNDLES)
def test_per_fold_dbn_is_the_default_and_requires_a_fold(
    target: str, condition: str
) -> None:
    bundle = load_trained_model_bundle("beatnet", target, condition)

    assert bundle.default_postprocessor == "tuned-dbn-per-fold"
    assert bundle.postprocessor_is_per_fold()
    assert not bundle.postprocessor_is_per_fold("tuned-dbn")
    record = bundle.tuned_postprocessors["tuned-dbn-per-fold"]
    assert record.metadata["selection_scope"] == "per_fold"
    assert record.metadata["selection_split"] == "validation"
    assert record.metadata["test_used_for_selection"] is False
    with pytest.raises(ValueError, match="pass fold_index"):
        bundle.postprocessor_path()
    with pytest.raises(KeyError, match="no fold 5"):
        bundle.postprocessor_path(fold_index=5)

    selection = json.loads(
        (bundle.root / record.relative_path).read_text(encoding="utf-8")
    )
    assert selection["selection_policy"] == "one_parameter_set_per_fold"
    selected = {row["fold_index"]: row["parameters"] for row in selection["fold_parameters"]}
    for fold_index in range(bundle.fold_count):
        path = bundle.postprocessor_path(fold_index=fold_index)
        assert path.name == f"fold_{fold_index}.json"
        parameters = json.loads(path.read_text(encoding="utf-8"))
        assert parameters == selected[fold_index]
        assert parameters["method"] == "dbn_downbeat"
        assert parameters["online"] is True and parameters["correct"] is False
        assert path == trained_postprocessor_path(
            "beatnet", target, condition, fold_index=fold_index
        )


def test_per_fold_selection_replaces_the_late_salsa_setting() -> None:
    bundle = load_trained_model_bundle("beatnet", "salsa", "finetune_latin_general")

    shared = json.loads(bundle.postprocessor_path("tuned-dbn").read_text(encoding="utf-8"))
    assert shared["observation_lambda"] == 16
    for fold_index in range(bundle.fold_count):
        parameters = json.loads(
            bundle.postprocessor_path(fold_index=fold_index).read_text(encoding="utf-8")
        )
        assert (parameters["observation_lambda"], parameters["transition_lambda"]) == (8, 50.0)


def test_shared_postprocessors_accept_a_fold_index() -> None:
    bundle = load_trained_model_bundle("beatnet", "salsa", "scratch")

    assert bundle.default_postprocessor == "tuned-dbn"
    assert not bundle.postprocessor_is_per_fold()
    assert bundle.postprocessor_path(fold_index=3) == bundle.postprocessor_path()
    with pytest.raises(KeyError, match="no fold 9"):
        bundle.postprocessor_path("stock-dbn", fold_index=9)


def test_unusable_legacy_1d_candidate_is_not_exposed_as_tuned() -> None:
    bundle = load_trained_model_bundle("beatnet", "salsa", "scratch")

    assert "stock-1d" in bundle.stock_postprocessors
    assert "tuned-1d" not in bundle.tuned_postprocessors


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
