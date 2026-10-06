#!/usr/bin/env python3
"""Rebuild and validate manifests for packaged trained model candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SCHEMA = "mir.trained-model-bundle/v2"
STOCK_SCHEMA = "mir.stock-beatnet-postprocessors/v1"
CONTRACT_HASH = "e2e-537f350dbaf7e925"
SEED = 42
FOLD_COUNT = 5
STOCK_POSTPROCESSOR_IDS = (
    "stock-1d",
    "stock-dbn",
    "stock-pf",
)
# Shared selections of the August searches. None is packaged any more; they
# were archived on 6 October 2026 when every tuned postprocessor became per-fold.
TUNED_POSTPROCESSOR_SOURCE_IDS = {
    "tuned-1d": "1d-causal-activation-v2-hybrid-joint",
    "tuned-dbn": "dbn-hybrid-joint",
    "tuned-pf": "particle-filter-fixed",
}
# Selected separately for every fold on that fold's validation songs.
PER_FOLD_POSTPROCESSOR_SOURCE_IDS = {
    "tuned-1d": "1d-live-validation-rerank-per-fold",
    "tuned-dbn": "dbn-live-validation-rerank-per-fold",
    "tuned-pf": "particle-filter-live-validation-rerank-per-fold",
}
REQUIRED_TUNED_POSTPROCESSOR_IDS = {
    "tuned-1d",
    "tuned-dbn",
    "tuned-pf",
}
DEFAULT_POSTPROCESSOR_ID = "tuned-dbn"
STOCK_CATALOG_PATH = Path("beatnet/stock/baseline/postprocessors.json")


@dataclass(frozen=True)
class Source:
    model_family: str
    target: str
    condition: str
    task: str
    experiment_hash: str
    scientific_config_hash: str
    attempt_id: str
    git_commit: str


SOURCES = (
    Source(
        "beatnet",
        "latin_general",
        "scratch",
        "beat_tracking",
        "btk-531c2f27e2d4e6dd",
        "531c2f27e2d4e6ddc24fb8bbf8c83f8c8b88aa1e2424479aba176125b1737ac5",
        "beat-final-online-v3-20260811T1715Z-latin-general",
        "629a9fb08c8334c861118c24f376cab624d67afa",
    ),
    Source(
        "beatnet",
        "brid",
        "scratch",
        "beat_tracking",
        "btk-16d65fb5265baec2",
        "16d65fb5265baec257a6ffb78613a1154b21b2b864c08c8c0cf62acd6e015069",
        "beat-final-online-v3-20260811T1715Z-specialists-scratch",
        "f31d8042c937c69b61011a519b9cc3609d3e8155",
    ),
    Source(
        "beatnet",
        "candombe",
        "scratch",
        "beat_tracking",
        "btk-519cedc1e0634d01",
        "519cedc1e0634d010f5069f5e6d3287ea2d058a6e7b6e60d537662330ac0d0be",
        "beat-final-online-v3-20260811T1715Z-specialists-scratch",
        "f31d8042c937c69b61011a519b9cc3609d3e8155",
    ),
    Source(
        "beatnet",
        "salsa",
        "scratch",
        "beat_tracking",
        "btk-745d23a56687ebe0",
        "745d23a56687ebe0d345aa382c07272d1a654b104772f8b8dcfce5fac97e00f3",
        "beat-final-online-v3-20260811T1715Z-specialists-scratch",
        "f31d8042c937c69b61011a519b9cc3609d3e8155",
    ),
    Source(
        "beatnet",
        "brid",
        "finetune_latin_general",
        "beat_tracking",
        "btk-331955aceed37404",
        "331955aceed374044f6f2fed6d4229d300133b7956e694918fd8135df1b62a46",
        "beat-final-online-v3-20260811T1715Z-specialists-finetune",
        "50a8707a74fad882329533a8aee2b63aaf7054d5",
    ),
    Source(
        "beatnet",
        "candombe",
        "finetune_latin_general",
        "beat_tracking",
        "btk-db25e6e89ff49b3f",
        "db25e6e89ff49b3f20c6eae374942f95e5545e85c5fbb0ec55e65bc45c126ff6",
        "beat-final-online-v3-20260811T1715Z-specialists-finetune",
        "629a9fb08c8334c861118c24f376cab624d67afa",
    ),
    Source(
        "beatnet",
        "salsa",
        "finetune_latin_general",
        "beat_tracking",
        "btk-013273597182a11d",
        "013273597182a11daf264e251b483d2106e36b7551ec4c2e94ae1a896e6c7a78",
        "beat-final-online-v3-20260811T1715Z-specialists-finetune",
        "50a8707a74fad882329533a8aee2b63aaf7054d5",
    ),
    Source(
        "classifier",
        "latin_router",
        "yamnet",
        "classification",
        "clf-839371022f8f797e",
        "839371022f8f797e7bde41f44198ec8797838e436925612ba75ee791ce8d36ed",
        "classifier-20260813T225050Z-0d27da6155",
        "0f7a5eaad2155e8edf8a52ddad4e9792b2b37957",
    ),
    Source(
        "classifier",
        "latin_router",
        "efficientat",
        "classification",
        "clf-b2fdbc86520b0d6c",
        "b2fdbc86520b0d6c78f4f7526a3bd53b7cd9f3fc69344c16a8e1df68e5c069ab",
        "classifier-20260814T045441Z-c14dd90c29",
        "0f7a5eaad2155e8edf8a52ddad4e9792b2b37957",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def rendered_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def rendered_file_record(relative_path: Path, rendered: str) -> dict[str, Any]:
    content = rendered.encode("utf-8")
    return {
        "path": relative_path.as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def load_stock_postprocessors(
    trained_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    catalog_path = trained_root / STOCK_CATALOG_PATH
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if payload.get("schema") != STOCK_SCHEMA:
        raise ValueError(f"Unexpected stock postprocessor schema: {catalog_path}")
    choices = payload.get("evaluation_postprocessors")
    if not isinstance(choices, list):
        raise ValueError(f"Stock postprocessor choices are not a list: {catalog_path}")
    by_id: dict[str, dict[str, Any]] = {}
    for choice in choices:
        if not isinstance(choice, dict):
            raise ValueError(f"Stock postprocessor choice is not an object: {catalog_path}")
        candidate = choice.get("id")
        if not isinstance(candidate, str) or candidate in by_id:
            raise ValueError(f"Stock postprocessor ids are invalid: {catalog_path}")
        by_id[candidate] = dict(choice)
    if set(by_id) != set(STOCK_POSTPROCESSOR_IDS):
        raise ValueError(f"Stock postprocessor inventory changed: {catalog_path}")
    source = payload.get("source")
    if not isinstance(source, dict) or not source:
        raise ValueError(f"Stock postprocessor source is missing: {catalog_path}")
    return {
        **source,
        "catalog": STOCK_CATALOG_PATH.as_posix(),
        "catalog_schema": STOCK_SCHEMA,
    }, by_id


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return payload


def checkpoint_fold(payload: dict[str, Any]) -> int:
    if isinstance(payload.get("fold_index"), int):
        return int(payload["fold_index"])
    split = payload.get("split_contract")
    if isinstance(split, dict) and isinstance(split.get("fold_index"), int):
        return int(split["fold_index"])
    raise ValueError("Checkpoint has no fold index.")


def split_summary(payload: dict[str, Any]) -> dict[str, Any]:
    split = payload.get("split_contract")
    plan = payload.get("split_plan")
    config = payload.get("config")
    if not isinstance(split, dict):
        raise ValueError("Checkpoint has no split contract.")
    if split.get("contract_hash") != CONTRACT_HASH:
        raise ValueError(f"Unexpected split contract: {split.get('contract_hash')}")
    universe = None
    plan_hash = split.get("plan_hash")
    membership_hash = split.get("plan_membership_hash")
    records_hash = split.get("records_hash")
    if isinstance(plan, dict):
        universe = plan.get("universe")
        plan_hash = plan_hash or plan.get("plan_hash")
        membership_hash = membership_hash or plan.get("plan_membership_hash")
        records_hash = records_hash or plan.get("records_hash")
    if isinstance(config, dict):
        configured_plan = config.get("data", {}).get("split_plan", {})
        universe = universe or configured_plan.get("universe")
    if universe != "latin-router-system-v1":
        raise ValueError(f"Unexpected split universe: {universe!r}")
    return {
        "contract_hash": CONTRACT_HASH,
        "universe": universe,
        "n_folds": FOLD_COUNT,
        "validation_fraction": 0.1,
        "plan_hash": plan_hash,
        "plan_membership_hash": membership_hash,
        "records_hash": records_hash,
    }


def model_metadata(source: Source, payload: dict[str, Any]) -> dict[str, Any]:
    if source.task == "beat_tracking":
        config = payload.get("config")
        if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
            raise ValueError("Beat checkpoint has no model config.")
        return dict(config["model"])
    return {
        "name": payload.get("arch"),
        "architecture": payload.get("model_config"),
        "feature_config": payload.get("feature_config"),
        "labels": payload.get("labels"),
        "runtime_policy_embedded_per_fold": True,
    }


def checkpoint_records(
    source: Source,
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model: dict[str, Any] | None = None
    common_split: dict[str, Any] | None = None
    for path in sorted((root / "checkpoints").glob("seed_42_fold_*.pt")):
        payload = load_checkpoint(path)
        fold = checkpoint_fold(payload)
        if payload.get("experiment_hash") != source.experiment_hash:
            raise ValueError(f"Experiment mismatch in {path}")
        if fold != len(records):
            raise ValueError(f"Non-contiguous fold inventory in {root}")
        split = split_summary(payload)
        if common_split is None:
            common_split = split
        elif split != common_split:
            raise ValueError(f"Common split identity changed in {root}")
        current_model = model_metadata(source, payload)
        if model is None:
            model = current_model
        elif current_model != model:
            raise ValueError(f"Model metadata changed across folds in {root}")
        record = {
            "fold_index": fold,
            "selector": "val_loss",
            "selection_split": "validation",
            **file_record(path, root),
        }
        for key in ("epoch", "val_loss", "best_metric", "best_value"):
            value = payload.get(key)
            if isinstance(value, str | int | float) and not isinstance(value, bool):
                record[key] = value
        if isinstance(payload.get("state_dict_sha256"), str):
            record["state_dict_sha256"] = payload["state_dict_sha256"]
        runtime_policy = payload.get("runtime_router_policy")
        if isinstance(runtime_policy, dict):
            policy_hash = runtime_policy.get("policy_hash")
            if not isinstance(policy_hash, str) or not policy_hash:
                raise ValueError(f"Classifier checkpoint has no runtime policy hash: {path}")
            record["runtime_policy_hash"] = policy_hash
        records.append(record)
    if len(records) != FOLD_COUNT or model is None or common_split is None:
        raise ValueError(f"Expected five complete checkpoints in {root}")
    return records, model, common_split


def tuned_postprocessor_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for params_path in sorted((root / "postprocessors").glob("*/params.json")):
        candidate = params_path.parent.name
        if candidate in STOCK_POSTPROCESSOR_IDS:
            continue
        source_path = params_path.with_name("source-manifest.json")
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing PP source manifest: {source_path}")
        params = json.loads(params_path.read_text(encoding="utf-8"))
        source = json.loads(source_path.read_text(encoding="utf-8"))
        stage = source.get("stage_config", {})
        source_candidate = source.get("postprocessor_id")
        expected_source_candidate = TUNED_POSTPROCESSOR_SOURCE_IDS.get(candidate)
        if (
            expected_source_candidate is None
            or source.get("status") != "completed"
            or source_candidate != expected_source_candidate
            or stage.get("postprocessors_all_online") is not True
            or stage.get("selection_split") != "validation"
            or stage.get("test_used_for_selection") is not False
        ):
            raise ValueError(f"Postprocessor is not a completed causal selection: {params_path}")
        source_record = file_record(source_path, root)
        records[candidate] = {
            "id": candidate,
            "kind": "tuned",
            "source_id": source_candidate,
            "method": params.get("method"),
            "causal": True,
            "selection_hash": source.get("selection_hash"),
            "selection_split": "validation",
            "test_used_for_selection": False,
            "online_contract_version": stage.get("online_causal_contract_version"),
            "source_manifest": source_record,
            **file_record(params_path, root),
        }
    return records


def per_fold_postprocessor_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for selection_path in sorted((root / "postprocessors").glob("*/selection.json")):
        candidate = selection_path.parent.name
        source_path = selection_path.with_name("source-manifest.json")
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing PP source manifest: {source_path}")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        source = json.loads(source_path.read_text(encoding="utf-8"))
        stage = source.get("stage_config", {})
        if (
            PER_FOLD_POSTPROCESSOR_SOURCE_IDS.get(candidate) is None
            or source.get("postprocessor_id")
            != PER_FOLD_POSTPROCESSOR_SOURCE_IDS[candidate]
            or source.get("status") != "completed"
            or selection.get("postprocessor_id") != candidate
            or selection.get("selection_policy") != "one_parameter_set_per_fold"
            or stage.get("selection_policy") != "one_parameter_set_per_fold"
            or stage.get("postprocessors_all_online") is not True
            or stage.get("selection_split") != "validation"
            or stage.get("test_used_for_selection") is not False
        ):
            raise ValueError(
                f"Postprocessor is not a completed per-fold causal selection: {selection_path}"
            )
        selected = {
            int(row["fold_index"]): row["parameters"]
            for row in selection.get("fold_parameters", [])
        }
        if sorted(selected) != list(range(FOLD_COUNT)):
            raise ValueError(f"Per-fold selection must cover every fold: {selection_path}")
        fold_records = []
        for fold in range(FOLD_COUNT):
            fold_path = selection_path.with_name(f"fold_{fold}.json")
            if not fold_path.is_file():
                raise FileNotFoundError(f"Missing fold parameters: {fold_path}")
            params = json.loads(fold_path.read_text(encoding="utf-8"))
            if params != selected[fold] or params.get("method") != selection.get("method"):
                raise ValueError(f"Fold parameters differ from their selection: {fold_path}")
            fold_records.append({"fold_index": fold, **file_record(fold_path, root)})
        records[candidate] = {
            "id": candidate,
            "kind": "tuned",
            "source_id": source.get("postprocessor_id"),
            "method": selection.get("method"),
            "causal": True,
            "selection_hash": source.get("selection_hash"),
            "selection_split": "validation",
            "selection_scope": "per_fold",
            "test_used_for_selection": False,
            "online_contract_version": stage.get("online_causal_contract_version"),
            "source_manifest": file_record(source_path, root),
            "fold_parameters": fold_records,
            **file_record(selection_path, root),
        }
    return records


def stock_postprocessor_records(
    stock_source: dict[str, Any],
    stock_choices: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for candidate in STOCK_POSTPROCESSOR_IDS:
        parameters = stock_choices[candidate]
        relative_path = Path("postprocessors") / candidate / "params.json"
        rendered = rendered_json(parameters)
        records[candidate] = {
            "id": candidate,
            "kind": "stock",
            "method": parameters.get("method"),
            "causal": True,
            "selection_split": None,
            "test_used_for_selection": False,
            "source": stock_source,
            **rendered_file_record(relative_path, rendered),
        }
    return records


def build_manifest(
    trained_root: Path,
    source: Source,
    stock_source: dict[str, Any],
    stock_choices: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    root = trained_root / source.model_family / source.target / source.condition
    checkpoints, model, split = checkpoint_records(source, root)
    if source.task == "beat_tracking":
        postprocessors = tuned_postprocessor_records(root)
        postprocessors.update(per_fold_postprocessor_records(root))
        postprocessors.update(stock_postprocessor_records(stock_source, stock_choices))
    else:
        postprocessors = {}
    if source.task == "beat_tracking" and not REQUIRED_TUNED_POSTPROCESSOR_IDS <= set(
        postprocessors
    ):
        raise ValueError(f"Beat bundle has an incomplete tuned inventory: {root}")
    manifest = {
        "schema": SCHEMA,
        "bundle_id": f"{source.model_family}/{source.target}/{source.condition}",
        "lifecycle": "candidate",
        "task": source.task,
        "model_family": source.model_family,
        "target": source.target,
        "condition": source.condition,
        "seed": SEED,
        "fold_count": FOLD_COUNT,
        "split_contract": split,
        "model": model,
        "source": {
            "experiment_hash": source.experiment_hash,
            "scientific_config_hash": source.scientific_config_hash,
            "attempt_id": source.attempt_id,
            "git_commit": source.git_commit,
            "checkpoint_selector": "val_loss",
            "checkpoint_selection_split": "validation",
            "test_used_for_checkpoint_selection": False,
        },
        "checkpoints": checkpoints,
        "postprocessors": postprocessors,
        "default_postprocessor": (
            DEFAULT_POSTPROCESSOR_ID if source.task == "beat_tracking" else None
        ),
    }
    return root / "manifest.json", manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trained-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "mir_core"
        / "checkpoints"
        / "trained",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if checked-in manifests differ instead of rewriting them",
    )
    args = parser.parse_args()
    trained_root = args.trained_root.expanduser().resolve()
    stock_source, stock_choices = load_stock_postprocessors(trained_root)
    changed: list[Path] = []
    for source in SOURCES:
        if source.task == "beat_tracking":
            root = trained_root / source.model_family / source.target / source.condition
            for candidate in STOCK_POSTPROCESSOR_IDS:
                params_path = root / "postprocessors" / candidate / "params.json"
                rendered = rendered_json(stock_choices[candidate])
                current = (
                    params_path.read_text(encoding="utf-8")
                    if params_path.is_file()
                    else None
                )
                if current != rendered:
                    changed.append(params_path)
                    if not args.check:
                        params_path.parent.mkdir(parents=True, exist_ok=True)
                        params_path.write_text(rendered, encoding="utf-8")
        path, payload = build_manifest(
            trained_root,
            source,
            stock_source,
            stock_choices,
        )
        rendered = rendered_json(payload)
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != rendered:
            changed.append(path)
            if not args.check:
                path.write_text(rendered, encoding="utf-8")
        print(f"validated {payload['bundle_id']}")
    if args.check and changed:
        print("stale trained registry artifacts:")
        for path in changed:
            print(path)
        return 1
    print(f"{'would update' if args.check else 'updated'} {len(changed)} artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
