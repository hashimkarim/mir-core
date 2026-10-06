"""Resolve hash-bound trained model bundles shipped inside :mod:`mir_core`.

The trained registry deliberately keeps component selection manual.  A bundle
identifies one model family, target, and training condition; callers choose the
fold checkpoint and (for beat trackers) the online postprocessor explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mir_core.postprocessing.state_space_1d import (
    normalize_state_space_1d_mode,
    state_space_1d_type,
)


TRAINED_MODEL_BUNDLE_SCHEMA = "mir.trained-model-bundle/v2"
TRAINED_MODELS_ROOT = Path(__file__).with_name("trained")

_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "bundle_id",
        "lifecycle",
        "task",
        "model_family",
        "target",
        "condition",
        "seed",
        "fold_count",
        "split_contract",
        "model",
        "source",
        "checkpoints",
        "postprocessors",
        "default_postprocessor",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def _identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(
            f"{name} must contain only lowercase letters, digits, '_' or '-'."
        )
    return value


def _relative_path(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}.path must be a non-empty string.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name}.path must stay within its trained bundle.")
    return path.as_posix()


def _file_record(value: Any, *, name: str) -> "TrainedFile":
    record = _mapping(value, name=name)
    sha256 = record.get("sha256")
    size_bytes = record.get("size_bytes")
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise ValueError(f"{name}.sha256 must be a lowercase SHA-256 digest.")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        raise ValueError(f"{name}.size_bytes must be a positive integer.")
    return TrainedFile(
        relative_path=_relative_path(record.get("path"), name=name),
        sha256=sha256,
        size_bytes=size_bytes,
        metadata=MappingProxyType(dict(record)),
    )


def _verify_file(path: Path, record: "TrainedFile", *, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Packaged {name} is missing: {path}")
    if path.stat().st_size != record.size_bytes:
        raise ValueError(f"{name} byte size does not match its bundle manifest.")
    if _sha256(path) != record.sha256:
        raise ValueError(f"{name} SHA-256 does not match its bundle manifest.")


def _fold_parameter_records(
    record: "TrainedFile",
    *,
    name: str,
) -> dict[int, "TrainedFile"] | None:
    """Return the per-fold parameter files of a postprocessor, if it has any.

    A tuned postprocessor selected separately for every fold lists one file per
    fold under ``fold_parameters``; its own ``path`` is then the selection
    record, not a parameter file.
    """

    rows = record.metadata.get("fold_parameters")
    if rows is None:
        if record.metadata.get("selection_scope") == "per_fold":
            raise ValueError(f"{name} is per-fold but lists no fold_parameters.")
        return None
    if record.metadata.get("selection_scope") != "per_fold":
        raise ValueError(f"{name} fold_parameters require selection_scope 'per_fold'.")
    if not isinstance(rows, list | tuple) or not rows:
        raise ValueError(f"{name}.fold_parameters must be a non-empty list.")
    folds: dict[int, TrainedFile] = {}
    for row in rows:
        entry = _mapping(row, name=f"{name} fold parameters")
        fold_index = entry.get("fold_index")
        if isinstance(fold_index, bool) or not isinstance(fold_index, int):
            raise ValueError(f"{name}.fold_parameters[].fold_index must be an integer.")
        if fold_index in folds:
            raise ValueError(f"{name} lists fold {fold_index} more than once.")
        parsed = _file_record(entry, name=f"{name} fold {fold_index}")
        # Fold files share the selection provenance of their postprocessor.
        folds[fold_index] = TrainedFile(
            relative_path=parsed.relative_path,
            sha256=parsed.sha256,
            size_bytes=parsed.size_bytes,
            metadata=record.metadata,
        )
    return folds


def _nonnegative_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a non-negative finite number.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return number


def _validate_causal_postprocessor(
    path: Path,
    record: "TrainedFile",
    *,
    name: str,
) -> None:
    metadata = record.metadata
    if metadata.get("causal") is not True:
        raise ValueError(f"{name} must be explicitly causal/online.")
    kind = metadata.get("kind")
    if kind == "tuned":
        if metadata.get("selection_split") != "validation":
            raise ValueError(f"{name} must be selected on validation data.")
    elif kind == "stock":
        if metadata.get("selection_split") is not None:
            raise ValueError(f"{name} stock parameters must not claim tuning.")
    else:
        raise ValueError(f"{name} must declare kind 'stock' or 'tuned'.")
    if metadata.get("test_used_for_selection") is not False:
        raise ValueError(f"{name} must not use test results for selection.")
    try:
        parameters = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{name} parameters must be an object.")
    method = parameters.get("method")
    if method != metadata.get("method"):
        raise ValueError(f"{name} method does not match its bundle manifest.")
    if method in {"dbn", "dbn_downbeat", "beast_dbn"}:
        if parameters.get("online") is not True:
            raise ValueError(f"{name} DBN must set online: true.")
        if method != "beast_dbn" and bool(parameters.get("correct", False)):
            raise ValueError(f"{name} DBN must set correct: false.")
        return
    if method == "heydari_1d_state_space":
        window = int(
            _nonnegative_number(
                parameters.get(
                    "peak_snap_window_frames",
                    parameters.get("snap_window_frames", 0),
                ),
                name=f"{name} peak-snap window",
            )
        )
        mode = str(parameters.get("peak_snap_mode", "center")).lower()
        if window > 0 and mode not in {"past", "causal"}:
            raise ValueError(f"{name} uses future-looking 1D peak snapping.")
        mode = normalize_state_space_1d_mode(
            parameters.get(
                "mode",
                parameters.get("event_trigger_mode", "at"),
            )
        )
        declared_type = parameters.get("1d_ss_type")
        if declared_type is not None and str(declared_type) != state_space_1d_type(mode):
            raise ValueError(
                f"{name} 1d_ss_type does not match its configured mode."
            )
        if mode == "at" and window:
            raise ValueError(
                f"{name} uses immediate activation with retrospective snapping."
            )
        return
    if method == "particle_filter":
        return
    raise ValueError(f"{name} uses unsupported causal method {method!r}.")


@dataclass(frozen=True, slots=True)
class TrainedFile:
    """One byte-bound file inside a trained model bundle."""

    relative_path: str
    sha256: str
    size_bytes: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TrainedModelBundle:
    """Validated paths and provenance for one packaged trained model."""

    root: Path
    manifest_path: Path
    bundle_id: str
    lifecycle: str
    task: str
    model_family: str
    target: str
    condition: str
    seed: int
    fold_count: int
    split_contract: Mapping[str, Any]
    model: Mapping[str, Any]
    source: Mapping[str, Any]
    checkpoints: Mapping[int, TrainedFile]
    postprocessors: Mapping[str, TrainedFile]
    stock_postprocessors: Mapping[str, TrainedFile]
    tuned_postprocessors: Mapping[str, TrainedFile]
    default_postprocessor: str | None

    def checkpoint_path(self, fold_index: int, *, verify: bool = True) -> Path:
        """Return the requested fold-matched checkpoint path."""

        if isinstance(fold_index, bool) or not isinstance(fold_index, int):
            raise TypeError("fold_index must be an integer.")
        try:
            record = self.checkpoints[fold_index]
        except KeyError as exc:
            raise KeyError(
                f"Bundle {self.bundle_id!r} has no fold {fold_index}."
            ) from exc
        path = self.root / record.relative_path
        if verify:
            _verify_file(path, record, name=f"checkpoint fold {fold_index}")
        return path

    def _postprocessor_record(self, name: str | None) -> tuple[str, TrainedFile]:
        selected = self.default_postprocessor if name is None else name
        if selected is None:
            raise ValueError(f"Bundle {self.bundle_id!r} has no postprocessors.")
        try:
            return selected, self.postprocessors[selected]
        except KeyError as exc:
            available = ", ".join(self.postprocessors) or "none"
            raise KeyError(
                f"Bundle {self.bundle_id!r} has no postprocessor {selected!r}; "
                f"available: {available}."
            ) from exc

    def postprocessor_is_per_fold(self, name: str | None = None) -> bool:
        """Whether the postprocessor has one parameter file for every fold."""

        selected, record = self._postprocessor_record(name)
        return (
            _fold_parameter_records(record, name=f"postprocessor {selected!r}")
            is not None
        )

    def postprocessor_path(
        self,
        name: str | None = None,
        *,
        fold_index: int | None = None,
        verify: bool = True,
    ) -> Path:
        """Return one explicitly selected online postprocessor parameter file.

        A per-fold postprocessor requires ``fold_index`` and returns the file
        selected on that fold's validation songs. For a postprocessor with one
        shared parameter file, ``fold_index`` is accepted and has no effect, so
        fold-aware callers can always pass it.
        """

        selected, record = self._postprocessor_record(name)
        if fold_index is not None:
            if isinstance(fold_index, bool) or not isinstance(fold_index, int):
                raise TypeError("fold_index must be an integer.")
            if fold_index not in self.checkpoints:
                raise KeyError(f"Bundle {self.bundle_id!r} has no fold {fold_index}.")
        folds = _fold_parameter_records(record, name=f"postprocessor {selected!r}")
        if folds is not None:
            if fold_index is None:
                raise ValueError(
                    f"Postprocessor {selected!r} of bundle {self.bundle_id!r} has "
                    "one parameter set per fold; pass fold_index."
                )
            record = folds[fold_index]
            selected = f"{selected} fold {fold_index}"
        path = self.root / record.relative_path
        if verify:
            _verify_file(path, record, name=f"postprocessor {selected!r}")
            _validate_causal_postprocessor(
                path,
                record,
                name=f"postprocessor {selected!r}",
            )
        return path


def _bundle_root(model_family: str, target: str, condition: str) -> Path:
    return TRAINED_MODELS_ROOT / model_family / target / condition


def load_trained_model_bundle(
    model_family: str,
    target: str,
    condition: str,
    *,
    verify_files: bool = True,
) -> TrainedModelBundle:
    """Load a bundle from ``trained/<model>/<target>/<condition>``."""

    model_family = _identifier(model_family, name="model_family")
    target = _identifier(target, name="target")
    condition = _identifier(condition, name="condition")
    root = _bundle_root(model_family, target, condition)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Trained model manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid trained model manifest: {manifest_path}") from exc
    manifest = _mapping(payload, name="trained model manifest")
    fields = frozenset(manifest)
    if fields != _TOP_LEVEL_FIELDS:
        raise ValueError(
            "Trained model manifest fields do not match the v2 schema; "
            f"missing={sorted(_TOP_LEVEL_FIELDS - fields)}, "
            f"extra={sorted(fields - _TOP_LEVEL_FIELDS)}."
        )
    if manifest.get("schema") != TRAINED_MODEL_BUNDLE_SCHEMA:
        raise ValueError(f"Unsupported trained model schema: {manifest.get('schema')!r}.")
    expected_bundle_id = f"{model_family}/{target}/{condition}"
    if manifest.get("bundle_id") != expected_bundle_id:
        raise ValueError("Trained model path does not match manifest bundle_id.")
    for field, expected in (
        ("model_family", model_family),
        ("target", target),
        ("condition", condition),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"Trained model path does not match manifest {field}.")
    lifecycle = manifest.get("lifecycle")
    if lifecycle not in {"candidate", "selected", "retired"}:
        raise ValueError("Trained model lifecycle is invalid.")
    task = manifest.get("task")
    if task not in {"beat_tracking", "classification"}:
        raise ValueError("Trained model task is invalid.")
    seed = manifest.get("seed")
    fold_count = manifest.get("fold_count")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Trained model seed must be a non-negative integer.")
    if (
        isinstance(fold_count, bool)
        or not isinstance(fold_count, int)
        or fold_count < 2
    ):
        raise ValueError("Trained model fold_count must be at least two.")
    split_contract = _mapping(
        manifest.get("split_contract"),
        name="trained model split_contract",
    )
    if (
        split_contract.get("n_folds") != fold_count
        or not str(split_contract.get("contract_hash") or "").strip()
    ):
        raise ValueError("Trained model split contract is inconsistent.")
    model = _mapping(manifest.get("model"), name="trained model model")
    source = _mapping(manifest.get("source"), name="trained model source")
    if not model or not source:
        raise ValueError("Trained model metadata must not be empty.")

    checkpoint_rows = manifest.get("checkpoints")
    if (
        isinstance(checkpoint_rows, (str, bytes))
        or not isinstance(checkpoint_rows, Sequence)
    ):
        raise ValueError("Trained model checkpoints must be a list.")
    checkpoints: dict[int, TrainedFile] = {}
    for index, value in enumerate(checkpoint_rows):
        row = _mapping(value, name=f"checkpoint {index}")
        fold_index = row.get("fold_index")
        if (
            isinstance(fold_index, bool)
            or not isinstance(fold_index, int)
            or fold_index in checkpoints
        ):
            raise ValueError("Trained checkpoint folds must be unique integers.")
        if row.get("selection_split") != "validation":
            raise ValueError("Trained checkpoints must be selected on validation data.")
        checkpoints[fold_index] = _file_record(
            row,
            name=f"checkpoint fold {fold_index}",
        )
    if set(checkpoints) != set(range(fold_count)):
        raise ValueError("Trained model must contain exactly one checkpoint per fold.")

    postprocessor_rows = _mapping(
        manifest.get("postprocessors"),
        name="trained model postprocessors",
    )
    postprocessors: dict[str, TrainedFile] = {}
    stock_postprocessors: dict[str, TrainedFile] = {}
    tuned_postprocessors: dict[str, TrainedFile] = {}
    for name, value in postprocessor_rows.items():
        postprocessor_name = _identifier(str(name), name="postprocessor name")
        record = _file_record(value, name=f"postprocessor {name!r}")
        if record.metadata.get("id") != postprocessor_name:
            raise ValueError("Postprocessor mapping key must match its id.")
        kind = record.metadata.get("kind")
        if kind == "stock":
            stock_postprocessors[postprocessor_name] = record
        elif kind == "tuned":
            tuned_postprocessors[postprocessor_name] = record
        else:
            raise ValueError("Postprocessors must declare kind 'stock' or 'tuned'.")
        if not postprocessor_name.startswith(f"{kind}-"):
            raise ValueError(
                "Postprocessor ids must use the '<kind>-<method>[-<variant>]' "
                "naming pattern."
            )
        folds = _fold_parameter_records(record, name=f"postprocessor {name!r}")
        if folds is not None and set(folds) != set(checkpoints):
            raise ValueError(
                f"Per-fold postprocessor {name!r} must cover exactly the bundle's folds."
            )
        postprocessors[postprocessor_name] = record
    default_postprocessor = manifest.get("default_postprocessor")
    if task == "beat_tracking":
        if not postprocessors:
            raise ValueError("Beat-tracking bundles require online postprocessors.")
        if not stock_postprocessors or not tuned_postprocessors:
            raise ValueError(
                "Beat-tracking bundles require both stock and tuned postprocessors."
            )
        if default_postprocessor not in tuned_postprocessors:
            raise ValueError(
                "default_postprocessor must name a tuned postprocessor."
            )
    elif postprocessors or default_postprocessor is not None:
        raise ValueError("Classification bundles must not define postprocessors.")

    bundle = TrainedModelBundle(
        root=root,
        manifest_path=manifest_path,
        bundle_id=expected_bundle_id,
        lifecycle=str(lifecycle),
        task=str(task),
        model_family=model_family,
        target=target,
        condition=condition,
        seed=seed,
        fold_count=fold_count,
        split_contract=MappingProxyType(dict(split_contract)),
        model=MappingProxyType(dict(model)),
        source=MappingProxyType(dict(source)),
        checkpoints=MappingProxyType(checkpoints),
        postprocessors=MappingProxyType(postprocessors),
        stock_postprocessors=MappingProxyType(stock_postprocessors),
        tuned_postprocessors=MappingProxyType(tuned_postprocessors),
        default_postprocessor=(
            None if default_postprocessor is None else str(default_postprocessor)
        ),
    )
    if verify_files:
        for fold_index in range(bundle.fold_count):
            bundle.checkpoint_path(fold_index)
        for name in bundle.postprocessors:
            if bundle.postprocessor_is_per_fold(name):
                for fold_index in range(bundle.fold_count):
                    bundle.postprocessor_path(name, fold_index=fold_index)
            else:
                bundle.postprocessor_path(name)
    return bundle


def list_trained_model_bundles(
    *,
    task: str | None = None,
    verify_files: bool = False,
) -> tuple[TrainedModelBundle, ...]:
    """List every packaged trained bundle, optionally filtered by task."""

    if task is not None and task not in {"beat_tracking", "classification"}:
        raise ValueError("task must be 'beat_tracking' or 'classification'.")
    bundles: list[TrainedModelBundle] = []
    for manifest_path in sorted(TRAINED_MODELS_ROOT.glob("*/*/*/manifest.json")):
        relative = manifest_path.relative_to(TRAINED_MODELS_ROOT)
        bundle = load_trained_model_bundle(
            relative.parts[0],
            relative.parts[1],
            relative.parts[2],
            verify_files=verify_files,
        )
        if task is None or bundle.task == task:
            bundles.append(bundle)
    return tuple(bundles)


def trained_checkpoint_path(
    model_family: str,
    target: str,
    condition: str,
    fold_index: int,
) -> Path:
    """Resolve and verify one packaged fold checkpoint."""

    return load_trained_model_bundle(
        model_family,
        target,
        condition,
        verify_files=False,
    ).checkpoint_path(fold_index)


def trained_postprocessor_path(
    model_family: str,
    target: str,
    condition: str,
    name: str | None = None,
    *,
    fold_index: int | None = None,
) -> Path:
    """Resolve and verify one packaged online postprocessor parameter file.

    Per-fold postprocessors require ``fold_index``.
    """

    return load_trained_model_bundle(
        model_family,
        target,
        condition,
        verify_files=False,
    ).postprocessor_path(name, fold_index=fold_index)


def beatnet_stock_postprocessor_selection_path() -> Path:
    """Return the shipped stock BeatNet online-postprocessor selection file."""

    path = (
        TRAINED_MODELS_ROOT
        / "beatnet"
        / "stock"
        / "baseline"
        / "postprocessors.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Packaged stock postprocessors are missing: {path}")
    return path


__all__ = [
    "TRAINED_MODEL_BUNDLE_SCHEMA",
    "TRAINED_MODELS_ROOT",
    "TrainedFile",
    "TrainedModelBundle",
    "beatnet_stock_postprocessor_selection_path",
    "list_trained_model_bundles",
    "load_trained_model_bundle",
    "trained_checkpoint_path",
    "trained_postprocessor_path",
]
