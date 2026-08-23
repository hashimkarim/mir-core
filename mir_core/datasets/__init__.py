"""
Data loading and dataset handling for beat tracking and genre classification.

Datasets:
    BeatTrackingDataset — Loads audio + beat/downbeat annotations (no feature extraction).
    GenreDataset        — Loads audio segments + genre labels with mel/mfcc features.

Loaders (via mirdata or custom loaders):
    load_dataset_tracks       — GTZAN, BRID, Candombe, and other mirdata datasets.
    load_salsaset_tracks      — SalsaSet (custom loader from GitHub).
    load_salsa_dataset_tracks — Salsa Dataset (custom loader from GitHub).
    load_genre_dataset        — Load audio paths and genre labels from multiple datasets.

Splitting:
    get_dataset_splits               — Simple train/val/test split.
    get_kfold_splits                 — K-fold CV (default 5-fold per Rapini & Jordanous, 2024).
    get_stratified_kfold_splits      — Stratified K-fold by label (e.g. tempo range).
    get_incremental_training_splits  — 1..N file incremental splits (Maia et al., 2023).
    CrossValidationRunner            — Orchestrates fold creation and result aggregation.
"""

from .annotations import (
    BeatAnnotation,
    read_beat,
    write_beat,
    with_dance_structure,
    from_candombe_csv,
    from_beats_tsv,
    from_salsa_dataset,
    from_salsaset_csv,
)
from .beat_tracking_dataset import BeatTrackingDataset
from .dance_annotations import (
    DANCE_ANNOTATION_IDENTITY_SCHEMA,
    DANCE_ANNOTATION_RULES_SCHEMA,
    DANCE_PHASE_STATUSES,
    DanceAnnotationIdentity,
    DanceAnnotationRegistry,
    DanceAnnotationRule,
    load_dance_annotation_rules,
)
from .genre_dataset import GenreDataset
from .splits import (
    get_dataset_splits,
    get_kfold_splits,
    get_stratified_kfold_splits,
    get_incremental_training_splits,
    create_dataloaders,
    CrossValidationRunner,
)

__all__ = [
    # Annotations
    "BeatAnnotation",
    "read_beat",
    "write_beat",
    "with_dance_structure",
    "from_candombe_csv",
    "from_beats_tsv",
    "from_salsa_dataset",
    "from_salsaset_csv",
    "DANCE_ANNOTATION_RULES_SCHEMA",
    "DANCE_ANNOTATION_IDENTITY_SCHEMA",
    "DANCE_PHASE_STATUSES",
    "DanceAnnotationIdentity",
    "DanceAnnotationRegistry",
    "DanceAnnotationRule",
    "load_dance_annotation_rules",
    # Datasets
    "BeatTrackingDataset",
    "GenreDataset",
    # Loaders
    "load_dataset_tracks",
    "load_salsaset_tracks",
    "load_salsa_dataset_tracks",
    "load_genre_dataset",
    "SALSASET_REPO_URL",
    "SALSA_DATASET_REPO_URL",
    # Splits
    "get_dataset_splits",
    "get_kfold_splits",
    "get_stratified_kfold_splits",
    "get_incremental_training_splits",
    "create_dataloaders",
    "CrossValidationRunner",
]


def __getattr__(name: str):
    """Load optional mirdata-backed helpers only when explicitly requested."""
    if name in {
        "load_dataset_tracks",
        "load_salsaset_tracks",
        "load_salsa_dataset_tracks",
        "load_genre_dataset",
        "SALSASET_REPO_URL",
        "SALSA_DATASET_REPO_URL",
    }:
        from . import loaders

        value = getattr(loaders, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
