from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mir_core.datasets.annotations import (
    BeatAnnotation,
    read_beat,
    with_dance_structure,
    write_beat,
)
from mir_core.datasets.dance_annotations import (
    DANCE_ANNOTATION_IDENTITY_SCHEMA,
    DANCE_ANNOTATION_RULES_SCHEMA,
    load_dance_annotation_rules,
)


def _four_four_annotation() -> BeatAnnotation:
    positions = np.tile(np.arange(1, 5, dtype=np.int32), 3)
    bars = np.repeat(np.arange(1, 4, dtype=np.int32), 4)
    return BeatAnnotation(
        times=np.arange(12, dtype=np.float64) * 0.5,
        positions=positions,
        bars=bars,
    )


def test_salsa_structure_preserves_musical_downbeats_and_adds_dancebeat() -> None:
    annotation = with_dance_structure(
        _four_four_annotation(),
        beats_per_bar=4,
        dance_cycle_beats=8,
        first_bar_dance_position=1,
    )

    np.testing.assert_array_equal(
        annotation.positions,
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
    )
    np.testing.assert_array_equal(
        annotation.dance_positions,
        [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4],
    )
    np.testing.assert_array_equal(annotation.downbeat_times, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(annotation.dancebeat_times, [0.0, 4.0])
    np.testing.assert_array_equal(
        annotation.dance_bars,
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    )


def test_salsa_phase_can_begin_on_dance_count_five() -> None:
    annotation = with_dance_structure(
        _four_four_annotation(),
        beats_per_bar=4,
        dance_cycle_beats=8,
        first_bar_dance_position=5,
    )

    np.testing.assert_array_equal(
        annotation.dance_positions,
        [5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
    )
    np.testing.assert_array_equal(annotation.dancebeat_times, [2.0])


def test_one_bar_dance_cycle_makes_dancebeat_equal_downbeat() -> None:
    annotation = with_dance_structure(
        _four_four_annotation(),
        beats_per_bar=4,
        dance_cycle_beats=4,
    )

    np.testing.assert_array_equal(annotation.dance_positions, annotation.positions)
    np.testing.assert_array_equal(annotation.dance_bars, annotation.bars)
    np.testing.assert_array_equal(annotation.dancebeat_times, annotation.downbeat_times)


def test_five_column_beats_round_trip(tmp_path: Path) -> None:
    annotation = with_dance_structure(
        _four_four_annotation(),
        beats_per_bar=4,
        dance_cycle_beats=8,
    )
    path = tmp_path / "track.beats"

    write_beat(path, annotation)
    loaded = read_beat(path)

    np.testing.assert_array_equal(loaded.times, annotation.times)
    np.testing.assert_array_equal(loaded.positions, annotation.positions)
    np.testing.assert_array_equal(loaded.bars, annotation.bars)
    np.testing.assert_array_equal(loaded.dance_positions, annotation.dance_positions)
    np.testing.assert_array_equal(loaded.dance_bars, annotation.dance_bars)


def test_dance_cycle_must_contain_whole_musical_bars() -> None:
    with pytest.raises(ValueError, match="multiple of beats_per_bar"):
        with_dance_structure(
            _four_four_annotation(),
            beats_per_bar=4,
            dance_cycle_beats=6,
        )


def test_dance_position_and_dancebar_columns_are_atomic() -> None:
    with pytest.raises(ValueError, match="present together"):
        BeatAnnotation(
            times=np.asarray([0.0]),
            positions=np.asarray([1]),
            bars=np.asarray([1]),
            dance_positions=np.asarray([1]),
        )


def test_rule_registry_binds_rule_to_source_annotation_hash(tmp_path: Path) -> None:
    path = tmp_path / "rules.csv"
    path.write_text(
        "schema,dataset_id,track_id,source_annotation_hash,title,artist,genre,"
        "dance_style,beats_per_bar,dance_cycle_beats,"
        "first_bar_dance_position,phase_status,genre_evidence_id,"
        "cycle_evidence_id,phase_source,notes\n"
        f"{DANCE_ANNOTATION_RULES_SCHEMA},salsaset_ft,001,"
        f"{'a' * 64},A Song,An Artist,salsa,salsa,4,8,5,assumed,"
        "salsaset,salsa-eight-count,phase-table,listen later\n",
        encoding="utf-8",
    )

    registry = load_dance_annotation_rules(path)
    identity = registry.identity_for(
        "salsaset_ft",
        "001",
        source_annotation_hash="a" * 64,
    )
    annotation = registry.apply(
        "salsaset_ft",
        "001",
        _four_four_annotation(),
        source_annotation_hash="a" * 64,
    )

    assert len(registry.sha256) == 64
    assert len(identity.dance_rule_hash) == 64
    assert len(identity.effective_annotation_hash) == 64
    assert identity.schema == DANCE_ANNOTATION_IDENTITY_SCHEMA
    assert identity.source_annotation_hash == "a" * 64
    assert identity.dance_rules_hash == registry.sha256
    np.testing.assert_array_equal(annotation.dancebeat_times, [2.0])
    with pytest.raises(ValueError, match="stale"):
        registry.apply(
            "salsaset_ft",
            "001",
            _four_four_annotation(),
            source_annotation_hash="b" * 64,
        )
    with pytest.raises(ValueError, match="stale"):
        registry.identity_for(
            "salsaset_ft",
            "001",
            source_annotation_hash="b" * 64,
        )


def test_effective_annotation_hash_changes_with_source_or_rule_bundle(
    tmp_path: Path,
) -> None:
    header = (
        "schema,dataset_id,track_id,source_annotation_hash,title,artist,genre,"
        "dance_style,beats_per_bar,dance_cycle_beats,"
        "first_bar_dance_position,phase_status,genre_evidence_id,"
        "cycle_evidence_id,phase_source,notes\n"
    )
    row = (
        f"{DANCE_ANNOTATION_RULES_SCHEMA},salsaset_ft,001,{'a' * 64},"
        "A Song,An Artist,salsa,salsa,4,8,1,assumed,salsaset,"
        "salsa-eight-count,phase-table,listen later\n"
    )
    first_path = tmp_path / "rules-a.csv"
    first_path.write_text(header + row, encoding="utf-8")
    second_path = tmp_path / "rules-b.csv"
    second_path.write_text(
        header + row.replace("listen later", "reviewed note"), encoding="utf-8"
    )

    first = load_dance_annotation_rules(first_path).identity_for(
        "salsaset_ft",
        "001",
        source_annotation_hash="a" * 64,
    )
    second = load_dance_annotation_rules(second_path).identity_for(
        "salsaset_ft",
        "001",
        source_annotation_hash="a" * 64,
    )

    assert first.source_annotation_hash == second.source_annotation_hash
    assert first.dance_rule_hash != second.dance_rule_hash
    assert first.dance_rules_hash != second.dance_rules_hash
    assert first.effective_annotation_hash != second.effective_annotation_hash
