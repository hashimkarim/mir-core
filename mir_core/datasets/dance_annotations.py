"""Versioned dance-count rules layered over canonical beat annotations."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .annotations import BeatAnnotation, with_dance_structure
from ..utils.hashing import stable_digest

DANCE_ANNOTATION_RULES_SCHEMA = "mir.dance-annotation-rules/v1"
DANCE_ANNOTATION_IDENTITY_SCHEMA = "mir.effective-dance-annotation/v1"
DANCE_PHASE_STATUSES = frozenset({"assumed", "derived", "reviewed"})


@dataclass(frozen=True)
class DanceAnnotationRule:
    """One auditable mapping from a track's musical meter to its dance count."""

    dataset_id: str
    track_id: str
    source_annotation_hash: str
    genre: str
    dance_style: str
    beats_per_bar: int
    dance_cycle_beats: int
    first_bar_dance_position: int
    phase_status: str
    genre_evidence_id: str
    cycle_evidence_id: str
    phase_source: str
    title: str = ""
    artist: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "track_id",
            "source_annotation_hash",
            "genre",
            "dance_style",
            "genre_evidence_id",
            "cycle_evidence_id",
            "phase_source",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"Dance annotation rule {name} cannot be empty.")
        if len(self.source_annotation_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_annotation_hash.lower()
        ):
            raise ValueError(
                "Dance annotation rules require a lowercase SHA-256 source "
                "annotation hash."
            )
        if self.phase_status not in DANCE_PHASE_STATUSES:
            raise ValueError(
                f"Unknown dance phase status {self.phase_status!r}; expected one "
                f"of {sorted(DANCE_PHASE_STATUSES)}."
            )
        # Reuse the structural validation without needing an annotation.
        if self.beats_per_bar <= 0:
            raise ValueError("beats_per_bar must be positive.")
        if self.dance_cycle_beats <= 0 or self.dance_cycle_beats % self.beats_per_bar:
            raise ValueError(
                "dance_cycle_beats must be a positive multiple of beats_per_bar."
            )
        valid_starts = range(1, self.dance_cycle_beats + 1, self.beats_per_bar)
        if self.first_bar_dance_position not in valid_starts:
            raise ValueError(
                "first_bar_dance_position must align with a musical bar boundary."
            )

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset_id, self.track_id

    @property
    def sha256(self) -> str:
        """Content hash of this rule, independent of CSV row ordering."""
        return stable_digest(
            {
                "schema": DANCE_ANNOTATION_RULES_SCHEMA,
                "rule": asdict(self),
            }
        )

    def apply(
        self,
        annotation: BeatAnnotation,
        *,
        source_annotation_hash: str | None = None,
    ) -> BeatAnnotation:
        """Return the canonical annotation with this rule's dance axis added."""
        if (
            source_annotation_hash is not None
            and source_annotation_hash != self.source_annotation_hash
        ):
            raise ValueError(
                "Dance annotation rule is stale for "
                f"{self.dataset_id}:{self.track_id}: expected source annotation "
                f"{self.source_annotation_hash}, got {source_annotation_hash}."
            )
        return with_dance_structure(
            annotation,
            beats_per_bar=self.beats_per_bar,
            dance_cycle_beats=self.dance_cycle_beats,
            first_bar_dance_position=self.first_bar_dance_position,
        )

    @classmethod
    def from_csv_row(cls, row: Mapping[str, str]) -> "DanceAnnotationRule":
        return cls(
            dataset_id=row["dataset_id"].strip(),
            track_id=row["track_id"].strip(),
            source_annotation_hash=row["source_annotation_hash"].strip().lower(),
            title=row.get("title", "").strip(),
            artist=row.get("artist", "").strip(),
            genre=row["genre"].strip(),
            dance_style=row["dance_style"].strip(),
            beats_per_bar=int(row["beats_per_bar"]),
            dance_cycle_beats=int(row["dance_cycle_beats"]),
            first_bar_dance_position=int(row["first_bar_dance_position"]),
            phase_status=row["phase_status"].strip(),
            genre_evidence_id=row["genre_evidence_id"].strip(),
            cycle_evidence_id=row["cycle_evidence_id"].strip(),
            phase_source=row["phase_source"].strip(),
            notes=row.get("notes", "").strip(),
        )


@dataclass(frozen=True)
class DanceAnnotationIdentity:
    """Reproducible identity for one effective five-column annotation.

    The original annotation remains content-addressed independently.  The rule
    and full rule-bundle hashes then identify exactly how the optional dance
    axis was derived without rewriting the source ``.beats`` file.
    """

    source_annotation_hash: str
    dance_rule_hash: str
    dance_rules_hash: str
    effective_annotation_hash: str
    schema: str = DANCE_ANNOTATION_IDENTITY_SCHEMA

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DanceAnnotationRegistry:
    """Immutable, content-addressed collection of per-track dance rules."""

    rules: Mapping[tuple[str, str], DanceAnnotationRule]
    source_path: Path
    sha256: str
    schema: str = DANCE_ANNOTATION_RULES_SCHEMA

    def rule_for(self, dataset_id: str, track_id: str) -> DanceAnnotationRule:
        key = (str(dataset_id), str(track_id))
        try:
            return self.rules[key]
        except KeyError as exc:
            raise KeyError(f"No dance annotation rule for {key[0]}:{key[1]}.") from exc

    def apply(
        self,
        dataset_id: str,
        track_id: str,
        annotation: BeatAnnotation,
        *,
        source_annotation_hash: str | None = None,
    ) -> BeatAnnotation:
        return self.rule_for(dataset_id, track_id).apply(
            annotation,
            source_annotation_hash=source_annotation_hash,
        )

    def identity_for(
        self,
        dataset_id: str,
        track_id: str,
        *,
        source_annotation_hash: str,
    ) -> DanceAnnotationIdentity:
        """Return the source + rule + bundle provenance chain for one track."""
        rule = self.rule_for(dataset_id, track_id)
        normalized_source_hash = str(source_annotation_hash).lower()
        if normalized_source_hash != rule.source_annotation_hash:
            raise ValueError(
                "Dance annotation rule is stale for "
                f"{rule.dataset_id}:{rule.track_id}: expected source annotation "
                f"{rule.source_annotation_hash}, got {normalized_source_hash}."
            )
        identity_payload = {
            "schema": DANCE_ANNOTATION_IDENTITY_SCHEMA,
            "source_annotation_hash": normalized_source_hash,
            "dance_rule_hash": rule.sha256,
            "dance_rules_hash": self.sha256,
        }
        return DanceAnnotationIdentity(
            source_annotation_hash=normalized_source_hash,
            dance_rule_hash=rule.sha256,
            dance_rules_hash=self.sha256,
            effective_annotation_hash=stable_digest(identity_payload),
        )


def load_dance_annotation_rules(
    path: str | Path,
) -> DanceAnnotationRegistry:
    """Load a strict per-track rule table and bind it to its content hash."""
    source_path = Path(path).expanduser().resolve()
    payload = source_path.read_bytes()
    rules: dict[tuple[str, str], DanceAnnotationRule] = {}
    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "schema" not in reader.fieldnames:
            raise ValueError("Dance annotation rules CSV must contain a schema column.")
        for line_number, row in enumerate(reader, start=2):
            if row.get("schema") != DANCE_ANNOTATION_RULES_SCHEMA:
                raise ValueError(
                    f"{source_path}:{line_number}: unsupported schema "
                    f"{row.get('schema')!r}."
                )
            rule = DanceAnnotationRule.from_csv_row(row)
            if rule.key in rules:
                raise ValueError(
                    f"{source_path}:{line_number}: duplicate dance rule for "
                    f"{rule.dataset_id}:{rule.track_id}."
                )
            rules[rule.key] = rule
    if not rules:
        raise ValueError(f"Dance annotation rule table is empty: {source_path}")
    return DanceAnnotationRegistry(
        rules=rules,
        source_path=source_path,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
