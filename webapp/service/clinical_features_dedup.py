"""Deduplication of clinical features that share a name across uploaded files.

Multiple clinical-feature CSVs in the same album can contain the same column
name (e.g. ``CenterID``). Feeding every copy to training would use the same
feature several times, so a single rule applies everywhere: **one definition
per name, the newest file wins** (highest ``clinical_feature_file_id``).

Both the training paths (``service.machine_learning``) and the advisory route
(``routes.clinical_features``) must import from this module so the warning
shown to the user can never drift from what training actually does.
"""

from typing import Dict, List, Optional

from quantimage2_backend_common.models import (
    ClinicalFeatureDefinition,
    ClinicalFeatureFile,
    ClinicalFeatureValue,
)

# Statuses reported per duplicated feature name
STATUS_IDENTICAL = "identical"
STATUS_COVERAGE_LOSS = "coverage_loss"
STATUS_CONFLICT = "conflict"

# Mirrors the missing-value detection used when building the training matrix
# (see service.machine_learning.get_clinical_features) plus the upload-time
# normalization in routes.clinical_features (empty string / "N/A" -> None).
MISSING_STRINGS = {"", "n/a", "n(a", "none", "nan"}


def is_missing_value(value: Optional[str]) -> bool:
    """True if a stored clinical feature value counts as missing for training."""
    if value is None:
        return True
    return str(value).strip().lower() in MISSING_STRINGS


def dedupe_definitions_by_name(
    definitions: List[ClinicalFeatureDefinition],
    ids_with_values: Optional[set] = None,
) -> List[ClinicalFeatureDefinition]:
    """Keep one definition per name: the newest file that actually has data.

    "Newest" is the highest ``clinical_feature_file_id`` (auto-increment, so
    higher id == more recently created file).

    A failed or partial upload can leave a newer definition with **no values**
    (e.g. a file whose patients didn't match the album); preferring it would
    shadow an older file that does have data — and feed training an empty
    column. So when ``ids_with_values`` (the set of definition ids that have at
    least one non-missing value, see :func:`definition_ids_with_values`) is
    provided, a definition that has values always beats one that doesn't; ties
    fall back to newest file. When it is ``None`` the function stays pure
    (no DB access) and applies plain newest-file-wins.
    """

    def has_values(definition):
        return ids_with_values is None or definition.id in ids_with_values

    def is_better(candidate, current):
        cand_has, cur_has = has_values(candidate), has_values(current)
        if cand_has != cur_has:
            return cand_has
        return candidate.clinical_feature_file_id > current.clinical_feature_file_id

    by_name: Dict[str, ClinicalFeatureDefinition] = {}
    for definition in definitions:
        current = by_name.get(definition.name)
        if current is None or is_better(definition, current):
            by_name[definition.name] = definition
    return list(by_name.values())


def _values_equal(a: str, b: str) -> bool:
    """Compare two stored values, tolerating numeric formatting ("1" == "1.0")."""
    a_str, b_str = str(a).strip(), str(b).strip()
    if a_str == b_str:
        return True
    try:
        return float(a_str) == float(b_str)
    except ValueError:
        return False


def definition_ids_with_values(definition_ids: List[int]) -> set:
    """Subset of ``definition_ids`` that have at least one non-missing value.

    Requires an app context (it queries the DB). Pass the result to
    :func:`dedupe_definitions_by_name` so a value-less definition can't shadow
    one that has data.
    """
    if not definition_ids:
        return set()
    return set(_non_missing_values_by_patient(definition_ids).keys())


def _non_missing_values_by_patient(
    definition_ids: List[int],
) -> Dict[int, Dict[str, str]]:
    """Batch-load values for many definitions: {definition_id: {patient: value}}."""
    values = ClinicalFeatureValue.find_by_clinical_feature_definition_ids(
        definition_ids
    )
    by_definition: Dict[int, Dict[str, str]] = {}
    for v in values:
        if not is_missing_value(v.value):
            by_definition.setdefault(v.clinical_feature_definition_id, {})[
                v.patient_id
            ] = v.value
    return by_definition


def compute_clinical_duplicate_advisories(user_id: str, album_id: str) -> List[Dict]:
    """Describe what newest-wins dedup does to each duplicated feature name.

    For every feature name present in more than one file of the album, compare
    the kept (newest-file) values against each older copy and classify:

    - ``identical``     — dropping the older copies loses no data
    - ``coverage_loss`` — some patients only have a value in an older copy;
                          those values are ignored by training
    - ``conflict``      — some patients have different values in the kept and
                          an older copy; the newest value is used

    ``coverage_loss`` and ``conflict`` can both apply to the same name.
    """
    definitions = ClinicalFeatureDefinition.find_by_user_id_and_album_id(
        user_id=user_id, album_id=album_id
    )
    files_by_id = {
        f.id: f
        for f in ClinicalFeatureFile.find_by_user_id_and_album_id(user_id, album_id)
    }

    definitions_by_name: Dict[str, List[ClinicalFeatureDefinition]] = {}
    for definition in definitions:
        definitions_by_name.setdefault(definition.name, []).append(definition)

    duplicated = {
        name: name_definitions
        for name, name_definitions in definitions_by_name.items()
        if len(name_definitions) >= 2
    }

    # One query for every duplicated definition's values, compared in memory.
    values_by_definition = _non_missing_values_by_patient(
        [d.id for name_definitions in duplicated.values() for d in name_definitions]
    )

    advisories = []
    for name, name_definitions in duplicated.items():
        # Mirror dedupe_definitions_by_name: a definition that has values beats
        # one that doesn't, then newest file wins. (values_by_definition only
        # holds definitions that have at least one non-missing value.)
        kept = max(
            name_definitions,
            key=lambda d: (d.id in values_by_definition, d.clinical_feature_file_id),
        )
        dropped = [d for d in name_definitions if d.id != kept.id]

        kept_values = values_by_definition.get(kept.id, {})

        conflict_patients = set()
        coverage_loss_patients = set()
        for dropped_definition in dropped:
            for patient_id, value in values_by_definition.get(
                dropped_definition.id, {}
            ).items():
                kept_value = kept_values.get(patient_id)
                if kept_value is None:
                    coverage_loss_patients.add(patient_id)
                elif not _values_equal(kept_value, value):
                    conflict_patients.add(patient_id)

        statuses = []
        if coverage_loss_patients:
            statuses.append(STATUS_COVERAGE_LOSS)
        if conflict_patients:
            statuses.append(STATUS_CONFLICT)
        if not statuses:
            statuses = [STATUS_IDENTICAL]

        def file_name(file_id):
            f = files_by_id.get(file_id)
            return f.name if f else f"File #{file_id}"

        advisories.append(
            {
                "name": name,
                "kept_file_id": kept.clinical_feature_file_id,
                "kept_file_name": file_name(kept.clinical_feature_file_id),
                "dropped_file_ids": [d.clinical_feature_file_id for d in dropped],
                "dropped_file_names": [
                    file_name(d.clinical_feature_file_id) for d in dropped
                ],
                "statuses": statuses,
                "coverage_loss_patient_count": len(coverage_loss_patients),
                "conflict_patient_count": len(conflict_patients),
                "affected_patient_count": len(
                    conflict_patients | coverage_loss_patients
                ),
            }
        )

    advisories.sort(key=lambda a: a["name"])
    return advisories
