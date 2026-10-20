"""
Prepare a set of class users (e.g. hecktor1..15@hevs.ch) for a new session.

For each user listed in the config file:

1. Resolve the Keycloak user ID from the email (Keycloak admin API). The album
   is the one of the user's latest feature extraction, unless the config gives
   an album_id for that user.
2. Check (report only, nothing is changed) that:
   - the outcomes match the reference outcomes CSV,
   - the clinical features match the reference clinical features CSV.
3. Clean: delete all trained models (rows + joblib files), all feature
   collections and every feature extraction except the latest one.
4. Set a manual train/test split on the latest extraction: the patients of the
   test patients CSV go to the test set, every other patient of the reference
   outcomes CSV goes to training.

Run inside the backend container (its working directory is webapp/):

    docker compose exec backend python -m scripts.prepare_class_users \\
        scripts/class_prep/hecktor.yml --dry-run

The Keycloak admin password is read from KEYCLOAK_ADMIN_PASSWORD, or asked
interactively. Paths in the config file are relative to the config file.
"""

import argparse
import getpass
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml

from config import FEATURES_CACHE_BASE_DIR
from quantimage2_backend_common.const import (
    DATA_SPLITTING_TYPES,
    TRAIN_TEST_SPLIT_TYPES,
)
from service.clinical_features_dedup import (
    _values_equal,
    dedupe_definitions_by_name,
    definition_ids_with_values,
    is_missing_value,
)
from service.feature_transformation import (
    OUTCOME_FIELD_CLASSIFICATION,
    OUTCOME_FIELD_SURVIVAL_EVENT,
    OUTCOME_FIELD_SURVIVAL_TIME,
)

PATIENT_ID = "PatientID"

# Max number of offending patients listed per issue (the count is always shown)
MAX_LISTED = 5


@dataclass
class UserReport:
    email: str
    album_id: Optional[str] = None
    user_id: Optional[str] = None
    extraction_id: Optional[int] = None
    outcome_issues: List[str] = field(default_factory=list)
    clinical_issues: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    deleted_models: int = 0
    deleted_collections: int = 0
    deleted_extractions: int = 0
    split_set: bool = False

    @property
    def ok(self) -> bool:
        return not (self.outcome_issues or self.clinical_issues or self.errors)


# --------------------------------------------------------------------------- #
# Reference files
# --------------------------------------------------------------------------- #


def read_csv_as_strings(path: Path) -> pd.DataFrame:
    """Read a CSV keeping every cell as a string, the way the frontend sends it."""
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    if df.columns[0] != PATIENT_ID:
        raise ValueError(f"{path}: first column must be {PATIENT_ID}")
    df[PATIENT_ID] = df[PATIENT_ID].str.strip()
    # Spreadsheet exports often end with empty ",,,," rows
    df = df[df[PATIENT_ID] != ""].reset_index(drop=True)
    duplicates = df[PATIENT_ID][df[PATIENT_ID].duplicated()].tolist()
    if duplicates:
        raise ValueError(f"{path}: duplicated patients {duplicates[:MAX_LISTED]}")
    return df


def load_reference_outcomes(path: Path) -> Dict[str, Dict[str, str]]:
    """{patient_id: {field: value}}, fields being Outcome or Time/Event."""
    df = read_csv_as_strings(path)
    fields = set(df.columns) - {PATIENT_ID}
    survival = {OUTCOME_FIELD_SURVIVAL_TIME, OUTCOME_FIELD_SURVIVAL_EVENT}
    if fields != {OUTCOME_FIELD_CLASSIFICATION} and fields != survival:
        raise ValueError(
            f"{path}: expected columns {PATIENT_ID},{OUTCOME_FIELD_CLASSIFICATION} "
            f"or {PATIENT_ID},{OUTCOME_FIELD_SURVIVAL_TIME},"
            f"{OUTCOME_FIELD_SURVIVAL_EVENT} - got {list(df.columns)}"
        )
    return df.set_index(PATIENT_ID).to_dict(orient="index")


def load_test_patients(path: Path) -> List[str]:
    """Patient IDs of a one-column CSV (a PatientID header is optional)."""
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    if lines and lines[0] == PATIENT_ID:
        lines = lines[1:]
    if len(set(lines)) != len(lines):
        raise ValueError(f"{path}: duplicated patients")
    return lines


# --------------------------------------------------------------------------- #
# Checks (pure functions, no DB access)
# --------------------------------------------------------------------------- #


def _listed(patients) -> str:
    patients = sorted(patients)
    suffix = ", ..." if len(patients) > MAX_LISTED else ""
    return ", ".join(patients[:MAX_LISTED]) + suffix


def compare_outcomes(
    reference: Dict[str, Dict[str, str]], labels: Dict[str, Dict[str, str]]
) -> List[str]:
    """Differences between the reference outcomes and a user's labels."""
    issues = []

    def is_empty(content):
        return not content or all(is_missing_value(v) for v in content.values())

    missing = [p for p in reference if is_empty(labels.get(p))]
    if missing:
        issues.append(f"{len(missing)} patients without outcome: {_listed(missing)}")

    extra = [p for p in labels if p not in reference and not is_empty(labels[p])]
    if extra:
        issues.append(
            f"{len(extra)} patients not in the reference file: {_listed(extra)}"
        )

    mismatched = [
        p
        for p, expected in reference.items()
        if not is_empty(labels.get(p))
        and any(not _values_equal(labels[p].get(k, ""), v) for k, v in expected.items())
    ]
    if mismatched:
        issues.append(
            f"{len(mismatched)} patients with a different outcome: "
            f"{_listed(mismatched)}"
        )

    return issues


def compare_clinical_features(
    reference: pd.DataFrame, values: Dict[str, Dict[str, str]]
) -> List[str]:
    """Differences between the reference CSV and {feature_name: {patient: value}}."""
    issues = []
    features = [c for c in reference.columns if c != PATIENT_ID]

    missing_features = [f for f in features if f not in values]
    if missing_features:
        issues.append(f"missing clinical features: {', '.join(missing_features)}")

    extra_features = sorted(set(values) - set(features))
    if extra_features:
        issues.append(
            f"clinical features not in the reference file: {', '.join(extra_features)}"
        )

    for feature in features:
        if feature not in values:
            continue
        stored = values[feature]
        mismatched = []
        for patient, expected in zip(reference[PATIENT_ID], reference[feature]):
            actual = stored.get(patient)
            if is_missing_value(expected) and is_missing_value(actual):
                continue
            if (
                is_missing_value(expected)
                or is_missing_value(actual)
                or not _values_equal(actual, expected)
            ):
                mismatched.append(patient)
        if mismatched:
            issues.append(
                f"'{feature}': {len(mismatched)} patients differ: "
                f"{_listed(mismatched)}"
            )

    return issues


def compute_split(
    all_patients: List[str], test_patients: List[str]
) -> Tuple[List[str], List[str]]:
    """(training, test) patients; every test patient must belong to the album."""
    unknown = set(test_patients) - set(all_patients)
    if unknown:
        raise ValueError(
            f"test patients not in the reference outcomes: {_listed(unknown)}"
        )
    test = set(test_patients)
    training = sorted(p for p in all_patients if p not in test)
    return training, sorted(test)


# --------------------------------------------------------------------------- #
# DB access
# --------------------------------------------------------------------------- #


def find_outcome_category(album_id: str, user_id: str, name: Optional[str]):
    """The outcome to check: by name if given, else the album's selected one."""
    from quantimage2_backend_common.models import AlbumOutcome, LabelCategory

    categories = LabelCategory.find_by_album(album_id, user_id)
    if name:
        matching = [c for c in categories if c.name == name]
        return matching[-1] if matching else None
    selected = AlbumOutcome.find_by_album_user_id(album_id, user_id)
    if selected and selected.outcome:
        return selected.outcome
    return categories[0] if len(categories) == 1 else None


def load_clinical_feature_values(
    album_id: str, user_id: str
) -> Dict[str, Dict[str, str]]:
    """{feature_name: {patient: value}}, deduplicated the same way as training."""
    from quantimage2_backend_common.models import (
        ClinicalFeatureDefinition,
        ClinicalFeatureValue,
    )

    definitions = ClinicalFeatureDefinition.find_by_user_id_and_album_id(
        user_id, album_id
    )
    ids_with_values = definition_ids_with_values([d.id for d in definitions])
    definitions = dedupe_definitions_by_name(definitions, ids_with_values)
    names_by_id = {d.id: d.name for d in definitions}

    values: Dict[str, Dict[str, str]] = {name: {} for name in names_by_id.values()}
    for v in ClinicalFeatureValue.find_by_clinical_feature_definition_ids(
        list(names_by_id)
    ):
        values[names_by_id[v.clinical_feature_definition_id]][v.patient_id] = v.value
    return values


def delete_extraction_rows(db, extraction_id: int):
    """Same deletion order as routes.features.cancel_extraction."""
    params = {"extraction_id": extraction_id}
    db.session.execute(
        db.text(
            "DELETE FROM feature_value WHERE feature_extraction_task_id IN "
            "(SELECT id FROM feature_extraction_task "
            "WHERE feature_extraction_id = :extraction_id)"
        ),
        params,
    )
    db.session.execute(
        db.text(
            "DELETE FROM feature_extraction_task "
            "WHERE feature_extraction_id = :extraction_id"
        ),
        params,
    )
    db.session.execute(
        db.text("DELETE FROM feature_extraction WHERE id = :extraction_id"), params
    )


def extraction_problem(db, extraction_id: int, expected_tasks: int) -> Optional[str]:
    """Why the extraction can't be kept as the only one, or None if it's complete.

    An extraction row is saved before its tasks are created, so a failed or
    still-running extraction can be the latest one. Require one task per
    reference patient, each with the same (non-zero) number of feature values.
    """
    counts = [
        row[1]
        for row in db.session.execute(
            db.text(
                "SELECT t.id, COUNT(fv.id) FROM feature_extraction_task t "
                "LEFT JOIN feature_value fv ON fv.feature_extraction_task_id = t.id "
                "WHERE t.feature_extraction_id = :extraction_id GROUP BY t.id"
            ),
            {"extraction_id": extraction_id},
        )
    ]
    if len(counts) != expected_tasks:
        return f"has {len(counts)} tasks, expected {expected_tasks}"
    empty = sum(c == 0 for c in counts)
    if empty:
        return f"{empty} tasks without feature values"
    if len(set(counts)) > 1:
        return (
            f"tasks have different numbers of feature values "
            f"({min(counts)} to {max(counts)})"
        )
    return None


def prepare_user(report: UserReport, refs: dict, dry_run: bool):
    from quantimage2_backend_common.models import (
        FeatureCollection,
        FeatureExtraction,
        Model,
        db,
    )

    user_id = report.user_id
    if report.album_id is None:
        latest_any = (
            FeatureExtraction.query.filter(FeatureExtraction.user_id == user_id)
            .order_by(db.desc(FeatureExtraction.id))
            .first()
        )
        if latest_any is None:
            report.errors.append("no feature extraction, cannot find the album")
            return
        report.album_id = latest_any.album_id
    album_id = report.album_id

    # Checks (report only)
    category = find_outcome_category(album_id, user_id, refs["outcome_name"])
    if category is None:
        report.outcome_issues.append(
            f"outcome '{refs['outcome_name']}' not found"
            if refs["outcome_name"]
            else "no selected outcome (set outcome_name in the config)"
        )
    else:
        labels = {l.patient_id: l.label_content or {} for l in category.labels}
        report.outcome_issues += [
            f"[{category.name}] {i}" for i in compare_outcomes(refs["outcomes"], labels)
        ]

    if refs["clinical"] is not None:
        report.clinical_issues += compare_clinical_features(
            refs["clinical"], load_clinical_feature_values(album_id, user_id)
        )

    extractions = (
        FeatureExtraction.query.filter(
            FeatureExtraction.user_id == user_id,
            FeatureExtraction.album_id == album_id,
        )
        .order_by(db.desc(FeatureExtraction.id))
        .all()
    )
    if not extractions:
        report.errors.append("no feature extraction on this album")
    latest, older = (extractions[0], extractions[1:]) if extractions else (None, [])

    # Safety check before deleting anything (dry run included): the extraction
    # we keep must be complete, otherwise skip this user entirely.
    if latest is not None:
        problem = extraction_problem(db, latest.id, len(refs["outcomes"]))
        if problem:
            report.extraction_id = latest.id
            report.errors.append(
                f"latest extraction {latest.id} {problem}: user skipped, "
                f"nothing deleted"
            )
            return
    report.extraction_id = latest.id if latest else None

    # Cleaning
    models = Model.query.filter(
        Model.user_id == user_id, Model.album_id == album_id
    ).all()
    collections = FeatureCollection.query.filter(
        FeatureCollection.feature_extraction_id.in_([e.id for e in extractions])
    ).all()
    report.deleted_models = len(models)
    report.deleted_collections = len(collections)
    report.deleted_extractions = len(older)

    if dry_run:
        report.split_set = latest is not None
        return

    model_paths = [m.model_path for m in models]
    for m in models:
        db.session.delete(m)
    for c in collections:
        db.session.delete(c)
    db.session.flush()
    older_ids = [e.id for e in older]
    # Expunge before the raw DELETEs so the ORM doesn't try to touch them later
    for e in older:
        db.session.expunge(e)
    for extraction_id in older_ids:
        delete_extraction_rows(db, extraction_id)

    if latest is not None:
        latest.data_splitting_type = DATA_SPLITTING_TYPES.TRAINTESTSPLIT.value
        latest.train_test_split_type = TRAIN_TEST_SPLIT_TYPES.MANUAL.value
        latest.training_patients = refs["training"]
        latest.test_patients = refs["test"]

    db.session.commit()
    report.split_set = latest is not None

    # Files only once the DB changes are committed
    for path in model_paths:
        try:
            os.unlink(path)
        except OSError:
            pass
    for extraction_id in older_ids:
        shutil.rmtree(
            f"{FEATURES_CACHE_BASE_DIR}/extraction-{extraction_id}", ignore_errors=True
        )


# --------------------------------------------------------------------------- #
# Keycloak
# --------------------------------------------------------------------------- #


def keycloak_user_ids(emails: List[str], args) -> Dict[str, Optional[str]]:
    from keycloak import KeycloakAdmin

    password = os.environ.get("KEYCLOAK_ADMIN_PASSWORD") or getpass.getpass(
        f"Keycloak password for '{args.keycloak_admin}': "
    )
    admin = KeycloakAdmin(
        server_url=args.keycloak_url,
        username=args.keycloak_admin,
        password=password,
        realm_name=args.keycloak_realm,
        user_realm_name="master",
    )
    ids = {}
    for email in emails:
        users = [
            u
            for u in admin.get_users({"email": email, "exact": True})
            if u.get("email", "").lower() == email.lower()
        ]
        ids[email] = users[0]["id"] if len(users) == 1 else None
    return ids


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def load_config(path: Path) -> Tuple[List[dict], dict]:
    config = yaml.safe_load(path.read_text())
    base = path.parent

    outcomes = load_reference_outcomes(base / config["outcomes_csv"])
    clinical = None
    if config.get("clinical_features_csv"):
        clinical = read_csv_as_strings(base / config["clinical_features_csv"])
        # The app only stores clinical features of the album's patients
        clinical = clinical[clinical[PATIENT_ID].isin(outcomes)].reset_index(drop=True)
    training, test = compute_split(
        list(outcomes), load_test_patients(base / config["test_patients_csv"])
    )
    refs = {
        "outcomes": outcomes,
        "outcome_name": config.get("outcome_name"),
        "clinical": clinical,
        "training": training,
        "test": test,
    }

    return config["users"], refs


def print_report(reports: List[UserReport], dry_run: bool):
    verb = "would delete" if dry_run else "deleted"
    print()
    for r in reports:
        status = "OK" if r.ok else "FAIL"
        print(f"[{status}] {r.email}  album={r.album_id}  user_id={r.user_id}")
        for issue in r.errors:
            print(f"    error: {issue}")
        for issue in r.outcome_issues:
            print(f"    outcomes: {issue}")
        for issue in r.clinical_issues:
            print(f"    clinical: {issue}")
        if r.user_id:
            print(
                f"    {verb}: {r.deleted_models} models, "
                f"{r.deleted_collections} collections, "
                f"{r.deleted_extractions} older extractions"
            )
        if r.split_set:
            print(
                f"    split on extraction {r.extraction_id}: "
                f"{'would be set' if dry_run else 'set'}"
            )
    failed = sum(not r.ok for r in reports)
    print(f"\n{len(reports) - failed}/{len(reports)} users OK")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare class users for a new session (see module docstring)."
    )
    parser.add_argument("config", type=Path, help="YAML config file")
    parser.add_argument(
        "--dry-run", action="store_true", help="report only, change nothing"
    )
    parser.add_argument("--yes", action="store_true", help="skip the confirmation")
    parser.add_argument("--keycloak-url", default=os.environ.get("KEYCLOAK_BASE_URL"))
    parser.add_argument(
        "--keycloak-realm", default=os.environ.get("KEYCLOAK_REALM_NAME")
    )
    parser.add_argument("--keycloak-admin", default="kcadmin")
    args = parser.parse_args(argv)

    users, refs = load_config(args.config)
    print(
        f"{len(users)} users, {len(refs['training'])} training / "
        f"{len(refs['test'])} test patients"
    )

    if not args.dry_run and not args.yes:
        answer = input(
            "This deletes all models, collections and older extractions of these "
            "users. Continue? [y/N] "
        )
        if answer.strip().lower() != "y":
            return 1

    user_ids = keycloak_user_ids([u["email"] for u in users], args)

    from quantimage2_backend_common.flask_init import create_app
    from quantimage2_backend_common.models import db

    app = create_app()
    reports = []
    with app.app_context():
        for u in users:
            report = UserReport(email=u["email"], album_id=u.get("album_id"))
            report.user_id = user_ids.get(u["email"])
            reports.append(report)
            if report.user_id is None:
                report.errors.append("user not found in Keycloak")
                continue
            try:
                prepare_user(report, refs, args.dry_run)
            except Exception as e:
                db.session.rollback()
                report.errors.append(f"{type(e).__name__}: {e} (nothing changed)")
                report.deleted_models = report.deleted_collections = 0
                report.deleted_extractions = 0
                report.split_set = False
            finally:
                if args.dry_run:
                    db.session.rollback()

    print_report(reports, args.dry_run)
    return 0 if all(r.ok for r in reports) else 2


if __name__ == "__main__":
    sys.exit(main())
