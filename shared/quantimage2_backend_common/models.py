from __future__ import annotations

import decimal, datetime
from typing import List

import sqlalchemy
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import ForeignKey, Table, Column, Integer
from sqlalchemy.orm import joinedload
from ttictoc import tic, toc

from quantimage2_backend_common.const import (
    featureIDMatcher,
    DATA_SPLITTING_TYPES,
    TRAIN_TEST_SPLIT_TYPES,
)
from quantimage2_backend_common.kheops_utils import dicomFields

db = SQLAlchemy()


def alchemyencoder(obj):
    if isinstance(obj, datetime.date):
        return obj.isoformat() + "Z"
    elif isinstance(obj, decimal.Decimal):
        return float(obj)


# Base Model with basic methods (find by id, find all, save to DB, etc.)
class BaseModel(db.Model):
    __abstract__ = True

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    @classmethod
    def find_by_id(cls, id):
        instance = db.session.query(cls).filter_by(id=id).one_or_none()
        return instance

    @classmethod
    def delete_by_id(cls, id, options=None):

        if options is not None:
            instance = (
                db.session.query(cls).options(options).filter_by(id=id).one_or_none()
            )
        else:
            instance = db.session.query(cls).filter_by(id=id).one_or_none()

        if instance:
            db.session.delete(instance)
            db.session.commit()
            return instance
        else:
            return None

    @classmethod
    def find_all(cls):
        instances = db.session.query(cls).all()
        return instances

    @classmethod
    def get_or_create(cls, criteria=None, defaults=None):
        instance = (
            db.session.query(cls).filter_by(**criteria).with_for_update().one_or_none()
        )
        if instance:
            return instance, False
        else:
            criteria.update(defaults)
            instance = cls(**criteria)
            db.session.add(instance)
            db.session.commit()
            return instance, True

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def flush_to_db(self):
        db.session.add(self)
        db.session.flush()

    def update(self, **kwargs):
        for key, value in kwargs.items():
            # Check that object has this property, and disregard creation & updated dates
            try:
                if hasattr(self, key) and key not in ["created_at", "updated_at"]:
                    if value is not None:
                        setattr(self, key, value)
                    else:
                        setattr(self, key, sqlalchemy.null())
            except TypeError:
                # Swallow unsuccesful attributions (like associations)
                print("impossible to set attribute " + key)
        self.save_to_db()


class BaseModelAssociation(object):
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def flush_to_db(self):
        db.session.add(self)
        db.session.flush()


# Modality (CT, PET, MRI, ...)
class Modality(BaseModel, db.Model):
    def __init__(self, name):
        self.name = name

    name = db.Column(db.String(255), nullable=False, unique=True)


# ROI (GTV_T, GTV_N, GTV_L, ...)
class ROI(BaseModel, db.Model):
    __tablename__ = "roi"

    def __init__(self, name):
        self.name = name

    name = db.Column(
        db.String(255, collation="latin1_bin"), nullable=False, unique=True
    )


# Feature extraction preset (YAML file)
class FeaturePreset(BaseModel, db.Model):
    def __init__(self, name, config_path):
        self.name = name
        self.config_path = config_path

    name = db.Column(db.String(255), nullable=False, unique=True)

    # Path to JSON configuration file
    config_path = db.Column(db.String(255), nullable=False, unique=True)

    @classmethod
    def find_by_name(cls, name):
        instance = cls.query.filter_by(name=name).one_or_none()
        return instance

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "name": self.name,
            "config_path": self.config_path,
        }


# One instance of running a feature extraction on an album
class FeatureExtraction(BaseModel, db.Model):
    def __init__(self, user_id, album_id):
        self.user_id = user_id
        self.album_id = album_id

    # Keycloak ID of the user that extracted the features
    user_id = db.Column(db.String(255), nullable=False)

    # Kheops album ID for the extraction
    album_id = db.Column(db.String(255), nullable=True)

    # Celery Result ID
    result_id = db.Column(db.String(255))

    # Extraction configuration file
    config_file = db.Column(db.String(255))

    # Data splitting type
    data_splitting_type = db.Column(
        db.String(255), default=DATA_SPLITTING_TYPES.TRAINTESTSPLIT.value
    )

    # Train/test split type (auto/manual)
    train_test_split_type = db.Column(
        db.String(255), default=TRAIN_TEST_SPLIT_TYPES.AUTO.value
    )

    # Train/test patients
    training_patients = db.Column(db.JSON, nullable=True)
    test_patients = db.Column(db.JSON, nullable=True)

    # Association to FeatureDefinition
    feature_definitions = db.relationship(
        "FeatureDefinition", secondary="feature_extraction_definition"
    )

    # Association to Modality
    modalities = db.relationship("Modality", secondary="feature_extraction_modality")

    # Association to ROI
    rois = db.relationship("ROI", secondary="feature_extraction_roi")

    # Tasks for this feature extraction
    tasks = db.relationship("FeatureExtractionTask")

    # Models for this feature extraction
    models = db.relationship("Model")

    # Collections for this feature extraction
    collections = db.relationship("FeatureCollection")

    def feature_names(self):
        feature_names = self.feature_definitions(
            list(map(lambda fd: fd.name, self.feature_definitions))
        )

        return sorted(list(feature_names))

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "user_id": self.user_id,
            "album_id": self.album_id,
            "feature_definitions": list(
                map(
                    lambda feature_definition: feature_definition.name,
                    self.feature_definitions,
                )
            ),
            "data_splitting_type": self.data_splitting_type,
            "train_test_split_type": self.train_test_split_type,
            "training_patients": self.training_patients,
            "test_patients": self.test_patients,
            "modalities": list(map(lambda modality: modality.name, self.modalities)),
            "rois": list(map(lambda roi: roi.name, self.rois)),
            "tasks": list(
                map(
                    lambda feature_extraction_task: feature_extraction_task.to_dict(),
                    self.tasks,
                )
            ),
            "result_id": self.result_id,
        }

    @classmethod
    def find_latest_by_user_and_album_id(cls, user_id, album_id):
        query_result = (
            cls.query.filter(cls.user_id == user_id, cls.album_id == album_id)
            .order_by(db.desc(FeatureExtraction.id))
            .first()
        )

        return query_result

    @classmethod
    def find_by_user(cls, user_id):
        query_results = cls.query.filter(cls.user_id == user_id).all()

        return query_results

    @classmethod
    def find_by_id_populate(cls, id):
        populated_extraction = (
            cls.query.filter(cls.id == id)
            .options(
                joinedload(cls.tasks)
                .subqueryload(FeatureExtractionTask.feature_values)
                .options(
                    joinedload(FeatureValue.modality),
                    joinedload(FeatureValue.roi),
                    joinedload(FeatureValue.feature_definition),
                )
            )
            .one_or_none()
        )

        return populated_extraction


# A link between a feature definition and a feature extraction
feature_extraction_definition = Table(
    "feature_extraction_definition",
    db.metadata,
    Column(
        "feature_extraction_id",
        Integer,
        ForeignKey("feature_extraction.id", ondelete="CASCADE", onupdate="CASCADE"),
    ),
    Column(
        "feature_definition_id",
        Integer,
        ForeignKey("feature_definition.id", ondelete="CASCADE", onupdate="CASCADE"),
    ),
)

# A link between a modality and a feature extraction
feature_extraction_modality = Table(
    "feature_extraction_modality",
    db.metadata,
    Column(
        "feature_extraction_id",
        Integer,
        ForeignKey("feature_extraction.id", ondelete="CASCADE", onupdate="CASCADE"),
    ),
    Column(
        "modality_id",
        Integer,
        ForeignKey("modality.id", ondelete="CASCADE", onupdate="CASCADE"),
    ),
)

# A link between a ROI and a feature extraction
feature_extraction_roi = Table(
    "feature_extraction_roi",
    db.metadata,
    Column(
        "feature_extraction_id",
        Integer,
        ForeignKey("feature_extraction.id", ondelete="CASCADE", onupdate="CASCADE"),
    ),
    Column(
        "roi_id", Integer, ForeignKey("roi.id", ondelete="CASCADE", onupdate="CASCADE")
    ),
)


# A specific feature extraction task for a given study
class FeatureExtractionTask(BaseModel, db.Model):
    def __init__(
        self,
        feature_extraction_id,
        study_uid,
        task_id,
    ):
        self.feature_extraction_id = feature_extraction_id
        self.study_uid = study_uid
        self.task_id = task_id

    # Kheops Study UID
    study_uid = db.Column(db.String(255), nullable=False)

    # Celery task ID to get information about the status etc.
    task_id = db.Column(db.String(255), nullable=True)

    # Associate feature extraction task with a feature extraction
    feature_extraction_id = db.Column(
        db.Integer,
        ForeignKey("feature_extraction.id", ondelete="CASCADE", onupdate="CASCADE"),
    )
    feature_extraction = db.relationship("FeatureExtraction", back_populates="tasks")

    # Associate feature extraction task with feature values
    feature_values = db.relationship("FeatureValue")

    @classmethod
    def find_by_user(cls, user_id):
        query_results = (
            cls.query.join(FeatureExtraction)
            .filter(FeatureExtraction.user_id == user_id)
            .all()
        )

        return query_results

    @classmethod
    def find_by_user_and_study(cls, user_id, study_uid):
        query_results = (
            cls.query.join(FeatureExtraction)
            .filter(FeatureExtraction.user_id == user_id, cls.study_uid == study_uid)
            .all()
        )

        return query_results

    @classmethod
    def find_latest_by_user_and_study(cls, user_id, study_uid):
        latest_task = (
            cls.query.join(FeatureExtraction)
            .filter(FeatureExtraction.user_id == user_id, cls.study_uid == study_uid)
            .order_by(db.desc(FeatureExtractionTask.id))
            .first()
        )

        return latest_task

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "feature_extraction_id": self.feature_extraction_id,
            "study_uid": self.study_uid,
            "task_id": self.task_id,
        }


# One type of feature
class FeatureDefinition(BaseModel, db.Model):
    def __init__(self, name):
        self.name = name

    # Name of the feature
    name = db.Column(db.String(255), nullable=False, unique=False)

    @classmethod
    def find_by_name(cls, feature_names):
        feature_definitions = cls.query.filter(cls.name.in_(feature_names)).all()

        return feature_definitions

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "name": self.name,
        }


# The value of a given feature
class FeatureValue(BaseModel, db.Model):
    def __init__(
        self,
        value,
        feature_definition_id,
        feature_extraction_task_id,
        modality_id,
        roi_id,
    ):
        self.value = value
        self.feature_definition_id = feature_definition_id
        self.feature_extraction_task_id = feature_extraction_task_id
        self.modality_id = modality_id
        self.roi_id = roi_id

    # Simplified class for bulk loading of features
    class SimpleFeatureValue(object):
        def __init__(
            self,
            feature_extraction_task_id,
            modality_id,
            roi_id,
            feature_definition_id,
            value,
        ):
            self.feature_extraction_task_id = feature_extraction_task_id
            self.modality_id = modality_id
            self.roi_id = roi_id
            self.feature_definition_id = feature_definition_id
            self.value = value

    @classmethod
    def save_features_batch(cls, feature_instances):
        db.session.bulk_insert_mappings(cls, feature_instances)
        db.session.commit()

    @classmethod
    def get_for_collection(cls, collection):

        modalities_map, rois_map, definitions_map = get_modality_roi_feature_maps()
        feature_tasks_map = get_tasks_map(collection.feature_extraction_id)

        tic()
        feature_collection_values = cls.fetch_feature_collection_values(collection.id)
        elapsed = toc()
        print("Getting the feature collection values from the DB took", elapsed)

        tic()
        features_formatted = [
            {
                "study_uid": feature_tasks_map[
                    feature_value.feature_extraction_task_id
                ],
                "modality": modalities_map[feature_value.modality_id],
                "roi": rois_map[feature_value.roi_id],
                "name": definitions_map[feature_value.feature_definition_id],
                "value": feature_value.value,
            }
            for feature_value in feature_collection_values
        ]
        names = list(dict.fromkeys([f["name"] for f in features_formatted]))
        elapsed = toc()
        print("Formatting the feature collection value took", elapsed)

        return features_formatted, names

    @classmethod
    def get_for_extraction(cls, feature_extraction):

        modalities_map, rois_map, definitions_map = get_modality_roi_feature_maps()
        feature_tasks_map = get_tasks_map(feature_extraction.id)

        feature_extraction_task_ids = list(feature_tasks_map.keys())

        tic()
        feature_values = cls.fetch_feature_values(feature_extraction_task_ids)
        elapsed = toc()
        print("Getting the feature values from the DB took", elapsed)

        tic()
        features_formatted = [
            {
                "study_uid": feature_tasks_map[
                    feature_value.feature_extraction_task_id
                ],
                "modality": modalities_map[feature_value.modality_id],
                "roi": rois_map[feature_value.roi_id],
                "name": definitions_map[feature_value.feature_definition_id],
                "value": feature_value.value,
            }
            for feature_value in feature_values
        ]
        names = list(dict.fromkeys([f["name"] for f in features_formatted]))

        elapsed = toc()
        print("Formatting the DB features took", elapsed)

        return features_formatted, names

    @classmethod
    def fetch_feature_values(cls, feature_extraction_task_ids):
        # Low-level DBAPI fetchall()
        compiled = (
            cls.__table__.select()
            .with_only_columns(
                [
                    cls.__table__.c.feature_extraction_task_id,
                    cls.__table__.c.modality_id,
                    cls.__table__.c.roi_id,
                    cls.__table__.c.feature_definition_id,
                    cls.__table__.c.value,
                ]
            )
            .where(
                cls.__table__.c.feature_extraction_task_id.in_(
                    feature_extraction_task_ids
                )
            )
            .compile(dialect=db.engine.dialect, compile_kwargs={"literal_binds": True})
        )

        return cls.fetch_values_compiled_query(compiled)

    @classmethod
    def fetch_feature_collection_values(cls, feature_collection_id):
        collection = FeatureCollection.find_by_id(feature_collection_id)
        modalities_map, rois_map, features_map = get_modality_roi_feature_maps_by_name()
        tasks_map = get_tasks_map(collection.feature_extraction_id)

        task_ids = list(tasks_map.keys())

        conditions = []

        for feature_id in collection.feature_ids:
            modality_name, roi_name, feature_name = featureIDMatcher.match(
                feature_id
            ).groups()
            conditions.append(
                (
                    modalities_map[modality_name],
                    rois_map[roi_name],
                    features_map[feature_name],
                )
            )

        # Low-level DBAPI fetchall()
        compiled = (
            cls.__table__.select()
            .with_only_columns(
                [
                    cls.__table__.c.feature_extraction_task_id,
                    cls.__table__.c.modality_id,
                    cls.__table__.c.roi_id,
                    cls.__table__.c.feature_definition_id,
                    cls.__table__.c.value,
                ]
            )
            .where(cls.__table__.c.feature_extraction_task_id.in_(task_ids))
            .where(
                db.tuple_(
                    cls.__table__.c.modality_id,
                    cls.__table__.c.roi_id,
                    cls.__table__.c.feature_definition_id,
                ).in_(conditions)
            )
            .compile(dialect=db.engine.dialect, compile_kwargs={"literal_binds": True})
        )

        return cls.fetch_values_compiled_query(compiled)

    @classmethod
    def find_id_by_collection_criteria_new(
        cls, feature_extraction, feature_studies, feature_ids, patients
    ):
        db_modality_map, db_roi_map, db_feature_map = get_modality_roi_feature_maps()

        # Map Patient IDs to corresponding study IDs
        study_uids = []
        for patient in patients:
            study_uid = next(
                s[dicomFields.STUDY_UID][dicomFields.VALUE][0]
                for s in feature_studies
                if s[dicomFields.PATIENT_ID][dicomFields.VALUE][0] == patient
            )
            study_uids.append(study_uid)

        # Get all task IDs related to the corresponding study UIDs
        task_ids = []
        for task in feature_extraction.tasks:
            if task.study_uid in study_uids:
                task_ids.append(task.id)

        # Transform feature IDs to triplets of Modality ID, ROI ID & Feature Definition ID
        conditions = []
        for feature_id in feature_ids:
            modality_name, roi_name, feature_name = featureIDMatcher.match(
                feature_id
            ).groups()
            conditions.append(
                (
                    db_modality_map[modality_name],
                    db_roi_map[roi_name],
                    db_feature_map[feature_name],
                )
            )

        feature_value_ids = (
            cls.query.with_entities(cls.id)
            .filter(
                cls.feature_extraction_task_id.in_(task_ids),
                db.tuple_(cls.modality_id, cls.roi_id, cls.feature_definition_id).in_(
                    conditions
                ),
            )
            .all()
        )

        return feature_value_ids

    @classmethod
    def find_by_collection_criteria(
        cls,
        feature_extraction,
        feature_studies,
        modalities,
        rois,
        patients,
        feature_names,
    ):
        # Get necessary info from the DB
        db_modalities = Modality.find_all()
        db_rois = ROI.find_all()

        # Map Patient IDs to corresponding study IDs
        study_uids = []
        for patient in patients:
            study_uid = next(
                s[dicomFields.STUDY_UID][dicomFields.VALUE][0]
                for s in feature_studies
                if s[dicomFields.PATIENT_ID][dicomFields.VALUE][0] == patient
            )
            study_uids.append(study_uid)

        # Get all task IDs related to the corresponding study UIDs
        task_ids = []
        for task in feature_extraction.tasks:
            if task.study_uid in study_uids:
                task_ids.append(task.id)

        # Map modality names to modality instance IDs
        modality_ids = []
        for modality in modalities:
            modality_id = next(
                db_modality.id
                for db_modality in db_modalities
                if db_modality.name == modality
            )
            modality_ids.append(modality_id)

        # Map ROI names to ROI instance IDs
        roi_ids = []
        for roi in rois:
            roi_id = next(db_roi.id for db_roi in db_rois if db_roi.name == roi)
            roi_ids.append(roi_id)

        # Map feature names to Feature Definition instance IDs
        feature_definitions = FeatureDefinition.find_by_name(feature_names)
        feature_definition_ids = list(map(lambda fd: fd.id, feature_definitions))

        feature_values = cls.query.filter(
            cls.feature_extraction_task_id.in_(task_ids),
            cls.modality_id.in_(modality_ids),
            cls.roi_id.in_(roi_ids),
            cls.feature_definition_id.in_(feature_definition_ids),
        ).all()

        return feature_values

    @classmethod
    def find_by_tasks_modality_roi_features(
        cls, task_ids, modality_id, roi_id, feature_definition_ids
    ):
        feature_values = cls.query.filter(
            cls.modality_id == modality_id,
            cls.roi_id == roi_id,
            cls.feature_definition_id.in_(feature_definition_ids),
            cls.feature_extraction_task_id.in_(task_ids),
        ).all()

        return feature_values

    # Value of the feature
    value = db.Column(db.Float)

    # Relationships
    feature_definition_id = db.Column(
        db.Integer,
        ForeignKey("feature_definition.id", ondelete="CASCADE", onupdate="CASCADE"),
    )
    feature_definition = db.relationship("FeatureDefinition")
    feature_extraction_task_id = db.Column(
        db.Integer,
        ForeignKey(
            "feature_extraction_task.id", ondelete="CASCADE", onupdate="CASCADE"
        ),
    )
    feature_extraction_task = db.relationship("FeatureExtractionTask")
    modality_id = db.Column(
        db.Integer, ForeignKey("modality.id", ondelete="CASCADE", onupdate="CASCADE")
    )
    modality = db.relationship("Modality")
    roi_id = db.Column(
        db.Integer, ForeignKey("roi.id", ondelete="CASCADE", onupdate="CASCADE")
    )
    roi = db.relationship("ROI")

    def to_formatted_dict(self, study_uid=None):
        return {
            "study_uid": study_uid
            if study_uid
            else self.feature_extraction_task.study_uid,
            "modality": self.modality.name,
            "roi": self.roi.name,
            "name": self.feature_definition.name,
            "value": self.value,
        }

    @classmethod
    def fetch_values_compiled_query(cls, compiled):
        sql = str(compiled)

        conn = db.engine.raw_connection()
        cursor = conn.cursor()
        cursor.execute(sql)

        feature_values = []
        for row in cursor.fetchall():
            # ensure that we fully fetch!
            feature_value = cls.SimpleFeatureValue(
                feature_extraction_task_id=row[0],
                modality_id=row[1],
                roi_id=row[2],
                feature_definition_id=row[3],
                value=row[4],
            )
            feature_values.append(feature_value)

        conn.close()

        return feature_values


# Customized Feature Collection (filtered rows & columns so far)
class FeatureCollection(BaseModel, db.Model):
    def __init__(
        self,
        name,
        feature_extraction_id,
        feature_ids,
        data_splitting_type,
        train_test_split_type,
        training_patients,
        test_patients,
    ):
        self.name = name
        self.feature_extraction_id = feature_extraction_id
        self.feature_ids = feature_ids
        self.data_splitting_type = data_splitting_type
        self.train_test_split_type = train_test_split_type
        self.training_patients = training_patients
        self.test_patients = test_patients

    @classmethod
    def find_by_extraction(cls, extraction_id):

        query = cls.query.filter(
            cls.feature_extraction_id == extraction_id,
        )

        feature_collections = query.all()

        return feature_collections

    # Name of the collection
    name = db.Column(db.String(255), nullable=False, unique=False)

    # Feature IDs of this collection
    feature_ids = db.Column(db.JSON, nullable=False, unique=False)

    # Data splitting type
    data_splitting_type = db.Column(
        db.String(255), default=DATA_SPLITTING_TYPES.TRAINTESTSPLIT.value
    )

    # Train/test split type (auto/manual)
    train_test_split_type = db.Column(
        db.String(255), default=TRAIN_TEST_SPLIT_TYPES.AUTO.value
    )

    # Train/test patients
    training_patients = db.Column(db.JSON, nullable=True)
    test_patients = db.Column(db.JSON, nullable=True)

    # Association to a FeatureExtraction
    feature_extraction_id = db.Column(
        db.Integer,
        ForeignKey("feature_extraction.id", ondelete="CASCADE", onupdate="CASCADE"),
    )
    feature_extraction = db.relationship(
        "FeatureExtraction", back_populates="collections"
    )

    # Association to Models
    models = db.relationship("Model", cascade="all, delete")

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "name": self.name,
            "feature_extraction_id": self.feature_extraction_id,
            "data_splitting_type": self.data_splitting_type,
            "train_test_split_type": self.train_test_split_type,
            "feature_ids": self.feature_ids,
            "training_patients": self.training_patients,
            "test_patients": self.test_patients,
        }

    def format_collection(self, with_values=False):

        tic()
        if with_values:
            (modalities, rois, features) = self.get_modalities_rois_features()
            result = {
                "collection": self.to_dict(),
                "modalities": modalities,
                "rois": rois,
                "features": features,
            }
        else:
            result = {"collection": self.to_dict()}
        elapsed = toc()
        print("Formatting collection took", elapsed)

        return result

    def get_modalities_rois_features(self):

        feature_ids = self.feature_ids

        # Count present modalities, ROIs & feature names based on
        # feature IDs

        modalities = set()
        rois = set()
        features = set()

        for feature_id in feature_ids:
            modality_name, roi_name, feature_name = featureIDMatcher.match(
                feature_id
            ).groups()

            modalities.add(modality_name)
            rois.add(roi_name)
            features.add(feature_name)

        return list(modalities), list(rois), list(features)


class ClinicalFeatureDefinition(BaseModel, db.Model):

    __tablename__ = "clinical_feature_definition"

    def __init__(self, name, feat_type, encoding, user_id):
        self.name = name
        self.user_id = user_id
        self.feat_type = feat_type
        self.encoding = encoding


    # Name of the feature
    name = db.Column(db.String(255), nullable=False, unique=False)

    feat_type = db.Column(db.String(255), nullable=False, unique=False)

    encoding = db.Column(db.String(255), nullable=False, unique=False)

    # User who created the clinical feature category
    user_id = db.Column(db.String(255), nullable=False, unique=False)

    @classmethod
    def find_by_name(cls, clinical_feature_names, user_id):
        clinical_feature_definitions = cls.query.filter(cls.name.in_(clinical_feature_names), cls.user_id.in_([user_id])).all()

        return clinical_feature_definitions
    
    @classmethod
    def find_by_user_id(cls, user_id) -> List[ClinicalFeatureDefinition]:
        return cls.query.filter(cls.user_id == user_id).all()

    @classmethod
    def insert(cls, name, feat_type, encoding, user_id):
        exisiting_definitions = cls.query.filter(cls.name == name, cls.user_id == user_id).all() # we enable updating the values of the feature
        if len(exisiting_definitions) > 0:
            new_definition = exisiting_definitions[0].update(feature_type=feat_type, encoding=encoding)
            db.session.commit()
            return new_definition
        else:
            clin_feat_def = ClinicalFeatureDefinition(name, feat_type, encoding, user_id)
            clin_feat_def.save_to_db()
            db.session.commit()
            return clin_feat_def


    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "Name": self.name,
            "Type": self.feat_type,
            "Encoding": self.encoding,
            "user_id": self.user_id,
        }

    @classmethod
    def delete_by_user_id(cls, user_id: str):
        cls.query.filter(cls.user_id == user_id).delete()
        db.session.commit()
    
# The value of a given feature
class ClinicalFeatureValue(BaseModel, db.Model):

    __tablenane__ = "clinical_feature_value"

    def __init__(
        self,
        value,
        clinical_feature_definition_id,
        patient_id,
    ):
        self.value = value
        self.clinical_feature_definition_id = clinical_feature_definition_id
        self.patient_id = patient_id

    # Value of the feature
    value = db.Column(db.String(255), nullable=False, unique=False)

    patient_id = db.Column(db.String(255), nullable=False, unique=False)

    # Relationships
    clinical_feature_definition_id = db.Column(
        db.Integer,
        ForeignKey("clinical_feature_definition.id", ondelete="CASCADE", onupdate="CASCADE"),
    )
    clinical_feature_definition = db.relationship("ClinicalFeatureDefinition")

    @classmethod
    def find_by_clinical_feature_definition_ids(cls, clinical_feature_definition_ids: List[str]):
        return cls.query.filter(cls.clinical_feature_definition_id.in_(clinical_feature_definition_ids)).all()
    
    @classmethod
    def insert_value(cls, value, clinical_feature_definition_id, patient_id):
        clinical_feature_value = cls.query.filter(cls.clinical_feature_definition_id == clinical_feature_definition_id, cls.patient_id == patient_id, cls.value == value).first()
        value = cls(value, clinical_feature_definition_id, patient_id)
        if not clinical_feature_value:
            value.save_to_db()
        else:
            clinical_feature_value.update(value=value)
        db.session.commit()
        return value

    @classmethod
    def find_by_patient_ids(cls, patient_ids, user_id):
        return db.session.query(ClinicalFeatureValue, ClinicalFeatureDefinition).join(ClinicalFeatureDefinition).filter(
            cls.patient_id.in_(patient_ids),
            ClinicalFeatureDefinition.user_id == user_id,
        ).all()
            

    @classmethod
    def delete_by_user_id(cls, user_id: str):
        cls.query.join(ClinicalFeatureDefinition).filter(ClinicalFeatureDefinition.user_id == user_id).delete()
        db.session.commit()

    def to_dict(self):
        return {
            "clinical_feature_definition_id": self.clinical_feature_definition_id,
            "value": self.value,
            "patient_id": self.patient_id,
        }
    
    
def process_query_single_column(query):
    compiled = query.compile(
        dialect=db.engine.dialect, compile_kwargs={"literal_binds": True}
    )

    sql = str(compiled)

    conn = db.engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute(sql)

    results = []
    for row in cursor.fetchall():
        results.append(row[0])

    conn.close()

    return results


# Machine learning model
class Model(BaseModel, db.Model):
    def __init__(
        self,
        name,
        best_algorithm,
        data_splitting_type,
        train_test_split_type,
        training_validation,
        test_validation,
        best_data_normalization,
        feature_selection,
        feature_names,
        training_patient_ids,
        test_patient_ids,
        model_path,
        training_metrics,
        test_metrics,
        user_id,
        album_id,
        label_category_id,
        feature_extraction_id,
        feature_collection_id=None,
    ):
        self.name = name
        self.best_algorithm = best_algorithm
        self.data_splitting_type = data_splitting_type
        self.train_test_split_type = train_test_split_type
        self.training_validation = training_validation
        self.test_validation = test_validation
        self.best_data_normalization = best_data_normalization
        self.feature_selection = feature_selection
        self.feature_names = feature_names
        self.training_patient_ids = training_patient_ids
        self.test_patient_ids = test_patient_ids
        self.model_path = model_path
        self.training_metrics = training_metrics
        self.test_metrics = test_metrics
        self.user_id = user_id
        self.album_id = album_id
        self.label_category_id = label_category_id
        self.feature_extraction_id = feature_extraction_id
        if feature_collection_id is not None:
            self.feature_collection_id = feature_collection_id

    # Name of the model
    name = db.Column(db.String(255), nullable=False, unique=False)

    # Algorithm used for the model (linear regression, random forests, SVM, etc.)
    best_algorithm = db.Column(db.String(255), nullable=False, unique=False)

    # Type of model validation (Full-Dataset CV, Train/Test split, etc.)
    data_splitting_type = db.Column(
        db.String(255),
        nullable=True,
        unique=False,
    )

    # Train/test split type (auto/manual)
    train_test_split_type = db.Column(
        db.String(255), default=TRAIN_TEST_SPLIT_TYPES.AUTO.value
    )

    # Validation strategy used for training the model (Stratified K-Fold, etc.)
    training_validation = db.Column(db.String(255), nullable=True, unique=False)

    # Validation strategy (data splitting) used on the test set (Bootstrap, etc.)
    test_validation = db.Column(db.String(255), nullable=True, unique=False)

    # Data normalization used for the model (L2 norm, standardization, etc.)
    best_data_normalization = db.Column(db.String(255), nullable=True, unique=False)

    # Feature selection used for the model (variance thresholding, correlation, etc.)
    feature_selection = db.Column(db.String(255), nullable=True, unique=False)

    # Feature names that were finally used for the model (after selection)
    feature_names = db.Column(db.JSON, nullable=True, unique=False)

    # Patients used for training the model
    training_patient_ids = db.Column(db.JSON, nullable=True, unique=False)

    # Patients used for testing the model (OPTIONAL)
    test_patient_ids = db.Column(db.JSON, nullable=True, unique=False)

    # Path to pickled version of the model
    model_path = db.Column(db.String(255), nullable=False, unique=True)

    # Model metrics (JSON) - Training
    training_metrics = db.Column(db.JSON, nullable=True, unique=False)

    # Model metrics (JSON) - Test
    test_metrics = db.Column(db.JSON, nullable=True, unique=False)

    # User who created the model
    user_id = db.Column(db.String(255), nullable=False, unique=False)

    # Album on which the model was created
    album_id = db.Column(db.String(255), nullable=False, unique=False)

    # Relationship - Label Category
    label_category_id = db.Column(
        db.Integer,
        ForeignKey("label_category.id", ondelete="CASCADE", onupdate="CASCADE"),
    )
    label_category = db.relationship("LabelCategory")

    # Relationship - Feature Extraction
    feature_extraction_id = db.Column(
        db.Integer,
        ForeignKey("feature_extraction.id", ondelete="CASCADE", onupdate="CASCADE"),
    )
    feature_extraction = db.relationship("FeatureExtraction", back_populates="models")

    # Relationship - Collection
    feature_collection_id = db.Column(
        db.Integer,
        ForeignKey("feature_collection.id", ondelete="CASCADE", onupdate="CASCADE"),
    )
    feature_collection = db.relationship("FeatureCollection", back_populates="models")

    @classmethod
    def find_by_album(cls, album_id, user_id):
        instances = cls.query.filter_by(album_id=album_id, user_id=user_id).all()
        return instances

    @classmethod
    def find_by_user(cls, user_id):
        instances = cls.query.filter_by(user_id=user_id).all()
        return instances

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "name": self.name,
            "best_algorithm": self.best_algorithm,
            "data_splitting_type": self.data_splitting_type,
            "train_test_split_type": self.train_test_split_type,
            "training_validation": self.training_validation,
            "test_validation": self.test_validation,
            "best_data_normalization": self.best_data_normalization,
            "feature_selection": self.feature_selection,
            "feature_names": self.feature_names,
            "training_patient_ids": self.training_patient_ids,
            "test_patient_ids": self.test_patient_ids,
            "model_path": self.model_path,
            "training_metrics": self.training_metrics,
            "test_metrics": self.test_metrics,
            "user_id": self.user_id,
            "album_id": self.album_id,
            "feature_extraction_id": self.feature_extraction_id,
            "feature_collection_id": self.feature_collection_id,
            "type": self.label_category.label_type,
            "label_category": self.label_category.name,
        }


# Data Label Category
class LabelCategory(BaseModel, db.Model):
    def __init__(self, album_id, label_type, name, user_id, pos_label=None):
        self.album_id = album_id
        self.label_type = label_type
        self.name = name
        self.user_id = user_id
        self.pos_label = pos_label

    # Album ID
    album_id = db.Column(db.String(255), nullable=False, unique=False)

    # Label Type (Classification, Survival, ...)
    label_type = db.Column(db.String(255), nullable=False, unique=False)

    # Label Category nam (PLC Status, ...)
    name = db.Column(db.String(255), nullable=False, unique=False)

    # The positive label for classification (in case we use textual labels)
    pos_label = db.Column(db.String(255), nullable=True)

    # User who created the label category
    user_id = db.Column(db.String(255), nullable=False, unique=False)

    # The labels of this category
    labels = db.relationship("Label")

    @classmethod
    def find_by_album(cls, album_id, user_id):
        instances = (
            cls.query.filter_by(album_id=album_id, user_id=user_id)
            .options(joinedload(cls.labels))
            .all()
        )
        return instances

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "album_id": self.album_id,
            "user_id": self.user_id,
            "label_type": self.label_type,
            "name": self.name,
            "labels": list(map(lambda label: label.to_dict(), self.labels)),
            "pos_label": self.pos_label,
        }


# Data Label
class Label(BaseModel, db.Model):
    def __init__(self, label_category_id, patient_id, label_content):
        self.label_category_id = label_category_id
        self.patient_id = patient_id
        self.label_content = label_content

    # Label Category
    label_category_id = db.Column(
        db.Integer,
        ForeignKey("label_category.id", ondelete="CASCADE", onupdate="CASCADE"),
    )
    label_category = db.relationship("LabelCategory", back_populates="labels")

    # Patient ID
    patient_id = db.Column(db.String(255), nullable=False, unique=False)

    # Label Content
    label_content = db.Column(db.JSON, nullable=False, unique=False)

    @classmethod
    def find_by_label_category(cls, label_category_id):
        instances = cls.query.filter_by(label_category_id=label_category_id).all()

        return instances

    @classmethod
    def save_labels(cls, label_category_id, labels_to_save):
        tic()

        # Delete any existing labels
        deleted_existing_labels = cls.query.filter_by(
            label_category_id=label_category_id
        ).delete()

        # Save labels in bulk
        db.session.bulk_save_objects(labels_to_save)
        db.session.commit()
        elapsed = toc()

        # Fetch newly saved labels
        labels = cls.find_by_label_category(label_category_id)

        print("Saving labels took", elapsed)

        return labels

    @classmethod
    def save_label(cls, label_category_id, patient_id, label_content):
        old_instance, created = Label.get_or_create(
            criteria={
                "label_category_id": label_category_id,
                "patient_id": patient_id,
            },
            defaults={
                "label_category_id": label_category_id,
                "patient_id": patient_id,
                "label_content": label_content,
            },
        )
        if not created:
            old_instance.label_content = label_content
            old_instance.save_to_db()

        return old_instance

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "patient_id": self.patient_id,
            "label_content": self.label_content,
        }


# Navigation History
class NavigationHistory(BaseModel, db.Model):
    def __init__(self, path, user_id):
        self.path = path
        self.user_id = user_id

    # Path
    path = db.Column(db.String(255), nullable=False, unique=False)

    # User who created the label
    user_id = db.Column(db.String(255), nullable=False, unique=False)

    @classmethod
    def find_by_user(cls, user_id):
        instances = cls.query.filter_by(user_id=user_id).all()
        return instances

    @classmethod
    def create_entry(cls, path, user_id):
        entry = NavigationHistory(path, user_id)
        entry.save_to_db()

        return entry

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "path": self.path,
            "user_id": self.user_id,
        }


class Album(BaseModel, db.Model):
    def __init__(self, album_id):
        self.album_id = album_id

    album_id = db.Column(db.String(255), nullable=False, unique=True)
    rois = db.Column(db.JSON, nullable=True, unique=False)
    studies = db.Column(db.JSON, nullable=True, unique=False)

    @classmethod
    def save_rois(cls, album_id, rois):
        album = cls.find_by_album_id(album_id)
        album.rois = rois
        album.save_to_db()

        return album

    @classmethod
    def find_by_album_id(cls, album_id):
        (instance, created) = cls.get_or_create(
            criteria={
                "album_id": album_id,
            },
            defaults={"album_id": album_id},
        )
        return instance

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "album_id": self.album_id,
        }


class AlbumOutcome(BaseModel, db.Model):
    def __init__(self, album_id, user_id, outcome_id):
        self.album_id = album_id
        self.user_id = user_id
        self.outcome_id = outcome_id

    album_id = db.Column(db.String(255), nullable=False, unique=False)
    user_id = db.Column(db.String(255), nullable=True, unique=False)
    outcome_id = db.Column(
        db.Integer,
        ForeignKey("label_category.id", ondelete="CASCADE", onupdate="CASCADE"),
    )
    outcome = db.relationship("LabelCategory")

    @classmethod
    def find_by_album_user_id(cls, album_id, user_id):
        instance = cls.query.filter_by(user_id=user_id, album_id=album_id).one_or_none()
        return instance

    @classmethod
    def save_current_outcome(cls, album_id, user_id, outcome_id):
        (instance, created) = cls.get_or_create(
            criteria={"album_id": album_id, "user_id": user_id},
            defaults={
                "album_id": album_id,
                "user_id": user_id,
                "outcome_id": outcome_id,
            },
        )
        instance.outcome_id = outcome_id
        instance.save_to_db()

        return instance

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "album_id": self.album_id,
            "user_id": self.user_id,
            "outcome_id": self.outcome_id,
        }


def get_modality_roi_feature_maps():
    # Get necessary info from the DB
    db_modalities = Modality.find_all()
    db_rois = ROI.find_all()
    db_feature_definitions = FeatureDefinition.find_all()

    # Map names to DB IDs
    db_modality_map = {
        db_modality.id: db_modality.name for db_modality in db_modalities
    }
    db_roi_map = {db_roi.id: db_roi.name for db_roi in db_rois}
    db_feature_map = {
        db_feature.id: db_feature.name for db_feature in db_feature_definitions
    }

    return db_modality_map, db_roi_map, db_feature_map


def get_modality_roi_feature_maps_by_name():
    # Get necessary info from the DB
    db_modalities = Modality.find_all()
    db_rois = ROI.find_all()
    db_feature_definitions = FeatureDefinition.find_all()

    # Map names to DB IDs
    db_modality_map = {
        db_modality.name: db_modality.id for db_modality in db_modalities
    }
    db_roi_map = {db_roi.name: db_roi.id for db_roi in db_rois}
    db_feature_map = {
        db_feature.name: db_feature.id for db_feature in db_feature_definitions
    }

    return db_modality_map, db_roi_map, db_feature_map


def get_tasks_map(extraction_id):
    # Get Study UIDs for all the feature extraction tasks
    feature_extraction_tasks = FeatureExtractionTask.query.filter_by(
        feature_extraction_id=extraction_id
    )

    feature_tasks_map = {task.id: task.study_uid for task in feature_extraction_tasks}

    return feature_tasks_map
