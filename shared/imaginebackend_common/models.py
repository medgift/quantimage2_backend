import decimal, datetime
import re

import pandas

from flask_sqlalchemy import SQLAlchemy
from more_itertools import first_true
from sqlalchemy import ForeignKey, Table, Column, Integer
from sqlalchemy.orm import joinedload, load_only
from ttictoc import tic, toc

from sqlalchemy.dialects.mysql import LONGTEXT

from imaginebackend_common.kheops_utils import dicomFields

db = SQLAlchemy()


def alchemyencoder(obj):
    if isinstance(obj, datetime.date):
        return obj.isoformat() + "Z"
    elif isinstance(obj, decimal.Decimal):
        return float(obj)


# Base Model with basic methods (find by id, find all, save to DB, etc.)
class BaseModel(object):
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
                    setattr(self, key, value)
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

    name = db.Column(db.String(255), nullable=False, unique=True)


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
    Column("feature_extraction_id", Integer, ForeignKey("feature_extraction.id")),
    Column("feature_definition_id", Integer, ForeignKey("feature_definition.id")),
)

# A link between a modality and a feature extraction
feature_extraction_modality = Table(
    "feature_extraction_modality",
    db.metadata,
    Column("feature_extraction_id", Integer, ForeignKey("feature_extraction.id")),
    Column("modality_id", Integer, ForeignKey("modality.id")),
)

# A link between a ROI and a feature extraction
feature_extraction_roi = Table(
    "feature_extraction_roi",
    db.metadata,
    Column("feature_extraction_id", Integer, ForeignKey("feature_extraction.id")),
    Column("roi_id", Integer, ForeignKey("roi.id")),
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
    feature_extraction_id = db.Column(db.Integer, ForeignKey("feature_extraction.id"))
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

    @classmethod
    def save_features_batch(cls, feature_instances):
        db.session.bulk_insert_mappings(cls, feature_instances)
        db.session.commit()

    @classmethod
    def get_for_collection(cls, collection):

        modalities = Modality.find_all()
        rois = ROI.find_all()
        definitions = FeatureDefinition.find_all()

        modalities_map = {modality.id: modality.name for modality in modalities}
        rois_map = {roi.id: roi.name for roi in rois}
        definitions_map = {definition.id: definition.name for definition in definitions}

        tic()
        collection_loaded = (
            db.session.query(FeatureCollection)
            .options(
                joinedload(FeatureCollection.values).options(
                    joinedload(FeatureValue.feature_extraction_task).options(
                        load_only("study_uid")
                    ),
                )
            )
            .filter(FeatureCollection.id == collection.id)
            .one_or_none()
        )
        elapsed = toc()
        print("Getting the feature collection values from the DB took", elapsed)

        tic()
        features_formatted = [
            {
                "study_uid": feature_value.feature_extraction_task.study_uid,
                "modality": modalities_map[feature_value.modality_id],
                "roi": rois_map[feature_value.roi_id],
                "name": definitions_map[feature_value.feature_definition_id],
                "value": feature_value.value,
            }
            for feature_value in collection_loaded.values
        ]
        names = list(dict.fromkeys([f["name"] for f in features_formatted]))
        elapsed = toc()
        print("Formatting the feature collection value took", elapsed)

        return features_formatted, names

    @classmethod
    def get_for_extraction(cls, feature_extraction):
        modalities = Modality.find_all()
        rois = ROI.find_all()
        definitions = FeatureDefinition.find_all()
        feature_extraction_tasks = FeatureExtractionTask.query.filter_by(
            feature_extraction_id=feature_extraction.id
        )

        modalities_map = {modality.id: modality.name for modality in modalities}
        rois_map = {roi.id: roi.name for roi in rois}
        definitions_map = {definition.id: definition.name for definition in definitions}
        feature_tasks_map = {
            task.id: task.study_uid for task in feature_extraction_tasks
        }

        feature_extraction_task_ids = list(
            map(
                lambda feature_extraction_task: feature_extraction_task.id,
                feature_extraction_tasks,
            )
        )

        tic()
        feature_values = FeatureValue.query.filter(
            FeatureValue.feature_extraction_task_id.in_(feature_extraction_task_ids)
        ).all()
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
        print("Formatting features took", elapsed)

        return features_formatted, names

    @classmethod
    def find_id_by_collection_criteria_new(
        cls, feature_extraction, feature_studies, feature_ids, patients
    ):
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
        matcher = re.compile(r"(?P<modality>.*?)-(?P<roi>.*?)-(?P<feature>.*)")
        for feature_id in feature_ids:
            modality_name, roi_name, feature_name = matcher.match(feature_id).groups()
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
    feature_definition_id = db.Column(db.Integer, ForeignKey("feature_definition.id"))
    feature_definition = db.relationship("FeatureDefinition")
    feature_extraction_task_id = db.Column(
        db.Integer, ForeignKey("feature_extraction_task.id")
    )
    feature_extraction_task = db.relationship("FeatureExtractionTask")
    modality_id = db.Column(db.Integer, ForeignKey("modality.id"))
    modality = db.relationship("Modality")
    roi_id = db.Column(db.Integer, ForeignKey("roi.id"))
    roi = db.relationship("ROI")

    # Association to FeatureCollectionValues
    # collections = db.relationship(
    #     "FeatureCollectionValue", back_populates="feature_value"
    # )

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


feature_collection_value = Table(
    "feature_collection_value",
    db.metadata,
    Column("feature_collection_id", Integer, ForeignKey("feature_collection.id")),
    Column("feature_value_id", Integer, ForeignKey("feature_value.id")),
)

# Customized Feature Collection (filtered rows & columns so far)
class FeatureCollection(BaseModel, db.Model):
    def __init__(self, name, feature_extraction_id):
        self.name = name
        self.feature_extraction_id = feature_extraction_id

    @classmethod
    def find_by_extraction(cls, extraction_id):
        feature_collections = cls.query.filter(
            cls.feature_extraction_id == extraction_id,
        ).all()

        return feature_collections

    @classmethod
    def save_feature_collection_values_batch(cls, feature_collection_value_instances):
        db.session.connection().execute(
            feature_collection_value.insert(), feature_collection_value_instances
        )
        db.session.commit()

    # Name of the collection
    name = db.Column(db.String(255), nullable=False, unique=False)

    # Association to a FeatureExtraction
    feature_extraction_id = db.Column(db.Integer, ForeignKey("feature_extraction.id"))
    feature_extraction = db.relationship(
        "FeatureExtraction", back_populates="collections"
    )

    # Association to FeatureCollectionValues
    values = db.relationship("FeatureValue", secondary="feature_collection_value")

    # Association to Models
    models = db.relationship("Model")

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "name": self.name,
            "feature_extraction_id": self.feature_extraction_id,
        }

    def format_collection(self):
        modalities = list(set(map(lambda v: v.modality.name, self.values)))
        rois = list(set(map(lambda v: v.roi.name, self.values)))
        features = list(set(map(lambda v: v.feature_definition.name, self.values)))
        return {
            "collection": self.to_dict(),
            "modalities": modalities,
            "rois": rois,
            "features": features,
        }


# Machine learning model
class Model(BaseModel, db.Model):
    def __init__(
        self,
        name,
        type,
        algorithm,
        validation_strategy,
        data_normalization,
        feature_selection,
        feature_names,
        modalities,
        rois,
        model_path,
        user_id,
        album_id,
        feature_extraction_id,
        feature_collection_id=None,
    ):
        self.name = name
        self.type = type
        self.algorithm = algorithm
        self.validation_strategy = validation_strategy
        self.data_normalization = data_normalization
        self.feature_selection = feature_selection
        self.feature_names = feature_names
        self.modalities = modalities
        self.rois = rois
        self.model_path = model_path
        self.user_id = user_id
        self.album_id = album_id
        self.feature_extraction_id = feature_extraction_id
        if feature_collection_id is not None:
            self.feature_collection_id = feature_collection_id

    # Name of the model
    name = db.Column(db.String(255), nullable=False, unique=False)

    # Type of the model (classification, survival)
    type = db.Column(db.String(255), nullable=False, unique=False)

    # Algorithm used for the model (linear regression, random forests, SVM, etc.)
    algorithm = db.Column(db.String(255), nullable=False, unique=False)

    # Validation strategy used for the model (Stratified K-Fold, Train/Test split, etc.)
    validation_strategy = db.Column(db.String(255), nullable=True, unique=False)

    # Data normalization used for the model (L2 norm, standardization, etc.)
    data_normalization = db.Column(db.String(255), nullable=True, unique=False)

    # Feature selection used for the model (variance thresholding, correlation, etc.)
    feature_selection = db.Column(db.String(255), nullable=True, unique=False)

    # Feature names that were finally used for the model (after selection)
    feature_names = db.Column(db.JSON, nullable=True, unique=False)

    # Modalities used for training the model
    modalities = db.Column(db.JSON, nullable=False, unique=False)

    # ROIs used for training the model
    rois = db.Column(db.JSON, nullable=False, unique=False)

    # Path to pickled version of the model
    model_path = db.Column(db.String(255), nullable=False, unique=True)

    # User who created the model
    user_id = db.Column(db.String(255), nullable=False, unique=False)

    # Album on which the model was created
    album_id = db.Column(db.String(255), nullable=False, unique=False)

    # Relationships
    feature_extraction_id = db.Column(db.Integer, ForeignKey("feature_extraction.id"))
    feature_extraction = db.relationship("FeatureExtraction", back_populates="models")

    # Collection
    feature_collection_id = db.Column(db.Integer, ForeignKey("feature_collection.id"))
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
            "type": self.type,
            "algorithm": self.algorithm,
            "validation_strategy": self.validation_strategy,
            "data_normalization": self.data_normalization,
            "feature_selection": self.feature_selection,
            "feature_names": self.feature_names,
            "modalities": self.modalities,
            "rois": self.rois,
            "model_path": self.model_path,
            "user_id": self.user_id,
            "album_id": self.album_id,
            "feature_collection_id": self.feature_collection_id,
        }


# Data Label
class Label(BaseModel, db.Model):
    def __init__(self, album_id, patient_id, label_type, label_content, user_id):
        self.album_id = album_id
        self.patient_id = patient_id
        self.label_type = label_type
        self.label_content = label_content
        self.user_id = user_id

    # Album ID
    album_id = db.Column(db.String(255), nullable=False, unique=False)

    # Patient ID
    patient_id = db.Column(db.String(255), nullable=False, unique=False)

    # Label Type (Classification, Survival, ...)
    label_type = db.Column(db.String(255), nullable=False, unique=False)

    # Label Content
    label_content = db.Column(db.JSON, nullable=False, unique=False)

    # User who created the label
    user_id = db.Column(db.String(255), nullable=False, unique=False)

    @classmethod
    def find_by_album(cls, album_id, user_id, label_type):
        instances = cls.query.filter_by(
            album_id=album_id, user_id=user_id, label_type=label_type
        ).all()
        return instances

    @classmethod
    def find_by_user(cls, user_id):
        instances = cls.query.filter_by(user_id=user_id).all()
        return instances

    @classmethod
    def save_label(cls, album_id, patient_id, label_type, label_content, user_id):
        old_instance, created = Label.get_or_create(
            criteria={
                "album_id": album_id,
                "patient_id": patient_id,
                "label_type": label_type,
                "user_id": user_id,
            },
            defaults={"label_content": label_content},
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
            "album_id": self.album_id,
            "patient_id": self.patient_id,
            "label_type": self.label_type,
            "label_content": self.label_content,
            "user_id": self.user_id,
        }


# Chart Annotation
class Annotation(BaseModel, db.Model):
    def __init__(self, parent_id, album_id, deleted, title, text, lines, user_id):
        self.parent_id = parent_id
        self.album_id = album_id
        self.deleted = deleted
        self.title = title
        self.text = text
        self.lines = lines
        self.user_id = user_id

    # Parent Annotation
    parent_id = db.Column(db.Integer, ForeignKey("annotation.id"))

    # Album ID
    album_id = db.Column(db.String(255), nullable=False, unique=False)

    # Deleted
    deleted = db.Column(db.Boolean, nullable=False, default=False)

    # Title
    title = db.Column(db.String(255), nullable=False)

    # Text
    text = db.Column(db.Text, nullable=False)

    # Lines
    lines = db.Column(LONGTEXT, nullable=True)

    # User who created the label
    user_id = db.Column(db.String(255), nullable=False, unique=False)

    @classmethod
    def find_by_album(cls, album_id, user_id):
        instances = cls.query.filter_by(album_id=album_id, user_id=user_id).all()
        return instances

    @classmethod
    def find_by_user(cls, user_id):
        instances = cls.query.filter_by(user_id=user_id).all()
        return instances

    @classmethod
    def create_annotation(
        cls, album_id, parent_id, deleted, title, text, lines, user_id
    ):

        annotation = Annotation(
            parent_id, album_id, deleted, title, text, lines, user_id
        )
        annotation.save_to_db()

        return annotation

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "album_id": self.album_id,
            "parent_id": self.parent_id,
            "parent": self.parent.to_dict() if self.parent_id else None,
            "deleted": self.deleted,
            "title": self.title,
            "text": self.text,
            "lines": self.lines,
            "user_id": self.user_id,
        }


# Establish link to parent annotation
# Needs to be defined outside because "remote_side" requires a Column-type expression
# which in this case is not possible inside of the Annotation class because the "id"
# is defined in the BaseModel superclass
Annotation.parent = db.relationship(
    "Annotation", uselist=False, remote_side=[Annotation.id], lazy="joined"
)

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
