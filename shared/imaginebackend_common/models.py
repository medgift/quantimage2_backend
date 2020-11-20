import decimal, datetime
import pandas

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import ForeignKey, Table, Column, Integer
from ttictoc import tic, toc

from sqlalchemy.dialects.mysql import LONGTEXT


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
    def delete_by_id(cls, id):
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
        instance = db.session.query(cls).filter_by(**criteria).one_or_none()
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


# Link between a feature extraction and feature families
class FeatureExtractionFamily(BaseModelAssociation, db.Model):
    # Left
    feature_family_id = db.Column(
        db.Integer, db.ForeignKey("feature_family.id"), primary_key=True,
    )

    # Right
    feature_extraction_id = db.Column(
        db.Integer, db.ForeignKey("feature_extraction.id"), primary_key=True,
    )

    # Extra Data
    family_config_path = db.Column(db.String(255))

    # Relationships
    feature_extraction = db.relationship("FeatureExtraction", back_populates="families")
    feature_family = db.relationship("FeatureFamily", back_populates="extractions")

    def to_dict(self):
        return {
            "feature_family": self.feature_family.to_dict(),
            "family_config_path": self.family_config_path,
        }


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


# Global family of features (Intensity, Texture, ...)
class FeatureFamily(BaseModel, db.Model):
    def __init__(self, name, config_path):
        self.name = name
        self.config_path = config_path

    # Name of the family
    name = db.Column(db.String(255), nullable=False, unique=True)

    # Path to JSON configuration file of the family
    config_path = db.Column(db.String(255), nullable=False, unique=True)

    # Association to FeatureExtractionFamily
    extractions = db.relationship(
        "FeatureExtractionFamily", back_populates="feature_family"
    )

    # Association to FeatureDefinition
    feature_definitions = db.relationship(
        "FeatureDefinition", back_populates="feature_family"
    )

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

    # Families selected for this extraction
    families = db.relationship(
        "FeatureExtractionFamily", back_populates="feature_extraction"
    )

    # Tasks for this feature extraction
    tasks = db.relationship("FeatureExtractionTask")

    # Models for this feature extraction
    models = db.relationship("Model")

    # Collections for this feature extraction
    collections = db.relationship("FeatureCollection")

    def feature_names(self):
        feature_names = []

        for feature_extraction_family in self.families:
            family_definitions = (
                feature_extraction_family.feature_family.feature_definitions
            )
            feature_names.extend(list(map(lambda fd: fd.name, family_definitions)))

        return feature_names

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "user_id": self.user_id,
            "album_id": self.album_id,
            "families": list(
                map(
                    lambda feature_extraction_family: feature_extraction_family.to_dict(),
                    self.families,
                )
            ),
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


# A specific feature extraction task for a given study
class FeatureExtractionTask(BaseModel, db.Model):
    def __init__(
        self, feature_extraction_id, study_uid, feature_family_id, task_id,
    ):
        self.feature_extraction_id = feature_extraction_id
        self.study_uid = study_uid
        self.feature_family_id = feature_family_id
        self.task_id = task_id

    # Kheops Study UID
    study_uid = db.Column(db.String(255), nullable=False)

    # Celery task ID to get information about the status etc.
    task_id = db.Column(db.String(255), nullable=True)

    # Associate feature extraction task with a feature family
    feature_family_id = db.Column(db.Integer, ForeignKey("feature_family.id"))
    feature_family = db.relationship("FeatureFamily", lazy="joined")

    # Associate feature extraction task with a feature extraction
    feature_extraction_id = db.Column(db.Integer, ForeignKey("feature_extraction.id"))
    feature_extraction = db.relationship(
        "FeatureExtraction", back_populates="tasks", lazy="joined"
    )

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
            "feature_family_id": self.feature_family_id,
            "study_uid": self.study_uid,
            "task_id": self.task_id,
        }


# One type of feature that is associated to a given family
class FeatureDefinition(BaseModel, db.Model):
    def __init__(self, name, feature_family_id):
        self.name = name
        self.feature_family_id = feature_family_id

    # Name of the feature
    name = db.Column(db.String(255), nullable=False, unique=False)

    # Relationships
    feature_family_id = db.Column(db.Integer, ForeignKey("feature_family.id"))
    feature_family = db.relationship(
        "FeatureFamily", back_populates="feature_definitions", lazy="joined"
    )

    @classmethod
    def find_by_name(cls, feature_names):
        feature_definitions = cls.query.filter(cls.name.in_(feature_names)).all()

        return feature_definitions


# class FeatureCollectionValue(BaseModelAssociation, db.Model):
#     # Left
#     feature_collection_id = db.Column(
#         db.Integer, db.ForeignKey("feature_collection.id"), primary_key=True,
#     )
#
#     # Right
#     feature_value_id = db.Column(
#         db.Integer, db.ForeignKey("feature_value.id"), primary_key=True,
#     )
#
#     # TODO : Extra Data (could contain modified value e.g.)
#
#     # Relationships
#     feature_collection = db.relationship("FeatureCollection", back_populates="values")
#     feature_value = db.relationship("FeatureValue", back_populates="collections")


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
    def get_for_collection(cls, collection):
        names = []
        features_formatted = []

        for i in collection.values:
            features_formatted.append(i.to_formatted_dict())
            if i.feature_definition.name not in names:
                names.append(i.feature_definition.name)

        return features_formatted, names

    @classmethod
    def get_for_extraction(cls, feature_extraction):
        extraction_task_ids = list(map(lambda task: task.id, feature_extraction.tasks))

        tic()
        instances = cls.query.filter(
            FeatureValue.feature_extraction_task_id.in_(extraction_task_ids)
        ).all()
        elapsed_db = toc()
        print("DB query took:", elapsed_db)

        names = []
        features_formatted = []

        for i in instances:
            features_formatted.append(i.to_formatted_dict())
            if i.feature_definition.name not in names:
                names.append(i.feature_definition.name)

        return features_formatted, names

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
    feature_definition = db.relationship("FeatureDefinition", lazy="joined")
    feature_extraction_task_id = db.Column(
        db.Integer, ForeignKey("feature_extraction_task.id")
    )
    feature_extraction_task = db.relationship("FeatureExtractionTask", lazy="joined")
    modality_id = db.Column(db.Integer, ForeignKey("modality.id"))
    modality = db.relationship("Modality", lazy="joined")
    roi_id = db.Column(db.Integer, ForeignKey("roi.id"))
    roi = db.relationship("ROI", lazy="joined")

    # Association to FeatureCollectionValues
    # collections = db.relationship(
    #     "FeatureCollectionValue", back_populates="feature_value"
    # )

    def to_formatted_dict(self):
        return {
            "study_uid": self.feature_extraction_task.study_uid,
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

    # Name of the collection
    name = db.Column(db.String(255), nullable=False, unique=False)

    # Association to a FeatureExtraction
    feature_extraction_id = db.Column(db.Integer, ForeignKey("feature_extraction.id"))
    feature_extraction = db.relationship(
        "FeatureExtraction", back_populates="collections", lazy="joined"
    )

    # Association to FeatureCollectionValues
    values = db.relationship("FeatureValue", secondary="feature_collection_value")

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
        return {"collection": self.to_dict(), "modalities": modalities, "rois": rois}


# Machine learning model
class Model(BaseModel, db.Model):
    def __init__(
        self,
        name,
        type,
        algorithm,
        validation_strategy,
        feature_selection,
        feature_names,
        modalities,
        rois,
        model_path,
        user_id,
        album_id,
        feature_extraction_id,
    ):
        self.name = name
        self.type = type
        self.algorithm = algorithm
        self.validation_strategy = validation_strategy
        self.feature_selection = feature_selection
        self.feature_names = feature_names
        self.modalities = modalities
        self.rois = rois
        self.model_path = model_path
        self.user_id = user_id
        self.album_id = album_id
        self.feature_extraction_id = feature_extraction_id

    # Name of the model
    name = db.Column(db.String(255), nullable=False, unique=False)

    # Type of the model (classification, survival)
    type = db.Column(db.String(255), nullable=False, unique=False)

    # Algorithm used for the model (linear regression, random forests, SVM, etc.)
    algorithm = db.Column(db.String(255), nullable=False, unique=False)

    # Validation strategy used for the model (Stratified K-Fold, Train/Test split, etc.)
    validation_strategy = db.Column(db.String(255), nullable=True, unique=False)

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
    feature_extraction = db.relationship(
        "FeatureExtraction", back_populates="models", lazy="joined"
    )

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
            "feature_selection": self.feature_selection,
            "feature_names": self.feature_names,
            "modalities": self.modalities,
            "rois": self.rois,
            "model_path": self.model_path,
            "user_id": self.user_id,
            "album_id": self.album_id,
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
