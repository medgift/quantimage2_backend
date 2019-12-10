import decimal, datetime

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import ForeignKey

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
        instance = cls.query.filter_by(id=id).one_or_none()
        return instance

    @classmethod
    def find_all(cls):
        instances = cls.query.all()
        return instances

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def update(self, **kwargs):
        db.session.query(type(self)).filter(type(self).id == self.id).update(**kwargs)
        db.session.commit()


class BaseModelAssociation(object):
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()


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


# One instance of running a feature extraction on a study or an album
class FeatureExtraction(BaseModel, db.Model):
    def __init__(self, user_id, album_id, study_uid=None):
        self.user_id = user_id
        self.album_id = album_id
        self.study_uid = study_uid

    # Keycloak ID of the user that extracted the features
    user_id = db.Column(db.String(255), nullable=False)

    # Kheops album ID for the extraction
    album_id = db.Column(db.String(255), nullable=True)

    # Kheops study ID for the extraction
    study_uid = db.Column(db.String(255), nullable=True)

    # Celery Result ID
    result_id = db.Column(db.String(255))

    # Families selected for this extraction
    families = db.relationship(
        "FeatureExtractionFamily", back_populates="feature_extraction"
    )

    # Tasks for this feature extraction
    tasks = db.relationship("FeatureExtractionTask")

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "user_id": self.user_id,
            "album_id": self.album_id,
            "study_uid": self.study_uid,
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
    def find_by_user_and_study_uid(cls, user_id, study_uid):
        query_results = cls.query.filter(
            cls.user_id == user_id, cls.study_uid == study_uid
        ).all()

        return query_results

    @classmethod
    def find_latest_by_user_and_study_uid(cls, user_id, study_uid):
        query_result = (
            cls.query.filter(cls.user_id == user_id, cls.study_uid == study_uid)
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
        self,
        feature_extraction_id,
        study_uid,
        feature_family_id,
        task_id,
        features_path,
    ):
        self.feature_extraction_id = feature_extraction_id
        self.study_uid = study_uid
        self.feature_family_id = feature_family_id
        self.task_id = task_id
        self.features_path = features_path

    # Kheops Study UID
    study_uid = db.Column(db.String(255), nullable=False)

    # Celery task ID to get information about the status etc.
    task_id = db.Column(db.String(255), nullable=True)

    # Path to the extracted feature file
    features_path = db.Column(db.String(255), nullable=False)

    # Associate feature extraction task with a feature family
    feature_family_id = db.Column(db.Integer, ForeignKey("feature_family.id"))
    feature_family = db.relationship("FeatureFamily", lazy="joined")

    # Associate feature extraction task with a feature extraction
    feature_extraction_id = db.Column(db.Integer, ForeignKey("feature_extraction.id"))
    feature_extraction = db.relationship(
        "FeatureExtraction", back_populates="tasks", lazy="joined"
    )

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

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "feature_extraction_id": self.feature_extraction_id,
            "feature_family_id": self.feature_family_id,
            "study_uid": self.study_uid,
            "task_id": self.task_id,
            "features_path": self.features_path,
        }


def get_or_create(model, **kwargs):
    instance = db.session.query(model).filter_by(**kwargs).one_or_none()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        db.session.add(instance)
        db.session.commit()
        return instance
