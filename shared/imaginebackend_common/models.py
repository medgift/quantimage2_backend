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

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def flush_to_db(self):
        db.session.add(self)
        db.session.flush()

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
            "features_path": self.features_path,
        }


# Machine learning model
class Model(BaseModel, db.Model):
    def __init__(self, name, type, algorithm, model_path, user_id, album_id):
        self.name = name
        self.type = type
        self.algorithm = algorithm
        self.model_path = model_path
        self.user_id = user_id
        self.album_id = album_id

    # Name of the model
    name = db.Column(db.String(255), nullable=False, unique=False)

    # Type of the model (classification, survival)
    type = db.Column(db.String(255), nullable=False, unique=False)

    # Algorithm used for the model (linear regression, random forests, SVM, etc.)
    algorithm = db.Column(db.String(255), nullable=False, unique=False)

    # Path to pickled version of the model
    model_path = db.Column(db.String(255), nullable=False, unique=True)

    # User who created the model
    user_id = db.Column(db.String(255), nullable=False, unique=False)

    # Album on which the album was created
    album_id = db.Column(db.String(255), nullable=False, unique=False)

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
            "model_path": self.model_path,
            "user_id": self.user_id,
            "album_id": self.album_id,
        }


# Patient/Region outcome
class Label(BaseModel, db.Model):
    def __init__(self, album_id, patient_id, roi, outcome, user_id):
        self.album_id = album_id
        self.patient_id = patient_id
        self.roi = roi
        self.outcome = outcome
        self.user_id = user_id

    # Album ID
    album_id = db.Column(db.String(255), nullable=False, unique=False)

    # Patient ID
    patient_id = db.Column(db.String(255), nullable=False, unique=False)

    # ROI Name
    roi = db.Column(db.String(255), nullable=False, unique=False)

    # Outcome
    outcome = db.Column(db.String(255), nullable=False, unique=False)

    # User who created the model
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
    def save_label(cls, album_id, patient_id, roi, outcome, user_id):
        result = get_or_create(
            Label,
            criteria={
                "album_id": album_id,
                "patient_id": patient_id,
                "roi": roi,
                "user_id": user_id,
            },
            defaults={"outcome": outcome},
        )
        if result["created"] == False:
            old_instance = result["instance"]
            old_instance.outcome = outcome
            old_instance.save_to_db()

        return result["instance"]

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "album_id": self.album_id,
            "patient_id": self.patient_id,
            "roi": self.roi,
            "outcome": self.outcome,
            "user_id": self.user_id,
        }


def get_or_create(model, criteria=None, defaults=None):
    instance = db.session.query(model).filter_by(**criteria).one_or_none()
    if instance:
        return {"instance": instance, "created": False}
    else:
        criteria.update(defaults)
        instance = model(**criteria)
        db.session.add(instance)
        db.session.commit()
        return {"instance": instance, "created": True}
