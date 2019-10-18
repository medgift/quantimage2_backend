import decimal, datetime
import json

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import ForeignKey
from sqlalchemy.orm import joinedload

db = SQLAlchemy()


def alchemyencoder(obj):
    if isinstance(obj, datetime.date):
        return obj.isoformat() + "Z"
    elif isinstance(obj, decimal.Decimal):
        return float(obj)


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

    # def to_json(self):
    #    json_object = self.__dict__.copy()
    #    del json_object["_sa_instance_state"]
    #    return json.dumps(json_object, default=alchemyencoder)


class Study(BaseModel, db.Model):
    uid = db.Column(db.String(255), nullable=False)
    features = db.relationship("FeatureExtraction", backref="study", lazy=True)

    def __init__(self, uid):
        self.uid = uid

    @classmethod
    def find_by_uid(cls, uid):
        return cls.query.filter_by(uid=uid).one_or_none()


# Global family of features (Intensity, Texture, ...)
class FeatureFamily(BaseModel, db.Model):
    def __init__(self, name, config_path):
        self.name = name
        self.config_path = config_path

    # Name of the family
    name = db.Column(db.String(255), nullable=False)

    # Path to JSON configuration file of the family
    config_path = db.Column(db.String(255), nullable=False)

    # Extractions done for this family
    extractions = db.relationship("FeatureExtraction")


# One set of extracted features for a given feature family & configuration
class FeatureExtraction(BaseModel, db.Model):
    def __init__(self, features_path, config_path, user_id, study_id):
        self.features_path = features_path
        self.config_path = config_path
        self.user_id = user_id
        self.study_id = study_id

    # Path to the extracted feature file
    features_path = db.Column(db.String(255), nullable=False)

    # Path of JSON file describing the selected features / parameters
    config_path = db.Column(db.String(255), nullable=False)

    # Keycloak ID of the user that extracted the features
    user_id = db.Column(db.String(255), nullable=False)

    # Celery task ID to get information about the status etc.
    task_id = db.Column(db.String(255), nullable=True)

    # Associate feature extraction with a Study in the DB
    study_id = db.Column(db.Integer, ForeignKey("study.id"))

    # Associate feature extraction with a FeatureFamily
    feature_family_id = db.Column(db.Integer, ForeignKey("feature_family.id"))

    @classmethod
    def find_by_path(cls, path):

        feature = (
            cls.query.options(joinedload("study"), joinedload("feature_family"))
            .filter_by(path=path)
            .one_or_none()
        )

        return feature

    @classmethod
    def find_by_user(cls, user_id):
        query_results = cls.query.join(Study).filter(cls.user_id == user_id).all()

        return query_results

    @classmethod
    def find_by_user_and_study_uid(cls, user_id, study_uid):

        study = Study.find_by_uid(study_uid)

        if study is None:
            return None

        query_results = (
            cls.query.join(Study)
            .filter(cls.user_id == user_id, Study.id == study.id)
            .all()
        )

        return query_results

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "name": self.name,
            "path": self.path,
            "user_id": self.user_id,
            "study_id": self.study_id,
            "task_id": self.task_id,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), default=alchemyencoder)


def get_or_create(model, **kwargs):
    instance = db.session.query(model).filter_by(**kwargs).one_or_none()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        db.session.add(instance)
        db.session.commit()
        return instance
