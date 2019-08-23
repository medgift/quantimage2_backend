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
    features = db.relationship("Feature", backref="study", lazy=True)

    def __init__(self, uid):
        self.uid = uid

    @classmethod
    def find_by_uid(cls, uid):
        return cls.query.filter_by(uid=uid).one_or_none()


class Feature(BaseModel, db.Model):
    def __init__(self, name, path, user_id, study_id):
        self.name = name
        self.path = path
        self.user_id = user_id
        self.study_id = study_id

    name = db.Column(db.String(255), nullable=False)
    path = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.String(255), nullable=False)
    study_id = db.Column(db.Integer, ForeignKey("study.id"))
    task_id = db.Column(db.String(255), nullable=True)

    @classmethod
    def find_by_path(cls, path):

        feature = (
            cls.query.options(joinedload("study")).filter_by(path=path).one_or_none()
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
