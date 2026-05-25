from __future__ import annotations

import json

import pytest
from flask import Flask
from sqlalchemy.exc import IntegrityError

from app.extensions import db


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    @event.listens_for(Engine, "connect")
    def _fk_on(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    with app.app_context():
        from app import models  # noqa: F401  确保 Template 被注册
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


def test_template_can_be_created(app):
    from app.models import FaceProfile, Template
    p = FaceProfile(name="alice")
    db.session.add(p)
    db.session.flush()
    t = Template(
        profile_id=p.id,
        strategy="mean_all",
        vector_json=json.dumps([0.1] * 512),
        source_count=10,
    )
    db.session.add(t)
    db.session.commit()
    assert t.id is not None
    assert p.templates[0].strategy == "mean_all"


def test_unique_profile_strategy(app):
    from app.models import FaceProfile, Template
    p = FaceProfile(name="bob")
    db.session.add(p)
    db.session.flush()
    db.session.add(Template(
        profile_id=p.id, strategy="mean_all",
        vector_json="[]", source_count=1,
    ))
    db.session.commit()
    db.session.add(Template(
        profile_id=p.id, strategy="mean_all",
        vector_json="[]", source_count=1,
    ))
    with pytest.raises(IntegrityError):
        db.session.commit()
    db.session.rollback()


def test_cascade_delete_removes_templates(app):
    from app.models import FaceProfile, Template
    p = FaceProfile(name="carol")
    db.session.add(p)
    db.session.flush()
    db.session.add(Template(
        profile_id=p.id, strategy="all_vectors",
        vector_json="[[0.1]]", source_count=2,
    ))
    db.session.commit()
    db.session.delete(p)
    db.session.commit()
    assert Template.query.count() == 0
