from __future__ import annotations

import json

import numpy as np
import pytest
from flask import Flask
from sqlalchemy import event
from sqlalchemy.engine import Engine

from app.extensions import db


@event.listens_for(Engine, "connect")
def _fk_on(dbapi_connection, connection_record):
    cur = dbapi_connection.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        from app import models  # noqa: F401
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


def _add_profile_with_template(name: str, vectors: np.ndarray, strategy: str):
    from app.models import FaceProfile, Template
    p = FaceProfile(name=name)
    db.session.add(p)
    db.session.flush()
    if vectors.ndim == 1:
        payload = vectors.tolist()
    else:
        payload = vectors.tolist()
    db.session.add(Template(
        profile_id=p.id, strategy=strategy,
        vector_json=json.dumps(payload),
        source_count=int(vectors.shape[0]) if vectors.ndim == 2 else 1,
    ))
    db.session.commit()
    return p


def test_match_returns_best_profile(app):
    from app.models import FaceProfile
    from app.services.face_service import match_with_templates

    target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    far = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    _add_profile_with_template("alice", target, strategy="mean_all")
    _add_profile_with_template("bob", far, strategy="mean_all")

    profiles = FaceProfile.query.all()
    best, score = match_with_templates(target, profiles, strategy="mean_all")
    assert best.name == "alice"
    assert score == pytest.approx(1.0, abs=1e-6)


def test_match_uses_max_cosine_over_multi_template(app):
    from app.models import FaceProfile
    from app.services.face_service import match_with_templates

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    multi = np.stack([
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),  # 完全匹配
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ], axis=0)
    _add_profile_with_template("alice", multi, strategy="all_vectors")

    profiles = FaceProfile.query.all()
    best, score = match_with_templates(query, profiles, strategy="all_vectors")
    assert best.name == "alice"
    assert score == pytest.approx(1.0, abs=1e-6)


def test_match_skips_profiles_without_strategy(app):
    from app.models import FaceProfile
    from app.services.face_service import match_with_templates

    target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    _add_profile_with_template("alice", target, strategy="mean_all")
    _add_profile_with_template("bob", target, strategy="kmeans_k3")

    profiles = FaceProfile.query.all()
    best, score = match_with_templates(target, profiles, strategy="mean_all")
    assert best.name == "alice"  # bob 的 mean_all 模板不存在，被跳过


def test_match_returns_none_when_no_templates(app):
    from app.models import FaceProfile
    from app.services.face_service import match_with_templates

    target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    _add_profile_with_template("alice", target, strategy="kmeans_k3")
    profiles = FaceProfile.query.all()
    best, score = match_with_templates(target, profiles, strategy="mean_all")
    assert best is None
    assert score == -1.0
