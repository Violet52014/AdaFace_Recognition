import json as _json
from datetime import datetime
from typing import Optional

from .extensions import db


class FaceProfile(db.Model):
    __tablename__ = "face_profiles"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False, unique=True)
    class_name = db.Column(db.String(64), nullable=True)
    avatar = db.Column(db.String(255), nullable=True)
    feature_vector = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class RecognitionRecord(db.Model):
    __tablename__ = "recognition_records"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(16), nullable=False)  # success / error
    class_name = db.Column(db.String(64), nullable=True)
    avatar = db.Column(db.String(255), nullable=True)
    description = db.Column(db.String(255), nullable=True)
    recognized_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_api_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "class": self.class_name or "",
            "avatar": self.avatar or "",
            "description": self.description or "",
            "time": self.recognized_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


class Template(db.Model):
    __tablename__ = "templates"

    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(
        db.Integer,
        db.ForeignKey("face_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    strategy = db.Column(db.String(32), nullable=False, index=True)
    vector_json = db.Column(db.Text, nullable=False)
    source_count = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    profile = db.relationship(
        "FaceProfile",
        backref=db.backref("templates", cascade="all, delete-orphan", lazy="select"),
    )

    __table_args__ = (
        db.UniqueConstraint("profile_id", "strategy", name="uq_profile_strategy"),
    )


def parse_template_vectors(vector_json: Optional[str]):
    """把 Template.vector_json 解析为 (M, 512) numpy float32 数组。

    支持两种 JSON 形态：1D 列表（M=1）或 2D 列表（M>1）。返回值始终二维。
    """
    import numpy as np
    if not vector_json:
        return np.zeros((0, 512), dtype=np.float32)
    arr = _json.loads(vector_json)
    np_arr = np.asarray(arr, dtype=np.float32)
    if np_arr.ndim == 1:
        np_arr = np_arr[None, :]
    return np_arr
