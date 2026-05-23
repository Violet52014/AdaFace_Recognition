from datetime import datetime

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
