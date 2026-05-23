import os

from flask import Flask

from .config import Config
from .extensions import cors, db
from .routes.face import face_bp
from .routes.records import records_bp
from .routes.stats import stats_bp
from .utils.response import api_error


def _bootstrap_adaface_env() -> None:
    """在首次 import face_alignment 之前设置 MTCNN 设备（与 AdaFace/face_alignment/align.py 一致）。"""
    if os.environ.get("ADAFACE_DEVICE"):
        return
    try:
        import torch

        os.environ["ADAFACE_DEVICE"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        os.environ["ADAFACE_DEVICE"] = "cpu"


def create_app() -> Flask:
    _bootstrap_adaface_env()
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    cors.init_app(app, resources={r"/api/*": {"origins": "*"}})

    app.register_blueprint(face_bp, url_prefix="/api")
    app.register_blueprint(records_bp, url_prefix="/api")
    app.register_blueprint(stats_bp, url_prefix="/api")

    @app.get("/api/health")
    def health():
        return {"code": 0, "message": "ok", "data": {"status": "healthy"}}

    @app.errorhandler(404)
    def not_found(_err):
        return api_error(message="接口不存在", http_status=404)

    with app.app_context():
        from . import models  # noqa: F401

        db.create_all()

    return app
