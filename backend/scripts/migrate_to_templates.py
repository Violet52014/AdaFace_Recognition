"""一次性迁移：扫已存在的 face_profiles.feature_vector，回填 strategy='mean_all' 的 Template 行。

适用场景：升级前已用旧版 build_face_gallery 跑过库的开发者。
对于全新仓库执行 build_face_gallery，本脚本可不跑。

用法（在 backend 下）:
    python scripts/migrate_to_templates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models import FaceProfile, Template  # noqa: E402


def main() -> int:
    app = create_app()
    with app.app_context():
        n_added = 0
        n_skipped = 0
        for p in FaceProfile.query.all():
            if not p.feature_vector:
                continue
            existing = Template.query.filter_by(profile_id=p.id, strategy="mean_all").first()
            if existing:
                n_skipped += 1
                continue
            db.session.add(Template(
                profile_id=p.id,
                strategy="mean_all",
                vector_json=p.feature_vector,
                source_count=1,
            ))
            n_added += 1
        db.session.commit()
        print(f"已迁移: {n_added} 行；跳过（已有）: {n_skipped} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
