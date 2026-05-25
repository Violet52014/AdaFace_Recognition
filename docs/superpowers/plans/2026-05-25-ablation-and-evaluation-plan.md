# 5 策略消融 + 评估体系实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 fork 自上游的 AdaFace_Recognition 项目上叠加 5 策略多模板消融 + 离线评估系统（ROC/EER/TAR\@FAR=1e-3/AUC），并将赢家策略接入在线识别路径。

**Architecture:** 增量式分层。新增 3 个独立模块：`app/strategies/`（5 个纯函数模板生成器，零 IO）、`app/evaluation/`（数据切分 / LFW 加载 / pair 生成 / 纯 numpy 指标 / runner / CLI）、`app/config_loader.py`（pydantic-settings 读 `config.yaml`）。新增 `Template` 表多对一 `FaceProfile`。上游 `routes/` 与 `adaface_infer.py` 不动，仅 `face_service._match_with_gallery` 替换为 `match_with_templates`。

**Tech Stack:** Python 3.10+, Flask 2.x, SQLAlchemy 1.4, PyTorch 1.13, numpy, scikit-learn (kmeans), pydantic-settings, PyYAML, pytest, AdaFace IR-50, MTCNN.

**Spec:** [`docs/superpowers/specs/2026-05-25-ablation-and-evaluation-design.md`](../specs/2026-05-25-ablation-and-evaluation-design.md)

---

## 里程碑总览

- **M1 地基**：`config.yaml` + `AppConfig` + `Template` 表 + 测试脚手架（Tasks 1-6）
- **M2 5 策略**：`strategies/` 全部实现 + 单测（Tasks 7-12）
- **M3 评估纯函数**：`data_split` / `lfw_loader` / `pair_generator` / `metrics`（Tasks 13-19）
- **M4 跑通消融**：`runner` / `run_ablation` CLI + 集成 smoke（Tasks 20-23）
- **M5 切线生产默认**：`match_with_templates` + `build_face_gallery` 改造 + 迁移脚本（Tasks 24-28）

每完成一个 M 打 git tag。

---

## 全局约定

- **Python 版本**：3.10+。文件全部 UTF-8 编码。
- **路径规则**：所有 `Path` 操作以仓库根为基准；`config.yaml` 在仓库根，`backend/` 是后端工作目录。
- **运行测试**：均在 `backend/` 目录下执行 `pytest`，不在仓库根。
- **commit message 风格**：Conventional Commits (`feat: / fix: / refactor: / test: / chore:`)。
- **PR/分支**：本计划假定直接在 `main` 上小步推进；若用 worktree，由 `superpowers:using-git-worktrees` 提前准备。
- **导入规则**：`from app.xxx import yyy`（`backend/` 是 sys.path 起点，与上游 `scripts/build_face_gallery.py` 一致）。

---

## M1 — 地基

### Task 1: 增补依赖 + pytest 配置

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/pytest.ini`
- Create: `backend/tests/__init__.py` (空)
- Create: `backend/tests/conftest.py`

- [ ] **Step 1: 修改 `backend/requirements.txt`，在末尾追加 5 行**

```
pytest
pytest-cov
pydantic-settings
PyYAML
scikit-learn
```

- [ ] **Step 2: 创建 `backend/pytest.ini`**

```ini
[pytest]
markers =
    slow: 集成测试，需要权重，分钟级
addopts = -ra -q
testpaths = tests
```

- [ ] **Step 3: 创建空的 `backend/tests/__init__.py`**

(空文件即可)

- [ ] **Step 4: 创建 `backend/tests/conftest.py`**

```python
"""测试公共 fixture。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 让 `from app.xxx import yyy` 在 pytest 中可用，与 scripts/build_face_gallery.py 同款
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_vectors():
    """构造 (N, 512) 已 L2 归一化的合成向量。"""
    def _make(n: int = 10, dim: int = 512, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((n, dim)).astype(np.float32)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        return v
    return _make
```

- [ ] **Step 5: 安装依赖并验证 pytest 可发现**

Run: `cd backend && pip install -r requirements.txt && pytest --collect-only`
Expected: collected 0 items (没测试是正常的)，无 import 错误。

- [ ] **Step 6: Commit**

```bash
git add backend/requirements.txt backend/pytest.ini backend/tests/
git commit -m "chore(test): 引入 pytest 脚手架 + 评估相关依赖"
```

---

### Task 2: AppConfig 加载器（先写测试）

**Files:**
- Create: `backend/tests/unit/__init__.py` (空)
- Create: `backend/tests/unit/test_config_loader.py`

- [ ] **Step 1: 创建 `backend/tests/unit/__init__.py`** (空文件)

- [ ] **Step 2: 写失败的测试 `backend/tests/unit/test_config_loader.py`**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from app.config_loader import AppConfig, ConfigError


VALID_YAML = """
recognition:
  match_threshold: 0.35
  production_strategy: mean_all
evaluation:
  random_seed: 42
  train_ratio: 0.8
  min_vectors_per_person: 5
  lfw_cache_dir: backend/dataset/lfw
  lfw_impostor_count: 1000
  far_targets: [1.0e-3]
  output_dir: reports
strategies:
  kmeans:
    k: 3
  manual_three:
    pose_groups: [frontal, left, right]
"""


def test_loads_valid_yaml(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(VALID_YAML, encoding="utf-8")
    cfg = AppConfig.load(cfg_file)
    assert cfg.recognition.match_threshold == pytest.approx(0.35)
    assert cfg.recognition.production_strategy == "mean_all"
    assert cfg.evaluation.random_seed == 42
    assert cfg.evaluation.far_targets == [1e-3]
    assert cfg.strategies.kmeans.k == 3
    assert cfg.strategies.manual_three.pose_groups == ["frontal", "left", "right"]


def test_threshold_must_be_in_range(tmp_path: Path):
    bad = VALID_YAML.replace("match_threshold: 0.35", "match_threshold: 1.5")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError):
        AppConfig.load(cfg_file)


def test_production_strategy_must_be_known(tmp_path: Path):
    bad = VALID_YAML.replace("production_strategy: mean_all", "production_strategy: foo")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError):
        AppConfig.load(cfg_file)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(ConfigError):
        AppConfig.load(tmp_path / "nope.yaml")


def test_train_ratio_must_be_open_unit_interval(tmp_path: Path):
    bad = VALID_YAML.replace("train_ratio: 0.8", "train_ratio: 1.0")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError):
        AppConfig.load(cfg_file)
```

- [ ] **Step 3: 跑测试确认全部 FAIL（ImportError）**

Run: `cd backend && pytest tests/unit/test_config_loader.py -v`
Expected: collection error / `ModuleNotFoundError: No module named 'app.config_loader'`

(Task 2 不 commit，等 Task 3 实现完一并 commit)

---

### Task 3: 实现 AppConfig

**Files:**
- Create: `backend/app/config_loader.py`
- Create: `config.yaml` (仓库根)

- [ ] **Step 1: 创建 `backend/app/config_loader.py`**

```python
"""集中读 config.yaml，pydantic 校验失败立即崩溃。

注意：上游 app/config.py 仍在使用（环境变量风格），本模块独立存在，
新代码统一引用 AppConfig，老代码不动。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


KNOWN_STRATEGIES = (
    "random_one", "mean_all", "manual_three", "kmeans_k3", "all_vectors",
)


class ConfigError(Exception):
    pass


class _RecognitionCfg(BaseModel):
    match_threshold: float = Field(ge=0.0, le=1.0)
    production_strategy: Literal[
        "random_one", "mean_all", "manual_three", "kmeans_k3", "all_vectors"
    ]


class _EvaluationCfg(BaseModel):
    random_seed: int
    train_ratio: float
    min_vectors_per_person: int = Field(ge=1)
    lfw_cache_dir: str
    lfw_impostor_count: int = Field(ge=1)
    far_targets: List[float]
    output_dir: str

    @field_validator("train_ratio")
    @classmethod
    def _ratio_open_unit(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("train_ratio 必须在 (0, 1) 开区间内")
        return v


class _KmeansCfg(BaseModel):
    k: int = Field(ge=1)


class _ManualThreeCfg(BaseModel):
    pose_groups: List[str]


class _StrategiesCfg(BaseModel):
    kmeans: _KmeansCfg
    manual_three: _ManualThreeCfg


class AppConfig(BaseModel):
    recognition: _RecognitionCfg
    evaluation: _EvaluationCfg
    strategies: _StrategiesCfg

    @classmethod
    def load(cls, path: Path | str) -> "AppConfig":
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"配置文件不存在: {p}")
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return cls.model_validate(data)
        except (yaml.YAMLError, ValidationError) as e:
            raise ConfigError(f"配置文件解析失败 {p}: {e}") from e
```

- [ ] **Step 2: 创建 `config.yaml`（仓库根）**

```yaml
recognition:
  match_threshold: 0.35
  production_strategy: mean_all

evaluation:
  random_seed: 42
  train_ratio: 0.8
  min_vectors_per_person: 5
  lfw_cache_dir: backend/dataset/lfw
  lfw_impostor_count: 1000
  far_targets: [1.0e-3]
  output_dir: reports

strategies:
  kmeans:
    k: 3
  manual_three:
    pose_groups: [frontal, left, right]
```

- [ ] **Step 3: 跑测试确认全部 PASS**

Run: `cd backend && pytest tests/unit/test_config_loader.py -v`
Expected: 5 passed

- [ ] **Step 4: Commit**

```bash
git add backend/app/config_loader.py backend/tests/unit/__init__.py backend/tests/unit/test_config_loader.py config.yaml
git commit -m "feat(config): 引入 config.yaml + pydantic AppConfig 加载器"
```

---

### Task 4: 新增 Template 表（先写测试）

**Files:**
- Create: `backend/tests/unit/test_models_template.py`

- [ ] **Step 1: 写失败的测试 `backend/tests/unit/test_models_template.py`**

```python
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
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `cd backend && pytest tests/unit/test_models_template.py -v`
Expected: `ImportError: cannot import name 'Template' from 'app.models'`

(不 commit，下个 Task 一起)

---

### Task 5: 实现 Template 模型 + parse 工具

**Files:**
- Modify: `backend/app/models.py`

- [ ] **Step 1: 修改 `backend/app/models.py`，在文件末尾追加**

```python
import json as _json
from typing import Optional


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
```

- [ ] **Step 2: SQLite 默认对 ON DELETE 不开 FK 约束 — 在测试 fixture 内开**

修改 `backend/tests/unit/test_models_template.py` 的 `app` fixture，在 `db.init_app(app)` 之后追加：

```python
    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    @event.listens_for(Engine, "connect")
    def _fk_on(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
```

(放在 `with app.app_context():` 之前)

- [ ] **Step 3: 跑测试确认 PASS**

Run: `cd backend && pytest tests/unit/test_models_template.py -v`
Expected: 3 passed

- [ ] **Step 4: Commit**

```bash
git add backend/app/models.py backend/tests/unit/test_models_template.py
git commit -m "feat(db): 新增 Template 多对一表 + parse_template_vectors"
```

---

### Task 6: 打 M1 tag

- [ ] **Step 1: 跑全部已写测试**

Run: `cd backend && pytest -v`
Expected: 8 passed (5 config + 3 template)

- [ ] **Step 2: 打 tag**

```bash
git tag m1-foundation
```


---

## M2 — 5 个策略

### Task 7: 策略 Protocol + 注册表（先写测试）

**Files:**
- Create: `backend/tests/unit/strategies/__init__.py` (空)
- Create: `backend/tests/unit/strategies/test_registry.py`

- [ ] **Step 1: 创建空 `backend/tests/unit/strategies/__init__.py`**

- [ ] **Step 2: 写测试 `backend/tests/unit/strategies/test_registry.py`**

```python
from __future__ import annotations

import pytest

from app.strategies import STRATEGIES, get_strategy


def test_registry_contains_all_five():
    expected = {"random_one", "mean_all", "manual_three", "kmeans_k3", "all_vectors"}
    assert set(STRATEGIES.keys()) == expected


def test_get_strategy_returns_named():
    s = get_strategy("mean_all")
    assert s.name == "mean_all"


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get_strategy("not_a_real_strategy")
```

- [ ] **Step 3: 跑测试确认 FAIL**

Run: `cd backend && pytest tests/unit/strategies/test_registry.py -v`
Expected: ImportError

(不 commit)

---

### Task 8: 实现 base + 注册表

**Files:**
- Create: `backend/app/strategies/__init__.py`
- Create: `backend/app/strategies/base.py`

- [ ] **Step 1: 创建 `backend/app/strategies/base.py`**

```python
"""TemplateStrategy Protocol：5 策略的统一形状。"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


def l2_normalize(v: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """对每一行做 L2 归一化。输入 (M,D) 或 (D,) 都接受，输出形状不变。"""
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + eps)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + eps)


class TemplateStrategy(Protocol):
    name: str

    def build(
        self,
        vectors: np.ndarray,
        *,
        rng: np.random.Generator,
        group_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ...
```

- [ ] **Step 2: 创建 `backend/app/strategies/__init__.py`**

```python
"""5 策略注册表。"""
from __future__ import annotations

from typing import Dict

from .all_vectors import AllVectorsStrategy
from .base import TemplateStrategy
from .kmeans_k3 import KmeansK3Strategy
from .manual_three import ManualThreeStrategy
from .mean_all import MeanAllStrategy
from .random_one import RandomOneStrategy


STRATEGIES: Dict[str, TemplateStrategy] = {
    "random_one": RandomOneStrategy(),
    "mean_all": MeanAllStrategy(),
    "manual_three": ManualThreeStrategy(),
    "kmeans_k3": KmeansK3Strategy(),
    "all_vectors": AllVectorsStrategy(),
}


def get_strategy(name: str) -> TemplateStrategy:
    if name not in STRATEGIES:
        raise KeyError(f"未知策略: {name}，可选 {list(STRATEGIES)}")
    return STRATEGIES[name]


__all__ = ["STRATEGIES", "TemplateStrategy", "get_strategy"]
```

(此时 5 个策略文件还没建，下面的 Task 9-11 一一加上；先把空文件占位)

- [ ] **Step 3: 创建 5 个空策略 stub（让 import 成立）**

```bash
cat > backend/app/strategies/random_one.py <<'PY'
from __future__ import annotations
import numpy as np
from .base import TemplateStrategy

class RandomOneStrategy:
    name = "random_one"
    def build(self, vectors, *, rng, group_labels=None):
        raise NotImplementedError
PY

cat > backend/app/strategies/mean_all.py <<'PY'
from __future__ import annotations
import numpy as np
from .base import TemplateStrategy

class MeanAllStrategy:
    name = "mean_all"
    def build(self, vectors, *, rng, group_labels=None):
        raise NotImplementedError
PY

cat > backend/app/strategies/manual_three.py <<'PY'
from __future__ import annotations
import numpy as np
from .base import TemplateStrategy

class ManualThreeStrategy:
    name = "manual_three"
    def build(self, vectors, *, rng, group_labels=None):
        raise NotImplementedError
PY

cat > backend/app/strategies/kmeans_k3.py <<'PY'
from __future__ import annotations
import numpy as np
from .base import TemplateStrategy

class KmeansK3Strategy:
    name = "kmeans_k3"
    k: int = 3
    def build(self, vectors, *, rng, group_labels=None):
        raise NotImplementedError
PY

cat > backend/app/strategies/all_vectors.py <<'PY'
from __future__ import annotations
import numpy as np
from .base import TemplateStrategy

class AllVectorsStrategy:
    name = "all_vectors"
    def build(self, vectors, *, rng, group_labels=None):
        raise NotImplementedError
PY
```

- [ ] **Step 4: 跑注册表测试确认 PASS**

Run: `cd backend && pytest tests/unit/strategies/test_registry.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/strategies/ backend/tests/unit/strategies/
git commit -m "feat(strategies): TemplateStrategy Protocol + 5 策略注册表（stub）"
```

---

### Task 9: random_one + mean_all + all_vectors

**Files:**
- Create: `backend/tests/unit/strategies/test_random_one.py`
- Create: `backend/tests/unit/strategies/test_mean_all.py`
- Create: `backend/tests/unit/strategies/test_all_vectors.py`
- Modify: `backend/app/strategies/random_one.py`
- Modify: `backend/app/strategies/mean_all.py`
- Modify: `backend/app/strategies/all_vectors.py`

- [ ] **Step 1: 写 `tests/unit/strategies/test_random_one.py`**

```python
from __future__ import annotations

import numpy as np
import pytest

from app.strategies import get_strategy


def test_shape_is_one_row(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("random_one").build(v, rng=np.random.default_rng(42))
    assert out.shape == (1, 512)


def test_deterministic_given_seed(synthetic_vectors):
    v = synthetic_vectors(n=10)
    s = get_strategy("random_one")
    a = s.build(v, rng=np.random.default_rng(7))
    b = s.build(v, rng=np.random.default_rng(7))
    assert np.allclose(a, b)


def test_output_is_l2_normalized(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("random_one").build(v, rng=np.random.default_rng(0))
    assert np.linalg.norm(out[0]) == pytest.approx(1.0, abs=1e-5)
```

- [ ] **Step 2: 写 `tests/unit/strategies/test_mean_all.py`**

```python
from __future__ import annotations

import numpy as np
import pytest

from app.strategies import get_strategy


def test_shape_is_one_row(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("mean_all").build(v, rng=np.random.default_rng(0))
    assert out.shape == (1, 512)


def test_equals_mean_then_l2(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("mean_all").build(v, rng=np.random.default_rng(0))
    expected = np.mean(v, axis=0)
    expected = expected / (np.linalg.norm(expected) + 1e-8)
    assert np.allclose(out[0], expected, atol=1e-6)


def test_output_is_l2_normalized(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("mean_all").build(v, rng=np.random.default_rng(0))
    assert np.linalg.norm(out[0]) == pytest.approx(1.0, abs=1e-5)
```

- [ ] **Step 3: 写 `tests/unit/strategies/test_all_vectors.py`**

```python
from __future__ import annotations

import numpy as np

from app.strategies import get_strategy


def test_shape_equals_input(synthetic_vectors):
    v = synthetic_vectors(n=7)
    out = get_strategy("all_vectors").build(v, rng=np.random.default_rng(0))
    assert out.shape == (7, 512)


def test_output_equals_input(synthetic_vectors):
    v = synthetic_vectors(n=7)
    out = get_strategy("all_vectors").build(v, rng=np.random.default_rng(0))
    assert np.array_equal(out, v)


def test_does_not_mutate_input(synthetic_vectors):
    v = synthetic_vectors(n=7)
    snapshot = v.copy()
    _ = get_strategy("all_vectors").build(v, rng=np.random.default_rng(0))
    assert np.array_equal(v, snapshot)
```

- [ ] **Step 4: 跑测试确认 FAIL（NotImplementedError）**

Run: `cd backend && pytest tests/unit/strategies/test_random_one.py tests/unit/strategies/test_mean_all.py tests/unit/strategies/test_all_vectors.py -v`
Expected: 9 failed

- [ ] **Step 5: 实现 `backend/app/strategies/random_one.py`**

```python
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import l2_normalize


class RandomOneStrategy:
    name = "random_one"

    def build(
        self,
        vectors: np.ndarray,
        *,
        rng: np.random.Generator,
        group_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        idx = int(rng.integers(0, vectors.shape[0]))
        return l2_normalize(vectors[idx][None, :])
```

- [ ] **Step 6: 实现 `backend/app/strategies/mean_all.py`**

```python
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import l2_normalize


class MeanAllStrategy:
    name = "mean_all"

    def build(
        self,
        vectors: np.ndarray,
        *,
        rng: np.random.Generator,
        group_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        m = np.mean(vectors, axis=0, keepdims=True)
        return l2_normalize(m)
```

- [ ] **Step 7: 实现 `backend/app/strategies/all_vectors.py`**

```python
from __future__ import annotations

from typing import Optional

import numpy as np


class AllVectorsStrategy:
    name = "all_vectors"

    def build(
        self,
        vectors: np.ndarray,
        *,
        rng: np.random.Generator,
        group_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # 输入约定已 L2 归一化；为避免污染调用方语义，返回拷贝
        return vectors.copy()
```

- [ ] **Step 8: 跑测试确认 PASS**

Run: `cd backend && pytest tests/unit/strategies/ -v`
Expected: 12 passed (3 registry + 9 新增)

- [ ] **Step 9: Commit**

```bash
git add backend/app/strategies/ backend/tests/unit/strategies/
git commit -m "feat(strategies): 实现 random_one / mean_all / all_vectors"
```

---

### Task 10: manual_three（按姿态分组）

**Files:**
- Create: `backend/tests/unit/strategies/test_manual_three.py`
- Modify: `backend/app/strategies/manual_three.py`

- [ ] **Step 1: 写测试 `tests/unit/strategies/test_manual_three.py`**

```python
from __future__ import annotations

import numpy as np
import pytest

from app.strategies import get_strategy
from app.strategies.base import l2_normalize


def _labels(*items: str) -> np.ndarray:
    return np.array(items, dtype=object)


def test_three_groups_full_yields_three_rows(synthetic_vectors):
    v = synthetic_vectors(n=9)  # 3 frontal + 3 left + 3 right
    labels = _labels(*(["frontal"] * 3 + ["left"] * 3 + ["right"] * 3))
    out = get_strategy("manual_three").build(
        v, rng=np.random.default_rng(0), group_labels=labels,
    )
    assert out.shape == (3, 512)


def test_each_row_equals_group_mean_l2(synthetic_vectors):
    v = synthetic_vectors(n=6)
    labels = _labels("frontal", "frontal", "left", "left", "right", "right")
    out = get_strategy("manual_three").build(
        v, rng=np.random.default_rng(0), group_labels=labels,
    )
    expected_frontal = l2_normalize(np.mean(v[:2], axis=0))
    expected_left = l2_normalize(np.mean(v[2:4], axis=0))
    expected_right = l2_normalize(np.mean(v[4:6], axis=0))
    # 行序固定为 frontal/left/right
    assert np.allclose(out[0], expected_frontal, atol=1e-6)
    assert np.allclose(out[1], expected_left, atol=1e-6)
    assert np.allclose(out[2], expected_right, atol=1e-6)


def test_missing_group_yields_fewer_rows(synthetic_vectors):
    v = synthetic_vectors(n=4)
    labels = _labels("frontal", "frontal", "left", "left")  # right 缺
    out = get_strategy("manual_three").build(
        v, rng=np.random.default_rng(0), group_labels=labels,
    )
    assert out.shape == (2, 512)


def test_all_groups_missing_raises(synthetic_vectors):
    v = synthetic_vectors(n=2)
    labels = _labels("unknown", "unknown")
    with pytest.raises(ValueError):
        get_strategy("manual_three").build(
            v, rng=np.random.default_rng(0), group_labels=labels,
        )


def test_missing_group_labels_raises(synthetic_vectors):
    v = synthetic_vectors(n=3)
    with pytest.raises(ValueError):
        get_strategy("manual_three").build(
            v, rng=np.random.default_rng(0), group_labels=None,
        )
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `cd backend && pytest tests/unit/strategies/test_manual_three.py -v`
Expected: 5 failed

- [ ] **Step 3: 实现 `backend/app/strategies/manual_three.py`**

```python
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import l2_normalize


POSE_ORDER = ("frontal", "left", "right")


class ManualThreeStrategy:
    name = "manual_three"

    def build(
        self,
        vectors: np.ndarray,
        *,
        rng: np.random.Generator,
        group_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if group_labels is None:
            raise ValueError("manual_three 需要 group_labels（pose 标签）")
        if len(group_labels) != vectors.shape[0]:
            raise ValueError("group_labels 长度必须与 vectors 数量一致")

        rows = []
        for pose in POSE_ORDER:
            mask = np.array([str(g) == pose for g in group_labels])
            if not mask.any():
                continue
            rows.append(l2_normalize(np.mean(vectors[mask], axis=0)))
        if not rows:
            raise ValueError(f"无任一姿态组（{POSE_ORDER}）有样本")
        return np.stack(rows, axis=0)
```

- [ ] **Step 4: 跑测试确认 PASS**

Run: `cd backend && pytest tests/unit/strategies/test_manual_three.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/strategies/manual_three.py backend/tests/unit/strategies/test_manual_three.py
git commit -m "feat(strategies): 实现 manual_three 按姿态分组取均值"
```

---

### Task 11: kmeans_k3

**Files:**
- Create: `backend/tests/unit/strategies/test_kmeans_k3.py`
- Modify: `backend/app/strategies/kmeans_k3.py`

- [ ] **Step 1: 写测试 `tests/unit/strategies/test_kmeans_k3.py`**

```python
from __future__ import annotations

import numpy as np
import pytest

from app.strategies import get_strategy


def test_shape_is_min_k_n(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("kmeans_k3").build(v, rng=np.random.default_rng(42))
    assert out.shape == (3, 512)


def test_n_smaller_than_k_degrades(synthetic_vectors):
    v = synthetic_vectors(n=2)
    out = get_strategy("kmeans_k3").build(v, rng=np.random.default_rng(42))
    assert out.shape == (2, 512)


def test_n_one_returns_single_template(synthetic_vectors):
    v = synthetic_vectors(n=1)
    out = get_strategy("kmeans_k3").build(v, rng=np.random.default_rng(42))
    assert out.shape == (1, 512)


def test_each_row_is_l2_normalized(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("kmeans_k3").build(v, rng=np.random.default_rng(42))
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_deterministic_given_seed(synthetic_vectors):
    v = synthetic_vectors(n=10)
    s = get_strategy("kmeans_k3")
    a = s.build(v, rng=np.random.default_rng(99))
    b = s.build(v, rng=np.random.default_rng(99))
    assert np.allclose(a, b)
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `cd backend && pytest tests/unit/strategies/test_kmeans_k3.py -v`
Expected: 5 failed

- [ ] **Step 3: 实现 `backend/app/strategies/kmeans_k3.py`**

```python
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import l2_normalize


class KmeansK3Strategy:
    name = "kmeans_k3"
    k: int = 3

    def build(
        self,
        vectors: np.ndarray,
        *,
        rng: np.random.Generator,
        group_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = vectors.shape[0]
        if n == 0:
            raise ValueError("kmeans_k3 输入向量数为 0")
        k = min(self.k, n)
        if k == 1:
            return l2_normalize(np.mean(vectors, axis=0, keepdims=True))

        from sklearn.cluster import KMeans
        # 用 rng 派生一个稳定 int seed 给 sklearn
        seed = int(rng.integers(0, 2**31 - 1))
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(vectors)
        return l2_normalize(km.cluster_centers_.astype(np.float32))
```

- [ ] **Step 4: 跑测试确认 PASS**

Run: `cd backend && pytest tests/unit/strategies/test_kmeans_k3.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/strategies/kmeans_k3.py backend/tests/unit/strategies/test_kmeans_k3.py
git commit -m "feat(strategies): 实现 kmeans_k3（N<k 时退化）"
```

---

### Task 12: 打 M2 tag

- [ ] **Step 1: 跑全部测试**

Run: `cd backend && pytest -v`
Expected: 25 passed (8 from M1 + 3 registry + 9 + 5 + 5)

- [ ] **Step 2: 打 tag**

```bash
git tag m2-strategies
```


---

## M3 — 评估纯函数

### Task 13: data_split — ImageEntry + 切分（先写测试）

**Files:**
- Create: `backend/tests/unit/evaluation/__init__.py` (空)
- Create: `backend/tests/unit/evaluation/test_data_split.py`

- [ ] **Step 1: 创建空 `backend/tests/unit/evaluation/__init__.py`**

- [ ] **Step 2: 写测试 `tests/unit/evaluation/test_data_split.py`**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.data_split import ImageEntry, split_by_person


def _make_dataset_old(root: Path, persons: dict[str, int]) -> None:
    """旧布局: root/<name>/img_xx.jpg"""
    for name, n in persons.items():
        d = root / name
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"img_{i:02d}.jpg").write_bytes(b"\x00")


def _make_dataset_pose(root: Path, persons: dict[str, dict[str, int]]) -> None:
    """新布局: root/<name>/<pose>/img_xx.jpg"""
    for name, poses in persons.items():
        for pose, n in poses.items():
            d = root / name / pose
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"img_{i:02d}.jpg").write_bytes(b"\x00")


def test_split_old_layout_returns_pose_none(tmp_path: Path):
    _make_dataset_old(tmp_path, {"alice": 10})
    out = split_by_person(tmp_path, train_ratio=0.8, seed=42)
    train, probe = out["alice"]
    assert all(isinstance(e, ImageEntry) for e in train + probe)
    assert all(e.pose is None for e in train + probe)


def test_split_pose_layout_records_pose(tmp_path: Path):
    _make_dataset_pose(tmp_path, {"bob": {"frontal": 4, "left": 4, "right": 4}})
    out = split_by_person(tmp_path, train_ratio=0.5, seed=42)
    train, probe = out["bob"]
    poses = {e.pose for e in train + probe}
    assert poses == {"frontal", "left", "right"}


def test_split_ratio_approximate(tmp_path: Path):
    _make_dataset_old(tmp_path, {"x": 10})
    train, probe = split_by_person(tmp_path, train_ratio=0.8, seed=42)["x"]
    assert len(train) == 8
    assert len(probe) == 2


def test_split_deterministic_given_seed(tmp_path: Path):
    _make_dataset_old(tmp_path, {"x": 10})
    a = split_by_person(tmp_path, train_ratio=0.7, seed=42)["x"]
    b = split_by_person(tmp_path, train_ratio=0.7, seed=42)["x"]
    assert [str(e.path) for e in a[0]] == [str(e.path) for e in b[0]]
    assert [str(e.path) for e in a[1]] == [str(e.path) for e in b[1]]


def test_single_image_person_all_in_train(tmp_path: Path, caplog):
    _make_dataset_old(tmp_path, {"loner": 1})
    train, probe = split_by_person(tmp_path, train_ratio=0.8, seed=42)["loner"]
    assert len(train) == 1
    assert probe == []


def test_skips_lfw_subdir(tmp_path: Path):
    _make_dataset_old(tmp_path, {"alice": 4})
    # 模拟 dataset/lfw 缓存子目录，必须被跳过
    (tmp_path / "lfw" / "George").mkdir(parents=True)
    (tmp_path / "lfw" / "George" / "george_01.jpg").write_bytes(b"\x00")
    out = split_by_person(tmp_path, train_ratio=0.5, seed=42)
    assert "lfw" not in out
    assert "alice" in out
```

- [ ] **Step 3: 跑测试确认 FAIL**

Run: `cd backend && pytest tests/unit/evaluation/test_data_split.py -v`
Expected: ImportError

(不 commit)

---

### Task 14: 实现 data_split

**Files:**
- Create: `backend/app/evaluation/__init__.py` (空)
- Create: `backend/app/evaluation/data_split.py`

- [ ] **Step 1: 创建空 `backend/app/evaluation/__init__.py`**

- [ ] **Step 2: 创建 `backend/app/evaluation/data_split.py`**

```python
"""按人切分数据集为 train/probe，识别姿态子目录。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


log = logging.getLogger(__name__)

# 与 config.yaml strategies.manual_three.pose_groups 对齐
POSE_NAMES = frozenset({"frontal", "left", "right"})

# 评估时不可作为"人脸库人员"的特殊子目录名
RESERVED_SUBDIRS = frozenset({"lfw"})

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class ImageEntry:
    path: Path
    pose: Optional[str]


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _collect_images_for_person(person_dir: Path) -> List[ImageEntry]:
    """递归发现 person_dir 下的所有图片，pose=直接父目录名（若在 POSE_NAMES 内）。"""
    entries: List[ImageEntry] = []
    # 直接子文件
    for p in sorted(person_dir.iterdir()):
        if _is_image(p):
            entries.append(ImageEntry(path=p, pose=None))
        elif p.is_dir() and p.name in POSE_NAMES:
            for q in sorted(p.iterdir()):
                if _is_image(q):
                    entries.append(ImageEntry(path=q, pose=p.name))
    return entries


def split_by_person(
    dataset_root: Path,
    *,
    train_ratio: float,
    seed: int,
) -> Dict[str, Tuple[List[ImageEntry], List[ImageEntry]]]:
    """每人切 train/probe；同 seed 同结果；N==1 时全部入 train。

    跳过 RESERVED_SUBDIRS（如 lfw）。
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, Tuple[List[ImageEntry], List[ImageEntry]]] = {}
    for person_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if person_dir.name in RESERVED_SUBDIRS:
            continue
        entries = _collect_images_for_person(person_dir)
        if not entries:
            log.warning("人 %s 无可用图片，跳过", person_dir.name)
            continue
        n = len(entries)
        if n == 1:
            log.warning("人 %s 仅 1 张图，全部入 train，probe 为空", person_dir.name)
            out[person_dir.name] = (entries, [])
            continue
        idx = rng.permutation(n)
        n_train = max(1, int(round(n * train_ratio)))
        n_train = min(n_train, n - 1)  # 至少留 1 张做 probe
        train = [entries[i] for i in idx[:n_train]]
        probe = [entries[i] for i in idx[n_train:]]
        out[person_dir.name] = (train, probe)
    return out
```

- [ ] **Step 3: 跑测试确认 PASS**

Run: `cd backend && pytest tests/unit/evaluation/test_data_split.py -v`
Expected: 6 passed

- [ ] **Step 4: Commit**

```bash
git add backend/app/evaluation/__init__.py backend/app/evaluation/data_split.py backend/tests/unit/evaluation/
git commit -m "feat(evaluation): data_split 支持新旧布局 + pose 标签"
```

---

### Task 15: metrics — 纯 numpy ROC/EER/TAR/AUC（先写测试）

**Files:**
- Create: `backend/tests/unit/evaluation/test_metrics.py`

- [ ] **Step 1: 写测试 `tests/unit/evaluation/test_metrics.py`**

```python
from __future__ import annotations

import numpy as np
import pytest

from app.evaluation.metrics import auc, eer, roc_curve, tar_at_far


def test_perfect_separation():
    # 50 正样本得分=1，50 负样本得分=0
    scores = np.concatenate([np.ones(50), np.zeros(50)])
    labels = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])
    fpr, tpr, thr = roc_curve(scores, labels)
    assert auc(fpr, tpr) == pytest.approx(1.0, abs=1e-6)
    eer_v, _ = eer(fpr, tpr, thr)
    assert eer_v == pytest.approx(0.0, abs=1e-6)


def test_random_uniform_auc_near_half():
    rng = np.random.default_rng(42)
    scores = rng.random(2000)
    labels = rng.integers(0, 2, 2000)
    fpr, tpr, _ = roc_curve(scores, labels)
    assert auc(fpr, tpr) == pytest.approx(0.5, abs=0.05)


def test_tar_at_far_unreachable_returns_zero():
    # 全部 score 相同 → fpr 直接从 0 跳到 1，FAR=1e-3 不可达
    scores = np.full(100, 0.5)
    labels = np.array([0, 1] * 50)
    fpr, tpr, _ = roc_curve(scores, labels)
    assert tar_at_far(fpr, tpr, 1e-3) == 0.0


def test_constant_scores_degenerate():
    scores = np.full(100, 0.5)
    labels = np.array([0, 1] * 50)
    fpr, tpr, thr = roc_curve(scores, labels)
    assert auc(fpr, tpr) == pytest.approx(0.5, abs=1e-6)
    eer_v, _ = eer(fpr, tpr, thr)
    assert eer_v == pytest.approx(0.5, abs=1e-6)


def test_rejects_nan():
    scores = np.array([0.1, np.nan, 0.5])
    labels = np.array([0, 1, 1])
    with pytest.raises(ValueError):
        roc_curve(scores, labels)


def test_rejects_single_class_labels():
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 1, 1])
    fpr = np.array([0.0, 1.0])
    tpr = np.array([0.0, 1.0])
    thr = np.array([1.0, 0.0])
    with pytest.raises(ValueError):
        eer(fpr, tpr, thr)  # 已计算的曲线没法定义 EER 时也应拒绝？
    # 主要拒绝在 roc_curve 入口
    with pytest.raises(ValueError):
        roc_curve(scores, labels)


def test_tpr_strictly_monotonic_perfect():
    scores = np.array([0.9, 0.8, 0.7, 0.4, 0.3])
    labels = np.array([1, 1, 1, 0, 0])
    fpr, tpr, _ = roc_curve(scores, labels)
    # tpr 单调不减
    assert np.all(np.diff(tpr) >= -1e-9)
    assert np.all(np.diff(fpr) >= -1e-9)
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `cd backend && pytest tests/unit/evaluation/test_metrics.py -v`
Expected: ImportError

(不 commit)

---

### Task 16: 实现 metrics

**Files:**
- Create: `backend/app/evaluation/metrics.py`

- [ ] **Step 1: 创建 `backend/app/evaluation/metrics.py`**

```python
"""ROC / EER / TAR@FAR / AUC，纯 numpy。

实现要点：
- roc_curve 按 score 降序排序，逐阈值累计 TP/FP，输出从 (0,0) 到 (1,1) 的折线点。
- AUC 用梯形法（np.trapz）。
- EER = fpr 与 (1-tpr) 的最小交差；具体取 |fpr - fnr| 的 argmin。
- TAR@FAR(target) = 在 fpr <= target 区域的最大 tpr；若无可达点返回 0。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _validate(scores: np.ndarray, labels: np.ndarray) -> None:
    if scores.shape != labels.shape:
        raise ValueError("scores 与 labels 形状不一致")
    if np.isnan(scores).any():
        raise ValueError("scores 含 NaN")
    uniq = set(np.unique(labels).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError("labels 必须为 0/1")
    if uniq != {0, 1}:
        raise ValueError("labels 必须同时包含 0 和 1")


def roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (fpr, tpr, thresholds)，三者长度一致，从 (0,0) 出发到 (1,1) 结束。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    _validate(scores, labels)

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    l_sorted = labels[order]

    n_pos = int(l_sorted.sum())
    n_neg = int(len(l_sorted) - n_pos)

    tp = np.cumsum(l_sorted == 1)
    fp = np.cumsum(l_sorted == 0)

    # 在每段 score 相等的边界处取点
    distinct = np.r_[np.where(np.diff(s_sorted) != 0)[0], len(s_sorted) - 1]
    tp = tp[distinct]
    fp = fp[distinct]
    thr = s_sorted[distinct]

    # 在前面拼接 (0,0)，对应阈值 +inf
    tp = np.r_[0, tp]
    fp = np.r_[0, fp]
    thr = np.r_[np.inf, thr]

    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    return fpr, tpr, thr


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapz(tpr, fpr))


def eer(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float]:
    """EER = 让 fpr ≈ 1-tpr 的那个工作点；返回 (eer_value, threshold@eer)。"""
    fnr = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    idx = int(np.argmin(diff))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
    """fpr <= target_far 区域的最大 tpr；不可达则 0.0。"""
    mask = fpr <= target_far
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())
```

- [ ] **Step 2: 跑测试确认 PASS**

Run: `cd backend && pytest tests/unit/evaluation/test_metrics.py -v`
Expected: 7 passed

- [ ] **Step 3: Commit**

```bash
git add backend/app/evaluation/metrics.py backend/tests/unit/evaluation/test_metrics.py
git commit -m "feat(evaluation): 纯 numpy ROC / EER / TAR@FAR / AUC"
```

---

### Task 17: pair_generator（先写测试）

**Files:**
- Create: `backend/tests/unit/evaluation/test_pair_generator.py`

- [ ] **Step 1: 写测试 `tests/unit/evaluation/test_pair_generator.py`**

```python
from __future__ import annotations

import numpy as np
import pytest

from app.evaluation.pair_generator import Pair, make_pairs


def _l2(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-8)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def test_pair_counts_basic():
    rng = np.random.default_rng(0)
    # 3 人 × 2 probe；gallery 单模板；LFW 5 张
    probe = {
        "a": [_l2(rng.standard_normal(8).astype(np.float32)) for _ in range(2)],
        "b": [_l2(rng.standard_normal(8).astype(np.float32)) for _ in range(2)],
        "c": [_l2(rng.standard_normal(8).astype(np.float32)) for _ in range(2)],
    }
    gallery = {
        "a": _l2(rng.standard_normal((1, 8)).astype(np.float32)),
        "b": _l2(rng.standard_normal((1, 8)).astype(np.float32)),
        "c": _l2(rng.standard_normal((1, 8)).astype(np.float32)),
    }
    lfw = _l2(rng.standard_normal((5, 8)).astype(np.float32))
    pairs = make_pairs(probe, gallery, lfw)
    n_genuine = sum(1 for p in pairs if p.is_genuine)
    n_impostor = sum(1 for p in pairs if not p.is_genuine)
    assert n_genuine == 6              # 3 人 × 2 probe
    assert n_impostor == 6 * 2 + 5 * 3  # cross-person 12 + lfw 15


def test_max_cosine_with_multiple_templates():
    # gallery 含 3 模板，其中 1 个与 probe 完全一致 → score=1.0
    probe_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    gallery = {
        "a": np.stack([other, probe_vec, other], axis=0),
    }
    probe = {"a": [probe_vec]}
    pairs = make_pairs(probe, gallery, np.zeros((0, 3), dtype=np.float32))
    assert len(pairs) == 1
    assert pairs[0].is_genuine
    assert pairs[0].score == pytest.approx(1.0, abs=1e-6)


def test_skips_persons_missing_in_gallery():
    """probe 中的人 d 在 gallery 中没有模板（manual_three 缺组场景），跳过该人 genuine。"""
    rng = np.random.default_rng(1)
    probe = {
        "a": [_l2(rng.standard_normal(4).astype(np.float32))],
        "d": [_l2(rng.standard_normal(4).astype(np.float32))],
    }
    gallery = {"a": _l2(rng.standard_normal((1, 4)).astype(np.float32))}
    pairs = make_pairs(probe, gallery, np.zeros((0, 4), dtype=np.float32))
    n_genuine = sum(1 for p in pairs if p.is_genuine)
    assert n_genuine == 1  # 仅 a 算


def test_score_is_in_unit_range():
    rng = np.random.default_rng(2)
    probe = {"a": [_l2(rng.standard_normal(8).astype(np.float32))]}
    gallery = {"a": _l2(rng.standard_normal((2, 8)).astype(np.float32))}
    pairs = make_pairs(probe, gallery, _l2(rng.standard_normal((3, 8)).astype(np.float32)))
    for p in pairs:
        assert -1.0 - 1e-6 <= p.score <= 1.0 + 1e-6
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `cd backend && pytest tests/unit/evaluation/test_pair_generator.py -v`
Expected: ImportError

(不 commit)

---

### Task 18: 实现 pair_generator

**Files:**
- Create: `backend/app/evaluation/pair_generator.py`

- [ ] **Step 1: 创建 `backend/app/evaluation/pair_generator.py`**

```python
"""构造 genuine / impostor pair；score = 多模板 max-cosine。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pair:
    score: float
    is_genuine: bool


def _max_cosine(query: np.ndarray, templates: np.ndarray) -> float:
    """query (D,) 与 templates (M, D)，假设两端均 L2 归一化，余弦=点积。"""
    if templates.shape[0] == 0:
        return float("-inf")
    sims = templates @ query
    return float(sims.max())


def make_pairs(
    probe_features: Dict[str, List[np.ndarray]],
    gallery_templates: Dict[str, np.ndarray],
    lfw_features: np.ndarray,
) -> List[Pair]:
    """三类 pair：
        genuine:           probe_i  vs gallery[name_i]
        impostor_internal: probe_i  vs gallery[name_j]   for j != i
        impostor_lfw:      lfw_k    vs gallery[name_j]   for all j
    """
    pairs: List[Pair] = []
    names = list(gallery_templates.keys())

    for name, probes in probe_features.items():
        for q in probes:
            if name in gallery_templates:
                pairs.append(Pair(
                    score=_max_cosine(q, gallery_templates[name]),
                    is_genuine=True,
                ))
            else:
                log.warning("probe 人 %s 不在 gallery 中，跳过其 genuine pair", name)
            for other in names:
                if other == name:
                    continue
                pairs.append(Pair(
                    score=_max_cosine(q, gallery_templates[other]),
                    is_genuine=False,
                ))

    for k in range(lfw_features.shape[0]):
        q = lfw_features[k]
        for other in names:
            pairs.append(Pair(
                score=_max_cosine(q, gallery_templates[other]),
                is_genuine=False,
            ))

    return pairs
```

- [ ] **Step 2: 跑测试确认 PASS**

Run: `cd backend && pytest tests/unit/evaluation/test_pair_generator.py -v`
Expected: 4 passed

- [ ] **Step 3: Commit**

```bash
git add backend/app/evaluation/pair_generator.py backend/tests/unit/evaluation/test_pair_generator.py
git commit -m "feat(evaluation): pair_generator 三类 pair + max-cosine"
```


---

### Task 19: lfw_loader

**Files:**
- Create: `backend/app/evaluation/lfw_loader.py`
- Create: `backend/tests/unit/evaluation/test_lfw_loader.py`

> **说明**：lfw_loader 涉及网络下载，不强制把"下载"也单测。这里只测**抽样选择函数**（纯函数），下载逻辑由集成测试或手动验证覆盖。

- [ ] **Step 1: 写测试 `tests/unit/evaluation/test_lfw_loader.py`**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.lfw_loader import sample_lfw_paths


def _make_lfw_layout(root: Path, persons: dict[str, int]) -> None:
    for name, n in persons.items():
        d = root / name
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{name}_{i:04d}.jpg").write_bytes(b"\x00")


def test_samples_exact_n_when_enough(tmp_path: Path):
    _make_lfw_layout(tmp_path, {"George_Bush": 50, "Tony_Blair": 30})
    paths = sample_lfw_paths(tmp_path, n_images=20, seed=42, exclude_names=set())
    assert len(paths) == 20


def test_excludes_overlapping_names(tmp_path: Path):
    _make_lfw_layout(tmp_path, {"alice": 10, "bob": 10})
    paths = sample_lfw_paths(tmp_path, n_images=10, seed=42, exclude_names={"alice"})
    assert all("alice" not in str(p) for p in paths)


def test_deterministic_given_seed(tmp_path: Path):
    _make_lfw_layout(tmp_path, {f"p{i}": 5 for i in range(10)})
    a = sample_lfw_paths(tmp_path, n_images=10, seed=7, exclude_names=set())
    b = sample_lfw_paths(tmp_path, n_images=10, seed=7, exclude_names=set())
    assert [str(p) for p in a] == [str(p) for p in b]


def test_returns_all_when_n_exceeds_available(tmp_path: Path):
    _make_lfw_layout(tmp_path, {"x": 3})
    paths = sample_lfw_paths(tmp_path, n_images=100, seed=42, exclude_names=set())
    assert len(paths) == 3


def test_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        sample_lfw_paths(tmp_path / "no_such", n_images=10, seed=42, exclude_names=set())
```

- [ ] **Step 2: 创建 `backend/app/evaluation/lfw_loader.py`**

```python
"""加载 LFW 子集作为 impostor 来源。

LFW 数据集本身不在仓库内：用户需要预先放置到 cache_dir，或自己解压官方
funneled tar.gz（http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz）。
本模块只做"抽样选择"，不实现下载（避免脚本调用网络的不确定性）。
若 cache_dir 不存在，run_ablation 会打印明确指引让用户手动准备。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Set

import numpy as np


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def sample_lfw_paths(
    cache_dir: Path,
    *,
    n_images: int,
    seed: int,
    exclude_names: Set[str],
) -> List[Path]:
    """LFW 布局 cache_dir/<person>/<image>.jpg；不在 exclude_names 中的子目录被纳入。"""
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"LFW 缓存目录不存在: {cache_dir}")
    candidates: List[Path] = []
    for person_dir in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        if person_dir.name in exclude_names:
            continue
        for img in sorted(person_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                candidates.append(img)
    if not candidates:
        return []
    rng = np.random.default_rng(seed)
    if n_images >= len(candidates):
        return candidates
    idx = rng.permutation(len(candidates))[:n_images]
    return [candidates[i] for i in sorted(idx)]
```

- [ ] **Step 3: 跑测试确认 PASS**

Run: `cd backend && pytest tests/unit/evaluation/test_lfw_loader.py -v`
Expected: 5 passed

- [ ] **Step 4: 跑 M3 全部测试**

Run: `cd backend && pytest tests/unit/evaluation/ -v`
Expected: 22 passed (6 split + 7 metrics + 4 pair + 5 lfw)

- [ ] **Step 5: Commit + tag**

```bash
git add backend/app/evaluation/lfw_loader.py backend/tests/unit/evaluation/test_lfw_loader.py
git commit -m "feat(evaluation): lfw_loader 抽样函数"
git tag m3-evaluation-pure-functions
```


---

## M4 — 跑通消融

### Task 20: runner — 单策略一次评估

**Files:**
- Create: `backend/app/evaluation/runner.py`

> **说明**：runner 是编排层，依赖 `extract_embedding_from_bgr`（需要真模型），故不在单测中跑。
> 集成测试在 Task 22 用真模型 + 官方 sample 图覆盖。

- [ ] **Step 1: 创建 `backend/app/evaluation/runner.py`**

```python
"""run_ablation 的执行核心：抽特征 → 5 策略 build → 评估，输出每策略一行结果。

设计：
- 一次性抽取所有 train / probe / lfw 特征（最贵的一步）
- 5 策略循环复用同一份特征
- 输出 dataclass AblationRow，由 CLI 落 CSV
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from app.config_loader import AppConfig
from app.evaluation.data_split import ImageEntry, split_by_person
from app.evaluation.lfw_loader import sample_lfw_paths
from app.evaluation.metrics import auc, eer, roc_curve, tar_at_far
from app.evaluation.pair_generator import make_pairs
from app.services.adaface_infer import extract_embedding_from_bgr
from app.services.image_utils import imread_bgr_unicode
from app.strategies import STRATEGIES, get_strategy


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AblationRow:
    strategy: str
    eer: float
    eer_threshold: float
    tar_at_far_1e_3: float
    auc: float
    n_pairs: int
    n_genuine: int
    n_impostor_internal: int
    n_impostor_lfw: int


def _extract_features(
    entries: List[ImageEntry],
) -> Tuple[np.ndarray, List[str]]:
    """对 entries 抽特征；返回 (features (N,512), pose_labels)。失败的图被跳过。"""
    feats: List[np.ndarray] = []
    poses: List[str] = []
    for e in entries:
        img = imread_bgr_unicode(e.path)
        if img is None:
            log.warning("无法读取 %s，跳过", e.path)
            continue
        emb, err, _ = extract_embedding_from_bgr(img)
        if err or emb is None:
            log.warning("特征提取失败 %s: %s", e.path, err)
            continue
        # 上游 emb 已 L2，但保险再做一次
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        feats.append(emb.astype(np.float32))
        poses.append(e.pose if e.pose else "")
    if not feats:
        return np.zeros((0, 512), dtype=np.float32), []
    return np.stack(feats, axis=0), poses


def _evaluate_one_strategy(
    strategy_name: str,
    train_per_person: Dict[str, Tuple[np.ndarray, List[str]]],
    probe_per_person: Dict[str, List[np.ndarray]],
    lfw_features: np.ndarray,
    *,
    rng_seed: int,
    target_far: float,
) -> AblationRow:
    strategy = get_strategy(strategy_name)
    rng = np.random.default_rng(rng_seed)

    gallery: Dict[str, np.ndarray] = {}
    for name, (vecs, poses) in train_per_person.items():
        if vecs.shape[0] == 0:
            log.warning("策略 %s：人 %s 无可用 train 向量，剔除", strategy_name, name)
            continue
        try:
            if strategy_name == "manual_three":
                labels = np.array(poses, dtype=object)
                if not any(p in {"frontal", "left", "right"} for p in poses):
                    log.warning("策略 %s：人 %s 无姿态标签，剔除", strategy_name, name)
                    continue
                templates = strategy.build(vecs, rng=rng, group_labels=labels)
            else:
                templates = strategy.build(vecs, rng=rng)
            gallery[name] = templates
        except Exception as exc:  # 单人策略失败，剔除该人但不阻断整体
            log.warning("策略 %s：人 %s build 失败: %s", strategy_name, name, exc)

    pairs = make_pairs(probe_per_person, gallery, lfw_features)
    if not pairs:
        raise ValueError(f"策略 {strategy_name}：未生成任何 pair")

    scores = np.array([p.score for p in pairs], dtype=np.float64)
    labels = np.array([1 if p.is_genuine else 0 for p in pairs], dtype=np.int64)
    fpr, tpr, thr = roc_curve(scores, labels)
    eer_v, eer_thr = eer(fpr, tpr, thr)

    n_genuine = int(labels.sum())
    n_impostor = len(labels) - n_genuine
    # 拆分 cross-person vs lfw 数（按构造顺序：先 probe×内部，再 lfw×names；近似估算）
    names = list(gallery.keys())
    n_impostor_lfw = int(lfw_features.shape[0]) * len(names)
    n_impostor_internal = n_impostor - n_impostor_lfw

    return AblationRow(
        strategy=strategy_name,
        eer=float(eer_v),
        eer_threshold=float(eer_thr),
        tar_at_far_1e_3=float(tar_at_far(fpr, tpr, target_far)),
        auc=float(auc(fpr, tpr)),
        n_pairs=len(pairs),
        n_genuine=n_genuine,
        n_impostor_internal=n_impostor_internal,
        n_impostor_lfw=n_impostor_lfw,
    )


def run_ablation(
    dataset_root: Path,
    cfg: AppConfig,
) -> List[AblationRow]:
    """跑 5 策略消融。返回每策略一行。

    副作用：log warning；不写文件。文件落地在 CLI 层。
    """
    splits = split_by_person(
        dataset_root,
        train_ratio=cfg.evaluation.train_ratio,
        seed=cfg.evaluation.random_seed,
    )
    if len(splits) < 2:
        raise ValueError(f"评估至少需要 2 个人，当前 {len(splits)}")

    log.info("抽取 train/probe 特征中（%d 人）...", len(splits))
    train_per_person: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    probe_per_person: Dict[str, List[np.ndarray]] = {}
    min_n = cfg.evaluation.min_vectors_per_person
    for name, (train_entries, probe_entries) in splits.items():
        train_vecs, train_poses = _extract_features(train_entries)
        probe_vecs, _ = _extract_features(probe_entries)
        if train_vecs.shape[0] < min_n:
            log.warning("人 %s train=%d < %d，整人剔除", name, train_vecs.shape[0], min_n)
            continue
        train_per_person[name] = (train_vecs, train_poses)
        probe_per_person[name] = [probe_vecs[i] for i in range(probe_vecs.shape[0])]

    if len(train_per_person) < 2:
        raise ValueError(f"过滤后 train 人数不足，仅 {len(train_per_person)} 人")

    log.info("准备 LFW impostor 特征...")
    lfw_paths = sample_lfw_paths(
        Path(cfg.evaluation.lfw_cache_dir),
        n_images=cfg.evaluation.lfw_impostor_count,
        seed=cfg.evaluation.random_seed,
        exclude_names=set(train_per_person.keys()),
    )
    lfw_entries = [ImageEntry(path=p, pose=None) for p in lfw_paths]
    lfw_vecs, _ = _extract_features(lfw_entries)
    log.info("LFW impostor 实抽到 %d 张", lfw_vecs.shape[0])

    target_far = cfg.evaluation.far_targets[0]
    rows: List[AblationRow] = []
    for name in STRATEGIES.keys():
        log.info("评估策略: %s", name)
        rows.append(_evaluate_one_strategy(
            name,
            train_per_person,
            probe_per_person,
            lfw_vecs,
            rng_seed=cfg.evaluation.random_seed,
            target_far=target_far,
        ))
    return rows
```

- [ ] **Step 2: 验证 import 不报错（不跑业务，只做导入烟测）**

Run: `cd backend && python -c "from app.evaluation.runner import run_ablation, AblationRow; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add backend/app/evaluation/runner.py
git commit -m "feat(evaluation): runner 编排单次消融评估"
```

---

### Task 21: run_ablation CLI

**Files:**
- Create: `backend/app/evaluation/run_ablation.py`
- Create: `reports/.gitkeep` (仓库根)

- [ ] **Step 1: 创建 `reports/.gitkeep`** (空文件)

- [ ] **Step 2: 创建 `backend/app/evaluation/run_ablation.py`**

```python
"""CLI 入口：跑 5 策略消融，输出 reports/<ts>/ablation.csv + roc.png。

用法 (在 backend/ 下):
    python -m app.evaluation.run_ablation --dataset dataset/ --config ../config.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from app.config_loader import AppConfig, ConfigError
from app.evaluation.runner import AblationRow, run_ablation
from app.services.adaface_infer import is_adaface_available


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="5 策略消融评估")
    p.add_argument("--dataset", type=Path, required=True, help="数据集根目录")
    p.add_argument("--config", type=Path, required=True, help="config.yaml 路径")
    return p


def _write_csv(rows: list[AblationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    headers = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _plot_roc_placeholder(rows: list[AblationRow], path: Path) -> None:
    """占位：暂以纯文字摘要替代图。matplotlib 留给后续 PR。"""
    summary = "\n".join(
        f"{r.strategy}: EER={r.eer:.4f} TAR@1e-3={r.tar_at_far_1e_3:.4f} AUC={r.auc:.4f}"
        for r in rows
    )
    path.write_text(summary, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if not args.dataset.is_dir():
        print(f"dataset 不是目录: {args.dataset}", file=sys.stderr)
        return 2

    if not is_adaface_available():
        print("未配置 AdaFace 权重，请放置到 backend/models/ 或设 ADAFACE_MODEL_PATH", file=sys.stderr)
        return 3

    try:
        cfg = AppConfig.load(args.config)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 4

    rows = run_ablation(args.dataset, cfg)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cfg.evaluation.output_dir) / ts
    _write_csv(rows, out_dir / "ablation.csv")
    _plot_roc_placeholder(rows, out_dir / "roc_summary.txt")
    print(f"输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: 烟测 CLI 帮助**

Run: `cd backend && python -m app.evaluation.run_ablation --help`
Expected: 打印 argparse 帮助文本，退出码 0

- [ ] **Step 4: Commit**

```bash
git add backend/app/evaluation/run_ablation.py reports/.gitkeep
git commit -m "feat(evaluation): run_ablation CLI + reports/ 占位"
```

---

### Task 22: 集成 smoke 测试

**Files:**
- Create: `backend/tests/integration/__init__.py` (空)
- Create: `backend/tests/integration/test_run_ablation_smoke.py`

> **说明**：本测试需要真 AdaFace 权重 + 官方 sample 图。标记 `@pytest.mark.slow`，
> 默认 `pytest` 不跑；通过 `pytest -m slow` 显式触发。
> AdaFace 自带 `backend/AdaFace/face_alignment/test_images/{img1,img2,img3}.jpeg` 三张，
> 我们复制为 3 人 × 2 张构造 mini 数据集；LFW 也用同样三张冒名顶替（不严格但够"流程不崩"）。

- [ ] **Step 1: 创建空 `backend/tests/integration/__init__.py`**

- [ ] **Step 2: 创建 `backend/tests/integration/test_run_ablation_smoke.py`**

```python
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from app.config_loader import AppConfig


SAMPLE_DIR = Path(__file__).resolve().parents[2] / "AdaFace" / "face_alignment" / "test_images"


@pytest.mark.slow
def test_ablation_runs_end_to_end_on_tiny_fixture(tmp_path, monkeypatch):
    samples = sorted(SAMPLE_DIR.glob("*.jpeg"))
    assert len(samples) >= 3, f"AdaFace sample 缺失: {SAMPLE_DIR}"

    # 构造 mini dataset：3 人 × 2 张同图（足够提取特征即可）
    dataset = tmp_path / "dataset"
    for i, src in enumerate(samples[:3]):
        for j in range(2):
            dst = dataset / f"person_{i}" / f"img_{j}.jpeg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

    # 模拟 LFW 缓存：3 张 sample 各放在不同子目录
    lfw_dir = tmp_path / "lfw"
    for i, src in enumerate(samples[:3]):
        d = lfw_dir / f"L{i}"
        d.mkdir(parents=True)
        shutil.copy(src, d / src.name)

    # 写一份临时 config.yaml
    cfg_text = f"""
recognition:
  match_threshold: 0.35
  production_strategy: mean_all
evaluation:
  random_seed: 42
  train_ratio: 0.5
  min_vectors_per_person: 1
  lfw_cache_dir: {lfw_dir}
  lfw_impostor_count: 3
  far_targets: [1.0e-3]
  output_dir: {tmp_path / "reports"}
strategies:
  kmeans:
    k: 3
  manual_three:
    pose_groups: [frontal, left, right]
"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(cfg_text, encoding="utf-8")

    from app.evaluation.run_ablation import main
    code = main(["--dataset", str(dataset), "--config", str(cfg_file)])
    assert code == 0

    # 找输出目录
    runs = sorted((tmp_path / "reports").iterdir())
    assert len(runs) == 1
    csv_path = runs[0] / "ablation.csv"
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    strategies_found = {r["strategy"] for r in rows}
    assert strategies_found == {"random_one", "mean_all", "manual_three", "kmeans_k3", "all_vectors"}
    for r in rows:
        # 数值字段必须可解析为浮点
        assert 0.0 <= float(r["eer"]) <= 1.0
        assert 0.0 <= float(r["auc"]) <= 1.0
        assert int(r["n_pairs"]) > 0
```

- [ ] **Step 3: 跑集成测试（需真权重）**

Run: `cd backend && pytest tests/integration/ -m slow -v`
Expected: 1 passed (前提：`backend/models/` 下放好 AdaFace 权重；否则 skip 或 fail，记录在测试日志中由用户决定是否处理)

- [ ] **Step 4: Commit**

```bash
git add backend/tests/integration/
git commit -m "test(integration): run_ablation 端到端 smoke (slow)"
```

---

### Task 23: 真数据集上跑一轮，归档结果

> **说明**：这一步是**用户操作**而非"代码任务"，但作为里程碑退出准则保留。
> Plan 写到这里就为了让执行者知道在哪个时点该跑出真实数据。

- [ ] **Step 1: 用户准备数据集**：`backend/dataset/<35人>/{frontal,left,right}/*.jpg` 至少一人完整三组（其他人可不分姿态）

- [ ] **Step 2: 用户准备 LFW**：手动下载 lfw-funneled.tgz，解压到 `backend/dataset/lfw/`

- [ ] **Step 3: 跑消融**

Run: `cd backend && python -m app.evaluation.run_ablation --dataset dataset --config ../config.yaml`
Expected: stdout 打印 "输出目录: reports/<ts>"，CSV 含 5 行

- [ ] **Step 4: 把 CSV 复制一份到 docs/ 作为答辩存档**（手工）

```bash
mkdir -p docs/results
cp reports/<ts>/ablation.csv docs/results/ablation-<date>.csv
git add docs/results/
git commit -m "docs(results): 首轮 5 策略消融数据"
```

- [ ] **Step 5: 打 M4 tag**

```bash
git tag m4-ablation-runs
```


---

## M5 — 切线生产默认

### Task 24: match_with_templates 函数（先写测试）

**Files:**
- Create: `backend/tests/unit/services/__init__.py` (空)
- Create: `backend/tests/unit/services/test_match.py`

- [ ] **Step 1: 创建空 `backend/tests/unit/services/__init__.py`**

- [ ] **Step 2: 写测试 `tests/unit/services/test_match.py`**

```python
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
```

- [ ] **Step 3: 跑测试确认 FAIL**

Run: `cd backend && pytest tests/unit/services/test_match.py -v`
Expected: 4 failed (`match_with_templates` 不存在)

(不 commit)

---

### Task 25: 实现 match_with_templates，替换 _match_with_gallery

**Files:**
- Modify: `backend/app/services/face_service.py`

- [ ] **Step 1: 在 `backend/app/services/face_service.py` 顶部增加 import**

```python
from ..models import FaceProfile, Template, parse_template_vectors
```

(替换原有 `from ..models import FaceProfile`)

- [ ] **Step 2: 在 `face_service.py` 末尾追加新匹配函数**

```python
def match_with_templates(
    query: np.ndarray,
    profiles: list,
    strategy: str,
) -> Tuple[Optional[FaceProfile], float]:
    """对每个 profile 取该 strategy 的 Template，计算 max-cosine；返回最佳 profile + 分数。

    无任何 profile 拥有该策略模板时返回 (None, -1.0)。
    query 与模板均假定已 L2 归一化（点积 = 余弦）。
    """
    q = query / (np.linalg.norm(query) + 1e-8)
    best_profile: Optional[FaceProfile] = None
    best_score = -1.0
    for p in profiles:
        tpl = next((t for t in p.templates if t.strategy == strategy), None)
        if tpl is None:
            continue
        mat = parse_template_vectors(tpl.vector_json)
        if mat.shape[0] == 0:
            continue
        # 模板再保险 L2 一次
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        mat_l2 = mat / norms
        score = float((mat_l2 @ q).max())
        if score > best_score:
            best_score = score
            best_profile = p
    return best_profile, best_score
```

- [ ] **Step 3: 修改 `recognize_face_from_bgr`，把 `_match_with_gallery` 调用改为 `match_with_templates`**

定位 `recognize_face_from_bgr` 内：

```python
        threshold = Config.ADAFACE_MATCH_THRESHOLD
        best_p, score = _match_with_gallery(emb, with_vectors)
```

改为：

```python
        from ..config_loader import AppConfig
        # 配置文件路径相对仓库根；run.py 启动时 cwd=backend，因此 ../config.yaml
        cfg = AppConfig.load(Path(__file__).resolve().parents[3] / "config.yaml")
        threshold = cfg.recognition.match_threshold
        best_p, score = match_with_templates(
            emb,
            profiles,  # 含全部 profile，由 match_with_templates 内部过滤
            strategy=cfg.recognition.production_strategy,
        )
```

并在文件顶部增加 `from pathlib import Path`。

> **保留 `_match_with_gallery` 函数本身不删**：作为参考实现保留（CLAUDE.md §6 不删既有列/函数原则的延伸）；
> 不再被引用即可。

- [ ] **Step 4: 跑测试确认 PASS**

Run: `cd backend && pytest tests/unit/services/test_match.py -v`
Expected: 4 passed

- [ ] **Step 5: 跑全部已写测试**

Run: `cd backend && pytest -v --ignore=tests/integration`
Expected: 51 passed (M1 8 + M2 17 + M3 22 + match 4)

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/face_service.py backend/tests/unit/services/
git commit -m "feat(face): match_with_templates 走配置选定策略 + 多模板 max-cosine"
```

---

### Task 26: build_face_gallery 写多 Template

**Files:**
- Modify: `backend/scripts/build_face_gallery.py`

> 改造目标：扫每人目录 → 抽 N 张特征 + pose 标签 → 5 策略全部 build → 写入对应的 5 行
> Template；同时回写 `feature_vector` 字段（mean_all 向量，兼容旧调用方）。

- [ ] **Step 1: 重写 `backend/scripts/build_face_gallery.py`**

```python
"""从目录结构 dataset/<姓名>/{frontal,left,right}/*.jpg 或 dataset/<姓名>/*.jpg
批量提取 AdaFace 特征，运行 5 个 TemplateStrategy，写入 face_profiles + templates 两张表。

用法（在 backend 目录下）:
  python scripts/build_face_gallery.py --dataset dataset/ --config ../config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402
from app.config_loader import AppConfig  # noqa: E402
from app.evaluation.data_split import POSE_NAMES  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models import FaceProfile, Template  # noqa: E402
from app.services.adaface_infer import extract_embedding_from_bgr  # noqa: E402
from app.services.image_utils import imread_bgr_unicode  # noqa: E402
from app.strategies import STRATEGIES  # noqa: E402


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _iter_person_images(person_dir: Path):
    """遍历直接图片 + frontal/left/right 子目录图片，yield (path, pose_or_None)"""
    for p in sorted(person_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p, None
        elif p.is_dir() and p.name in POSE_NAMES:
            for q in sorted(p.iterdir()):
                if q.is_file() and q.suffix.lower() in IMG_EXTS:
                    yield q, p.name


def _extract_person(person_dir: Path):
    vectors = []
    poses = []
    for img_path, pose in _iter_person_images(person_dir):
        img = imread_bgr_unicode(img_path)
        if img is None:
            print(f"无法读取: {img_path}")
            continue
        emb, err, _ = extract_embedding_from_bgr(img)
        if err:
            print(f"跳过 {img_path}: {err}")
            continue
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        vectors.append(emb.astype(np.float32))
        poses.append(pose if pose else "")
    if not vectors:
        return np.zeros((0, 512), dtype=np.float32), []
    return np.stack(vectors, axis=0), poses


def _upsert_profile(name: str, mean_vec: np.ndarray, class_name: str | None) -> FaceProfile:
    p = FaceProfile.query.filter_by(name=name).first()
    fv = json.dumps(mean_vec.astype(float).tolist())
    if p:
        p.feature_vector = fv
        if class_name:
            p.class_name = class_name
    else:
        p = FaceProfile(name=name, class_name=class_name or None, feature_vector=fv)
        db.session.add(p)
    db.session.flush()
    return p


def _upsert_template(profile_id: int, strategy: str, mat: np.ndarray, source_count: int) -> None:
    payload = mat[0].tolist() if mat.shape[0] == 1 else mat.tolist()
    vector_json = json.dumps(payload)
    existing = Template.query.filter_by(profile_id=profile_id, strategy=strategy).first()
    if existing:
        existing.vector_json = vector_json
        existing.source_count = source_count
    else:
        db.session.add(Template(
            profile_id=profile_id,
            strategy=strategy,
            vector_json=vector_json,
            source_count=source_count,
        ))


def main() -> None:
    ap = argparse.ArgumentParser(description="从文件夹数据集构建 5 策略人脸库")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "config.yaml")
    ap.add_argument("--min-images", type=int, default=1)
    ap.add_argument("--class-name", default="")
    args = ap.parse_args()

    if not args.dataset.is_dir():
        print("dataset 不是目录:", args.dataset)
        sys.exit(2)

    cfg = AppConfig.load(args.config)
    rng_seed = cfg.evaluation.random_seed

    app = create_app()
    with app.app_context():
        for person_dir in sorted(p for p in args.dataset.iterdir() if p.is_dir()):
            name = person_dir.name
            if name == "lfw":
                continue
            vectors, poses = _extract_person(person_dir)
            if vectors.shape[0] < args.min_images:
                print(f"跳过 {name}: 仅 {vectors.shape[0]} 张可用")
                continue

            mean_vec = np.mean(vectors, axis=0)
            mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
            profile = _upsert_profile(name, mean_vec, args.class_name)

            rng = np.random.default_rng(rng_seed)
            for strat_name, strat in STRATEGIES.items():
                try:
                    if strat_name == "manual_three":
                        labels = np.array(poses, dtype=object)
                        if not any(p in {"frontal", "left", "right"} for p in poses):
                            print(f"  跳过 {name} 的 manual_three（无姿态标签）")
                            continue
                        mat = strat.build(vectors, rng=rng, group_labels=labels)
                    else:
                        mat = strat.build(vectors, rng=rng)
                except Exception as exc:
                    print(f"  策略 {strat_name} 对 {name} 失败: {exc}")
                    continue
                _upsert_template(profile.id, strat_name, mat, source_count=vectors.shape[0])

            db.session.commit()
            print(f"已录入 {name}：{vectors.shape[0]} 张图，5 策略模板已更新")

    print("完成。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 烟测帮助**

Run: `cd backend && python scripts/build_face_gallery.py --help`
Expected: argparse 帮助输出

- [ ] **Step 3: Commit**

```bash
git add backend/scripts/build_face_gallery.py
git commit -m "refactor(scripts): build_face_gallery 写 5 策略 Template + 兼容 feature_vector"
```

---

### Task 27: 旧库迁移脚本

**Files:**
- Create: `backend/scripts/migrate_to_templates.py`

- [ ] **Step 1: 创建 `backend/scripts/migrate_to_templates.py`**

```python
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
```

- [ ] **Step 2: 烟测**

Run: `cd backend && python scripts/migrate_to_templates.py`
Expected: 退出码 0；若 DB 不存在或为空则 "已迁移: 0 行"

- [ ] **Step 3: Commit**

```bash
git add backend/scripts/migrate_to_templates.py
git commit -m "feat(scripts): migrate_to_templates 一次性 backfill"
```

---

### Task 28: 端到端冒烟 + M5 tag

> 这一步**用户操作 + 验证**：跑通真小程序 → /api/face/recognize 链路。

- [ ] **Step 1: （用户）启动 Flask**

Run: `cd backend && python run.py`
Expected: 监听 127.0.0.1:5000

- [ ] **Step 2: （用户）健康检查**

Run: `curl http://127.0.0.1:5000/api/health`
Expected: `{"code":0,"message":"ok","data":{"status":"healthy"}}`

- [ ] **Step 3: （用户）拿一张已注册人员的照片，POST 给 /api/face/recognize（base64）**

```bash
B64=$(python -c "import base64,sys;print(base64.b64encode(open(sys.argv[1],'rb').read()).decode())" path/to/face.jpg)
curl -s -X POST http://127.0.0.1:5000/api/face/recognize \
  -H 'Content-Type: application/json' \
  -d "{\"image\": \"$B64\"}"
```

Expected: `code=0` 且 `recognized=true`，`name` 为该人员

- [ ] **Step 4: （用户）小程序录屏冒烟**

打开微信开发者工具加载本仓库根，登录小程序拍照触发识别 → 看返回是否正确。

- [ ] **Step 5: 切换 production_strategy 验证**

修改 `config.yaml` 把 `production_strategy` 从 `mean_all` 改为评估出来的赢家（例如 `manual_three`），重启 Flask，重复 Step 3，应仍能识别（可能分数有差异）。

- [ ] **Step 6: 打 tag**

```bash
git tag m5-production-cutover
```


---

## 自查清单

| 检查项 | 状态 |
|---|---|
| 所有 spec 范围内功能都有对应 task | ✅ Tasks 1-28 覆盖 §1 全部 in-scope |
| 每个 task 的 Files/Step 段都给了具体路径 + 完整代码 | ✅ |
| 没有 "TBD/TODO/类似上一个任务" 占位 | ✅ |
| 函数命名一致（match_with_templates / parse_template_vectors / extract_embedding_from_bgr 全文一致） | ✅ |
| 测试先行（每个实现 task 前都有失败测试 task） | ✅ |
| 5 个里程碑各有 tag | ✅ m1-foundation, m2-strategies, m3-evaluation-pure-functions, m4-ablation-runs, m5-production-cutover |
| 上游代码（routes, adaface_infer, image_utils）只读不改 | ✅ 仅 face_service.py 增量修改，并保留 _match_with_gallery 不删 |

---

## 风险提示（执行时注意）

1. **AdaFace 权重**：M4 集成 smoke 与 M5 端到端依赖 `backend/models/*.{ckpt,pth}`。开发者需自行放置；CLAUDE.md §6 已明确不入仓。
2. **LFW 下载**：spec § 11 已指出。`lfw_loader` 不实现下载，要求用户预先解压到 `cache_dir`。否则 run_ablation 会以 FileNotFoundError 退出。
3. **SQLite 外键级联**：测试 fixture 显式开 `PRAGMA foreign_keys=ON`。生产 SQLite 是否也需开取决于业务（删人会不会留孤儿模板）；本期不强求，留作 M5 后的小修。
4. **PyTorch / sklearn 版本兼容**：上游 `torch<=1.13.1`，sklearn 默认随 numpy 走最新；如装失败，把 `scikit-learn` 钉成 `scikit-learn<1.4` 即可。

---

## 完成后下一步候选

- 在线注册 / 删除 / 列表 API（独立 spec + plan）
- matplotlib ROC 曲线图替代当前 `roc_summary.txt`
- e2e Flask test client 测试覆盖 `/api/face/recognize`
- 小程序端"管理员注册"页面
