# 5 策略消融 + 评估体系设计 spec

> 起草于 2026-05-25 · 状态：drafted, awaiting user review · 作者: brainstorming session
>
> 本 spec 仅覆盖**主线"5 策略消融 + 评估体系"**及最小必要的工程附属（`config.yaml` 外置 + `strategies/`/`evaluation/` 单元测试）。
> 不在范围内：在线注册/删除 API、测试金字塔向上游已有代码扩张、模型量化/onnx 等。
>
> 本 spec 的实施计划交由 `superpowers:writing-plans` 技能产出，独立文档存于 `docs/superpowers/plans/`。

---

## 0. 背景与目标

本项目 fork 自 [Violet52014/AdaFace_Recognition](https://github.com/Violet52014/AdaFace_Recognition)，
上游已有 Flask 后端 + 微信小程序 + AdaFace 推理胶水 + MTCNN 对齐。
本 spec 要在上游基础上叠加**学术差异化卖点**：

1. 比较 5 种"用 N 张照片为一个人构造模板"的策略
2. 在档案内（35 人 × ≥20 张）+ LFW 库外人脸 上跑统一评估协议
3. 产出 ROC / EER / TAR\@FAR=1e-3 / AUC 四组指标，作为答辩报告核心数据
4. 评估完后在 `config.yaml` 选定一个策略作为生产默认，识别路径切到该策略

完成后，答辩展示链路：**"5 策略 × 评估表 × ROC 曲线 × 选定策略 × 在线小程序识别"**。

---

## 1. 范围

### 在范围内
- 新增 `Template` 表，多 Template 多对一 `FaceProfile`
- `app/strategies/` —— 5 个 `TemplateStrategy` 纯函数实现
- `app/evaluation/` —— `data_split` / `lfw_loader` / `pair_generator` / `metrics` / `runner` / `run_ablation`
- `config.yaml` + `app/config_loader.py`（pydantic-settings）
- 上游 `face_service._match_with_gallery` 替换为支持多模板 max-cosine 的 `match_with_templates`
- `scripts/build_face_gallery.py` 改造：调 5 策略写多 Template
- 一次性迁移脚本 `scripts/migrate_to_templates.py`：把现有 `feature_vector` 写成 `strategy='mean_all'` 的 Template
- `tests/unit/{strategies,evaluation}` 完整单元测试
- 1 个集成 smoke 测试（`@pytest.mark.slow`，使用 AdaFace 自带 sample 图）
- `reports/` 目录（gitignore 内容，仅 `.gitkeep`）

### 不在范围内（YAGNI 边界，本 spec 不做）
- 在线注册 / 删除 / 列表 API（`routes/persons.py`）
- 上游 `routes/{face,records,stats}.py` 的 HTTP 契约不动
- 上游 `services/adaface_infer.py` / `image_utils.py` 不修改既有签名
- Alembic 迁移工具
- FAISS / Milvus / 任何工业级向量库
- 在线推理路径"动态切换策略"（生产策略由 `config.yaml` 静态决定）
- 评估结果落 DB（仅落 `reports/<timestamp>/` CSV+PNG）
- 模型训练 / fine-tune（CLAUDE.md §2 严禁）

---

## 2. 关键决策记录

| # | 决策 | 取代项 | 原因 |
|---|---|---|---|
| D1 | 主线选"5 策略消融 + 评估"，其他增量延后 | 一次做完 6 项 | 单 spec 单 plan 原则；学术卖点优先 |
| D2 | 新增 `templates` 表，多对一 | feature_vector 内嵌 dict | 可走 SQL，可记元信息，符合 CLAUDE.md §2 |
| D3 | `vector_json` 一字段统一存"单向量 / 向量列表" | 两个字段或两张表 | 上层 `parse_template_vectors` 总返回 `(M,512)`，消除分支 |
| D4 | 多模板匹配用 `max-cosine` | mean / sum / vote | 简单、可解释、答辩好讲；与 manual_three / kmeans_k3 / all_vectors 的语义直接对应 |
| D5 | 评估协议：档案内 80/20 + LFW impostor，固定 seed=42 | LFW 6000-pair 协议 / 仅档案内 | 档案场景最贴合实际；LFW 提供"完全陆生人"impostor 提升 FAR=1e-3 的统计显著性 |
| D6 | 5 策略仅离线评估，生产单选其一 | 在线投票 / 在线动态切换 | YAGNI；35 人规模演示价值低 |
| D7 | `manual_three` 的"质量分"= MTCNN bbox prob | 手工标注 / 模糊度算子 | MTCNN 已有现成 prob，不引入新依赖 |
| D8 | `kmeans_k3` 的 k 来自 config | 写死 3 | 答辩时讲超参可调更专业；写死等价于硬编码魔数 |
| D9 | 上游 `feature_vector` 字段保留不删 | 删除 | CLAUDE.md §6 "不删既有列"，保兼容 |
| D10 | `metrics.py` 全部纯 numpy 实现 | 调 sklearn.metrics | 数学层是项目核心声明，必须可被合成数据测出来 |
| D11 | 评估结果不入 DB | 写 `evaluation_runs` 表 | 无答辩价值，复杂化 schema |
| D12 | 不引入 Alembic | 用 Alembic 迁移 | 上游本来就 `db.create_all()`，YAGNI |

---

## 3. 架构

### 3.1 目录布局（增量部分加 `[+]`）

```
backend/
├── app/
│   ├── strategies/                  [+] 纯函数层，零外部依赖除 numpy/sklearn
│   │   ├── __init__.py              [+] 注册表 STRATEGIES: dict[str, TemplateStrategy]
│   │   ├── base.py                  [+] Protocol TemplateStrategy
│   │   ├── random_one.py            [+]
│   │   ├── mean_all.py              [+]
│   │   ├── manual_three.py          [+]
│   │   ├── kmeans_k3.py             [+]
│   │   └── all_vectors.py           [+]
│   ├── evaluation/                  [+] 离线脚本 + 纯函数
│   │   ├── __init__.py              [+]
│   │   ├── data_split.py            [+]
│   │   ├── lfw_loader.py            [+]
│   │   ├── pair_generator.py        [+]
│   │   ├── metrics.py               [+]
│   │   ├── runner.py                [+]
│   │   └── run_ablation.py          [+] CLI 入口
│   ├── config_loader.py             [+] pydantic-settings 读 config.yaml
│   ├── models.py                    [~] 新增 Template 表
│   ├── services/
│   │   ├── face_service.py          [~] _match_with_gallery → match_with_templates
│   │   └── adaface_infer.py         [~] 新增 extract_embedding_with_quality；老函数 extract_embedding_from_bgr 完全不动
│   ├── routes/                      [-] 不动
│   ├── extensions.py                [-] 不动
│   └── config.py                    [-] 不动（上游环境变量风格保留）
├── scripts/
│   ├── build_face_gallery.py        [~] 调 5 策略写多 Template
│   └── migrate_to_templates.py      [+] 一次性迁移脚本
├── tests/                           [+]
│   ├── conftest.py                  [+]
│   ├── unit/
│   │   ├── strategies/              [+] 5 文件
│   │   ├── evaluation/              [+] 4 文件
│   │   └── test_config_loader.py    [+]
│   └── integration/
│       └── test_run_ablation_smoke.py  [+] @pytest.mark.slow
└── requirements.txt                 [~] +pytest, pytest-cov, pydantic-settings, PyYAML, scikit-learn
config.yaml                          [+] 仓库根
docs/superpowers/specs/2026-05-25-ablation-and-evaluation-design.md   [+] 本文件
reports/.gitkeep                     [+]
```

### 3.2 依赖规则（强制）

```
evaluation/ ──→ strategies/ ──→ numpy
evaluation/ ──→ services/adaface_infer.py    （仅离线时调用提特征）
evaluation/ 不依赖 routes/ 不依赖 models/
strategies/ 不依赖任何 app 内其他模块（只用 numpy / sklearn）
```

违反规则的代码即"放错了地方"，必须停下重新放置。

---

## 4. 数据模型

### 4.1 新表 `templates`

```python
class Template(db.Model):
    __tablename__ = "templates"

    id            = db.Column(db.Integer, primary_key=True)
    profile_id    = db.Column(db.Integer,
                              db.ForeignKey("face_profiles.id", ondelete="CASCADE"),
                              nullable=False, index=True)
    strategy      = db.Column(db.String(32), nullable=False, index=True)
    vector_json   = db.Column(db.Text,        nullable=False)
    source_count  = db.Column(db.Integer,     nullable=False)
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow,
                                              nullable=False)

    profile = db.relationship(
        "FaceProfile",
        backref=db.backref("templates", cascade="all, delete-orphan", lazy="select"),
    )

    __table_args__ = (
        db.UniqueConstraint("profile_id", "strategy", name="uq_profile_strategy"),
    )
```

### 4.2 字段语义

- **`strategy`**：取值 ∈ {`random_one`, `mean_all`, `manual_three`, `kmeans_k3`, `all_vectors`}，由策略注册表强约束。
- **`vector_json`**：1D 向量（M=1）或 2D 向量列表（M>1）的 JSON 序列化。
  - 1D：`json.dumps([f0, f1, ..., f511])`
  - 2D：`json.dumps([[...], [...], ...])`
  - 读取通过 `parse_template_vectors() -> np.ndarray of shape (M, 512)`，M=1 也返回二维。
- **`source_count`**：构造该模板时所用的原图（成功提取出特征的）数量。
- **`(profile_id, strategy)` 唯一**：build_face_gallery 重跑时 upsert，避免脏数据。

### 4.3 兼容字段

`FaceProfile.feature_vector` **保留不删**：
- 本轮新代码不再读它
- `build_face_gallery.py` 改造后仍写 `mean_all` 向量到该字段（兼容未升级调用方）
- 同时写一行 `Template(strategy='mean_all', vector_json=...)`

### 4.4 schema 迁移方案

- 开发期使用 `db.create_all()` 自动建 `templates` 表（与上游风格一致）
- 一次性 backfill 脚本 `scripts/migrate_to_templates.py`：扫现有 `face_profiles.feature_vector` → 写为 `strategy='mean_all'` 的 Template 行
- 不引入 Alembic（YAGNI；本科期末项目）

---

## 5. 核心接口

### 5.1 `strategies/base.py`

```python
from typing import Protocol
import numpy as np

class TemplateStrategy(Protocol):
    name: str

    def build(self, vectors: np.ndarray, *,
              rng: np.random.Generator,
              quality_scores: np.ndarray | None = None) -> np.ndarray:
        """
        参数:
          vectors: (N, 512) 已 L2 归一化的同一人原始特征
          rng: numpy 随机源（确定性）
          quality_scores: (N,) 浮点，仅 manual_three 使用，其他策略忽略
        返回:
          (M, 512)，M ∈ {1, 3, N}，每行已 L2 归一化
        """
        ...
```

### 5.2 五个策略实现要点

| `name` | M | 实现 |
|---|---|---|
| `random_one` | 1 | `vectors[rng.integers(N)][None, :]`；输入已 L2，返回保持 |
| `mean_all` | 1 | `mean(vectors, axis=0)` → L2 归一化（与上游 build_face_gallery L77-78 等价） |
| `manual_three` | min(3, N) | 取 `quality_scores` 最高的前 K=3 行；返回各行（已 L2） |
| `kmeans_k3` | min(3, N) | `sklearn.cluster.KMeans(n_clusters=k, random_state=cfg.evaluation.random_seed).fit(vectors).cluster_centers_` → 各行 L2 归一化 |
| `all_vectors` | N | 原样返回输入（已 L2） |

K 值（`top_k`、`k`）由 `config.yaml` `strategies.*` 提供，默认值 3。

### 5.3 `evaluation/` 关键函数签名

```python
# data_split.py
def split_by_person(dataset_root: Path, *, train_ratio: float, seed: int
                    ) -> dict[str, tuple[list[Path], list[Path]]]:
    """每人切 train/probe；同 seed → 同结果。每人至少 1 张时全部入 train，probe=空且记 warning。"""

# lfw_loader.py
def load_lfw_impostors(cache_dir: Path, *, n_images: int, seed: int,
                       exclude_names: set[str]) -> list[Path]:
    """首次调用自动下载 LFW funneled 子集到 cache_dir；按 seed 抽样 n_images 张；不与 exclude_names 重名。"""

# pair_generator.py
@dataclass(frozen=True)
class Pair:
    score: float
    is_genuine: bool

def make_pairs(probe_features: dict[str, list[np.ndarray]],
               gallery_templates: dict[str, np.ndarray],   # value shape=(M_i, 512)
               lfw_features: np.ndarray) -> list[Pair]:
    """
    构造三类 pair（score = max-cosine）：
      genuine:           probe_i vs gallery_templates[name_i]
      impostor_internal: probe_i vs gallery_templates[name_j]   for j != i
      impostor_lfw:      lfw_k   vs gallery_templates[name_j]   for all j
    """

# metrics.py（纯 numpy）
def roc_curve(scores: np.ndarray, labels: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...   # fpr, tpr, thresholds
def eer(fpr, tpr, thresholds) -> tuple[float, float]: ...           # eer, threshold@eer
def tar_at_far(fpr, tpr, target_far: float) -> float: ...
def auc(fpr, tpr) -> float: ...
```

### 5.4 在线匹配函数（替换 `face_service._match_with_gallery`）

```python
def match_with_templates(query_l2: np.ndarray,
                         profiles: list[FaceProfile],
                         strategy: str
                         ) -> tuple[FaceProfile | None, float]:
    """
    对每个 profile：取该 strategy 的 Template，解析为 (M_i, 512)，计算 max-cosine。
    返回得分最高 profile + 最高分；profile 没有该 strategy 模板则跳过并 log.warning。
    """
```

---

## 6. 数据流

### 6.1 离线评估 (`run_ablation`)

```
dataset/35人          ←── data_split (80/20, seed=42)
   ├─→ train: 抽特征(L2) → 5 策略 build() → 5 套 gallery_templates
   └─→ probe: 抽特征(L2) → probe_features

dataset/lfw  ──→ lfw_loader → 抽特征(L2) → lfw_features

  ┌─ for strategy_name in 5 策略:
  │     pairs = pair_generator(probe_features,
  │                            gallery_templates[strategy_name],
  │                            lfw_features)
  │     scores, labels = unpack(pairs)
  │     fpr, tpr, thr = metrics.roc_curve(scores, labels)
  │     row = {strategy, EER, TAR@1e-3, AUC, threshold@EER, n_pairs}
  └─ rows → reports/<ts>/ablation.csv + roc.png
```

**性能优化**：probe / LFW 特征**只抽一次**，5 策略复用同一份。

### 6.2 在线识别（M5 切线后）

```
HTTP /api/face/recognize        (上游契约不动)
  → adaface_infer.extract_embedding_from_bgr(image)        (上游不动)
  → match_with_templates(query, profiles,
                         strategy=cfg.recognition.production_strategy)
  → 与 cfg.recognition.match_threshold 比较 → 返回结果
```

切换生产策略 = 改 `config.yaml` 一行 + 重启服务，**无需改代码**。

---

## 7. 配置 (`config.yaml`)

仓库根 `config.yaml`：

```yaml
recognition:
  match_threshold: 0.35
  production_strategy: mean_all     # ∈ {random_one, mean_all, manual_three, kmeans_k3, all_vectors}

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
  manual:
    top_k: 3
```

`app/config_loader.py` 用 `pydantic-settings` 加载为 `AppConfig` 单例。
启动时 / CLI 入口处 `AppConfig.load()` 校验失败立即 `sys.exit`。
**上游 `config.py` 不动**，新代码全部走 `AppConfig`。

---

## 8. 错误处理

按 CLAUDE.md §4 "只在系统边界做校验，内部相互信任"。

### 8.1 CLI 入口边界

| 失败场景 | 行为 |
|---|---|
| `--dataset` 不存在或非目录 | 打印错误 + `sys.exit(2)` |
| 数据集人数 < 2 | 拒绝，明确报错 cross-person impostor 至少需要 2 人 |
| 缺 AdaFace 权重 | 复用上游 `is_adaface_available()`，缺失时 `sys.exit(3)` |
| `config.yaml` 缺失 / 类型错 | pydantic `ValidationError` → 打印 + `sys.exit(4)` |
| LFW 下载失败 | 抛带原因的异常，不重试不静默 |

### 8.2 单张照片处理失败

唯一允许"warning + skip"的层。当某人最终可用图 < `min_vectors_per_person`（默认 5），整人剔除并 warning。

### 8.3 在线 HTTP 边界

保持上游 `routes/face.py` 错误契约不变。`match_with_templates` 内部对"profile 缺该 strategy 模板"的情况 warning + 跳过；若全部缺，返回 `recognized=False, message="人脸库未构建 X 策略模板"`。

### 8.4 metrics 边界值（**测试硬性 lock**）

| 输入 | 期望 |
|---|---|
| labels 单一类（全 0 或全 1） | `eer` / `auc` 抛 `ValueError("labels 必须同时包含 0 和 1")` |
| `target_far` 比所有 fpr 都小 | `tar_at_far` 返回 `0.0` |
| 所有 score 完全相同 | `auc=0.5`；`eer=0.5` |
| scores 含 nan | `roc_curve` 抛 `ValueError`，不静默剔除 |

### 8.5 不做

- 不引入自定义异常层次（用 stdlib `ValueError` / `FileNotFoundError`）
- 不写 retry / fallback
- 不接 Sentry / structlog（`logging.basicConfig(level=INFO)` 即可）

---

## 9. 测试策略

### 9.1 单元（30+ test，毫秒级，CI 默认）

**`tests/unit/strategies/`** —— 5 文件

- `test_random_one`：shape=(1,512)；同 seed → 同行索引；范数 ≈ 1
- `test_mean_all`：与 `np.mean+l2` 等价；与上游 `build_face_gallery` mean 行为一致（**回归测试**）
- `test_manual_three`：行序与 quality_scores 排序一致；N<3 时 M=N；prob 全相等时取前 3 稳定
- `test_kmeans_k3`：shape=(min(3,N),512)；同 seed 可复现；N=2 时退化 k=2
- `test_all_vectors`：输出与输入相等；不修改输入数组

**`tests/unit/evaluation/test_metrics.py`** —— 数学锚点

- `test_eer_perfect_separation`：完美可分 → EER=0, AUC=1
- `test_eer_random_uniform`：均匀随机标签 → AUC≈0.5（容差 0.05，2000 样本）
- `test_tar_at_far_unreachable_returns_zero`
- `test_roc_curve_rejects_nan`
- `test_eer_rejects_single_class_labels`
- `test_roc_curve_constant_scores_degenerate`：常数 scores → AUC=0.5, EER=0.5

**`tests/unit/evaluation/test_pair_generator.py`** —— pair 计数

- `test_pair_counts`：3 人 × 2 probe × 单模板 + LFW 5 张 → genuine=6, impostor_internal=12, impostor_lfw=15
- `test_multi_template_uses_max_cosine`：(M=3) 模板含 1 个与 probe 完全一致 → score=1.0（不是 mean）

**`tests/unit/evaluation/test_data_split.py`**

- `test_split_is_deterministic_given_seed`
- `test_split_ratio` ≈ 80/20
- `test_split_is_per_person`
- `test_minimum_two_per_person_falls_back`：单张照片的人 → 全部入 train + warning

**`tests/unit/test_config_loader.py`**

- 默认 yaml 加载成功
- 拒绝非法 strategy 名
- threshold 必须 ∈ [0, 1]

### 9.2 集成（`-m slow`）

```python
# tests/integration/test_run_ablation_smoke.py
@pytest.mark.slow
def test_ablation_runs_end_to_end_on_tiny_fixture(tmp_path):
    """
    用 backend/AdaFace/face_alignment/test_images/ 官方样图构造 mini 数据集（3 人 × 2 张）。
    跑 run_ablation，验证：
      - reports/<ts>/ablation.csv 存在且 5 行
      - 每行 EER / TAR / AUC 数值合法
      - 5 策略产生的模板数符合 (1, 1, ≤3, ≤3, N) 模式
    """
```

### 9.3 不做

- 不测 AdaFace 库本身
- 不为 `build_face_gallery.py` / `migrate_to_templates.py` 写测试（一次性 CLI）
- 不为 `runner.py` 写单测（薄编排，集成测试覆盖即可）
- 不写 e2e（M5 切线后另起 PR 写 Flask test client smoke）

### 9.4 工具链增量

`backend/requirements.txt`：
```
pytest
pytest-cov
pydantic-settings
PyYAML
scikit-learn
```

`backend/pytest.ini`（或 `pyproject.toml [tool.pytest.ini_options]`）：
```ini
[pytest]
markers = slow: 集成测试，需要权重，~分钟级
addopts = -ra -q
testpaths = tests
```

---

## 10. 里程碑

| M | 交付 | 退出准则 |
|---|---|---|
| **M1** | 地基 | `config.yaml` + `AppConfig` 加载工作；`Template` 表 `db.create_all` 出来；`tests/unit/test_config_loader.py` 全绿；测试脚手架（conftest, pytest.ini）就位 |
| **M2** | 5 策略 | `app/strategies/` 5 个实现 + `STRATEGIES` 注册表；`tests/unit/strategies/*` 全绿（≈18 用例） |
| **M3** | 评估纯函数 | `data_split` / `lfw_loader` / `pair_generator` / `metrics` 实现；`tests/unit/evaluation/*` 全绿（≈14 用例） |
| **M4** | run_ablation 跑通 | `runner.py` + `run_ablation.py` CLI；在真 35 人 + LFW 上跑出 `reports/<ts>/ablation.csv` + `roc.png`；集成 smoke 测试在 mini fixture 上绿 |
| **M5** | 切线生产默认 | 选定赢家策略写入 `config.yaml`；`face_service` 切到 `match_with_templates`；`build_face_gallery.py` 写多 Template；`migrate_to_templates.py` 跑通；上游 4 个 HTTP 接口的人工冒烟验证（小程序 → /api/face/recognize 跑通） |

每完成一个 M 打 git tag（`m1-foundation`, `m2-strategies`, ...）。

---

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| LFW 下载源不稳定 | `lfw_loader` 接受用户预先放置的 `cache_dir`；首次失败明确报错指引手动下载 |
| 35 人 × 20% probe ≈ 140 个 genuine pair，FAR=1e-3 统计噪声大 | 加 LFW × 35 ≈ 35000 个 impostor，FAR 分母拉大；输出 CSV 标注 `n_pairs` 让答辩可查 |
| sklearn KMeans 在 N=1 时崩溃 | `kmeans_k3` 内部判断 N<k 时退化为 `k=N`，N=1 时直接退化为 `mean_all` 的输出语义并 warning |
| `extract_embedding_from_bgr` 改返回值会破坏上游调用方 | 不改既有元组形状，新增第 4 元素 `prob: float` 仅追加；旧调用方解包前 3 元素仍工作 → **本 spec 改为：质量分通过新增独立函数 `extract_embedding_with_quality` 暴露**，老函数完全不动（更安全） |
| 数据集私照不能 commit | `.gitignore` 已含 `dataset/`；新增 `reports/*` 也入 gitignore；CI fixtures 仅用 AdaFace 自带 sample |

---

## 12. 后续

本 spec 落地完成后，下一轮 brainstorming 候选主题（独立 spec）：
- 在线注册 / 删除 / 列表 API（`routes/persons.py`）
- 上游 routes 的 e2e 测试覆盖
- 小程序端"管理员注册"页面

下一步：`superpowers:writing-plans` 产出实施计划文档 `docs/superpowers/plans/2026-05-25-ablation-and-evaluation-plan.md`。
