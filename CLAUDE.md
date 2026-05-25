# 项目指令 — 基于 AdaFace 的人脸识别系统（本科期末项目）

> 这是一份给 Claude Code（以及其他 AI 协作者）的项目级指令。
> **任何在本仓库工作的 AI 助手都必须先读这份文件**。
>
> 本项目是从开源仓库 [Violet52014/AdaFace_Recognition](https://github.com/Violet52014/AdaFace_Recognition) fork 而来的二次开发。
> 原仓库提供了：Flask 后端 + 微信小程序前端 + AdaFace 推理胶水代码 + MTCNN 对齐。
> 本次开发的目标是**在它的基础上做改进**，重点是**消融实验 + 评估体系 + 工程规范**。

---

## 1. 项目目的

构建一个**支持动态增删人员、有完整评估体系的开放集人脸识别系统**，作为本科期末项目交付。

- **底座**：上游仓库已有的 Flask + AdaFace + 小程序前端
- **库内规模**：约 35 人（一个班级量级），每人 ≥ 20 张照片
- **应用场景**：小型门禁 / 公司考勤 / 个人相册管理
- **核心增量能力**（在上游基础上要新加的）：
  1. **5 策略多模板消融对比**（random_one / mean_all / manual_three / kmeans_k3 / all_vectors）
  2. **离线评估系统**（ROC / EER / TAR\@FAR=1e-3 三组指标 + LFW 库外 impostor）
  3. **动态增删人员的注册接口**（当前只有 `build_face_gallery.py` 离线脚本，要补在线 API）
  4. **配置外置**（当前阈值散落在 `config.py` 和环境变量，要统一到 `config.yaml`）
  5. **测试金字塔**（当前零测试，要补单元 + 集成 + 端到端）
- **学术目标**：用消融实验对比 5 种模板生成策略，产出 ROC / EER / TAR\@FAR 三组指标，作为答辩报告核心数据。

---

## 2. 关键决策与边界（极重要）

### ✅ 必须做（已与用户对齐）

| 维度 | 决定 | 说明 |
| --- | --- | --- |
| **后端框架** | **保留 Flask** | 上游已有 6 路由 + 2 表 + ORM，重构没有加分点；新增功能用 Blueprint 增量挂载 |
| **前端形态** | **保留微信小程序为主** | 答辩演示直接用小程序录屏；管理后台如必要再加单文件 HTML，不上 Vue/React |
| **模型** | **AdaFace（IR-50 backbone）+ MTCNN 对齐** | 上游 stack，权重冻结。**禁止训练 / fine-tune** |
| **架构演进** | **增量式分层** | 不动现有 `app/{routes,services}` 结构。新增 `app/evaluation/` 和 `app/application/` 入口点 |
| **消融实验** | 5 模板策略 + ROC/EER/TAR | 这是项目主差异化卖点 |
| **向量库** | **SQLite + numpy 暴力检索** | 35 人规模 numpy 矩阵乘已经毫秒级 |
| **特征向量** | 512 维，**始终 L2 归一化**，余弦相似度 = 点积 | 上游 `_match_with_gallery` 已遵循，新代码必须保持 |
| **数据切分** | 按人 80/20，固定 `RANDOM_SEED = 42` | 评估可复现 |
| **配置** | `config.yaml` + `pydantic-settings` | 阈值 / 路径 / 超参集中在一处，**禁止散落在代码里** |
| **数据库扩展** | 新增字段不破坏现有表 | `feature_vector`(JSON 单向量) 改为多 `Template` 多对一关系，需写迁移脚本 |
| **依赖管理** | **uv + pyproject.toml + uv.lock，三件套在仓库根** | 命令一律从仓库根跑，不需要 `cd backend`；IDE 自动识别根目录 `.venv` |
| **PyTorch 来源** | **平台分流：Win → cu128，Linux → CPU，mac → PyPI** | Win 走 `https://download.pytorch.org/whl/cu128` 拿 GPU wheel（驱动需 ≥ 525.60）；Linux 走 CPU index 避免拖 ~2GB CUDA；mac fallback PyPI 拿 arm64 + MPS。Win 端 lock 在 `2.11.0+cu128`，mac/Linux 在 `2.12.0`/`+cpu` |
| **fallback 依赖** | **`backend/requirements.txt` 仅作 fallback** | 跨机一致性靠 `uv.lock`；`requirements.txt` 不锁版本，仅在装不上 uv 时用 |

### ❌ 严禁做（YAGNI 边界）

下列内容曾在 brainstorming 中被明确否决，**不要"主动加上"它们**——若觉得需要，先停下来问用户：

- 任何形式的**模型训练或微调**（包括 fine-tune 分类头）
- **FAISS / Milvus / Qdrant / Pinecone** 等工业级向量数据库
- **Vue / React + Vite / 独立管理后台 SPA** 等额外前端框架
- **Gradio / Streamlit** 演示 wrapper
- **重试 / 降级 / 熔断 / Sentinel** 等分布式系统模式
- **structlog / loguru** 等花哨日志库（用 `logging.basicConfig` 即可）
- **utils/ / helpers/ / services/** 万能桶目录（上游已有 services/ 是历史遗留，新增的不要再走这个名）
- 独立的 `Detector` / `Encoder` / `SimilarityCalculator` 抽象（AdaFace 推理 + 余弦点积一行，拆开是过度设计）
- `Photo` / `Image` 等图像包装实体（直接用 `np.ndarray`）
- 模型推理性能优化（量化 / TensorRT / ONNX Runtime）
- **重写上游 Flask 为 FastAPI**（用户已明确否决）
- **重写上游 wxapp 为 H5/PWA**（用户已明确否决）
- 任何"为了未来可能扩展"而加的接口、抽象、配置项

---

## 3. 仓库结构（fork 自上游 + 我们的增量）

```
AdaFace_Recognition/
│
├── 🟦 微信小程序前端（上游 fork 自带，必须在仓库根）
│   ├── app.js / app.json / app.wxss        # 入口 + 全局配置
│   ├── pages/{index,logs}/                 # 拍照页 / 记录页
│   ├── images/                             # 图标资源
│   ├── utils/api.js                        # BASE_URL 配置
│   └── project.config.json / sitemap.json  # 微信工具配置
│
├── 🟩 Python 后端
│   └── backend/
│       ├── AdaFace/                        # 上游：模型源码 + MTCNN 对齐（**禁止改动**）
│       ├── app/
│       │   ├── __init__.py                 # create_app + Blueprint 装配
│       │   ├── config.py                   # 上游：环境变量驱动的旧配置（保留）
│       │   ├── config_loader.py            # 我们：pydantic-settings 读 config.yaml
│       │   ├── extensions.py               # SQLAlchemy / CORS 实例
│       │   ├── models.py                   # ORM：FaceProfile / Template / RecognitionRecord
│       │   ├── routes/                     # 上游：face / records / stats Blueprint
│       │   ├── services/                   # 上游：face_service / adaface_infer / image_utils
│       │   ├── strategies/                 # 我们：5 个 TemplateStrategy + 注册表
│       │   ├── evaluation/                 # 我们：data_split / metrics / pair_generator / lfw_loader / runner / run_ablation
│       │   └── utils/response.py
│       ├── scripts/
│       │   ├── build_face_gallery.py       # 上游脚本，已改造写 5 策略 Template
│       │   ├── migrate_to_templates.py     # 我们：从 feature_vector 一次性 backfill mean_all
│       │   └── prepare_lfw_subset.py       # 我们：从 LFW 抽 gallery + impostor 子集
│       ├── tests/{unit,integration}/       # 测试金字塔
│       ├── models/                         # AdaFace 权重（gitignore）
│       ├── dataset/                        # 私照 + LFW 缓存（gitignore）
│       ├── instance/                       # SQLite（gitignore）
│       ├── run.py                          # Flask 启动入口
│       └── requirements.txt                # uv 不可用时的 fallback 依赖列表
│
├── 🟨 依赖与配置（仓库根）
│   ├── pyproject.toml                      # uv 项目声明 + pytest / uv.sources 配置
│   ├── uv.lock                             # uv 版本锁（含全平台 CPU torch wheel）
│   ├── .venv/                              # uv 创建的虚拟环境（gitignore）
│   └── config.yaml                         # 业务配置：阈值 / 模型路径 / 数据集路径
│
├── 🟪 文档与产物
│   ├── docs/superpowers/{specs,plans}/     # 设计文档 + 实施计划
│   └── reports/                            # 评估输出（gitignore，入仓 .gitkeep）
│
├── README.md                               # 公开交付物（含目录地图 + 启动）
└── CLAUDE.md                               # 本文件：给 AI 协作者的项目级指令（已入仓）
```

### 增量落地状态

| 路径 | 状态 | 备注 |
| --- | --- | --- |
| `backend/app/strategies/` | ✅ 已落地（M2） | 5 策略 + 注册表 |
| `backend/app/evaluation/` | ✅ 已落地（M3） | data_split / metrics / pair_generator / lfw_loader / runner / run_ablation CLI |
| `backend/app/config_loader.py` + `config.yaml` | ✅ 已落地（M1） | pydantic-settings |
| `backend/scripts/migrate_to_templates.py` | ✅ 已落地 | 从 feature_vector 反填 mean_all |
| `backend/scripts/prepare_lfw_subset.py` | ✅ 已落地 | LFW 抽 gallery + impostor |
| `backend/tests/{unit,integration}/` | ✅ 已落地 | 56 单测 + slow 集成 |
| `backend/app/application/` | 🟡 暂未落地 | 计划中："注册 / 识别"用例编排层；目前直接走 `services/face_service.py` |
| `backend/app/routes/persons.py` | 🟡 暂未落地 | 计划中：在线注册 / 删除 / 列表 Blueprint |

### 增量层依赖规则

```
routes/  ──→  application/  ──→  models / strategies
                  ↑                    ↑
                  └────  services/adaface_infer.py（上游推理胶水）
```

- **新写的 application/ 层只依赖 models 和 strategies**：不允许直接 import `routes/` 或写 SQL
- **strategies/ 层零外部依赖**（除 numpy）：每个策略只接受向量数组，不碰数据库 / 不碰文件系统
- **evaluation/ 层独立**：不被 routes / application 引用，是离线脚本
- **上游 services/ 视为"已存在的设施"**：新代码可以调它，但不要在它里面塞业务逻辑

如果一段代码不知道该放哪——大概率是它**违反了某条规则**，先停下来想清楚再写。

---

## 4. 编码规范

### Python 风格

- **Python 版本**：`>=3.10,<3.12`（在 `pyproject.toml` 锁定）；torch 已升到 `>=2.0,<3.0` 拿到 macOS arm64 原生 wheel
- **注释风格**：中文注释，复杂逻辑处必有注释
- **类型注解**：所有公开函数必须有类型注解，包括返回值
- **数据类**：新写的实体一律 `@dataclass(frozen=True)`，容器用 `tuple` 而非 `list`
- **接口**：用 `typing.Protocol`，不用 `abc.ABC`
- **导入顺序**：标准库 → 第三方 → 本项目（用 ruff 自动整理）

### 命名

- 模块文件：`snake_case.py`
- 类：`PascalCase`
- 函数 / 变量：`snake_case`
- 常量：`UPPER_SNAKE`
- **领域术语统一**：`FaceEncoding` 不是 `Embedding`，`Template` 不是 `Centroid`，`PersonRepository` 不是 `FaceDB`
- **不要重命名上游已有的字段**：`FaceProfile.name`、`feature_vector`、`recognition_records.recognized_at` 这些是上游 ORM 字段，前端 / DB 已经依赖

### 文件大小

- **单文件 ≤ 200 行**为目标，超过 300 行强烈考虑拆分
- 一个文件一个核心责任，不做"杂物间"

### 注释

- **默认不写注释**——清晰的命名自带文档
- 仅在 **WHY 非显然** 时写一行注释（隐藏约束、非直觉的算法选择、为绕过某个 bug 的 workaround）
- **绝不写解释 WHAT 的注释**（"# 加载模型" 这种废话）
- **绝不引用当前任务 / fix / issue**（"# 用于 X 流程"、"# 修复 issue #123"）——这种东西属于 commit message
- 公开 API 用一行 docstring，超过两行先想想是不是函数职责太杂
- **plans/specs 文档例外**：写 plan 时给零基础学生看，新概念出现时要详细教学

### 错误处理

- 领域错误用统一异常层次（在 `app/application/errors.py` 里定义）
- **不要捕获后吞掉错误**，除非业务上明确"单张失败可继续"（例：注册时单张照片无脸 → 记 warning 跳过）
- **不在边界外做防御性校验**：内部代码相互信任，只在系统边界（HTTP 请求、CLI 参数、文件读取）做校验

### 配置

- 任何"可能要调"的参数必须进 `config.yaml`
- 代码里**绝不出现**像 `0.35`、`'instance/face_access.db'`、`(640, 640)` 这种魔数 / 硬编码路径
- 用 `pydantic-settings` 加载，启动时类型校验失败立即崩溃
- 上游 `app/config.py` 用环境变量风格——**不动它**，新增参数走 `config.yaml`

---

## 5. 测试约定

### 测试金字塔

```
端到端 (3~5)    注册→识别→评估全链路冒烟（Flask test client + 真模型）
集成 (10+)     真模型 + 真 SQLite，不联网
单元 (30+)     纯逻辑，全 mock，毫秒级
```

### 测试边界（重要）

- **测自己写的逻辑**：5 策略产出形状、ROC/EER 计算的数学正确性、用例编排逻辑
- **不测外部库**：不写"`extract_embedding_from_bgr` 在某图上返回 512 维"——那是测 AdaFace 库，不是测我们
- **集成测试用 AdaFace 自带 sample**：`backend/AdaFace/face_alignment/test_images/` 已有示例图，方便构造确定性测试

### TDD

- 实现前先写失败的测试
- 红 → 绿 → 重构
- commit 颗粒度：每个绿灯都是一个 commit 的好时机

### 运行命令

```bash
# 命令一律从仓库根目录执行（pyproject.toml 已配 testpaths=backend/tests）
uv run pytest                            # 全部（默认排除 slow）
uv run pytest backend/tests/unit/        # 单元（默认运行，秒级）
uv run pytest -m slow                    # 集成（需权重 + ~分钟级）
uv run pytest backend/tests/unit/evaluation/test_metrics.py::test_eer_at_intersection -v   # 单条
```

> pytest / pytest-cov 已在 `pyproject.toml` 的 `[dependency-groups] dev` 锁定，`uv sync` 自动装齐。

---

## 6. 工作流约定

### Commit

- **小步提交**：一个完整的 TDD 循环（红→绿）就 commit
- 用 Conventional Commits：`feat:` / `fix:` / `refactor:` / `test:` / `docs:` / `chore:`
- Commit message 写"why"不写"what"
- 跟随 spec：每完成一个里程碑（M1~M6）打 tag

### 计划与执行

- 任何**新增功能 / 重大改动**之前，先看 `docs/superpowers/specs/` 是否已有 spec
- 若 spec 缺失，先用 `superpowers:brainstorming` 技能补齐 spec，**不要直接写代码**
- 实施时按 `docs/superpowers/plans/` 中的实施计划逐项推进
- 每完成一个任务做一次 `git commit`

### 与上游代码相处

- **上游 `backend/AdaFace/` 子目录不要改动**：是 AdaFace 官方代码，便于将来 sync upstream
- **上游 `app/services/` 中已有的函数**：可以扩展（加新函数）但谨慎修改既有签名——前端 / 路由依赖它
- **上游 `app/models.py` 表结构**：增加字段 / 新增表 OK，**不删既有列**——会破坏 wxapp
- **API 契约**：上游 4 个接口（`POST /api/face/recognize` / `GET /api/records/list` / `GET /api/stats/today` / `POST /api/records/clear`）的入参和返回结构**保持兼容**

### 数据安全

`.gitignore` 已配如下（动它前看清楚）：

- `backend/dataset/*` 私人照片 + LFW 缓存（仅入仓 `.txt` 说明 + `.gitkeep`）
- `backend/models/*.{pth,ckpt,pt}` ~85MB 的 AdaFace 权重
- `backend/instance/`、`*.db` 识别记录数据库
- `.venv/`、`__pycache__/`、`*.pyc`、`.DS_Store`
- `reports/*`（仅入仓 `.gitkeep`）
- `.claude/`（Claude Code 工作目录，agent 内部状态）
- `*.local.yaml`、`*.mp4`

---

## 7. 常用命令速查

```bash
# === 后端（uv 管理；首次：curl -LsSf https://astral.sh/uv/install.sh | sh）===
# pyproject.toml / uv.lock / .venv 都在仓库根目录；命令一律从根跑，不用 cd backend
uv sync                            # 按 uv.lock 创建 .venv 并装齐依赖

# 启动 Flask 服务
uv run python backend/run.py       # 默认 http://127.0.0.1:5000

# 健康检查
curl http://127.0.0.1:5000/api/health

# 离线建库（上游脚本）
uv run python backend/scripts/build_face_gallery.py --dataset backend/dataset/ --config config.yaml

# 离线评估（5 策略消融）
uv run python -m app.evaluation.run_ablation \
    --dataset backend/dataset --config config.yaml

# 跑测试（pytest 已在 pyproject.toml 配置 testpaths=backend/tests）
uv run pytest                      # 全部（默认排除 slow）
uv run pytest backend/tests/unit/  # 单元

# 类型检查 / 格式
uv run mypy backend/app/application backend/app/strategies backend/app/evaluation
uv run ruff check --fix .
uv run ruff format .

# === 前端（小程序）===
# 用微信开发者工具打开仓库根目录
# project.config.json 已配好 appid（开发者本人）
```

---

## 8. 给 AI 协作者的特别指引

1. **先读 spec 再动手**：每次会话开始先读根 `README.md` 了解目录布局，再读 `docs/superpowers/specs/`（如已写）确认上下文
2. **每个非平凡决策都需用户确认**：用户偏好"对比 + 推荐 + 反方观点"的协作方式，不要替用户拍板
3. **直接挑战次优决定**：用户欢迎被指出失败模式与潜在风险，不需要顺从
4. **教学化沉淀**：用户是本科生，希望把方法论沉淀到 spec / md，方便写报告引用——遇到值得教学的概念主动写进文档
5. **YAGNI 是硬规则**：上面 §2 的"严禁做"清单是真严禁，不是参考意见
6. **本项目不是企业级**：用户已多次强调期末项目不需要工业级方案；过度工程会扣分而不是加分
7. **尊重上游已运行的代码**：我们是 fork，目标是"在工作的东西上面叠改进"，不是"推翻重写"
8. **答辩友好**：每个增量都要能在答辩时讲清楚"为什么加 + 加了之后怎么验证 + 数据上的差异"

---

## 9. 项目历史背景（迁移自前一个仓库）

本项目用户上一个仓库 `~/Desktop/work/`（基于 ArcFace + buffalo_l + FastAPI + 单文件 HTML）已临时归档。
那边商讨过的设计决策（5 策略消融、ROC/EER/TAR 评估、清洁架构、TDD、教学风格的 plans/specs）将**继续在本仓库延续**。

那边写的 spec / plans 在 `~/Desktop/work/docs/superpowers/`，可作为本仓库 spec 起草的参考蓝本（**不要直接抄**——技术栈不同，需要重新适配 Flask / AdaFace / 小程序）。

---

## 10. 文档版本

| 日期 | 变更 |
| --- | --- |
| 2026-05-25 | 初版：fork 自 Violet52014/AdaFace_Recognition，写本协作指令 |
| 2026-05-25 | M1/M2/M3 落地（config_loader / strategies / evaluation）；迁 uv（三件套在仓库根）；torch 全平台锁 CPU wheel；新增根 README；§3 结构图与 §4/§5/§6 命令对齐当前事实 |
