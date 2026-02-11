# MLOps 端到端完整流程 - 面试准备指南

> 适用于准备大厂 ML/MLOps 岗位面试的完整参考文档

---

## 目录

1. [数据准备与版本控制](#1-数据准备与版本控制)
2. [模型训练与实验跟踪](#2-模型训练与实验跟踪)
3. [模型评估与选择](#3-模型评估与选择)
4. [模型版本控制与注册](#4-模型版本控制与注册)
5. [模型部署](#5-模型部署cicd容器化云部署)
6. [模型推理](#6-模型推理在线服务-vs-批量推理)
7. [监控、漂移检测与持续训练](#7-监控漂移检测与持续训练)
8. [端到端流程图](#8-端到端流程图)
9. [可运行示例代码](#9-可运行示例代码)
10. [学习资源](#10-学习资源)

---

## 1. 数据准备与版本控制

### 核心目标

确保数据的**可追溯性**、**可复现性**和**一致性**，让每次实验都能精确定位使用的数据版本。

### 具体步骤

#### 1.1 数据采集与清洗
```python
# 原始数据 → 清洗 → 验证 → 分割
raw_data → clean_data → validate_data → train/val/test split
```

#### 1.2 数据版本控制
```bash
# 方式 1: DVC (推荐)
dvc add data/train.csv
dvc push
git add data/.gitignore
git commit -m "add train data v1.0"

# 方式 2: Delta Lake (表格式)
delta_table.write.mode("overwrite").versionAsOf(1)

# 方式 3: MLflow Dataset
mlflow.data.log_dataset(dataset)
```

#### 1.3 数据特征存储
```python
# 计算特征 → 存储为特征组 → 训练时读取
features = compute_features()
fs = FeatureStoreClient()
fs.create_feature_group(
    name="user_features",
    version=1,
    features=features
)
```

#### 1.4 数据质量检查
```python
# 训练前验证
assert data["label"].notna().all()
assert data["age"].between(0, 120).all()
# Great Expectations / Pandera / Deequ
```

### 常用工具/技术栈

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| **DVC** | Git-like 数据版本控制 | 中小型数据，文件级别版本 |
| **Delta Lake** | ACID 事务，时间旅行 | 大数据，表格式版本 |
| **Pachyderm** | 数据管道版本化 | 复杂 ETL 流程 |
| **Feast** | 特征存储 | 线上/线下特征一致性 |
| **Great Expectations** | 数据质量测试 | 数据验证 |

### 最佳实践

1. **数据与代码分离**：数据不进 Git，用 DVC/对象存储
2. **不可变性**：数据版本不可修改，只能创建新版本
3. **血缘追踪**：记录数据来源和转换过程
4. **自动化验证**：每次数据更新自动运行质量检查
5. **样本量管理**：小版本号更新数据，大版本号更新 schema

### 常见坑 & 解决方案

| 问题 | 解决方案 |
|------|----------|
| 数据文件太大，无法版本控制 | 用 DVC + S3/GCS，只存元数据 |
| 数据漂移导致模型退化 | 建立数据漂移监控，定期重新标注 |
| 特征穿越（leakage） | 严格划分训练/推理特征计算逻辑 |
| 数据不一致 | 使用特征存储保证一致性 |

### 面试常考点

**Q1: 如何处理数据版本控制？**
```
A: 分层策略
1. 原始数据：DVC 管理 + 对象存储
2. 特征数据：Feast/Hopsworks 特征存储
3. 验证数据：固定测试集，不变更版本
4. 每个实验记录使用的数据版本 hash/ID
```

**Q2: 训练-推理数据不一致怎么办？**
```
A: 三个层面解决
1. 特征存储：保证计算逻辑统一
2. 数据契约：定义 schema，契约测试
3. 监控对比：训练/推理数据分布对比
```

**Q3: 如何检测数据漂移？**
```
A: 统计检测方法
1. 单变量：KS test, Population Stability Index (PSI)
2. 多变量：Classifier-based, PCA-based
3. 嵌入空间：深度模型的特征分布
工具：Evidently AI, NannyML, WhyLogs
```

---

## 2. 模型训练与实验跟踪

### 核心目标

**复现性**：任意实验都能完全重现
**可比较**：轻松对比不同实验结果
**自动化**：减少手动记录工作

### 具体步骤

#### 2.1 实验配置管理
```python
# config.py 或 YAML
@dataclass
class ExperimentConfig:
    model_type: str = "resnet50"
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    data_version: str = "v1.2"

# 用 Hydra/OmegaConf 管理配置
```

#### 2.2 设置实验跟踪
```python
import mlflow

mlflow.set_experiment("fraud_detection")
mlflow.start_run()

# 自动记录参数和指标
mlflow.log_params({
    "learning_rate": 0.001,
    "batch_size": 32
})

mlflow.log_metrics({
    "train_loss": 0.123,
    "val_loss": 0.145
})
```

#### 2.3 训练循环中记录
```python
for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss = validate()

    # 每个epoch记录
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss
    }, step=epoch)

    # 记录模型检查点
    if val_loss < best_loss:
        mlflow.pytorch.log_model(model, "model")
```

#### 2.4 Artifacts 记录
```python
# 记录模型文件、图表、配置
mlflow.log_artifact("config.yaml")
mlflow.log_artifact("confusion_matrix.png")
mlflow.log_figure(roc_curve, "roc_curve.png")
```

### 常用工具/技术栈

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| **MLflow** | 轻量，开源生态好 | 通用，中小团队 |
| **Weights & Biases** | 可视化强，云端体验佳 | 深度学习，重视可视化 |
| **Neptune.ai** | 元数据管理强 | 复杂实验管理 |
| **TensorBoard** | TensorFlow 生态，实时 | 深度学习训练监控 |
| **Sacred** | 轻量级，Python 原生 | 小项目，简单实验 |

### 最佳实践

1. **自动记录一切**：参数、指标、artifacts、环境信息
2. **父子运行**：超参数调优用 nested runs
3. **标签系统**：用 tags 标记实验状态、模型类型
4. **检查点管理**：只保留最优检查点，节省空间
5. **Git commit 关联**：每次实验自动记录 commit hash

### 常见坑 & 解决方案

| 问题 | 解决方案 |
|------|----------|
| 实验太多无法管理 | 用命名规范、tags、良好的 UI 过滤 |
| 环境无法复现 | 用 `mlflow.log_environment()` 或 Docker |
| 超参数搜索混乱 | 用 MLflow Projects 或 Optuna 集成 |
| 指标不一致 | 统一指标计算逻辑，封装成库 |

### 面试常考点

**Q1: 如何保证实验可复现？**
```
A: 四个层面
1. 随机种子：固定 numpy/torch/random seed
2. 环境：记录 Python 版本、包版本 (requirements.txt)
3. 数据：锁定数据版本 ID
4. 代码：记录 Git commit hash
```

**Q2: 如何管理超参数搜索？**
```
A: 三种策略
1. Grid Search: 小空间，穷举
2. Random Search: 大空间，随机采样
3. Bayesian: 顺序优化，用前期结果指导后期
工具：Optuna, Ray Tune, MLflow Projects
```

**Q3: 实验太多怎么选最佳模型？**
```
A: 系统化方法
1. 定义评估框架：业务指标 × 资源消耗 × 延迟
2. Pareto 前沿：找非支配解
3. A/B 测试：线上验证业务指标
```

---

## 3. 模型评估与选择

### 核心目标

**客观评估**：用科学方法评估模型性能
**风险控制**：识别模型短板和失败模式
**决策支持**：为上线选择提供依据

### 具体步骤

#### 3.1 建立评估框架
```python
@dataclass
class EvaluationConfig:
    # 数据集
    test_set_path: str
    # 指标
    metrics: List[str] = ["accuracy", "precision", "recall", "f1"]
    # 业务约束
    max_latency_ms: int = 100
    min_recall: float = 0.95
    # 公平性
    demographic_parity: float = 0.05
```

#### 3.2 多维度评估
```python
# 1. 性能指标
metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred),
    "recall": recall_score(y_true, y_pred),
    "f1": f1_score(y_true, y_pred),
    "auc_roc": roc_auc_score(y_true, y_proba)
}

# 2. 混淆矩阵分析
cm = confusion_matrix(y_true, y_pred)
# 分析 FP/FN 分布

# 3. 校准度
reliability_diag = calibration_diagram(y_true, y_proba)
# 预测概率是否与真实概率匹配

# 4. 亚组分析
for subgroup in demographic_groups:
    subgroup_metrics = compute_metrics(y_true[subgroup], y_pred[subgroup])
```

#### 3.3 错误分析
```python
# 识别错误模式
errors = y_true != y_pred

# 错误聚类：看看什么类型的样本容易错
error_features = X_test[errors]
cluster_analysis(error_features)

# 困难样本挖掘
hard_samples = get_low_confidence_samples(model, X_test)
```

#### 3.4 压力测试
```python
# 对抗样本
adversarial_samples = generate_adversarial(X_test)
model.evaluate(adversarial_samples)

# OOD 检测
ood_score = detect_out_of_distribution(model, X_test)

# 噪声鲁棒性
for noise_level in [0.1, 0.2, 0.3]:
    noisy_X = add_noise(X_test, noise_level)
    robust_metrics = model.evaluate(noisy_X)
```

### 常用工具/技术栈

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| **MLflow Evaluate** | 内置对比视图 | 快速模型对比 |
| **Evidently AI** | 数据/模型漂移 | 持续监控 |
| **Lime/SHAP** | 可解释性 | 模型解释 |
| **Fairlearn** | 公平性评估 | 公平性要求场景 |
| **Deepchecks** | 全面验证套件 | 模型验证 |

### 最佳实践

1. **分层评估**：先离线，再影子测试，最后 A/B
2. **业务指标对齐**：将 ML 指标映射到业务指标
3. **长期追踪**：建立模型卡片 (Model Card)
4. **负面测试**：主动寻找失败案例
5. **自动报告**：生成评估报告，减少人工

### 常见坑 & 解决方案

| 问题 | 解决方案 |
|------|----------|
| 训练集泄露 | 严格时间分割，交叉验证用时间序列 CV |
| 测试集过拟合 | 留 held-out 测试集，只在最终评估用 |
| 指标与业务脱节 | 直接定义业务损失函数 |
| 类别不平衡 | 用 stratified split, SMOTE, 合适指标 |

### 面试常考点

**Q1: 如何评估分类模型？**
```
A: 分层评估
1. 整体指标：Accuracy, F1, AUC-ROC
2. 阈值相关：Precision-Recall 曲线，选业务阈值
3. 概率校准：Brier score, reliability diagram
4. 代价敏感：混淆矩阵 × 业务代价矩阵
```

**Q2: 如何处理数据不平衡？**
```
A: 三种策略
1. 数据层面：过采样少数类，欠采样多数类
2. 算法层面：class_weight, focal loss
3. 评估层面：用 PR-AUC 非 ROC-AUC，关注 F1/Recall
```

**Q3: 什么是模型校准，为什么重要？**
```
A: 校准定义与重要性
1. 定义：预测概率=0.8 的样本，80% 应该是正类
2. 重要性：风险评估、阈值设置、用户信任
3. 方法：Platt scaling, Temperature scaling, Isotonic regression
4. 评估：Brier score, Expected Calibration Error
```

---

## 4. 模型版本控制与注册

### 核心目标

**统一管理**：所有模型集中管理，避免散落各地
**可追溯**：每个模型的来源、性能、使用情况清晰
**审批流程**：模型上线需要审核，符合合规要求

### 具体步骤

#### 4.1 模型注册
```python
import mlflow

# 训练完成后
with mlflow.start_run() as run:
    model.fit(X_train, y_train)

    # 记录模型
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name="fraud_detection",
        signature=signature,
        input_example=input_example
    )

    # 添加描述和标签
    mlflow.set_tag("stage", "staging")
    mlflow.set_tag("approval_status", "pending")
```

#### 4.2 模型阶段管理
```python
client = mlflow.tracking.MlflowClient()

# 阶段转换：Staging → Production
client.transition_model_version_stage(
    name="fraud_detection",
    version=3,
    stage="Production",
    archive_existing_versions=True
)

# 记录审批信息
client.update_model_version(
    name="fraud_detection",
    version=3,
    description="Approved by Jane Doe, A/B test +5% lift"
)
```

#### 4.3 模型元数据
```python
# 模型卡片
model_card = {
    "model_name": "fraud_detection",
    "version": 3,
    "training_data": "transactions_2024_h1",
    "performance": {
        "precision": 0.92,
        "recall": 0.89,
        "auc_roc": 0.96
    },
    "limitations": [
        "Only works for credit card transactions",
        "Degrades with new merchant categories"
    ],
    "intended_use": "Real-time fraud scoring",
    "fairness": "Demographic parity < 0.05 across all groups"
}

mlflow.set_model_card(model_card)
```

#### 4.4 模型加载
```python
import mlflow.pyfunc

# 加载 Production 版本
model_uri = "models:/fraud_detection/Production"
model = mlflow.pyfunc.load_model(model_uri)

# 或加载特定版本
model_uri = "models:/fraud_detection/3"
model = mlflow.pyfunc.load_model(model_uri)
```

### 常用工具/技术栈

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| **MLflow Model Registry** | 轻量，与跟踪集成 | 中小项目，简单场景 |
| **Amazon SageMaker Model Registry** | AWS 集成，AutoML | AWS 生态 |
| **Azure ML Model Registry** | 企业级，权限细粒度 | Azure 生态 |
| **Databricks Model Registry** | Unity Catalog，治理完善 | Databricks 平台 |
| **Hugging Face Hub** | 开源模型共享 | NLP/CV 开源社区 |

### 最佳实践

1. **语义版本控制**：Major.Minor.Patch (1.2.3)
2. **阶段命名规范**：Development → Staging → Production → Archived
3. **自动化签名**：每次部署自动签名记录
4. **访问控制**：区分角色权限（数据科学家、工程师、审计）
5. **定期清理**：归档旧模型，节省存储

### 常见坑 & 解决方案

| 问题 | 解决方案 |
|------|----------|
| 模型太多无法管理 | 命名规范、生命周期策略、自动归档 |
| 环境依赖丢失 | 用 MLflow Model 包含环境 (conda env) |
| 模型格式不兼容 | 统一格式 (MLflow/ONNX)，标准化接口 |
| 上线后不知道效果 | 阶段转换时关联业务指标 |

### 面试常考点

**Q1: 如何管理模型版本？**
```
A: 分级管理
1. 注册中心：MLflow Model Registry 统一管理
2. 版本号：每个训练自动递增版本号
3. 阶段：Dev → Staging → Production 流程
4. 元数据：记录训练数据、参数、性能、审批人
```

**Q2: 如何实现模型回滚？**
```
A: 快速回滚机制
1. 保留前 N 个 Production 版本
2. Model Registry 支持一键切换
3. 蓝绿部署：两版本同时在线，切流量
4. 监控告警：新版本问题立即回滚
```

**Q3: 模型注册和模型存储的区别？**
```
A: 两层架构
1. 模型存储 (Artifact Store)：存实际文件，S3/NFS
2. 模型注册 (Metadata Store)：存元数据和指针
好处：存储可扩展，注册中心轻量查询
```

---

## 5. 模型部署（CI/CD、容器化、云部署）

### 核心目标

**自动化**：从训练到部署全自动化
**可扩展**：支持流量扩展和负载均衡
**可监控**：部署后可观察性能

### 具体步骤

#### 5.1 容器化
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 复制依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY model_server.py .

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8080/health || exit 1

# 启动服务
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "model_server:app"]
```

#### 5.2 CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy Model

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Build Docker image
        run: docker build -t model:${{ github.sha }} .

      - name: Run tests
        run: |
          docker run model:${{ github.sha }} \
            python -m pytest tests/

      - name: Push to registry
        run: |
          docker tag model:${{ github.sha }} registry/model:latest
          docker push registry/model:latest

      - name: Deploy to production
        run: |
          kubectl set image deployment/model \
            model=registry/model:${{ github.sha }}
```

#### 5.3 推理服务器
```python
# model_server.py
from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()

# 加载模型
model = mlflow.pyfunc.load_model("models:/fraud_detection/Production")

@app.post("/predict")
async def predict(request: PredictRequest):
    features = request.features
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

#### 5.4 Kubernetes 部署
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model
  template:
    metadata:
      labels:
        app: model
    spec:
      containers:
      - name: model
        image: registry/model:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

### 常用工具/技术栈

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| **Docker** | 标准化容器 | 任何场景 |
| **Kubernetes** | 编排、自动扩缩容 | 大规模生产 |
| **SageMaker** | AWS 托管服务 | AWS 用户 |
| **BentoML** | 模型服务框架 | 快速部署 |
| **Seldon/Core** | K8s 原生 ML 部署 | K8s 环境 |

### 最佳实践

1. **多阶段部署**：Dev → Staging → Production
2. **金丝雀发布**：先小流量，逐步放量
3. **资源限制**：设置 CPU/内存 limit，避免 OOM
4. **健康检查**：liveness + readiness probe
5. **优雅关闭**：处理完现有请求再退出

### 常见坑 & 解决方案

| 问题 | 解决方案 |
|------|----------|
| 依赖冲突 | 用容器固化环境，poetry/conda-lock |
| 冷启动慢 | 模型预加载，多实例，GPU 预热 |
| 请求积压 | 异步队列，自动扩容 |
| 内存泄漏 | 定期重启，监控内存使用 |

### 面试常考点

**Q1: 描述一个完整的部署流程**
```
A: 标准 ML 部署流程
1. 代码提交 → Git
2. CI 构建 → Docker 镜像
3. 推送镜像 → 容器 Registry
4. 更新部署 → K8s Deployment
5. 健康检查 → Readiness Probe
6. 流量切换 → Service/Ingress
7. 监控验证 → Prometheus/Grafana
```

**Q2: 金丝雀发布 vs 蓝绿部署？**
```
A: 两种渐进式发布
金丝雀：
- 5% → 25% → 50% → 100% 流量切换
- 发现问题快速回滚
- 适合高可靠性要求

蓝绿：
- 两套完整环境，同时在线
- 瞬间切换流量
- 回滚也瞬间
- 资源需求翻倍
```

**Q3: 如何处理模型冷启动？**
```
A: 冷启动优化策略
1. 模型预加载：容器启动时加载到内存
2. 多实例：始终保持足够实例
3. 预热：发送假请求，初始化缓存
4. 懒加载：首次请求时才加载（牺牲首次）
5. 模型量化/蒸馏：减小模型大小
```

---

## 6. 模型推理（在线服务 vs 批量推理）

### 核心目标

**在线服务**：低延迟、高可用、自动扩缩容
**批量推理**：高吞吐、成本优化、容错重试

### 具体步骤

#### 6.1 在线推理架构
```python
# 同步 REST API
@app.post("/predict")
async def predict(request: PredictRequest):
    start = time.time()
    prediction = model.predict(request.features)
    latency = time.time() - start
    record_latency(latency)  # 监控
    return {"prediction": prediction}

# 异步任务队列
@app.post("/predict_async")
async def predict_async(request: PredictRequest):
    job_id = str(uuid.uuid4())
    await queue.enqueue(job_id, request.features)
    return {"job_id": job_id}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    result = await queue.get_result(job_id)
    return result
```

#### 6.2 批量推理管道
```python
# Spark 批量推理
df = spark.read.parquet("s3://data/features/")

# 批量预测，自动并行
udf_predict = pandas_udf(predict_batch, returnType=DoubleType())
df = df.withColumn("prediction", udf_predict("features"))

# 写回结果
df.write.parquet("s3://results/predictions/")
```

#### 6.3 推理优化
```python
# 批处理
def predict_batch(features_list):
    # 将多个请求打包成一次推理
    features_array = np.array(features_list)
    predictions = model.predict(features_array)
    return predictions

# 缓存
@lru_cache(maxsize=1000)
def predict_cached(features_hash):
    return model.predict(features)

# 量化
import torch.quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

### 常用工具/技术栈

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| **FastAPI** | 高性能异步 | 在线服务 |
| **TensorFlow Serving** | TF 专用，批量优化 | TensorFlow 模型 |
| **TorchServe** | PyTorch 官方 | PyTorch 模型 |
| **AWS Batch** | 托管批量计算 | AWS 批量 |
| **Ray Serve** | 弹性，多模型 | 复杂推理场景 |

### 最佳实践

1. **批处理打包**：累积请求到一定大小再推理
2. **超时控制**：设置合理超时，避免资源占用
3. **降级策略**：模型挂了返回规则结果
4. **请求批量化**：自动合并小请求
5. **结果缓存**：相同输入直接返回缓存

### 常见坑 & 解决方案

| 问题 | 解决方案 |
|------|----------|
| 延迟突增 | 批量推理限制大小，队列限流 |
| OOM | 限制并发，减小 batch size |
| GPU 利用率低 | 动态 batching，多实例 |
| 批量任务失败 | 任务 checkpoint，断点续算 |

### 面试常考点

**Q1: 在线推理 vs 批量推理？**
```
A: 场景对比
在线推理：
- 实时性要求高，ms 级延迟
- REST/gRPC API，自动扩缩容
- 适合：风控、推荐、搜索

批量推理：
- 吞吐优先，延迟不重要
- 离线任务，Spark/Dask
- 适合：打标签、报表生成、数据增强
```

**Q2: 如何优化推理延迟？**
```
A: 多层次优化
1. 模型层面：量化、蒸馏、剪枝
2. 框架层面：ONNX, TensorRT, OpenVINO
3. 系统层面：批处理、缓存、异步
4. 硬件层面：GPU、TPU、专用芯片
5. 架构层面：模型并行、流水线
```

**Q3: 如何设计高并发推理服务？**
```
A: 并发设计模式
1. 异步框架：FastAPI + asyncio
2. 负载均衡：Nginx/Kong 分发
3. 自动扩容：K8s HPA 基于 CPU/内存
4. 限流降级：令牌桶，超时返回默认
5. 分层缓存：Redis 缓存热点数据
```

---

## 7. 监控、漂移检测与持续训练

### 核心目标

**可观测性**：知道系统在做什么
**及时发现问题**：模型退化时立即告警
**自动化响应**：检测到问题自动触发流程

### 具体步骤

#### 7.1 监控指标
```python
# Prometheus 指标
from prometheus_client import Counter, Histogram

# 请求计数
request_counter = Counter(
    'model_requests_total',
    'Total model requests',
    ['model_version', 'status']
)

# 延迟分布
latency_histogram = Histogram(
    'model_latency_seconds',
    'Model prediction latency',
    ['model_version']
)

# 使用
request_counter.labels(version='v1', status='success').inc()
latency_histogram.labels(version='v1').observe(latency)
```

#### 7.2 数据漂移检测
```python
from alibi_detect import CDRift

# 训练时保存参考分布
cd = CDRift(X_train, p_val=0.05)

# 定期检测新数据
drift_result = cd.detect(X_new)

if drift_result['data']['is_drift']:
    alert("Data drift detected!")
    trigger_retraining_pipeline()
```

#### 7.3 模型性能监控
```python
# 线上监控
def monitor_prediction(y_true, y_pred, timestamp):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "prediction_distribution": pd.Series(y_pred).value_counts()
    }

    # 发送到监控平台
    send_to_prometheus(metrics)

    # 检查是否需要告警
    if metrics['recall'] < 0.8:
        alert(f"Recall dropped to {metrics['recall']}")
```

#### 7.4 持续训练触发
```python
def should_retrain():
    # 检查多个条件
    checks = {
        "data_drift": detect_drift() > threshold,
        "performance_degradation": get_current_recall() < 0.8,
        "time_since_last": days_since_training() > 30,
        "data_volume": new_data_size() > min_data_size
    }

    # 任意条件满足则触发
    return any(checks.values())

# 定期检查
scheduler.add_job(
    check_and_retrain,
    trigger='interval',
    hours=6
)
```

### 常用工具/技术栈

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| **Prometheus + Grafana** | 指标收集可视化 | 通用监控 |
| **Evidently AI** | 数据/模型漂移 | MLOps 监控 |
| **Arize** | 端到端 ML 监控 | 企业级 |
| **WhyLabs** | 无代码监控 | 快速上手 |
| **NannyML** | 性能估计（无标签） | 无真实标签监控 |

### 最佳实践

1. **分层监控**：基础设施 → 应用 → 业务 → ML
2. **金指标**：延迟、流量、错误、饱和度 (RED/SAT)
3. **自动化告警**：阈值告警 + 趋势告警 + 异常检测
4. **定期复盘**：每周 review 告警，调整阈值
5. **可解释性**：不仅告警，还要告警原因

### 常见坑 & 解决方案

| 问题 | 解决方案 |
|------|----------|
| 告警太多 | 调整阈值，聚合告警，区分 severity |
| 缺少真实标签 | 采样标注，弱监督，无标签估计 |
| 漂移误报 | 多指标综合，统计检验 + 规则 |
| 重训练成本高 | 增量训练，选择性重训练 |

### 面试常考点

**Q1: 如何监控生产模型？**
```
A: 四层监控体系
1. 基础设施：CPU、内存、GPU、网络
2. 服务指标：QPS、延迟、错误率、可用性
3. 模型特有：预测分布、置信度、漂移
4. 业务指标：转化率、用户反馈、A/B 结果

工具：Prometheus + Grafana + 告警
```

**Q2: 什么是数据漂移 vs 模型漂移？**
```
A: 两种漂移类型
数据漂移：输入分布变化
- Covariate Shift: P(X) 变化
- Prior Probability Shift: P(Y) 变化
- 检测：KS test, PSI, JS Divergence

模型漂移：P(Y|X) 关系变化
- 概念漂移：原本的关系变了
- 检测：需要真实标签，监控性能下降
```

**Q3: 如何设计持续训练流程？**
```
A: 自动化闭环
1. 监控：定期检测漂移/性能下降
2. 触发：自动或人工触发重训练
3. 训练：使用新数据 + 旧数据（防遗忘）
4. 验证：离线评估 + 影子测试
5. 上线：金丝雀发布，逐步切换
6. 反馈：收集线上数据，继续循环

工具：Airflow/Prefect 调度 + MLflow 跟踪
```

---

## 8. 端到端流程图

### 架构流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          数据层                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                   │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│    │原始数据  │───→│清洗转换  │───→│特征存储  │                    │
│    │S3/HDFS  │    │Spark/SQL │    │Feast   │                    │
│    └─────────┘    └─────────┘    └─────────┘                    │
│         │              │               │                           │
│         ▼              ▼               ▼                           │
│    DVC版本管理      Data Quality    特征组                        │
│                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          训练层                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                   │
│    ┌──────────────────────────────────────────────────┐              │
│    │           实验/训练 (MLflow)                   │              │
│    │                                                │              │
│    │   代码 ──→ Hyperparam ──→ 训练 ──→ 评估              │
│    │    ↓           ↓              ↓         ↓                  │              │
│    │  Git仓库    Optuna        GPU       指标                    │
│    │                                            │                  │              │
│    └──────────────────────────────────────────────────┘              │
│                         │                                         │
│                         ▼                                         │
│    ┌──────────────────────────────────────────────────┐              │
│    │        模型注册 (Model Registry)              │              │
│    │                                                │              │
│    │   ┌──────┐  ┌──────┐  ┌──────┐                   │              │
│    │   │Dev   │→ │Staging│→ │Prod  │                   │              │
│    │   │v1.2  │  │v1.3  │  │v1.0  │                   │              │
│    │   └──────┘  └──────┘  └──────┘                   │              │
│    └──────────────────────────────────────────────────┘              │
│                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          部署层                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                   │
│    ┌────────────────────────────────────────────┐                    │
│    │         CI/CD Pipeline                     │                    │
│    │                                            │                    │
│    │   推送模型 ──→ 构建镜像 ──→ 测试 ──→ 部署          │
│    │                                            │                    │
│    └────────────────────────────────────────────┘                    │
│                         │                                       │
│                         ▼                                       │
│    ┌────────────────────────────────────────────┐                    │
│    │        Kubernetes 集群                      │                    │
│    │                                            │                    │
│    │   ┌────────┐    ┌────────┐    ┌────────┐               │                    │
│    │   │服务A   │    │服务B   │    │服务C   │               │                    │
│    │   │v1.0    │    │v1.3    │    │v0.9    │               │                    │
│    │   │(金丝雀) │    │(稳定版)  │    │(旧版)   │               │                    │
│    │   └────────┘    └────────┘    └────────┘               │                    │
│    │                                            │                    │
│    │   ←────── 流量分割 ──────→                        │                    │
│    │                                            │                    │
│    └────────────────────────────────────────────┘                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          监控层                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                   │
│    ┌────────────────┐  ┌────────────────┐                       │
│    │  Prometheus    │  │  Grafana      │                       │
│    │  (指标收集)     │←→│  (可视化)       │                       │
│    └────────────────┘  └────────────────┘                       │
│            ↑                                                │           │
│            │                                                │           │
│    ┌───────┴────────┬───────────┬────────────┐             │
│    │                │           │            │             │
│ ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│ │ 服务指标   │  │ 模型特有   │  │ 业务指标    │            │
│ │ QPS/Latency│  │ 漂移/预测  │  │ 转化率     │            │
│ └──────────┘  └──────────┘  └──────────┘            │
│       │                │            │             │
│       └────────────────┴────────────┘             │
│                      │                          │
│                ┌──────────▼──────────┐                │
│                │  告警规则          │                │
│                │  (AlertManager)    │                │
│                └──────────┬──────────┘                │
│                           │                          │
│                ┌──────────▼──────────┐                │
│                │  响应动作          │                │
│                │  - 告警通知        │                │
│                │  - 自动回滚        │                │
│                │  - 触发重训练      │                │
│                └────────────────────┘                │
│                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 数据流

```
原始数据 → 版本控制 → 特征工程 → 训练集/测试集
    │
    ▼
模型训练 → 超参数搜索 → 最佳模型 → 模型注册
    │
    ▼
容器构建 → CI/CD → Kubernetes 部署 → 在线服务
    │
    ▼
请求 → 推理 → 响应 + 日志 → 监控
    │
    ▼
漂移检测 → 触发重训练 → 回到训练阶段
```

---

## 9. 可运行示例代码

### 9.1 项目结构

```
mlops-example/
├── data/
│   └── raw/
│       └── iris.csv
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── models/
│   └── (MLflow 会在这里存储)
├── tests/
│   └── test_model.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── mlflow_tracking_example.py
└── README.md
```

### 9.2 完整代码示例

```python
# mlflow_tracking_example.py
"""
MLflow 端到端示例：训练、跟踪、部署

运行方式：
1. 启动 MLflow: mlflow ui
2. 运行训练: python mlflow_tracking_example.py train
3. 运行预测: python mlflow_tracking_example.py predict
"""

import os
import pickle
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import docker
import click


# MLflow 配置
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "iris_classifier"


def prepare_data():
    """准备数据，记录数据版本"""
    print("准备数据...")

    # 加载数据
    X, y = load_iris(return_X_y=True)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 记录数据集信息
    data_info = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

    return X_train, X_test, y_train, y_test, data_info


def train_model(n_estimators: int = 100, max_depth: int = 10):
    """训练模型并记录到 MLflow"""
    print(f"训练模型 (n_estimators={n_estimators}, max_depth={max_depth})...")

    # 设置 MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 准备数据
    X_train, X_test, y_train, y_test, data_info = prepare_data()

    # 开始运行
    with mlflow.start_run():
        # 记录参数
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42
        })

        # 记录数据信息
        mlflow.log_params(data_info)

        # 训练模型
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"准确率: {accuracy:.4f}")

        # 记录指标
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"]
        })

        # 记录模型
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="iris_classifier",
            input_example=X_train[:5]
        )

        # 记录 artifacts
        with open("metrics.txt", "w") as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact("metrics.txt")

        run_id = mlflow.active_run().info.run_id
        print(f"训练完成！Run ID: {run_id}")

        return run_id


def predict_model(features: list):
    """加载模型并预测"""
    print(f"预测特征: {features}")

    # 加载模型
    model_uri = "models:/iris_classifier/Production"
    model = mlflow.sklearn.load_model(model_uri)

    # 预测
    prediction = model.predict([features])[0]
    print(f"预测结果: {prediction}")

    return prediction


def serve_model():
    """启动模型服务（FastAPI）"""
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="Iris Classifier API")

    class PredictRequest(BaseModel):
        features: list

    class PredictResponse(BaseModel):
        prediction: int
        confidence: float

    # 加载模型
    model_uri = "models:/iris_classifier/Production"
    model = mlflow.sklearn.load_model(model_uri)

    @app.get("/")
    def read_root():
        return {"message": "Iris Classifier API", "model_version": "1.0.0"}

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        features = np.array(request.features).reshape(1, -1)

        # 预测
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = proba[prediction]

        return PredictResponse(
            prediction=int(prediction),
            confidence=float(confidence)
        )

    print("启动服务...")
    uvicorn.run(app, host="0.0.0.0", port=8080)


# CLI 入口
@click.group()
def cli():
    """MLflow 示例工具"""
    pass


@cli.command()
@click.option("--n-estimators", default=100, help="树的数量")
@click.option("--max-depth", default=10, help="最大深度")
def train(n_estimators, max_depth):
    """训练模型"""
    train_model(n_estimators, max_depth)


@cli.command()
@click.option("--sepal-length", type=float, required=True)
@click.option("--sepal-width", type=float, required=True)
@click.option("--petal-length", type=float, required=True)
@click.option("--petal-width", type=float, required=True)
def predict(sepal_length, sepal_width, petal_length, petal_width):
    """预测"""
    features = [sepal_length, sepal_width, petal_length, petal_width]
    result = predict_model(features)
    print(f"预测类别: {result}")


@cli.command()
def serve():
    """启动服务"""
    serve_model()


if __name__ == "__main__":
    cli()
```

### 9.3 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY mlflow_tracking_example.py .

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8080/health || exit 1

# 启动命令
CMD ["python", "mlflow_tracking_example.py", "serve"]
```

### 9.4 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  # MLflow Tracking Server
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.0
    ports:
      - "5000:5000"
    command:
      - server
      - --backend-store-uri
      - sqlite:///mlflow/mlflow.db
      - --default-artifact-root
      - /mlflow/artifacts
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
      - mlflow_db:/mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5000

  # 模型服务
  model:
    build: .
    ports:
      - "8080:8080"
    depends_on:
      - mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    restart: unless-stopped

volumes:
  mlflow_artifacts:
  mlflow_db:
```

### 9.5 快速启动

```bash
# 1. 安装依赖
pip install mlflow scikit-learn click fastapi uvicorn docker

# 2. 启动服务
docker-compose up -d

# 3. 训练模型
python mlflow_tracking_example.py train --n-estimators 100

# 4. 预测
python mlflow_tracking_example.py predict \
  --sepal-length 5.1 \
  --sepal-width 3.5 \
  --petal-length 1.4 \
  --petal-width 0.2

# 5. 访问 UI
# http://localhost:5000

# 6. API 调用
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

---

## 10. 学习资源

### 书籍

1. **《Introducing MLOps》** by Mark Treveil
   - 系统性介绍 MLOps 全流程
   - 适合入门

2. **《Designing Machine Learning Systems》** by Chip Huyen
   - 深入讲解 ML 系统设计
   - 包含大量实战经验

3. **《Building Machine Learning Pipelines》** by Hannes Hapke
   - 聚焦管道构建
   - TFX 等工具详解

### 在线课程

1. **DeepLearning.AI MLOps Specialization** (Coursera)
   - Andrew Ng 主讲
   - 覆盖全流程，含项目

2. **Databricks Academy MLOps**
   - 免费在线课程
   - 基于实战平台

### 实践平台

1. **Kaggle**
   - 数据集和 competitions
   - Learn 分区有教程

2. **Hugging Face**
   - 模型 hub 和 datasets
   - Spaces 可部署 demo

3. **AWS/Azure/GCP 免费层**
   - 实际云平台操作
   - 各有 $200-300 免费额度

---

## 附录：面试速记

### 关键概念对比

| 概念 | 定义 | 关键点 |
|------|------|--------|
| **数据版本控制** | 追踪数据变化 | DVC, Delta Lake, 不可变 |
| **实验跟踪** | 记录训练过程 | MLflow, W&B, 参数+指标+artifacts |
| **模型注册** | 模型生命周期管理 | 版本, 阶段, 审批 |
| **金丝雀发布** | 渐进式流量切换 | 5% → 50% → 100% |
| **数据漂移** | 输入分布变化 | P(X) 变化, PSI, KS test |
| **模型漂移** | 输出关系变化 | P(Y\|X) 变化, 需标签 |
| **A/B 测试** | 线上效果验证 | 对照组, 统计显著性 |
| **影子测试** | 并行部署但不服务流量 | 验证性能, 零风险 |

### 常用命令速查

```bash
# DVC
dvc init                              # 初始化
dvc add data/train.csv                # 添加数据
dvc push                             # 推送远程
dvc checkout data.dvc                # 拉取数据

# MLflow
mlflow ui                            # 启动 UI
mlflow runs list                     # 列出运行
mlflow experiments list              # 列出实验
mlflow models serve -m models:/name/1  # 服务模型

# Docker
docker build -t model:latest .       # 构建镜像
docker run -p 8080:8080 model      # 运行容器
docker push registry/model            # 推送镜像

# Kubernetes
kubectl apply -f deployment.yaml    # 部署
kubectl rollout status deploy/model   # 查看状态
kubectl scale deploy/model --replicas=3  # 扩容
```

---

## 结语

MLOps 的核心是**自动化**和**标准化**。记住：

1. **一切皆版本**：数据、代码、模型、环境
2. **一切可追溯**：每个决策都有记录
3. **一切自动化**：减少人工操作
4. **一切可监控**：了解系统在做什么

面试时，先讲框架，再深入细节，最后举实例。祝你面试顺利！
