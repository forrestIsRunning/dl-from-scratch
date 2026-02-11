# MLOps 端到端示例

这是一个完整的 MLOps 示例，展示如何使用 MLflow 进行模型训练、跟踪和部署。

## 快速开始

### 1. 启动 MLflow 服务

```bash
# 使用 Docker Compose 启动 MLflow 和模型服务
docker-compose -f docker-compose.mlflow.yml up -d

# 访问 MLflow UI
# http://localhost:5000
```

### 2. 训练模型

```bash
# 安装依赖
pip install -r mlflow-requirements.txt

# 训练模型（会自动记录到 MLflow）
python mlflow_tracking_example.py train --n-estimators 100 --max-depth 10

# 训练多个实验（不同参数）
python mlflow_tracking_example.py train --n-estimators 50 --max-depth 5
python mlflow_tracking_example.py train --n-estimators 200 --max-depth 15
```

### 3. 预测

```bash
# 使用训练好的模型进行预测
python mlflow_tracking_example.py predict \
  --sepal-length 5.1 \
  --sepal-width 3.5 \
  --petal-length 1.4 \
  --petal-width 0.2
```

### 4. 启动推理服务

```bash
# 启动 REST API 服务
python mlflow_tracking_example.py serve

# 访问健康检查
curl http://localhost:8080/health

# 发送预测请求
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## 项目结构

```
mlops-example/
├── mlflow_tracking_example.py   # 主程序（含 train/predict/serve 命令）
├── Dockerfile.mlflow            # 容器化定义
├── docker-compose.mlflow.yml    # 服务编排
├── mlflow-requirements.txt      # Python 依赖
└── MLFLOW_EXAMPLE_README.md    # 详细使用说明
```

## MLflow 核心概念

### 1. Experiment（实验）
一组相关的运行，按项目或任务组织。

```python
mlflow.set_experiment("iris_classifier")
```

### 2. Run（运行）
单次训练执行，记录参数、指标和 artifacts。

```python
with mlflow.start_run():
    mlflow.log_param("lr", 0.001)
    mlflow.log_metric("accuracy", 0.95)
```

### 3. Model Registry（模型注册）
管理模型生命周期，支持版本和阶段管理。

```python
# 注册模型
mlflow.sklearn.log_model(model, "model", registered_model_name="iris_classifier")

# 加载模型
model = mlflow.sklearn.load_model("models:/iris_classifier/Production")
```

## 命令参考

```bash
# 训练相关命令
python mlflow_tracking_example.py train --help
python mlflow_tracking_example.py train --n-estimators 100 --max-depth 10

# 预测命令
python mlflow_tracking_example.py predict --help

# 启动服务
python mlflow_tracking_example.py serve
```

## 清理

```bash
# 停止服务
docker-compose -f docker-compose.mlflow.yml down

# 清理卷
docker volume rm mlflow_artifacts mlflow_db
```
