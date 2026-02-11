"""
MLflow 端到端示例：训练、跟踪、部署

运行方式：
1. 启动 MLflow: mlflow ui
2. 运行训练: python mlflow_tracking_example.py train
3. 运行预测: python mlflow_tracking_example.py predict
"""

import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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
        confidence = float(proba[prediction])

        return PredictResponse(
            prediction=int(prediction),
            confidence=confidence
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
