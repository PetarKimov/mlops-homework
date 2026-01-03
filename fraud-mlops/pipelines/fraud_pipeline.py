from kfp import dsl
from kfp.dsl import Artifact, Dataset, Model, Output, Input


REGISTRY = "host.minikube.internal:5000"
TAG = "0.1"


@dsl.container_component
def download_data(dataset: str, raw_data: Output[Dataset]):
    return dsl.ContainerSpec(
        image=f"{REGISTRY}/fraud-download:{TAG}",
        command=["python", "-m", "fraud.data_download"],
        args=["--dataset", dataset, "--out_dir", raw_data.path],
    )


@dsl.container_component
def preprocess_data(
    raw_data: Input[Dataset],
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
    scaler: Output[Artifact],
    schema: Output[Artifact],
):
    return dsl.ContainerSpec(
        image=f"{REGISTRY}/fraud-preprocess:{TAG}",
        command=["python", "-m", "fraud.preprocess"],
        args=[
            "--input_csv", f"{raw_data.path}/creditcard.csv",
            "--out_train", f"{train_data.path}/train.parquet",
            "--out_val", f"{val_data.path}/val.parquet",
            "--out_test", f"{test_data.path}/test.parquet",
            "--out_scaler", f"{scaler.path}/scaler.pkl",
            "--out_schema", f"{schema.path}/schema.json",
        ],
    )


@dsl.container_component
def train_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model: Output[Model],
    train_metrics: Output[Artifact],
    hidden: int = 128,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 512,
    epochs: int = 5,
):
    return dsl.ContainerSpec(
        image=f"{REGISTRY}/fraud-train:{TAG}",
        command=["python", "-m", "fraud.train"],
        args=[
            "--train_parquet", f"{train_data.path}/train.parquet",
            "--val_parquet", f"{val_data.path}/val.parquet",
            "--out_model", f"{model.path}/model.pt",
            "--out_metrics", f"{train_metrics.path}/train_metrics.json",
            "--hidden", hidden,
            "--dropout", dropout,
            "--lr", lr,
            "--batch_size", batch_size,
            "--epochs", epochs,
        ],
    )


@dsl.container_component
def evaluate_model(
    test_data: Input[Dataset],
    model: Input[Model],
    eval_metrics: Output[Artifact],
    threshold: Output[Artifact],
):
    return dsl.ContainerSpec(
        image=f"{REGISTRY}/fraud-evaluate:{TAG}",
        command=["python", "-m", "fraud.evaluate"],
        args=[
            "--test_parquet", f"{test_data.path}/test.parquet",
            "--model_path", f"{model.path}/model.pt",
            "--out_metrics", f"{eval_metrics.path}/eval_metrics.json",
            "--out_threshold", f"{threshold.path}/threshold.json",
        ],
    )


@dsl.pipeline(name="creditcard-fraud-train-eval", description="Download -> preprocess -> train -> evaluate")
def fraud_pipeline(
    dataset: str = "mlg-ulb/creditcardfraud",
    hidden: int = 128,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 512,
    epochs: int = 5,
):
    dl = download_data(dataset=dataset)
    pp = preprocess_data(raw_data=dl.outputs["raw_data"])
    tr = train_model(
        train_data=pp.outputs["train_data"],
        val_data=pp.outputs["val_data"],
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
    )
    evaluate_model(test_data=pp.outputs["test_data"], model=tr.outputs["model"])
