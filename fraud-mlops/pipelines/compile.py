from kfp import compiler
from fraud_pipeline import fraud_pipeline

compiler.Compiler().compile(
    pipeline_func=fraud_pipeline,
    package_path="fraud_pipeline.yaml",
)
print("Wrote fraud_pipeline.yaml")