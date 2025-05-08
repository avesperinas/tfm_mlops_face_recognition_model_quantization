"""Module to define the pipeline for the penguins problem."""

import os
from pathlib import Path

from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.steps import CacheConfig


# AWS Session
sagemaker_session = None
if os.environ.get("AWS_SM_LOCAL_SESSION"):
    from sagemaker.workflow.pipeline_context import LocalPipelineSession

    sagemaker_session = LocalPipelineSession()
else:
    from sagemaker.workflow.pipeline_context import PipelineSession

    sagemaker_session = PipelineSession(
        default_bucket=os.environ.get("AWS_DEFAULT_BUCKET"),
        default_bucket_prefix="pipelines",
    )


# AWS config
role = os.environ.get("AWS_ROLE_ARN")
cache_config = CacheConfig(enable_caching=True, expire_after="15d")
pipeline_definition_config = PipelineDefinitionConfig(use_custom_job_prefix=False)

# S3 bucket
s3_base_path = f"s3://{os.environ.get('AWS_DEFAULT_BUCKET')}"

# Input dataset
dataset_path = f"{s3_base_path}/reduced_dataset"

# Models
original_model_name = "model_float32.tflite"
quantized_model_name = "model_dynamic_wi8_afp32.tflite"
quantization_recipe_name = "dynamic_wi8_afp32"

# Detecting and normalizing faces
detector_name = "mtcnn"
normalizer_name = "simtrans"
normalizing_criteria_name = "most-centered-and-largest"

# Docker image URI
docker_face_image_tag = os.environ.get("DOCKER_FACE_IMAGE_TAG")
docker_quant_image_tag = os.environ.get("DOCKER_QUANT_IMAGE_TAG")

# Local paths
utils_path = Path(__file__).resolve().parent.parent.parent.parent / "utils"
scripts_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "processing"
docker_base_path = "/opt/ml/processing"
