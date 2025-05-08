"""Steps for the face model quantization testing."""

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep

from . import constants


utils_module = ProcessingInput(
    source=str(constants.utils_path),
    destination=f"{constants.docker_base_path}/code/utils",
)


quantization_processor = ScriptProcessor(
    image_uri=constants.docker_quant_image_tag,
    sagemaker_session=constants.sagemaker_session,
    role=constants.role,
    command=["python3"],
    instance_count=1,
    instance_type="ml.m5.2xlarge",
)

general_face_processor = ScriptProcessor(
    image_uri=constants.docker_face_image_tag,
    sagemaker_session=constants.sagemaker_session,
    role=constants.role,
    command=["python"],
    instance_count=1,
    instance_type="ml.m5.2xlarge",
    env={"PYTHONPATH": f"{constants.docker_base_path}/code"},
)


model_quantization_step = ProcessingStep(
    name="ModelQuantizationStep",
    processor=quantization_processor,
    inputs=[
        ProcessingInput(
            source=f"{constants.s3_base_path}/model",
            destination=f"{constants.docker_base_path}/input",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source=f"{constants.docker_base_path}/output",
            destination=(f"{constants.s3_base_path}/model"),
        ),
    ],
    cache_config=constants.cache_config,
    code=str(constants.scripts_path / "model_quantization.py"),
    job_arguments=[
        "--input_model_name",
        constants.original_model_name,
        "--quantized_model_name",
        constants.quantized_model_name,
        "--quantization_recipe_name",
        constants.quantization_recipe_name,
    ],
)

face_detection_step = ProcessingStep(
    name="FaceDetectionStep",
    processor=general_face_processor,
    inputs=[
        utils_module,
        ProcessingInput(
            source=constants.dataset_path,
            destination=f"{constants.docker_base_path}/input",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source=f"{constants.docker_base_path}/output",
            destination=f"{constants.s3_base_path}/processed",
        ),
    ],
    cache_config=constants.cache_config,
    code=str(constants.scripts_path / "face_detection.py"),
    job_arguments=[
        "--detector_name",
        constants.detector_name,
    ],
)

face_normalization_step = ProcessingStep(
    name="FaceNormalizationStep",
    processor=general_face_processor,
    inputs=[
        utils_module,
        ProcessingInput(
            source=constants.dataset_path,
            destination=f"{constants.docker_base_path}/input",
        ),
        ProcessingInput(
            source=f"{constants.s3_base_path}/processed",
            destination=f"{constants.docker_base_path}/processed",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source=f"{constants.docker_base_path}/output",
            destination=f"{constants.s3_base_path}/processed",
        ),
    ],
    cache_config=constants.cache_config,
    depends_on=[face_detection_step],
    code=str(constants.scripts_path / "face_normalization.py"),
    job_arguments=[
        "--normalizer_name",
        constants.normalizer_name,
        "--criteria_name",
        constants.normalizing_criteria_name,
    ],
)

original_model_inference_step = ProcessingStep(
    name="OriginalModelInferenceStep",
    processor=general_face_processor,
    inputs=[
        utils_module,
        ProcessingInput(
            source=f"{constants.s3_base_path}/model",
            destination=f"{constants.docker_base_path}/model",
        ),
        ProcessingInput(
            source=constants.dataset_path,
            destination=f"{constants.docker_base_path}/input",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source=f"{constants.docker_base_path}/output",
            destination=f"{constants.s3_base_path}/processed",
        ),
    ],
    cache_config=constants.cache_config,
    depends_on=[face_normalization_step],
    code=str(constants.scripts_path / "face_inference.py"),
    job_arguments=[
        "--model_name",
        constants.original_model_name,
    ],
)

quantized_model_inference_step = ProcessingStep(
    name="QuantizedModelInferenceStep",
    processor=general_face_processor,
    inputs=[
        utils_module,
        ProcessingInput(
            source=f"{constants.s3_base_path}/model",
            destination=f"{constants.docker_base_path}/model",
        ),
        ProcessingInput(
            source=constants.dataset_path,
            destination=f"{constants.docker_base_path}/input",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source=f"{constants.docker_base_path}/output",
            destination=f"{constants.s3_base_path}/processed",
        ),
    ],
    cache_config=constants.cache_config,
    depends_on=[face_normalization_step, model_quantization_step],
    code=str(constants.scripts_path / "face_inference.py"),
    job_arguments=[
        "--model_name",
        constants.quantized_model_name,
    ],
)

original_pairs_evaluation_step = ProcessingStep(
    name="OriginalModelPairsEvaluationStep",
    processor=general_face_processor,
    inputs=[
        ProcessingInput(
            source=constants.dataset_path,
            destination=f"{constants.docker_base_path}/input",
        ),
        ProcessingInput(
            f"{constants.s3_base_path}/processed",
            destination=f"{constants.docker_base_path}/processed",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source=f"{constants.docker_base_path}/output",
            destination=f"{constants.s3_base_path}/processed",
        ),
    ],
    cache_config=constants.cache_config,
    depends_on=[original_model_inference_step],
    code=str(constants.scripts_path / "pairs_evaluation.py"),
    job_arguments=[
        "--model_name",
        constants.original_model_name,
    ],
)

quantized_pairs_evaluation_step = ProcessingStep(
    name="QuantizedModelPairsEvaluationStep",
    processor=general_face_processor,
    inputs=[
        ProcessingInput(
            source=constants.dataset_path,
            destination=f"{constants.docker_base_path}/input",
        ),
        ProcessingInput(
            f"{constants.s3_base_path}/processed",
            destination=f"{constants.docker_base_path}/processed",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source=f"{constants.docker_base_path}/output",
            destination=f"{constants.s3_base_path}/processed",
        ),
    ],
    cache_config=constants.cache_config,
    depends_on=[quantized_model_inference_step],
    code=str(constants.scripts_path / "pairs_evaluation.py"),
    job_arguments=[
        "--model_name",
        constants.quantized_model_name,
    ],
)

original_verification_metrics_step = ProcessingStep(
    name="OriginalModelMetricsVerificationStep",
    processor=general_face_processor,
    inputs=[
        utils_module,
        ProcessingInput(
            f"{constants.s3_base_path}/processed",
            destination=f"{constants.docker_base_path}/processed",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source=f"{constants.docker_base_path}/output",
            destination=f"{constants.s3_base_path}/metric",
        ),
    ],
    cache_config=constants.cache_config,
    depends_on=[original_pairs_evaluation_step],
    code=str(constants.scripts_path / "verification_metrics.py"),
    job_arguments=[
        "--model_name",
        constants.original_model_name,
    ],
)

quantized_verification_metrics_step = ProcessingStep(
    name="QuantizedModelMetricsVerificationStep",
    processor=general_face_processor,
    inputs=[
        utils_module,
        ProcessingInput(
            f"{constants.s3_base_path}/processed",
            destination=f"{constants.docker_base_path}/processed",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source=f"{constants.docker_base_path}/output",
            destination=f"{constants.s3_base_path}/metric",
        ),
    ],
    cache_config=constants.cache_config,
    depends_on=[quantized_pairs_evaluation_step],
    code=str(constants.scripts_path / "verification_metrics.py"),
    job_arguments=[
        "--model_name",
        constants.quantized_model_name,
    ],
)
