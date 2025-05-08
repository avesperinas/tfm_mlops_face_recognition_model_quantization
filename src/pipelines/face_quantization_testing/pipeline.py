"""Pipeline definition for the face biometrics training."""

from sagemaker.workflow.pipeline import Pipeline

from . import constants, steps


role = constants.role
pipeline = Pipeline(
    name="FaceQuantizationTesting",
    steps=[
        # Quantize the original model
        steps.model_quantization_step,
        # Preprocess data
        steps.face_detection_step,
        steps.face_normalization_step,
        # Original Model evaluation
        steps.original_model_inference_step,
        steps.original_pairs_evaluation_step,
        steps.original_verification_metrics_step,
        # Quantized Model evaluation
        steps.quantized_model_inference_step,
        steps.quantized_pairs_evaluation_step,
        steps.quantized_verification_metrics_step,
    ],
    pipeline_definition_config=constants.pipeline_definition_config,
    sagemaker_session=constants.sagemaker_session,
)
