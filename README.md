**Master's Thesis – Data Science**

**Author:** Ander Vesperinas Calderón

**Title:** *Machine Learning Pipelines in Cloud Infrastructure: A Case Study on Face Recognition Model Quantization*

**Experimental results:** [Link](https://drive.google.com/file/d/1ho5HL4ivZ5kBAkmbwOpqfHN7HDxrUtK-/view?usp=sharing)

---

## 🧾 Summary

This contains an experiment in the cloud using AWS SageMaker, with a focus on **model quantization** to improve inference efficiency and reduce model size.

It leverages **AWS SageMaker Pipelines**, **Docker**, and custom **ScriptProcessor** steps to build a modular ML pipeline architecture.

---

## ⚠️ Disclaimer

Some pipeline steps are **non-functional** due to corporate confidentiality constraints. Parts of the code are **anonymized**, and certain logic relies on **internal proprietary libraries** that are not included in the repository.

---

## 📁 Project's src Structure

* **Docker/**: Custom Docker images, built and pushed to Amazon ECR for reuse across steps.
* **Pipelines/**: Core pipeline orchestration using SageMaker.
  * `constants.py`: Centralized configuration (e.g., URIs, paths).
  * `steps.py`: Defines each step and its dependencies.
  * `pipeline.py`: Constructs the overall pipeline.
* **Scripts/**: Step-specific logic in standalone Python scripts.
* **Utils/**: Shared helper functions and utilities.
* **Auxiliar/**: Auxiliary scripts for visualizing results and performance.

---

## ✅ Highlights

* Modular, cloud-ready MLOps pipeline.
* Face recognition model quantization with inference performance analysis.
* Containerized steps and reproducible experiment tracking.

