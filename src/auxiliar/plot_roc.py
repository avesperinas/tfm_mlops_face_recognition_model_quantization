import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


MAX_FPR = 1e-4
BASE_PATH = Path("/home/avesperinas/Escritorio/metrics/")
MODELS = ["original", "quantized"]

def plot_roc_curve(model_name):

    csv_file = BASE_PATH / F"roc_{model_name}.csv"
    png_file = BASE_PATH / F"roc_{model_name}.png"

    roc_data = pd.read_csv(csv_file)
    filtered_roc = roc_data[roc_data['False Positive Rate'] <= MAX_FPR]

    plt.figure(figsize=(8, 6))
    plt.plot(
        filtered_roc['False Positive Rate'],
        filtered_roc['True Positive Rate'],
        marker='o',
        label='ROC Curve (FPR ≤ {:.2f})'.format(MAX_FPR)
    )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Filtered ROC Curve {model_name.capitalize()} model')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig(png_file)


def compare_roc_curves(models):

    plt.figure(figsize=(8, 6))
    for model_name in models:
        csv_file = BASE_PATH / f"roc_{model_name}.csv"
        roc_data = pd.read_csv(csv_file)
        filtered_roc = roc_data[roc_data['False Positive Rate'] <= MAX_FPR]

        plt.plot(
            filtered_roc['False Positive Rate'],
            filtered_roc['True Positive Rate'],
            label=f'{model_name.capitalize()} Model'
        )

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison (FPR ≤ {:.2e})'.format(MAX_FPR))
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig(BASE_PATH / "roc_comparison.png")
    plt.close()


if __name__ == "__main__":
    compare_roc_curves(MODELS)
