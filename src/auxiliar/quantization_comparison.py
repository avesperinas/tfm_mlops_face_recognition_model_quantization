import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

np.random.seed(42)
torch.manual_seed(42)

num_weights = 1000
original_weights = np.random.normal(0, 0.5, num_weights)

def uniform_quantization(weights, bits=8):
    """Function for uniform quantization (simulates 8-bit quantization in ML)."""
    min_val, max_val = weights.min(), weights.max()
    step = (max_val - min_val) / (2**bits - 1)
    quantized = min_val + step * np.round((weights - min_val) / step)
    return quantized, step

def non_uniform_quantization(weights, clusters=256):
    """Non-uniform quantization based on weight distribution (similar to distribution-aware quantization)."""
    weights_reshaped = weights.reshape(-1, 1)
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit(weights_reshaped)
    centroids = kmeans.cluster_centers_.flatten()
    quantized = np.zeros_like(weights)
    labels = kmeans.labels_
    
    for i in range(len(weights)):
        quantized[i] = centroids[labels[i]]
    
    return quantized, centroids

uniform_quantized, uniform_step = uniform_quantization(original_weights, bits=3)
nonuniform_quantized, centroids = non_uniform_quantization(original_weights, clusters=8)

# Calculate quantization errors
uniform_error = np.abs(original_weights - uniform_quantized)
nonuniform_error = np.abs(original_weights - nonuniform_quantized)

bins = 50

# Create a single figure with both representations
plt.figure(figsize=(12, 10))

# Top half - Uniform Quantization
plt.subplot(2, 1, 1)
plt.hist(original_weights, bins=bins, alpha=0.5, label='Original Weights', color='blue')
plt.hist(uniform_quantized, bins=bins, alpha=0.5, label='Uniform Quantized Weights', color='red')
for level in np.unique(uniform_quantized):
    plt.axvline(x=level, color='r', linestyle='--', alpha=0.7)
plt.title("Uniform Weight Quantization (3-bit)")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

# Bottom half - Non-Uniform Quantization
plt.subplot(2, 1, 2)
plt.hist(original_weights, bins=bins, alpha=0.5, label='Original Weights', color='blue')
plt.hist(nonuniform_quantized, bins=bins, alpha=0.5, label='Non-Uniform Quantized Weights', color='green')
for level in np.unique(nonuniform_quantized):
    plt.axvline(x=level, color='g', linestyle='--', alpha=0.7)
plt.title("Non-Uniform Weight Quantization (K-means, 8 clusters)")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("quantization_comparison.png", dpi=300)
plt.show()

print("Image generated: quantization_comparison.png")