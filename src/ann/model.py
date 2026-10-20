"""
Feature Extraction and ANN Classifier for Adaptive Source Coding.
Features extracted per data chunk (payload):
1. Empirical Shannon Entropy (bits/symbol, normalized 0..1)
2. Normalized Byte Variance (spread of values across 0..255)
3. Zero-byte density (fraction of 0x00 bytes)
4. Consecutive byte repeat ratio (run-length / smoothness marker)
5. Lag-1 Autocorrelation (degree of temporal/spatial correlation)
6. Unique symbol count / symbol alphabet cardinality (out of 256)
7. Upper-half byte ratio (fraction of bytes >= 128)
8. High-frequency symbol concentration (top-4 most frequent symbols ratio)

Target Classes:
- 0: Huffman Coding (High throughput, memoryless skewed symbol distribution)
- 1: Arithmetic Coding (High compression efficiency near Shannon bound)
- 2: Dictionary Coding (LZ77 Sliding Window, optimal for repetitive strings/text/JSON)
- 3: Raw Pass-through (Identity, prevents expansion on high-entropy/random payloads)
"""

from collections import Counter
import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict


def extract_features(data: bytes) -> np.ndarray:
    """Extract an 8-dimensional normalized statistical feature vector from raw byte chunk."""
    n = len(data)
    if n == 0:
        return np.zeros(8, dtype=np.float32)
        
    counts = Counter(data)
    
    # 1. Shannon Entropy
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values()) / 8.0
    
    # 2. Byte Variance
    arr = np.frombuffer(data, dtype=np.uint8)
    mean_val = np.mean(arr)
    # Normalize by max possible variance for U[0,255]: Var_max = (255/2)^2 = 16256.25
    var_val = np.var(arr) / 16256.25
    
    # 3. Zero-byte density
    zero_density = counts.get(0, 0) / n
    
    # 4. Consecutive byte repeat ratio
    repeats = sum(1 for i in range(n - 1) if data[i] == data[i + 1])
    repeat_ratio = repeats / max(1, n - 1)
    
    # 5. Lag-1 Autocorrelation
    if n > 1 and np.std(arr) > 1e-5:
        centered = arr - mean_val
        autocorr = np.sum(centered[:-1] * centered[1:]) / (np.sum(centered ** 2) + 1e-8)
        autocorr = (autocorr + 1.0) / 2.0
    else:
        autocorr = 0.5
        
    # 6. Distinct symbol alphabet cardinality
    cardinality = len(counts) / 256.0
    
    # 7. Upper-half byte ratio
    upper_ratio = sum(1 for b in data if b >= 128) / n
    
    # 8. Top-4 symbol concentration
    top4 = sum(c for _, c in counts.most_common(4)) / n
    
    features = np.array([
        entropy,
        var_val,
        zero_density,
        repeat_ratio,
        autocorr,
        cardinality,
        upper_ratio,
        top4
    ], dtype=np.float32)
    
    return features


class AdaptiveCodingClassifier(nn.Module):
    """
    Feedforward Artificial Neural Network (Multilayer Perceptron) with LayerNorm
    Input: 8-dim feature vector
    Hidden: 64 -> LayerNorm -> ReLU -> Dropout -> 32 -> LayerNorm -> ReLU -> 16 -> ReLU -> 4
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_classes: int = 4, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
