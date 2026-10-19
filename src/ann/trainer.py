"""
Ground Truth Oracle & Label Generator for ANN Training:
Evaluates compression efficiency and determines the optimal source codec for each sample:
0: Huffman (High throughput prefix coding)
1: Arithmetic (High compression near theoretical Shannon limit)
2: LZ77 Dictionary (Optimal for repetitive patterns, structured code, text)
3: Raw Pass-through (High-entropy incompressible data)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

from src.source_coding.codecs import huffman_encode, arithmetic_encode, dictionary_lz77_encode, raw_encode
from src.ann.model import extract_features

CODEC_NAMES = ["Huffman", "Arithmetic", "Dictionary_LZ77", "Raw_Passthrough"]


def determine_optimal_codec(data: bytes) -> Tuple[int, Dict[str, int]]:
    """Evaluates candidates and picks the one minimizing bitstream length."""
    h_bytes, _ = huffman_encode(data)
    a_bytes, _ = arithmetic_encode(data)
    d_bytes, _ = dictionary_lz77_encode(data)
    raw_len = len(data)
    
    lens = {
        "Huffman": len(h_bytes),
        "Arithmetic": len(a_bytes),
        "Dictionary_LZ77": len(d_bytes),
        "Raw_Passthrough": raw_len
    }
    
    min_comp_len = min(lens["Huffman"], lens["Arithmetic"], lens["Dictionary_LZ77"])
    
    # 1. Incompressible / high-entropy payloads where compression expands or yields < 1.5% gain
    if min_comp_len >= raw_len * 0.985:
        return 3, lens
        
    # 2. Dictionary LZ77 wins decisively when repetitive structure yields >= 5% smaller footprint than symbol codecs
    if lens["Dictionary_LZ77"] < min(lens["Huffman"], lens["Arithmetic"]) - 20:
        return 2, lens

    # 3. Arithmetic wins on stationary distributions where fractional bit allocation saves >= 1.0%
    if (lens["Huffman"] - lens["Arithmetic"]) >= 8:
        return 1, lens

    # 4. Huffman wins when symbol frequencies are skewed and its low complexity gives near-identical compression
    return 0, lens


def build_ann_training_dataset(samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Extracts features and optimal labels for all dataset items.
    Returns: (X, y, detailed_meta)
    """
    X_list = []
    y_list = []
    meta_list = []
    
    for item in samples:
        data = item["data"]
        feat = extract_features(data)
        label, sizes = determine_optimal_codec(data)
        
        X_list.append(feat)
        y_list.append(label)
        meta_list.append({
            "sample_id": item["sample_id"],
            "type": item["type"],
            "features": feat,
            "optimal_label": label,
            "optimal_codec": CODEC_NAMES[label],
            "sizes": sizes
        })
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), meta_list


class CodingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
