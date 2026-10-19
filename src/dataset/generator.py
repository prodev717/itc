"""
Data Generator for Multi-Domain Lossless Communication Benchmarks.
Generates diverse data types covering key Information Theory regimes:
1. Natural English Text (Skewed letter/word frequency, dictionary/LZ77 optimal)
2. Structured JSON / Source Code (Repetitive nested tokens, dictionary/LZ77 optimal)
3. Dyadic Skewed Memoryless Source (Powers of 1/2 probabilities, exactly optimal for Huffman prefix trees)
4. Continuous Non-Dyadic Sensor Telemetry (Stationary skewed distribution, arithmetic coding optimal)
5. Correlated 8-bit Raster Image / Audio PCM (Smooth spatial gradients and runs)
6. Executable Binary / Machine Code (Mixed opcodes and addresses)
7. Cryptographic Random Payload (Uniform max entropy ~8.0 bits/byte, incompressible, Raw pass-through optimal)
"""

import math
import os
import random
import struct
import numpy as np
from typing import Dict, List, Tuple


def generate_text_sample(size_bytes: int, rng: random.Random) -> bytes:
    """Natural English text with Zipfian word distribution."""
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "adaptive", "coding", "channel", "entropy", "neural", "network", "ldpc", "lossless",
        "communication", "information", "theory", "transmission", "parity", "syndrome"
    ]
    weights = [1.0 / (i + 1) ** 0.85 for i in range(len(words))]
    total = sum(weights)
    weights = [w / total for w in weights]
    
    parts = []
    current_len = 0
    while current_len < size_bytes:
        w = rng.choices(words, weights=weights, k=1)[0]
        punct = rng.choice(["", "", "", " ", ", ", ". ", "\n"])
        token = (w + punct).encode("ascii")
        parts.append(token)
        current_len += len(token)
    
    data = b"".join(parts)
    return data[:size_bytes]


def generate_json_code_sample(size_bytes: int, rng: random.Random) -> bytes:
    """Structured JSON / code telemetry data with recurrent keys."""
    keys = ["\"device_id\"", "\"timestamp\"", "\"temperature\"", "\"status\"", "\"voltage\"", "\"metrics\"", "\"ip_address\""]
    statuses = ["\"ACTIVE\"", "\"IDLE\"", "\"CALIBRATING\"", "\"STANDBY\""]
    ips = ["\"192.168.1.10\"", "\"10.0.0.42\"", "\"172.16.5.101\""]
    
    parts = [b"[\n"]
    curr = len(parts[0])
    while curr < size_bytes:
        item = (
            f'  {{"id": {rng.randint(100, 999)}, '
            f'{rng.choice(keys)}: {round(rng.uniform(18.0, 45.0), 2)}, '
            f'{rng.choice(keys)}: {rng.choice(statuses)}, '
            f'{rng.choice(keys)}: {rng.choice(ips)}}},\n'
        ).encode("utf-8")
        parts.append(item)
        curr += len(item)
    parts.append(b"]\n")
    data = b"".join(parts)
    return data[:size_bytes]


def generate_dyadic_sample(size_bytes: int, rng: random.Random) -> bytes:
    """
    Dyadic Memoryless Source:
    Probabilities are negative powers of 2: p = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/128].
    By Shannon Information Theory, Huffman coding achieves exact entropy H(X) with 0 redundancy,
    and dictionary coding expands due to lack of repeated n-grams.
    """
    probs = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.0078125]
    symbols = list(range(8))
    # Draw independent symbols without recurrent phrases
    chosen = rng.choices(symbols, weights=probs, k=size_bytes)
    return bytes(chosen)


def generate_sensor_sample(size_bytes: int, rng: random.Random) -> bytes:
    """
    Continuous Non-Dyadic Stationary Sensor Telemetry:
    Non-dyadic probabilities where fractional bits in Arithmetic Coding outperform Huffman.
    """
    # Truncated Gaussian distribution over byte values with non-integer log2(1/p)
    out = bytearray()
    center = rng.randint(90, 150)
    std = rng.uniform(12.0, 25.0)
    for _ in range(size_bytes):
        val = int(rng.gauss(center, std))
        val = max(0, min(255, val))
        out.append(val)
    return bytes(out)


def generate_raster_image_sample(size_bytes: int, rng: random.Random) -> bytes:
    """Correlated 2D raster scan lines with smooth gradients."""
    out = bytearray()
    curr_line_val = rng.randint(40, 200)
    width = 64
    while len(out) < size_bytes:
        curr_line_val = max(0, min(255, curr_line_val + rng.randint(-8, 8)))
        v = curr_line_val
        for _ in range(min(width, size_bytes - len(out))):
            v = max(0, min(255, v + rng.randint(-3, 3)))
            out.append(v)
    return bytes(out[:size_bytes])


def generate_binary_sample(size_bytes: int, rng: random.Random) -> bytes:
    """Compiled binary machine code with repeated headers and alignment."""
    common_opcodes = [0x55, 0x48, 0x89, 0xE5, 0x48, 0x83, 0xEC, 0x10, 0xC3, 0x90, 0x00, 0x00, 0xFF, 0xE8]
    out = bytearray()
    while len(out) < size_bytes:
        mode = rng.random()
        if mode < 0.35:
            seq_len = rng.randint(4, 16)
            out.extend(rng.choice(common_opcodes) for _ in range(min(seq_len, size_bytes - len(out))))
        elif mode < 0.65:
            pad_len = rng.randint(2, 8)
            out.extend([0x00] * min(pad_len, size_bytes - len(out)))
        else:
            out.append(rng.randrange(256))
    return bytes(out[:size_bytes])


def generate_random_sample(size_bytes: int, rng: random.Random) -> bytes:
    """Maximum entropy uniform distribution (8.0 bits/byte, incompressible)."""
    return bytes(rng.randrange(256) for _ in range(size_bytes))


DATA_TYPE_GENERATORS = {
    "dyadic_skewed": generate_dyadic_sample,
    "sensor_gaussian": generate_sensor_sample,
    "text_natural": generate_text_sample,
    "json_structured": generate_json_code_sample,
    "binary_code": generate_binary_sample,
    "random_payload": generate_random_sample,
}


def build_benchmark_dataset(
    samples_per_type: int = 50,
    sample_size: int = 2048,
    seed: int = 42
) -> List[Dict]:
    """
    Build benchmark dataset across diverse information-theoretic domains.
    """
    dataset = []
    
    for type_id, (dtype, generator) in enumerate(DATA_TYPE_GENERATORS.items()):
        for s_idx in range(samples_per_type):
            sample_seed = seed + type_id * 10000 + s_idx
            sub_rng = random.Random(sample_seed)
            data = generator(sample_size, sub_rng)
            dataset.append({
                "sample_id": f"{dtype}_{s_idx:03d}",
                "type": dtype,
                "type_id": type_id,
                "data": data,
                "size_bytes": len(data),
            })
            
    return dataset
