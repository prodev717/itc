"""
Main entry point for ANN-Based Adaptive Source Coding with LDPC Channel Coding.
Provides both an interactive demonstration and programmatic research benchmark launcher.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_experiments import run_full_benchmark, train_and_evaluate_ann
from src.dataset.generator import build_benchmark_dataset
from src.ann.model import AdaptiveCodingClassifier
from src.ann.trainer import build_ann_training_dataset
from src.channel_coding.ldpc import LDPCCodec
from src.channel.channel import WiredChannel
from src.pipeline import AdaptiveTransmissionPipeline


def run_interactive_demo():
    print("=" * 80)
    print("ANN-Adaptive Source Coding with (128,64) LDPC Channel Coding Demo")
    print("=" * 80)

    # 1. Quick dataset build & ANN training
    print("\n[1] Initializing multi-domain dataset and lightweight training...")
    dataset = build_benchmark_dataset(samples_per_type=20, sample_size=2048, seed=42)
    X, y, meta = build_ann_training_dataset(dataset)
    model, eval_ann = train_and_evaluate_ann(X, y, meta, epochs=45, seed=42)
    print(f"-> ANN Classifier Ready | Accuracy: {eval_ann['accuracy']*100:.1f}% | Latency: {eval_ann['inference_latency_us']:.2f} us")

    # 2. Pipeline setup
    ldpc = LDPCCodec(n=128, k=64, dv=3, seed=42)
    channel = WiredChannel(snr_db=8.0, impulse_prob=0.0005, seed=42)
    pipeline = AdaptiveTransmissionPipeline(model, ldpc, channel)

    # 3. Demonstration across 3 representative sources
    demo_indices = [0, 40, 60]  # Dyadic, JSON, Text
    print("\n[2] Executing live adaptive transmission over wired channel (SNR = 8.0 dB)...")

    for idx in demo_indices:
        sample = dataset[idx]
        data = sample["data"]
        dtype = sample["type"]
        
        # Adaptive run
        res_adapt = pipeline.run_pipeline(data, forced_codec_id=None)
        # Fixed Huffman run
        res_huff = pipeline.run_pipeline(data, forced_codec_id=0)

        print("-" * 70)
        print(f"Data Category          : {dtype}")
        print(f"Original Byte Length   : {res_adapt['orig_bytes']} bytes")
        print(f"ANN Selected Codec     : {res_adapt['codec_selected']}")
        print(f"Adaptive Comp. Ratio   : {res_adapt['compression_ratio']:.2f}x ({res_adapt['bit_saving_pct']:.1f}% bit saving)")
        print(f"Huffman Comp. Ratio    : {res_huff['compression_ratio']:.2f}x ({res_huff['bit_saving_pct']:.1f}% bit saving)")
        print(f"Transmitted LDPC Bits  : {res_adapt['ldpc_transmitted_bits']} bits")
        print(f"Channel Raw BER        : {res_adapt['raw_channel_ber']:.4e}")
        print(f"Post-LDPC Residual BER : {res_adapt['post_ldpc_ber']:.4e}")
        print(f"Lossless Reconstructed : {'[YES] 100% Exact Match' if res_adapt['lossless_verified'] else '[NO] Corrupted'}")

    print("-" * 70)
    print("\nTo run the full research paper benchmark suite with SNR sweeps and figure generation:")
    print("    python experiments/run_experiments.py")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANN Adaptive Coding + LDPC Comms")
    parser.add_argument("--benchmark", action="store_true", help="Run full scientific benchmark suite")
    args = parser.parse_args()

    if args.benchmark:
        run_full_benchmark()
    else:
        run_interactive_demo()
