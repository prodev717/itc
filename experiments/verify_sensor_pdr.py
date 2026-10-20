"""
Sensor Domain PDR & BLER Empirical Verification Script.
Logs exact block counts, message block errors, bit errors, and packet delivery rates
for all 50 sensor payloads under SNR = 7.0 dB, P_impulse = 5e-4, alpha_emi = 3.0.
Saves complete raw logs to results/sensor_pdr_verification.log.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.dataset.generator import build_benchmark_dataset
from src.ann.trainer import build_ann_training_dataset
from src.ann.model import AdaptiveCodingClassifier
from src.channel_coding.ldpc import LDPCCodec
from src.channel.channel import WiredChannel
from src.pipeline import AdaptiveTransmissionPipeline, CODEC_DISPATCH
from src.source_coding.codecs import arithmetic_encode, arithmetic_decode


def run_sensor_verification():
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "sensor_pdr_verification.log"

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 80)
    log("SENSOR TELEMETRY DOMAIN: PDR & BLER MEASUREMENT & RECONCILIATION")
    log("=" * 80)
    log("Channel parameters: SNR = 7.0 dB, P_impulse = 5e-4, alpha_emi = 3.0")
    log("LDPC Codec: Systematic (128, 64) rate R=0.5, normalized Min-Sum BP (alpha=0.85, max_iter=25)")
    log("Payload size: 2048 bytes per packet, N = 50 packets\n")

    # 1. Build benchmark dataset
    log("[1/3] Generating benchmark dataset (seed=42)...")
    dataset = build_benchmark_dataset(samples_per_type=50, sample_size=2048, seed=42)
    sensor_samples = [s for s in dataset if s["type"] == "sensor_gaussian"]
    log(f"Extracted {len(sensor_samples)} sensor_gaussian payloads.\n")

    # 2. Standalone Arithmetic Coding Transmission Test (seed=42)
    log("[2/3] Evaluating Standalone Arithmetic Coding over Channel (seed=42)...")
    ldpc = LDPCCodec(n=128, k=64, dv=3, seed=42, max_iter=25)
    channel = WiredChannel(snr_db=7.0, impulse_prob=5e-4, impulse_magnitude=3.0, seed=42)

    total_blocks_trial1 = 0
    msg_block_errors_trial1 = 0
    packet_successes_trial1 = 0
    total_bits_trial1 = 0
    bit_errors_trial1 = 0

    log(f"{'Packet':<8}{'CompBytes':<12}{'CR':<8}{'Blocks':<10}{'BlkErrs':<10}{'BitErrs':<10}{'Lossless':<10}")
    log("-" * 68)

    for i, s in enumerate(sensor_samples):
        raw_bytes = s["data"]
        comp_bytes, meta = arithmetic_encode(raw_bytes)
        cr = len(raw_bytes) / len(comp_bytes)
        bits_str = "".join(f"{b:08b}" for b in comp_bytes)
        
        ldpc_bits, pad = ldpc.encode_bitstream(bits_str)
        rx_signal, rx_llr = channel.transmit(ldpc_bits)

        num_blocks = len(ldpc_bits) // ldpc.n
        total_blocks_trial1 += num_blocks
        total_bits_trial1 += num_blocks * ldpc.k

        reconstructed_bits = []
        p_block_errors = 0
        p_bit_errors = 0

        for b_idx in range(num_blocks):
            block_llr = rx_llr[b_idx * ldpc.n : (b_idx + 1) * ldpc.n]
            true_m = ldpc_bits[b_idx * ldpc.n : b_idx * ldpc.n + ldpc.k]

            dec_codeword, valid, iters = ldpc.decode_llr_minsum(block_llr)
            dec_m = dec_codeword[:ldpc.k]

            diffs = int(np.sum(dec_m != true_m))
            p_bit_errors += diffs
            if diffs > 0:
                p_block_errors += 1

            reconstructed_bits.extend(dec_m)

        msg_block_errors_trial1 += p_block_errors
        bit_errors_trial1 += p_bit_errors

        if pad > 0:
            reconstructed_bits = reconstructed_bits[:-pad]

        rec_bits_str = "".join(str(b) for b in reconstructed_bits)
        rec_bytes = bytearray()
        for bi in range(0, len(rec_bits_str), 8):
            c = rec_bits_str[bi:bi+8]
            if len(c) == 8:
                rec_bytes.append(int(c, 2))
        rec_bytes = bytes(rec_bytes)

        lossless = False
        try:
            dec_src = arithmetic_decode(rec_bytes, meta)
            lossless = (dec_src == raw_bytes)
        except Exception:
            lossless = False

        if lossless:
            packet_successes_trial1 += 1

        log(f"{i:<8}{len(comp_bytes):<12}{cr:<8.2f}{num_blocks:<10}{p_block_errors:<10}{p_bit_errors:<10}{str(lossless):<10}")

    bler1 = msg_block_errors_trial1 / total_blocks_trial1
    pdr1 = packet_successes_trial1 / len(sensor_samples)
    ber1 = bit_errors_trial1 / total_bits_trial1
    mean_blocks1 = total_blocks_trial1 / len(sensor_samples)
    pred_pdr1 = (1.0 - bler1) ** mean_blocks1

    log("\n" + "=" * 68)
    log("STANDALONE ARITHMETIC CODING RESULTS SUMMARY (Trial 1):")
    log(f"  - Total Transmitted LDPC Blocks: {total_blocks_trial1}")
    log(f"  - Mean Blocks per Payload:       {mean_blocks1:.2f} (approx {round(mean_blocks1)} blocks)")
    log(f"  - Message Block Errors:          {msg_block_errors_trial1} / {total_blocks_trial1}")
    log(f"  - Measured Message BLER:         {bler1:.5e} ({bler1*100:.3f}%)")
    log(f"  - Post-LDPC Bit Errors:          {bit_errors_trial1} / {total_bits_trial1} (BER = {ber1:.5e})")
    log(f"  - Independent Model Prediction:  (1 - {bler1:.5f})^{round(mean_blocks1)} = {pred_pdr1*100:.2f}%")
    log(f"  - Empirical Packet Delivery:     {packet_successes_trial1}/50 ({pdr1*100:.1f}%)")
    log("=" * 68 + "\n")

    # 3. Replicated Trial with seed=123 (Different noise realization)
    log("[3/3] Running Replicated Trial with seed=123...")
    channel2 = WiredChannel(snr_db=7.0, impulse_prob=5e-4, impulse_magnitude=3.0, seed=123)

    total_blocks_trial2 = 0
    msg_block_errors_trial2 = 0
    packet_successes_trial2 = 0
    total_bits_trial2 = 0
    bit_errors_trial2 = 0

    for i, s in enumerate(sensor_samples):
        raw_bytes = s["data"]
        comp_bytes, meta = arithmetic_encode(raw_bytes)
        bits_str = "".join(f"{b:08b}" for b in comp_bytes)
        ldpc_bits, pad = ldpc.encode_bitstream(bits_str)
        rx_signal, rx_llr = channel2.transmit(ldpc_bits)

        num_blocks = len(ldpc_bits) // ldpc.n
        total_blocks_trial2 += num_blocks
        total_bits_trial2 += num_blocks * ldpc.k

        reconstructed_bits = []
        p_block_errors = 0
        p_bit_errors = 0

        for b_idx in range(num_blocks):
            block_llr = rx_llr[b_idx * ldpc.n : (b_idx + 1) * ldpc.n]
            true_m = ldpc_bits[b_idx * ldpc.n : b_idx * ldpc.n + ldpc.k]

            dec_codeword, valid, iters = ldpc.decode_llr_minsum(block_llr)
            dec_m = dec_codeword[:ldpc.k]

            diffs = int(np.sum(dec_m != true_m))
            p_bit_errors += diffs
            if diffs > 0:
                p_block_errors += 1

            reconstructed_bits.extend(dec_m)

        msg_block_errors_trial2 += p_block_errors
        bit_errors_trial2 += p_bit_errors

        if pad > 0:
            reconstructed_bits = reconstructed_bits[:-pad]

        rec_bits_str = "".join(str(b) for b in reconstructed_bits)
        rec_bytes = bytearray()
        for bi in range(0, len(rec_bits_str), 8):
            c = rec_bits_str[bi:bi+8]
            if len(c) == 8:
                rec_bytes.append(int(c, 2))
        rec_bytes = bytes(rec_bytes)

        lossless = False
        try:
            dec_src = arithmetic_decode(rec_bytes, meta)
            lossless = (dec_src == raw_bytes)
        except Exception:
            lossless = False

        if lossless:
            packet_successes_trial2 += 1

    bler2 = msg_block_errors_trial2 / total_blocks_trial2
    pdr2 = packet_successes_trial2 / len(sensor_samples)
    ber2 = bit_errors_trial2 / total_bits_trial2
    mean_blocks2 = total_blocks_trial2 / len(sensor_samples)
    pred_pdr2 = (1.0 - bler2) ** mean_blocks2

    log("=" * 68)
    log("REPLICATED ARITHMETIC CODING RESULTS SUMMARY (Trial 2):")
    log(f"  - Total Transmitted LDPC Blocks: {total_blocks_trial2}")
    log(f"  - Mean Blocks per Payload:       {mean_blocks2:.2f}")
    log(f"  - Message Block Errors:          {msg_block_errors_trial2} / {total_blocks_trial2}")
    log(f"  - Measured Message BLER:         {bler2:.5e} ({bler2*100:.3f}%)")
    log(f"  - Post-LDPC Bit Errors:          {bit_errors_trial2} / {total_bits_trial2} (BER = {ber2:.5e})")
    log(f"  - Independent Model Prediction:  (1 - {bler2:.5f})^{round(mean_blocks2)} = {pred_pdr2*100:.2f}%")
    log(f"  - Empirical Packet Delivery:     {packet_successes_trial2}/50 ({pdr2*100:.1f}%)")
    log("=" * 68)

    # Save to file
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\n[DONE] Full verification log saved to: {log_file}")


if __name__ == "__main__":
    run_sensor_verification()
