"""
End-to-End System Integration:
ANN-Based Adaptive Source Coding + LDPC Channel Coding Pipeline.
Integrates:
1. Feature extraction
2. ANN inference for codec selection (Huffman vs Arithmetic vs LZW vs Raw)
3. Source Encoding with optimal selected algorithm
4. Systematic (N,K) LDPC Channel Encoding
5. Transmission over wired channel (AWGN + impulse EMI noise)
6. Iterative Min-Sum Belief Propagation LDPC Channel Decoding with syndrome checking
7. Source Decoding using matched codec
8. Full metric collection (BER, FER, Compression Ratio, Throughput, Lossless Reconstruction Verification)
"""

import time
import numpy as np
import torch
from typing import Dict, Any, Tuple

from src.ann.model import extract_features, AdaptiveCodingClassifier
from src.source_coding.codecs import (
    huffman_encode, huffman_decode,
    arithmetic_encode, arithmetic_decode,
    dictionary_lz77_encode, dictionary_lz77_decode,
    raw_encode, raw_decode
)
from src.channel_coding.ldpc import LDPCCodec
from src.channel.channel import WiredChannel

CODEC_DISPATCH = {
    0: ("Huffman", huffman_encode, huffman_decode),
    1: ("Arithmetic", arithmetic_encode, arithmetic_decode),
    2: ("Dictionary_LZ77", dictionary_lz77_encode, dictionary_lz77_decode),
    3: ("Raw_Passthrough", raw_encode, raw_decode)
}


class AdaptiveTransmissionPipeline:
    def __init__(self, ann_model: AdaptiveCodingClassifier, ldpc_codec: LDPCCodec, channel: WiredChannel):
        self.model = ann_model
        self.ldpc = ldpc_codec
        self.channel = channel
        self.model.eval()

    def run_pipeline(self, source_bytes: bytes, forced_codec_id: int = None) -> Dict[str, Any]:
        """
        Runs the complete transmitter -> channel -> receiver pipeline.
        If forced_codec_id is None, uses the ANN to adaptively select the codec.
        """
        t0 = time.perf_counter()
        orig_len = len(source_bytes)
        orig_bits = orig_len * 8

        # --- 1. ANN Feature Extraction & Adaptive Decision ---
        t_feat0 = time.perf_counter()
        features = extract_features(source_bytes)
        t_feat = time.perf_counter() - t_feat0

        t_ann0 = time.perf_counter()
        with torch.no_grad():
            feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits = self.model(feat_tensor)
            probs = torch.softmax(logits, dim=1).numpy().flatten()
            predicted_codec_id = int(np.argmax(probs))
        t_ann = time.perf_counter() - t_ann0

        codec_id = predicted_codec_id if forced_codec_id is None else forced_codec_id
        codec_name, enc_fn, dec_fn = CODEC_DISPATCH[codec_id]

        # --- 2. Source Encoding ---
        t_src_enc0 = time.perf_counter()
        source_compressed, meta = enc_fn(source_bytes)
        t_src_enc = time.perf_counter() - t_src_enc0

        compressed_len = len(source_compressed)
        comp_ratio = orig_len / max(1, compressed_len)
        bit_saving_pct = (1.0 - compressed_len / max(1, orig_len)) * 100.0

        # Convert compressed payload to bitstring
        bits_str = "".join(f"{b:08b}" for b in source_compressed)

        # --- 3. LDPC Channel Encoding ---
        t_ldpc_enc0 = time.perf_counter()
        ldpc_codeword_bits, pad = self.ldpc.encode_bitstream(bits_str)
        t_ldpc_enc = time.perf_counter() - t_ldpc_enc0

        # --- 4. Wired Channel Transmission ---
        t_chan0 = time.perf_counter()
        rx_signal, rx_llr = self.channel.transmit(ldpc_codeword_bits)
        t_chan = time.perf_counter() - t_chan0

        # Raw channel bit errors (hard-decision threshold at 0)
        hard_rx_bits = np.where(rx_llr < 0, 1, 0).astype(np.uint8)
        channel_bit_errors = int(np.sum(hard_rx_bits != ldpc_codeword_bits))
        raw_ber = channel_bit_errors / max(1, len(ldpc_codeword_bits))

        # --- 5. LDPC Belief Propagation Decoding ---
        t_ldpc_dec0 = time.perf_counter()
        reconstructed_bits_str, ldpc_stats = self.ldpc.decode_bitstream(rx_llr, pad)
        t_ldpc_dec = time.perf_counter() - t_ldpc_dec0

        # Verify decoded bits vs compressed source bits
        residual_bit_errors = sum(b1 != b2 for b1, b2 in zip(bits_str, reconstructed_bits_str))
        post_ldpc_ber = residual_bit_errors / max(1, len(bits_str))

        # Reconstruct compressed payload bytes
        reconstructed_payload = bytearray()
        for i in range(0, len(reconstructed_bits_str), 8):
            chunk = reconstructed_bits_str[i:i+8]
            if len(chunk) == 8:
                reconstructed_payload.append(int(chunk, 2))
        reconstructed_payload = bytes(reconstructed_payload)

        # --- 6. Source Decoding ---
        t_src_dec0 = time.perf_counter()
        decode_success = True
        try:
            reconstructed_source = dec_fn(reconstructed_payload, meta)
            lossless_verified = (reconstructed_source == source_bytes)
        except Exception as e:
            decode_success = False
            lossless_verified = False
            reconstructed_source = b""
        t_src_dec = time.perf_counter() - t_src_dec0

        t_total = time.perf_counter() - t0

        return {
            "codec_selected": codec_name,
            "codec_id": codec_id,
            "ann_probabilities": probs,
            "orig_bytes": orig_len,
            "compressed_bytes": compressed_len,
            "compression_ratio": comp_ratio,
            "bit_saving_pct": bit_saving_pct,
            "ldpc_transmitted_bits": len(ldpc_codeword_bits),
            "raw_channel_ber": raw_ber,
            "post_ldpc_ber": post_ldpc_ber,
            "frame_success_rate": ldpc_stats["frame_success_rate"],
            "avg_ldpc_iterations": ldpc_stats["avg_iterations"],
            "lossless_verified": lossless_verified,
            "decode_success": decode_success,
            "timings_ms": {
                "ann_feature": t_feat * 1000.0,
                "ann_inference": t_ann * 1000.0,
                "source_enc": t_src_enc * 1000.0,
                "ldpc_enc": t_ldpc_enc * 1000.0,
                "channel_sim": t_chan * 1000.0,
                "ldpc_dec": t_ldpc_dec * 1000.0,
                "source_dec": t_src_dec * 1000.0,
                "total_pipeline": t_total * 1000.0,
            }
        }
