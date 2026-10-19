# ANN-Based Adaptive Source Coding with LDPC Channel Coding for Lossless Wired Communication

An end-to-end research project combining **Neural Network Adaptive Source Coding** with **Systematic (128, 64) Low-Density Parity-Check (LDPC) Channel Coding** for high-efficiency, noise-resilient wired data transmission.

---

## 1. Project Overview

In traditional communication systems, source coding and channel coding are configured statically regardless of input data distributions:
- **Source Coding**: Static Huffman or static Arithmetic coding cannot adapt to changes between memoryless discrete alphabets, structured repetitive tokens, and incompressible encrypted data. Compressing uniform data with static prefix codes often results in bit expansion.
- **Channel Coding**: Classical Hamming(7,4) block codes are restricted to correcting single-bit errors ($d_{\min}=3$). In the presence of electromagnetic interference (EMI) or burst noise, Hamming codes suffer catastrophic failure.

### Proposed Architecture:
1. **Multi-Domain Lossless Source Coding**:
   - **Canonical Huffman Coding**: Optimal prefix coding for discrete dyadic distributions.
   - **Integer Arithmetic Coding**: Near-entropy coding for continuous stationary distributions.
   - **LZ77 Sliding-Window Dictionary Coding**: Length-distance sequence matching for repetitive text and structured telemetry.
   - **Raw Pass-Through**: Identity transmission preventing coding expansion on high-entropy/random payloads.
2. **ANN Adaptive Codec Selector**:
   - Extracts an 8-dimensional feature vector ($O(N)$ empirical moments: Shannon entropy, variance, zero-byte density, autocorrelation, etc.).
   - Multi-Layer Perceptron (LayerNorm + Dropout) classifies payloads in **$56.3\ \mu\text{s}$**, achieving **$93.33\%$** test accuracy and **$100\%$** Top-2 accuracy.
3. **Channel Coding Replacement (LDPC)**:
   - Systematic $(128, 64)$ rate $R=1/2$ LDPC code ($H = [P \mid I_{N-K}]$, $G = [I_K \mid P^T]$).
   - Iterative Min-Sum Belief Propagation (BP) decoder over bipartite Tanner graphs with real-time syndrome verification ($s = H \cdot \hat{c}^T \equiv 0 \pmod 2$).
4. **Wired Channel Simulation**:
   - Differential BPSK over Additive White Gaussian Noise (AWGN) channel corrupted by impulse EMI burst errors.

---

## 2. Directory Structure

```
├── experiments/
│   └── run_experiments.py          # Master benchmark, SNR sweep, and plotting suite
├── results/
│   ├── fig1_ann_confusion_matrix.png
│   ├── fig2_compression_ratio_by_domain.png
│   ├── fig3_ldpc_ber_waterfall.png
│   └── research_tables.md           # Publication tables (LaTeX/Markdown)
├── src/
│   ├── ann/
│   │   ├── model.py                 # Feature extractor & AdaptiveCodingClassifier (MLP)
│   │   └── trainer.py               # Ground truth oracle generator & dataset loader
│   ├── channel/
│   │   └── channel.py               # Wired physical channel with AWGN + EMI noise
│   ├── channel_coding/
│   │   └── ldpc.py                  # Systematic LDPC encoder & Min-Sum BP decoder
│   ├── dataset/
│   │   └── generator.py             # Multi-domain dataset synthesis across 6 classes
│   ├── pipeline.py                  # End-to-end adaptive transmission pipeline
│   └── source_coding/
│       └── codecs.py                # Huffman, Arithmetic, LZ77, and Raw codecs
├── main.py                          # Interactive demo & benchmark CLI
├── pyproject.toml                   # Project dependencies and packaging
└── README.md
```

---

## 3. Quick Start & Execution

### Running the Live Interactive Demo:
```bash
uv run python main.py
```

### Running the Full Research Benchmark Suite:
To execute the multi-modal dataset synthesis, ANN cross-validation, baseline comparisons, and channel SNR sweep:
```bash
uv run python experiments/run_experiments.py
```
This generates publication figures and tables in `results/`.

---

## 4. Key Scientific Results

### Table I: ANN Adaptive Selector Metrics
| Metric | Measured Value |
| :--- | :--- |
| **Accuracy** | 93.33% |
| **Top-2 Accuracy** | 100.00% |
| **Macro F1-Score** | 93.16% |
| **Weighted F1-Score** | 93.31% |
| **Validation Loss** | 0.1256 |
| **Inference Latency** | **$56.3\ \mu\text{s}$ / payload** |

### Table II: Compression Ratio Across Diverse Domains
| Domain | Static Huffman | Static Arithmetic | Static LZ77 Dict | **Proposed Adaptive** | Bit Saving (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dyadic Skewed** | **4.02x** | **4.02x** | 2.93x | **4.02x** | 75.1% |
| **Structured JSON** | 1.63x | 1.64x | **4.41x** | **4.41x** | 77.3% |
| **Natural English** | 1.90x | 1.91x | **2.32x** | **2.32x** | 56.9% |
| **Binary Executable** | 2.21x | **2.25x** | 1.87x | **2.25x** | 55.4% |
| **Sensor Telemetry** | 1.29x | **1.30x** | 1.24x | **1.29x** | 22.5% |
| **Random Payload** | 1.01x | 1.01x | 1.00x | **1.00x** | 0.0% |
| **Overall Mean** | **2.01x** | **2.02x** | **2.30x** | **2.55x** | **47.9%** |

### Table III: (128, 64) LDPC Channel Coding Gain
| SNR (dB) | Raw Channel BER | Post-LDPC Residual BER | Syndrome Verification Rate |
| :---: | :---: | :---: | :---: |
| **2.0 dB** | $3.83 \times 10^{-2}$ | $6.78 \times 10^{-4}$ | 97.7% |
| **5.0 dB** | $6.10 \times 10^{-3}$ | $1.22 \times 10^{-4}$ | 98.9% |
| **8.0 dB** | $3.77 \times 10^{-4}$ | **$0.00$** | 99.3% |
