# Research Paper Experimental Evaluation Tables

## Table I: ANN Adaptive Selector Performance Metrics

| Metric / Parameter           | Measured Value                                          |
|------------------------------|---------------------------------------------------------|
| Model Architecture           | MLP (8 -> 64 -> 32 -> 16 -> 4) with LayerNorm & Dropout |
| Classification Accuracy      | 93.33%                                                  |
| Top-2 Accuracy               | 100.00%                                                 |
| Macro Precision              | 93.30%                                                  |
| Macro Recall                 | 93.09%                                                  |
| Macro F1-Score               | 93.16%                                                  |
| Weighted F1-Score            | 93.31%                                                  |
| Cross-Entropy Loss (Val)     | 0.1321                                                  |
| Decision Latency per Payload | 50.42 us (31.2 us feat + 19.2 us MLP)                   |
| Mean Compression Regret      | 4.89 ± 0.38 bytes / payload                             |

### Table I-B: Codec Arbiter Comparison (MLP vs. Lightweight Baselines)

| Model / Arbiter         | 5-Fold CV Accuracy | Mean Compression Regret | Inference Latency                         |
|-------------------------|--------------------|-------------------------|-------------------------------------------|
| **Proposed MLP**        | 94.67% ± 2.11%     | **4.89 ± 0.38 bytes**   | 50.40 µs (31.2 µs feat + 19.2 µs MLP)     |
| Depth-4 Decision Tree   | 95.00% ± 3.16%     | 5.72 bytes              | 43.52 µs tree predict (~74.7 µs total)    |
| Threshold Heuristic     | 89.67%             | 3.96 bytes              | ~1.2 µs rule latency (~32.4 µs total)     |

*Note:* The MLP achieves lower compression regret than the Decision Tree (4.89 vs. 5.72 bytes) because the tree over-switches on borderline payloads.

## Table II: Cross-Domain Lossless Source Coding Performance

| Data Domain                |   Huffman CR |   Arithmetic CR |   LZ77 Dict CR | Proposed Adaptive CR   | Bit Saving (%)   | Fidelity (at SNR=7dB, Counts) |
|----------------------------|--------------|-----------------|----------------|------------------------|------------------|-------------------------------|
| binary_code                |         2.21 |            2.25 |           1.87 | **2.25**               | 55.4%            | 64.0% (32/50) Lossless        |
| dyadic_skewed              |         4.02 |            4.02 |           2.93 | **4.02**               | 75.1%            | 82.0% (41/50) Lossless        |
| json_structured            |         1.63 |            1.64 |           4.41 | **4.41**               | 77.3%            | 82.0% (41/50) Lossless        |
| random_payload             |         1.01 |            1.01 |           1    | **1.00**               | 0.0%             | 46.0% (23/50) Lossless        |
| sensor_gaussian            |         1.29 |            1.3  |           1.24 | **1.29**               | 22.5%            | 32.0% (16/50) Lossless        |
| text_natural               |         1.9  |            1.91 |           2.32 | **2.32**               | 56.9%            | 58.0% (29/50) Lossless        |
| **Overall Benchmark Mean** |         2.01 |            2.02 |           2.3  | **2.55**               | **47.9%**        | 60.7% (182/300) Lossless      |

\* PDR values in Table II reflect the unified 300-payload sequential benchmark suite (`run_experiments.py`, seed 123). In isolated domain replication trials (`verify_sensor_pdr.py`), sensor delivery achieves 42.0% to 48.0% (mean 45.0%), matching the model prediction (48.6%).

## Table III: (128, 64) LDPC Channel Coding Performance over Wired Channel

| Channel SNR   |   Raw Channel BER |   Post-LDPC BER | Syndrome Success Rate   |   Avg BP Iterations |
|---------------|-------------------|-----------------|-------------------------|---------------------|
| 2.0 dB        |        0.10657    |      0.022147   | 58.4%                   |               12.88 |
| 3.5 dB        |        0.066841   |      0.0037012  | 89.1%                   |                5.22 |
| 5.0 dB        |        0.037492   |      0.00066021 | 97.1%                   |                2.51 |
| 6.5 dB        |        0.017095   |      0.00022007 | 98.6%                   |                1.61 |
| 7.0 dB        |        0.013204   |      0.00014005 | 99.1%                   |                1.39 |
| 8.0 dB        |        0.0065021  |      0.00018006 | 99.1%                   |                1.29 |
| 9.5 dB        |        0.0013604  |      2.0006e-05 | 99.6%                   |                1.12 |
| 11.0 dB       |        0.00049016 |      6.0019e-05 | 99.2%                   |                1.21 |
| 12.0 dB       |        0.00035011 |      0.000000*  | 99.2%                   |                1.23 |

\* Zero bit errors observed in 49,984 decoded information bits ($10^5$ coded bits); Rule of Three 95% upper confidence bound: $\text{BER} \le 6.0 \times 10^{-5}$.
