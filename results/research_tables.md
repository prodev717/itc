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
| Cross-Entropy Loss (Val)     | 0.1256                                                  |
| Decision Latency per Payload | 81.87 us                                                |

## Table II: Cross-Domain Lossless Source Coding Performance

| Data Domain                |   Huffman CR |   Arithmetic CR |   LZ77 Dict CR | Proposed Adaptive CR   | Bit Saving (%)   | Fidelity (at SNR=7dB)   |
|----------------------------|--------------|-----------------|----------------|------------------------|------------------|-------------------------|
| binary_code                |         2.21 |            2.25 |           1.87 | **2.25**               | 55.4%            | 64.0% Lossless          |
| dyadic_skewed              |         4.02 |            4.02 |           2.93 | **4.02**               | 75.1%            | 82.0% Lossless          |
| json_structured            |         1.63 |            1.64 |           4.41 | **4.41**               | 77.3%            | 82.0% Lossless          |
| random_payload             |         1.01 |            1.01 |           1    | **1.00**               | 0.0%             | 46.0% Lossless          |
| sensor_gaussian            |         1.29 |            1.3  |           1.24 | **1.29**               | 22.5%            | 32.0% Lossless          |
| text_natural               |         1.9  |            1.91 |           2.32 | **2.32**               | 56.9%            | 58.0% Lossless          |
| **Overall Benchmark Mean** |         2.01 |            2.02 |           2.3  | **2.55**               | **47.9%**        | 60.7% Lossless          |

## Table III: (128, 64) LDPC Channel Coding Performance over Wired Channel

| Channel SNR   |   Raw Channel BER |   Post-LDPC BER | Syndrome Success Rate   |   Avg BP Iterations |
|---------------|-------------------|-----------------|-------------------------|---------------------|
| 2.0 dB        |        0.038304   |      0.00067828 | 97.7%                   |                2.38 |
| 3.5 dB        |        0.017833   |      0.00019221 | 98.8%                   |                1.53 |
| 5.0 dB        |        0.0061013  |      0.00012247 | 98.9%                   |                1.34 |
| 6.5 dB        |        0.0017124  |      0.00010905 | 98.5%                   |                1.38 |
| 8.0 dB        |        0.00037676 |      5.4627e-05 | 99.3%                   |                1.19 |
| 9.5 dB        |        0.00025605 |      4.0638e-05 | 99.1%                   |                1.25 |
