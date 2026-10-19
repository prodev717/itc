"""
Low-Density Parity-Check (LDPC) Channel Coding Implementation.
Features:
- Systematic (N, K) LDPC parity-check matrix H generation and generator matrix G construction over GF(2).
- Systematic Encoder: x = [m | p] where p satisfies H * x^T = 0 (mod 2).
- Iterative Belief Propagation / Log-Likelihood Ratio (LLR) Sum-Product Decoder and Min-Sum Decoder.
- Syndrome verification: valid codeword if H * c^T == 0 (mod 2).
- Flexible rate R = K/N (e.g. Rate 1/2 with N=128, K=64 or N=64, K=32).
- Fast block processing for arbitrary bitstreams with zero-padding.
"""

import math
from typing import List, Optional, Tuple
import numpy as np


class LDPCCodec:
    """
    Constructs a systematic (N, K) LDPC code using Gallager-style or PEG-inspired sparse parity check matrix.
    H = [P | I_(N-K)] in standard systematic form, allowing direct generator matrix G = [I_K | P^T].
    """
    def __init__(self, n: int = 128, k: int = 64, dv: int = 3, seed: int = 42, max_iter: int = 25):
        self.n = n
        self.k = k
        self.m = n - k  # number of parity check equations
        self.dv = dv
        self.max_iter = max_iter
        self.seed = seed
        self.H, self.G = self._generate_matrices()
        
        # Precompute sparse check and variable node connectivity graph
        self.check_to_vars = [np.where(self.H[c, :] == 1)[0] for c in range(self.m)]
        self.var_to_checks = [np.where(self.H[:, v] == 1)[0] for v in range(self.n)]

    def _generate_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.seed)
        
        # We need H = [P | I_m] of dimension (m, n) where P is (m, k).
        # To ensure low density, each column of P has approximately `dv` ones.
        while True:
            P = np.zeros((self.m, self.k), dtype=np.uint8)
            for col in range(self.k):
                rows = rng.choice(self.m, size=min(self.dv, self.m), replace=False)
                P[rows, col] = 1
                
            # Avoid any all-zero rows in P
            for row in range(self.m):
                if np.sum(P[row, :]) == 0:
                    P[row, rng.choice(self.k)] = 1
                    
            I_m = np.eye(self.m, dtype=np.uint8)
            H = np.hstack([P, I_m])
            
            # Since H = [P | I_m], G = [I_k | P.T] satisfies:
            # G * H^T = [I_k | P.T] * [P^T \\ I_m] = P^T + P^T = 0 (mod 2)
            G = np.hstack([np.eye(self.k, dtype=np.uint8), P.T])
            
            # Verify orthogonality
            check = np.dot(G, H.T) % 2
            if not np.any(check):
                return H, G

    def encode_block(self, message_bits: np.ndarray) -> np.ndarray:
        """
        Encode a single block of length K bits into N bits.
        Since G = [I_K | P^T], the first K bits are exactly message_bits!
        """
        if len(message_bits) != self.k:
            raise ValueError(f"Expected {self.k} bits, got {len(message_bits)}")
        codeword = np.dot(message_bits, self.G) % 2
        return codeword.astype(np.uint8)

    def check_syndrome(self, codeword: np.ndarray) -> bool:
        """Returns True if codeword is a valid codeword (syndrome is all zeros)."""
        s = np.dot(self.H, codeword) % 2
        return not np.any(s)

    def decode_llr_minsum(self, llr_channel: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        """
        Normalized Min-Sum iterative Belief Propagation decoder.
        llr_channel: log-likelihood ratios L(y) = 2*y / sigma^2, where positive means 0 is more likely.
        Returns: (decoded_codeword_bits, syndrome_valid, iterations_taken)
        """
        # q_v_c: messages from variable nodes to check nodes
        # r_c_v: messages from check nodes to variable nodes
        llr = np.copy(llr_channel)
        
        # Initialize check messages to 0
        # Fast dictionary/sparse representation
        r = {}
        for c in range(self.m):
            for v in self.check_to_vars[c]:
                r[(c, v)] = 0.0
                
        # Variable messages initialized to channel LLR
        q = {}
        for v in range(self.n):
            for c in self.var_to_checks[v]:
                q[(v, c)] = llr[v]

        alpha = 0.85  # Normalization factor for Min-Sum approximation

        for iteration in range(1, self.max_iter + 1):
            # Check node update
            for c in range(self.m):
                vars_in_c = self.check_to_vars[c]
                q_vals = [q[(v, c)] for v in vars_in_c]
                signs = [np.sign(val) if val != 0 else 1.0 for val in q_vals]
                magnitudes = [abs(val) for val in q_vals]
                
                prod_sign = np.prod(signs)
                
                for idx, v in enumerate(vars_in_c):
                    sign_ex = prod_sign * signs[idx]
                    # min magnitude excluding current variable
                    min_mag = min([magnitudes[j] for j in range(len(vars_in_c)) if j != idx], default=1e6)
                    r[(c, v)] = alpha * sign_ex * min_mag

            # Variable node update & marginal computation
            decoded_bits = np.zeros(self.n, dtype=np.uint8)
            for v in range(self.n):
                total_llr = llr[v] + sum(r[(c, v)] for c in self.var_to_checks[v])
                decoded_bits[v] = 1 if total_llr < 0 else 0
                
                for c in self.var_to_checks[v]:
                    q[(v, c)] = llr[v] + sum(r[(c_prime, v)] for c_prime in self.var_to_checks[v] if c_prime != c)

            # Check parity syndrome
            if self.check_syndrome(decoded_bits):
                return decoded_bits, True, iteration

        # Return best effort after max_iter
        return decoded_bits, self.check_syndrome(decoded_bits), self.max_iter

    def encode_bitstream(self, bitstring: str) -> Tuple[np.ndarray, int]:
        """Encodes arbitrary length binary string by chunking into blocks of K bits with zero-padding."""
        pad = (self.k - (len(bitstring) % self.k)) % self.k
        padded = bitstring + ("0" * pad)
        
        num_blocks = len(padded) // self.k
        encoded_blocks = []
        for i in range(num_blocks):
            block = np.array([int(b) for b in padded[i * self.k : (i + 1) * self.k]], dtype=np.uint8)
            encoded_blocks.append(self.encode_block(block))
            
        return np.concatenate(encoded_blocks), pad

    def decode_bitstream(self, received_llr: np.ndarray, pad: int) -> Tuple[str, dict]:
        """
        Decodes concatenated N-bit blocks using Min-Sum BP decoder.
        Returns: (reconstructed_bitstring, stats)
        """
        num_blocks = len(received_llr) // self.n
        decoded_data_bits = []
        
        corrected_blocks = 0
        valid_syndrome_blocks = 0
        total_iterations = 0

        for i in range(num_blocks):
            block_llr = received_llr[i * self.n : (i + 1) * self.n]
            dec_codeword, valid, iters = self.decode_llr_minsum(block_llr)
            total_iterations += iters
            if valid:
                valid_syndrome_blocks += 1
            # Systematic portion is first K bits
            decoded_data_bits.extend(dec_codeword[:self.k])

        bitstring = "".join(str(b) for b in decoded_data_bits)
        if pad > 0:
            bitstring = bitstring[:-pad]
            
        stats = {
            "num_blocks": num_blocks,
            "valid_syndromes": valid_syndrome_blocks,
            "avg_iterations": total_iterations / max(1, num_blocks),
            "frame_success_rate": valid_syndrome_blocks / max(1, num_blocks)
        }
        return bitstring, stats
