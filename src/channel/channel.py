"""
Realistic Wired Communication Channel Simulation:
- Differential transmission / Baseband BPSK (+1.0 / -1.0)
- Additive White Gaussian Noise (AWGN) parameterized by SNR (dB)
- Impulse noise / burst bit flips simulating electromagnetic interference (EMI) or power glitches
- Conversion to Log-Likelihood Ratios (LLR) for soft-decision LDPC decoding:
    LLR = 2 * y / sigma^2
"""

import numpy as np
from typing import Tuple, Optional


class WiredChannel:
    def __init__(self, snr_db: float = 6.0, impulse_prob: float = 0.001, impulse_magnitude: float = 3.0, seed: int = 42):
        self.snr_db = snr_db
        self.impulse_prob = impulse_prob
        self.impulse_magnitude = impulse_magnitude
        self.rng = np.random.RandomState(seed)

    def transmit(self, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transmits binary bits (0 or 1) through the simulated wired channel.
        Mapping: bit 0 -> +1.0, bit 1 -> -1.0.
        Returns: (received_signal, llr)
        """
        # BPSK modulation: 0 -> +1.0, 1 -> -1.0
        modulated = np.where(bits == 0, 1.0, -1.0)
        
        # AWGN calculation
        # SNR = 10^(SNR_dB / 10) = Es / (2 * sigma^2), where Es = 1.0
        snr_linear = 10.0 ** (self.snr_db / 10.0)
        sigma = np.sqrt(1.0 / (2.0 * snr_linear))
        
        noise = self.rng.normal(0.0, sigma, size=modulated.shape)
        
        # Impulse / burst noise model for wired channel
        if self.impulse_prob > 0.0:
            burst_mask = self.rng.random(size=modulated.shape) < self.impulse_prob
            impulses = self.rng.choice([-1.0, 1.0], size=modulated.shape) * self.impulse_magnitude
            noise += burst_mask * impulses
            
        received_signal = modulated + noise
        
        # Soft-decision LLR: L(y) = 2 * y / sigma^2
        # (Positive LLR => 0 is more probable, Negative LLR => 1 is more probable)
        llr = 2.0 * received_signal / (sigma ** 2)
        
        return received_signal, llr
