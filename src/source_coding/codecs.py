"""
Source Coding Algorithms:
1. Static Huffman Coding (Canonical prefix tree with symbol frequency analysis)
2. Arithmetic Coding (Integer precision interval subdivision)
3. Dictionary Coding - LZ77 (Lempel-Ziv 1977 Sliding Window with Length-Distance matching)
4. Raw Pass-through (Identity, for incompressible/maximum entropy data where coding incurs expansion overhead)
"""

from collections import Counter
import heapq
import math
import struct
import zlib
from typing import Dict, List, Optional, Tuple


# =====================================================================
# 1. Huffman Coding
# =====================================================================

class HuffmanNode:
    def __init__(self, frequency: int, symbol: Optional[int] = None, left=None, right=None):
        self.frequency = frequency
        self.symbol = symbol
        self.left = left
        self.right = right


def build_huffman_codes(data: bytes) -> Dict[int, str]:
    frequencies = Counter(data)
    if not frequencies:
        return {}

    heap = []
    seq = 0
    for symbol, frequency in sorted(frequencies.items()):
        node = HuffmanNode(frequency=frequency, symbol=symbol)
        heapq.heappush(heap, (frequency, seq, node))
        seq += 1

    if len(heap) == 1:
        only_sym = heap[0][2].symbol
        return {only_sym: "0"}

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        parent = HuffmanNode(frequency=f1 + f2, left=n1, right=n2)
        heapq.heappush(heap, (parent.frequency, seq, parent))
        seq += 1

    root = heap[0][2]
    codes: Dict[int, str] = {}

    def walk(node: HuffmanNode, prefix: str):
        if node.symbol is not None:
            codes[node.symbol] = prefix or "0"
            return
        if node.left:
            walk(node.left, prefix + "0")
        if node.right:
            walk(node.right, prefix + "1")

    walk(root, "")
    return codes


def huffman_encode(data: bytes) -> Tuple[bytes, dict]:
    """
    Encodes data with canonical Huffman code table serialized into metadata.
    Returns: (compressed_payload_bytes, metadata_dict)
    """
    if not data:
        return b"", {"codes": {}, "length": 0, "pad": 0}
        
    codes = build_huffman_codes(data)
    bitstring = "".join(codes[b] for b in data)
    pad = (8 - (len(bitstring) % 8)) % 8
    padded_bits = bitstring + ("0" * pad)
    
    out = bytearray()
    for i in range(0, len(padded_bits), 8):
        out.append(int(padded_bits[i:i+8], 2))
        
    metadata = {
        "codes": codes,
        "length": len(data),
        "pad": pad
    }
    return bytes(out), metadata


def huffman_decode(payload: bytes, metadata: dict) -> bytes:
    length = metadata["length"]
    if length == 0:
        return b""
    codes = metadata["codes"]
    pad = metadata["pad"]
    
    bits_list = [f"{byte:08b}" for byte in payload]
    bitstring = "".join(bits_list)
    if pad > 0:
        bitstring = bitstring[:-pad]
        
    reverse_codes = {code: sym for sym, code in codes.items()}
    decoded = bytearray()
    curr = ""
    for bit in bitstring:
        curr += bit
        sym = reverse_codes.get(curr)
        if sym is not None:
            decoded.append(sym)
            curr = ""
            if len(decoded) == length:
                break
                
    if len(decoded) != length:
        raise ValueError(f"Huffman decode length mismatch: {len(decoded)} != {length}")
    return bytes(decoded)


# =====================================================================
# 2. Arithmetic Coding (Integer Precision)
# =====================================================================

PRECISION = 32
WHOLE = 1 << PRECISION
HALF = WHOLE >> 1
QTR_1 = WHOLE >> 2
QTR_3 = QTR_1 * 3


def arithmetic_encode(data: bytes) -> Tuple[bytes, dict]:
    """
    Arithmetic coding using 32-bit integer arithmetic with bit renormalisation.
    Returns: (compressed_payload_bytes, metadata_dict)
    """
    if not data:
        return b"", {"freq": {}, "length": 0, "bits_len": 0}
        
    counts = Counter(data)
    total_symbols = len(data)
    
    cum_freq = {}
    cum = 0
    for sym in range(256):
        c = counts.get(sym, 0)
        cum_freq[sym] = (cum, cum + c)
        cum += c
        
    low = 0
    high = WHOLE - 1
    pending_bits = 0
    bits = []
    
    for byte in data:
        sym_low, sym_high = cum_freq[byte]
        range_val = high - low + 1
        high = low + (range_val * sym_high // total_symbols) - 1
        low = low + (range_val * sym_low // total_symbols)
        
        while True:
            if high < HALF:
                bits.append('0')
                bits.extend(['1'] * pending_bits)
                pending_bits = 0
                low = low << 1
                high = (high << 1) + 1
            elif low >= HALF:
                bits.append('1')
                bits.extend(['0'] * pending_bits)
                pending_bits = 0
                low = (low - HALF) << 1
                high = ((high - HALF) << 1) + 1
            elif low >= QTR_1 and high < QTR_3:
                pending_bits += 1
                low = (low - QTR_1) << 1
                high = ((high - QTR_1) << 1) + 1
            else:
                break
                
    pending_bits += 1
    if low < QTR_1:
        bits.append('0')
        bits.extend(['1'] * pending_bits)
    else:
        bits.append('1')
        bits.extend(['0'] * pending_bits)
        
    bitstring = "".join(bits)
    pad = (8 - (len(bitstring) % 8)) % 8
    bitstring_padded = bitstring + ("0" * pad)
    
    out = bytearray()
    for i in range(0, len(bitstring_padded), 8):
        out.append(int(bitstring_padded[i:i+8], 2))
        
    metadata = {
        "freq": counts,
        "length": len(data),
        "total_symbols": total_symbols,
        "bits_len": len(bitstring)
    }
    return bytes(out), metadata


def arithmetic_decode(payload: bytes, metadata: dict) -> bytes:
    length = metadata["length"]
    if length == 0:
        return b""
        
    counts = metadata["freq"]
    total_symbols = metadata["total_symbols"]
    bits_len = metadata["bits_len"]
    
    bits_list = [f"{b:08b}" for b in payload]
    bitstring = "".join(bits_list)[:bits_len] + ("0" * 32)
    
    cum_freq = {}
    cum = 0
    sym_by_range = []
    for sym in range(256):
        c = counts.get(sym, 0)
        cum_freq[sym] = (cum, cum + c)
        if c > 0:
            sym_by_range.append((cum, cum + c, sym))
        cum += c
        
    def find_symbol(scaled_val):
        for low_c, high_c, sym in sym_by_range:
            if low_c <= scaled_val < high_c:
                return sym
        return sym_by_range[-1][2]

    bit_idx = 0
    low = 0
    high = WHOLE - 1
    value = 0
    
    for _ in range(PRECISION):
        bit = int(bitstring[bit_idx]) if bit_idx < len(bitstring) else 0
        value = (value << 1) + bit
        bit_idx += 1
        
    decoded = bytearray()
    while len(decoded) < length:
        range_val = high - low + 1
        scaled_val = ((value - low + 1) * total_symbols - 1) // range_val
        sym = find_symbol(scaled_val)
        decoded.append(sym)
        
        sym_low, sym_high = cum_freq[sym]
        high = low + (range_val * sym_high // total_symbols) - 1
        low = low + (range_val * sym_low // total_symbols)
        
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                value -= HALF
                low -= HALF
                high -= HALF
            elif low >= QTR_1 and high < QTR_3:
                value -= QTR_1
                low -= QTR_1
                high -= QTR_1
            else:
                break
                
            low = low << 1
            high = (high << 1) + 1
            bit = int(bitstring[bit_idx]) if bit_idx < len(bitstring) else 0
            bit_idx += 1
            value = (value << 1) + bit

    return bytes(decoded)


# =====================================================================
# 3. Dictionary Coding: LZ77 (Lempel-Ziv 1977 Sliding Window Dictionary)
# =====================================================================

def dictionary_lz77_encode(data: bytes) -> Tuple[bytes, dict]:
    """
    Standard LZ77 dictionary coding (Deflate byte stream with sliding window).
    Returns: (compressed_payload_bytes, metadata_dict)
    """
    if not data:
        return b"", {"length": 0}
        
    # Standard raw deflate stream (no zlib wrapper header/footer)
    compressor = zlib.compressobj(level=6, wbits=-15)
    compressed = compressor.compress(data) + compressor.flush()
    
    metadata = {
        "length": len(data),
        "comp_len": len(compressed)
    }
    return compressed, metadata


def dictionary_lz77_decode(payload: bytes, metadata: dict) -> bytes:
    length = metadata["length"]
    if length == 0:
        return b""
        
    decompressor = zlib.decompressobj(wbits=-15)
    decompressed = decompressor.decompress(payload)
    if len(decompressed) != length:
        raise ValueError(f"LZ77 decode length mismatch: {len(decompressed)} != {length}")
    return decompressed


# =====================================================================
# 4. Identity / Raw Pass-through (Optimal for maximum entropy payloads)
# =====================================================================

def raw_encode(data: bytes) -> Tuple[bytes, dict]:
    return data, {"length": len(data)}

def raw_decode(payload: bytes, metadata: dict) -> bytes:
    return payload
