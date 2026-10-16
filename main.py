"""
Huffman Source Coding + Hamming (7,4) Linear Block Code demo.

Pipeline:
    saturated byte source -> Huffman encode -> Hamming(7,4) encode
    -> simulated wired channel with controlled bit flips
    -> Hamming decode/correct -> Huffman decode -> metrics

Dependencies: Python standard library only.

Notes:
- "Saturated source" is modeled here as uniformly distributed byte values (0..255).
- Hamming(7,4) corrects one bit error per 7-bit codeword. It does not reliably
  correct multiple errors within the same codeword.
- The channel is otherwise lossless; errors are injected only for demonstration.
"""

from collections import Counter
from dataclasses import dataclass
import heapq
import math
import random
import time
from typing import Optional


# ----------------------------- Huffman coding -----------------------------

@dataclass
class HuffmanNode:
    frequency: int
    symbol: Optional[int] = None
    left: Optional["HuffmanNode"] = None
    right: Optional["HuffmanNode"] = None


def build_huffman_codes(data: bytes) -> dict[int, str]:
    """Build deterministic Huffman codes for byte symbols in data."""
    frequencies = Counter(data)
    if not frequencies:
        return {}

    heap = []
    sequence = 0
    for symbol, frequency in sorted(frequencies.items()):
        node = HuffmanNode(frequency=frequency, symbol=symbol)
        heapq.heappush(heap, (frequency, sequence, node))
        sequence += 1

    if len(heap) == 1:
        only_symbol = heap[0][2].symbol
        return {only_symbol: "0"}

    while len(heap) > 1:
        freq_a, _, node_a = heapq.heappop(heap)
        freq_b, _, node_b = heapq.heappop(heap)
        parent = HuffmanNode(
            frequency=freq_a + freq_b,
            left=node_a,
            right=node_b,
        )
        heapq.heappush(heap, (parent.frequency, sequence, parent))
        sequence += 1

    root = heap[0][2]
    codes: dict[int, str] = {}

    def walk(node: HuffmanNode, prefix: str) -> None:
        if node.symbol is not None:
            codes[node.symbol] = prefix or "0"
            return
        assert node.left is not None and node.right is not None
        walk(node.left, prefix + "0")
        walk(node.right, prefix + "1")

    walk(root, "")
    return codes


def huffman_encode(data: bytes) -> tuple[str, dict[int, str]]:
    codes = build_huffman_codes(data)
    bitstream = "".join(codes[value] for value in data)
    return bitstream, codes


def huffman_decode(bitstream: str, codes: dict[int, str], output_length: int) -> bytes:
    """Decode using the codebook and expected original output length."""
    if output_length == 0:
        return b""

    reverse_codes = {code: symbol for symbol, code in codes.items()}
    decoded = bytearray()
    current = ""

    for bit in bitstream:
        current += bit
        symbol = reverse_codes.get(current)
        if symbol is not None:
            decoded.append(symbol)
            current = ""
            if len(decoded) == output_length:
                break

    if len(decoded) != output_length or current:
        raise ValueError(
            "Huffman decoding failed: corrupted stream, invalid codebook, or length mismatch."
        )
    return bytes(decoded)


# ------------------------- Hamming (7,4) LBC ------------------------------

def hamming74_encode_nibble(bits4: str) -> str:
    """
    Encode 4 data bits d1,d2,d3,d4 into positions:
    [p1, p2, d1, p4, d2, d3, d4]
    using even parity.
    """
    if len(bits4) != 4 or any(bit not in "01" for bit in bits4):
        raise ValueError("Expected exactly four binary bits.")

    d1, d2, d3, d4 = map(int, bits4)
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4
    return "".join(map(str, [p1, p2, d1, p4, d2, d3, d4]))


def hamming74_decode_codeword(codeword: str) -> tuple[str, int, bool]:
    """
    Decode one 7-bit codeword.
    Returns (four data bits, syndrome position, whether a bit was flipped).
    Syndrome 1..7 identifies the bit position to correct; 0 means no error.
    """
    if len(codeword) != 7 or any(bit not in "01" for bit in codeword):
        raise ValueError("Expected exactly seven binary bits.")

    b = [0] + list(map(int, codeword))  # one-indexed
    s1 = b[1] ^ b[3] ^ b[5] ^ b[7]
    s2 = b[2] ^ b[3] ^ b[6] ^ b[7]
    s4 = b[4] ^ b[5] ^ b[6] ^ b[7]
    syndrome = s1 + (s2 << 1) + (s4 << 2)

    corrected = syndrome != 0
    if corrected:
        b[syndrome] ^= 1

    data_bits = "".join(str(b[position]) for position in (3, 5, 6, 7))
    return data_bits, syndrome, corrected


def bits_to_bytes(bits: str, byte_count: int) -> bytes:
    if len(bits) < byte_count * 8:
        raise ValueError("Not enough bits to reconstruct requested bytes.")
    return bytes(
        int(bits[index:index + 8], 2)
        for index in range(0, byte_count * 8, 8)
    )


def lbc_encode(bitstream: str) -> tuple[str, int]:
    """Pad to a multiple of 4 bits, then encode each nibble to 7 bits."""
    pad_bits = (-len(bitstream)) % 4
    padded = bitstream + ("0" * pad_bits)
    encoded = "".join(
        hamming74_encode_nibble(padded[i:i + 4])
        for i in range(0, len(padded), 4)
    )
    return encoded, pad_bits


def lbc_decode(channel_bits: str, pad_bits: int) -> tuple[str, list[dict]]:
    if len(channel_bits) % 7 != 0:
        raise ValueError("Received LBC stream length must be a multiple of 7.")

    decoded_bits = []
    records = []
    for block_index, start in enumerate(range(0, len(channel_bits), 7)):
        received = channel_bits[start:start + 7]
        data_bits, syndrome, corrected = hamming74_decode_codeword(received)
        decoded_bits.append(data_bits)
        records.append({
            "block": block_index,
            "received_codeword": received,
            "syndrome": syndrome,
            "corrected": corrected,
        })

    combined = "".join(decoded_bits)
    if pad_bits:
        combined = combined[:-pad_bits]
    return combined, records


# ------------------------------ Metrics -----------------------------------

def entropy_bits_per_symbol(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def bit_error_count(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Bit strings must have equal length for BER comparison.")
    return sum(x != y for x, y in zip(a, b))


def print_metric(name: str, value: object) -> None:
    print(f"{name:<42}: {value}")


def run_demo(
    source_bytes: int = 2048,
    seed: int = 42,
    flips: Optional[list[int]] = None,
) -> None:
    """
    flips contains absolute bit positions in the transmitted LBC stream.
    By default, flip one bit in each of three different codewords.
    """
    rng = random.Random(seed)

    # Uniform byte source: each byte is independently selected from 0..255.
    source = bytes(rng.randrange(256) for _ in range(source_bytes))

    start_time = time.perf_counter()
    source_bits = len(source) * 8

    huffman_bits, codebook = huffman_encode(source)
    encoded_channel, padding_bits = lbc_encode(huffman_bits)

    # Wired channel: normally lossless; only specified positions are flipped.
    transmitted = list(encoded_channel)
    if flips is None:
        # One flipped bit in each of three distinct Hamming codewords.
        flips = [2, 7 * 10 + 4, 7 * 25 + 6] if len(transmitted) > 7 * 25 + 6 else [2]
    flips = sorted(set(position for position in flips if 0 <= position < len(transmitted)))

    for position in flips:
        transmitted[position] = "1" if transmitted[position] == "0" else "0"
    received_channel = "".join(transmitted)

    lbc_decoded_bits, block_records = lbc_decode(received_channel, padding_bits)
    recovered = huffman_decode(lbc_decoded_bits, codebook, len(source))
    elapsed = time.perf_counter() - start_time

    # Compare actual bit-level channel behavior and reconstruction.
    injected_error_count = bit_error_count(encoded_channel, received_channel)
    corrected_blocks = sum(record["corrected"] for record in block_records)
    residual_lbc_bit_errors = bit_error_count(huffman_bits, lbc_decoded_bits)
    source_equal = recovered == source
    source_bit_errors = bit_error_count(
        "".join(f"{byte:08b}" for byte in source),
        "".join(f"{byte:08b}" for byte in recovered),
    )

    source_bytes_count = len(source)
    huffman_bytes_equivalent = len(huffman_bits) / 8
    lbc_bytes_equivalent = len(encoded_channel) / 8
    source_entropy = entropy_bits_per_symbol(source)
    avg_huffman_length = len(huffman_bits) / source_bytes_count if source_bytes_count else 0.0
    huffman_ratio = len(huffman_bits) / source_bits if source_bits else 0.0
    lbc_rate = len(huffman_bits) / len(encoded_channel) if encoded_channel else 0.0
    overall_ratio = len(encoded_channel) / source_bits if source_bits else 0.0

    print("\n" + "=" * 76)
    print("HUFFMAN + HAMMING (7,4) LINEAR BLOCK CODE DEMONSTRATION")
    print("=" * 76)
    print("\nSOURCE AND SOURCE-CODING METRICS")
    print_metric("Source model", "Uniform random byte values (0..255)")
    print_metric("Random seed", seed)
    print_metric("Source size", f"{source_bytes_count:,} bytes ({source_bits:,} bits)")
    print_metric("Observed source entropy", f"{source_entropy:.4f} bits/byte")
    print_metric("Huffman bitstream length", f"{len(huffman_bits):,} bits")
    print_metric("Average Huffman code length", f"{avg_huffman_length:.4f} bits/byte")
    print_metric("Huffman/source bit ratio", f"{huffman_ratio:.4f}")
    print_metric("Huffman size change", f"{(1 - huffman_ratio) * 100:.2f}% reduction (negative means expansion)")
    print_metric("Unique symbols in codebook", len(codebook))

    print("\nCHANNEL-CODING METRICS")
    print_metric("Hamming code", "Systematic Hamming (7,4), even parity")
    print_metric("Huffman padding bits", padding_bits)
    print_metric("Transmitted LBC length", f"{len(encoded_channel):,} bits")
    print_metric("LBC code rate (4/7 nominal)", f"{lbc_rate:.4f}")
    print_metric("Total transmitted/source bit ratio", f"{overall_ratio:.4f}")
    print_metric("Channel bit positions flipped", flips)
    print_metric("Injected channel bit errors", injected_error_count)
    print_metric("Received codewords", len(block_records))
    print_metric("Codewords with nonzero syndrome/correction", corrected_blocks)
    print_metric("Residual errors in decoded Huffman bitstream", residual_lbc_bit_errors)

    print("\nRECONSTRUCTION AND PERFORMANCE")
    print_metric("Recovered source exactly matches original", source_equal)
    print_metric("Residual source bit errors", source_bit_errors)
    print_metric("Source bit error rate after decoding", f"{source_bit_errors / source_bits:.8f}" if source_bits else "N/A")
    print_metric("End-to-end processing time", f"{elapsed * 1000:.3f} ms")
    print_metric("End-to-end processing throughput", f"{source_bytes_count / elapsed / 1024:.2f} KiB/s" if elapsed else "N/A")
    print_metric("Source size", f"{source_bytes_count:,} bytes")
    print_metric("Huffman stream equivalent size", f"{huffman_bytes_equivalent:,.2f} bytes")
    print_metric("LBC stream equivalent size", f"{lbc_bytes_equivalent:,.2f} bytes")
    print_metric("Net source-to-channel overhead", f"{(len(encoded_channel) / source_bits - 1) * 100:.2f}%" if source_bits else "N/A")

    print("\nPER-BLOCK ERROR-CORRECTION LOG (only blocks with nonzero syndrome)")
    changed_records = [r for r in block_records if r["syndrome"] != 0]
    if not changed_records:
        print("No nonzero syndromes detected.")
    else:
        for record in changed_records[:20]:
            print(
                f"Block {record['block']:>5}: "
                f"syndrome={record['syndrome']}, "
                f"single-bit correction attempted"
            )
        if len(changed_records) > 20:
            print(f"... {len(changed_records) - 20} additional corrected blocks omitted.")

    print("\nINTERPRETATION")
    if source_equal:
        print("PASS: the original source was reconstructed exactly after the injected errors.")
    else:
        print("FAIL: the source differs from the original. Check the flip pattern and decoder.")
    print(
        "Note: Hamming(7,4) corrects one error per codeword. Multiple flips in one "
        "codeword can cause miscorrection; the simple syndrome decoder is not a "
        "general double-error detector."
    )
    print("=" * 76)


if __name__ == "__main__":
    run_demo(
        source_bytes=2048,
        seed=42,
        # Absolute bit positions in the encoded channel stream.
        # These positions fall in three different 7-bit codewords.
        flips=[2, 74, 181],
    )
