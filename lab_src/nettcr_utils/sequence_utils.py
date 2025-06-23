import numpy as np


def get_cdr_indices(sequence: str, cdr1: str, cdr2: str, cdr3: str) -> list:
    """Get indices of CDRs in a sequence.

    Args:
        sequence (str): Amino acid sequence.
        cdr1 (str): CDR1 sequence.
        cdr2 (str): CDR2 sequence.
        cdr3 (str): CDR3 sequence.

    Returns:
        list: List of indices for CDRs.
    """
    indices = []
    for cdr in [cdr1, cdr2, cdr3]:
        start_idx = sequence.find(cdr)
        end_idx = start_idx + len(cdr)
        indices.append(np.arange(start_idx, end_idx))
    return indices


def get_aa_map():
    return {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}


def count_residues(sequence: str, cdr_indices: list, aa_map: dict) -> np.ndarray:
    """Count residues in a sequence.

    Args:
        sequence (str): Amino acid sequence.
        cdr_indices (list): List of indices for CDRs.

    Returns:
        np.ndarray: Residue counts.
    """
    aa_map = get_aa_map()
    counts = np.zeros((3, 20))
    for i, cdr_idx in enumerate(cdr_indices):
        for idx in cdr_idx:
            residue = sequence[idx]
            counts[i, aa_map[residue]] += 1
    return counts


def pad_sequence(
    sequence: np.ndarray,
    padding_length: int,
    padding_value=-5.0,
    padding_side="right",
) -> np.ndarray:
    """Pad a single sequence to the specified length.

    Args:
        sequence: A numpy array representing the sequence to pad.
        padding_length: An integer representing the length to pad the sequence to.
        padding_value: The value to use for padding.
        padding_side: A string specifying the side to pad on: 'left', 'right', or 'center'.

    Returns:
        A numpy array representing the padded sequence.
    """
    if padding_length <= sequence.shape[0]:
        return sequence

    pad_size = padding_length - sequence.shape[0]

    if padding_side == "left":
        pad_width = ((pad_size, 0), (0, 0))
    elif padding_side == "right":
        pad_width = ((0, pad_size), (0, 0))
    elif padding_side == "center":
        if pad_size % 2 != 0:
            raise ValueError(
                "For center padding, the padding length must result in equal padding on both sides."
            )
        pad_left = pad_size // 2
        pad_right = pad_size // 2
        pad_width = ((pad_left, pad_right), (0, 0))
    else:
        raise ValueError(
            "Invalid padding_side. Choose from 'left', 'right', or 'center'."
        )

    return np.pad(
        sequence,
        pad_width,
        mode="constant",
        constant_values=padding_value,
    )


def encode_sequence(sequence: str, encoding_scheme: dict) -> np.ndarray:
    """Encode a single sequence using the specified encoding scheme.

    Args:
        sequence: A string representing the sequence to encode.
        encoding_scheme: A dictionary mapping amino acids to their encoding.

    Returns:
        A numpy array representing the encoded sequence.
    """
    return np.vstack([encoding_scheme[aa] for aa in sequence], dtype=np.float32)
    