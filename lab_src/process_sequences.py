import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nettcr_utils.sequence_encodings import ENCODING_SCHEMES
from nettcr_utils.sequence_utils import encode_sequence, pad_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode sequences using a specified encoding scheme."
    )
    parser.add_argument(
        "-i", dest="input", type=Path, required=True, help="Path to input file."
    )
    parser.add_argument(
        "-o", dest="out_dir", type=Path, required=True, help="Path to output directory."
    )
    parser.add_argument(
        "-e",
        dest="encoding",
        type=str,
        required=True,
        choices=ENCODING_SCHEMES.keys(),
        help="Type of encoding to use.",
    )
    parser.add_argument(
        "-n",
        dest="num_processes",
        type=int,
        default=1,
        help="Number of processes to use.",
    )
    parser.add_argument(
        "--peptide_esm_dir",
        type=Path,
        help="Path to the directory containing the peptide ESM embeddings.",
    )
    parser.add_argument(
        "--cdr_esm_dir",
        type=Path,
        help="Path to the directory containing the CDR ESM embeddings.",
    )
    parser.add_argument(
        "--save_to_dict",
        action="store_true",
        help="Save encoded data to a dictionary instead of files.",
    )
    return parser.parse_args()


def get_esm_embeddings(seqs, peptide_esm_dir, cdr_esm_dir):
    peptide_embedding = torch.load(peptide_esm_dir / f"{seqs[0]}.pt")
    cdr_embedding = torch.load(
        cdr_esm_dir / f"{seqs[1]}_{seqs[2]}_{seqs[3]}_{seqs[4]}_{seqs[5]}_{seqs[6]}.pt"
    )
    return [
        peptide_embedding["representations"][33],
        cdr_embedding["A1"],
        cdr_embedding["A2"],
        cdr_embedding["A3"],
        cdr_embedding["B1"],
        cdr_embedding["B2"],
        cdr_embedding["B3"],
    ]


def process_sequence(args):
    padding_value = -5.0
    (
        seqs,
        encoding_scheme,
        padding_lengths,
        out_path,
        peptide_esm_dir,
        cdr_esm_dir,
        save_to_dict,
    ) = args

    if encoding_scheme == "esm":
        # data = get_esm_embeddings(seqs, peptide_esm_dir, cdr_esm_dir)
        pass
    else:
        data = [
            (
                encode_sequence(seq, encoding_scheme)
                if seq
                else np.array([np.repeat(padding_value, 20)], dtype=np.int8)
            )
            for seq in seqs
        ]
    data = [
        pad_sequence(seq, length, padding_value).T
        for seq, length in zip(data, padding_lengths)
    ]

    # Convert to numpy instead of torch tensors before returning
    data = [
        np.array(seq, dtype=np.float32) for seq in data
    ]  # Avoid huge tensors in memory

    if save_to_dict:
        return out_path.stem, data  # Return numpy instead of tensor
    else:
        torch.save(data, str(out_path))
        return None


def encode_sequences(
    df,
    encoding_scheme,
    out_dir,
    num_processes=1,
    full_seq=False,
    peptide_esm_dir=None,
    cdr_esm_dir=None,
    save_to_dict=False,
):
    if full_seq:
        sequences = df[["peptide", "TRA", "TRB", "MHC"]].to_numpy()
    else:
        sequences = df[["peptide", "A1", "A2", "A3", "B1", "B2", "B3"]].to_numpy()
        for i in range(sequences.shape[1]):
            sequences[:, i] = np.where(
                np.char.find(sequences[:, i].astype(str), "X") >= 0, "", sequences[:, i]
            )
    if "name" not in df.columns:
        df["name"] = (
            df["peptide"]
            + "_"
            + df[["A1", "A2", "A3", "B1", "B2", "B3"]].astype(str).agg("_".join, axis=1)
        )

    out_paths = [out_dir / f"{name}.pt" for name in df["name"].values]
    padding_lengths = np.max(
        np.array(
            [np.vectorize(len)(sequences[:, i]) for i in range(sequences.shape[1])]
        ),
        axis=1,
    )

    with Pool(num_processes) as pool:
        args_list = [
            (
                seqs,
                encoding_scheme,
                padding_lengths,
                out_path,
                peptide_esm_dir,
                cdr_esm_dir,
                save_to_dict,
            )
            for seqs, out_path in zip(sequences, out_paths)
        ]
        results = list(
            tqdm(pool.imap(process_sequence, args_list), total=len(args_list))
        )

    if save_to_dict:
        return {name: data for name, data in results if name is not None}


def main():
    args = parse_args()
    df = pd.read_csv(args.input).fillna("")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = encode_sequences(
        df,
        ENCODING_SCHEMES[args.encoding],
        args.out_dir,
        args.num_processes,
        peptide_esm_dir=args.peptide_esm_dir,
        cdr_esm_dir=args.cdr_esm_dir,
        save_to_dict=args.save_to_dict,
    )

    if args.save_to_dict:
        torch.save(result, args.out_dir / f"{args.encoding}.pt")


if __name__ == "__main__":
    main()
