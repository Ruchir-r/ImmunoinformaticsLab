import argparse
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Predict cross-validation partitions using a TCR classification model.")

    parser.add_argument(
        "--dataset_path", type=Path, required=True, help="Path to the dataset file.")
    parser.add_argument(
        "--outdir_path", type=Path, required=True, help="Output directory for predictions.")
    parser.add_argument(
        "--run_name", type=str, default="experiment", help="Name identifier for the models (default: 'experiment').")

    return parser.parse_args()


def main():
    """Main entry point for training script."""
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    args.outdir_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.dataset_path)

    partitions = [
        p for p in df["partition"].unique()
    ]

    pred_list = []

    # Loop over each model by excluding the test partition defined in each CV round
    for t in partitions:

        test_df = df[df['partition']==t].copy()

        avg_prediction = [] # Reset prediction

        n = 0
        # Loop over each validation fold that is not the same of the test fold
        for v in partitions:

            if v!=t:

                # Load the prediction
                try:
                    pred = torch.load(args.outdir_path / f"test_pred_{args.run_name}_{v}_{t}.pt", weights_only=False)
                except FileNotFoundError:
                    print(f"Prediction file for {v} in {t} not found. Skipping...")
                    continue

                y_pred = torch.sigmoid(pred[1])

                avg_prediction.append(y_pred)

                n += 1

        # Averaging the predictions between all models in the inner loop
        avg_prediction = sum(avg_prediction)/len(avg_prediction)

        test_df.loc[:,'prediction'] = avg_prediction

        pred_list.append(test_df)

    pred_df = pd.concat(pred_list, ignore_index=True)

    pred_df.to_csv(args.outdir_path / f"kfold_pred_{args.run_name}.csv",index=False)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
