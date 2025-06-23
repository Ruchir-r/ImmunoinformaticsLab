import argparse
from pathlib import Path

import numpy as np
import torch

from nettcr_torch.dataset import TCRDatasetInMemory
from nettcr_torch.model_lstm import BindingPredictionBiModel
from nettcr_torch.train import train_one_fold

def parse_args():
    parser = argparse.ArgumentParser(description="Train a TCR classification model.")

    parser.add_argument(
        "--dataset_path", type=Path, required=True, help="Path to the dataset file.")
    parser.add_argument(
        "--datadict_path", type=Path, required=True, help="Path to the data dictionary file.")
    parser.add_argument(
        "--sample_weighting", action="store_true", help="Enable sample weighting.")
    parser.add_argument(
        "--valid_partition", type=int, required=True, help="Partition number for validation.")
    parser.add_argument(
        "--test_partition", type=int, required=True, help="Partition number for testing.")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument(
        "--patience", type=int, default=50, help="Patience for early stopping.")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument(
        "--outdir_path", type=Path, required=True, help="Output directory for saving model and predictions.")
    parser.add_argument(
        "--run_name", type=str, default="experiment", help="Name identifier for the run (default: 'experiment').")

    return parser.parse_args()


def main():
    """Main entry point for training script."""
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    args.outdir_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    dataset = TCRDatasetInMemory(
        dataset_path=args.dataset_path,
        datadict_path=args.datadict_path,
        sample_weighting=args.sample_weighting,
        inference=False,
    )

    train_partitions = [
        p for p in dataset.df["partition"].unique()
        if p not in [
            args.valid_partition,
            args.test_partition,
        ]
    ]

    model = BindingPredictionBiModel(
        hidden_size=32,
        num_layers=1,
        num_seeds=1,
        batch_size=64,
        lstm_dropout=0.4,
	ffn_dropout=0.4
    )
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)  # Placeholder
    criterion = torch.nn.BCEWithLogitsLoss()

    model, test_true, test_pred, valid_true, valid_pred = train_one_fold(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_dataset=dataset.get_subset(train_partitions),
        valid_dataset=dataset.get_subset(args.valid_partition, sample_weighting=False),
        test_dataset=dataset.get_subset(args.test_partition, sample_weighting=False),
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )

    # Save model
    torch.save(
        model,
        args.outdir_path
        / f"model_{args.run_name}_{args.valid_partition}_{args.test_partition}.pt",
    )
    torch.save(
        torch.vstack([torch.as_tensor(test_true), torch.as_tensor(test_pred)]),
        args.outdir_path
        / f"test_pred_{args.run_name}_{args.valid_partition}_{args.test_partition}.pt",
    )
    torch.save(
        torch.vstack([torch.as_tensor(valid_true), torch.as_tensor(valid_pred)]),
        args.outdir_path
        / f"valid_pred_{args.run_name}_{args.valid_partition}_{args.test_partition}.pt",
    )

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
