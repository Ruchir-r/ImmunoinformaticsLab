from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

class TCRDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        datadir_path: Union[Path, str],
        sample_weighting: bool = False,
        inference: bool = False,
    ):
        """Map style dataset class for TCR specificity data. Expects data points to be a list of format:
        [peptide, A1, A2, A3, B1, B2, B3], where each element is a NxH tensor where
        N is sequence length and H is embedding dimension.

        Args:
            dataset_path (Union[Path, str]): Path to the dataset .csv file.
            datadir_path (Union[Path, str]): Path to the directory containing the data.
            sample_weighting (bool, optional): Whether to use sample weighting.
            inference: (bool) Whether to return labels and sample weights in __getitem__.
        """
        self.dataset_path = Path(dataset_path)
        self.datadir_path = Path(datadir_path)
        self.df = pd.read_csv(dataset_path).reset_index(drop=True)
        self.files = self._get_file_names()
        self.labels = torch.tensor(self.df["binder"].to_numpy()).float()
        self.sample_weighting = sample_weighting
        self.inference = inference

        if sample_weighting:
            self.sample_weights = self.compute_sample_weights(self.df)
        else:
            self.sample_weights = torch.ones(len(self.df))

    def _get_file_names(self):
        return [
            self.datadir_path / f"{peptide}_{a1}_{a2}_{a3}_{b1}_{b2}_{b3}.pt"
            for peptide, a1, a2, a3, b1, b2, b3 in zip(
                self.df["peptide"],
                self.df["A1"],
                self.df["A2"],
                self.df["A3"],
                self.df["B1"],
                self.df["B2"],
                self.df["B3"],
            )
        ]

    def compute_sample_weights(self, df: pd.DataFrame) -> torch.tensor:
        """Compute sample weights for a dataset based on peptide frequency.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset.

        Returns:
            torch.tensor: Sample weights.
        """
        weights = np.log2(df.shape[0] / (df.peptide.value_counts())) / np.log2(
            df.peptide.nunique()
        )
        weights = weights * (df.shape[0] / np.sum(weights * df.peptide.value_counts()))
        df["sample_weights"] = df.peptide.map(weights)
        return torch.tensor(df.sample_weights.to_numpy()).float()

    def get_subset(
        self,
        partitions: Union[int, str, list, np.ndarray],
        sample_weighting: bool = None,
    ) -> Subset:
        """Get a subset of the dataset based on a partition or list of partitions.

        Args:
            partitions (Union[int, list, np.ndarray]): Partition(s) to include in the subset.
            sample_weighting (bool, optional): Whether sample_weighting needs to be recomputed.

        Returns:
            Subset: Subset of the dataset.
        """
        if not isinstance(partitions, np.ndarray) and not isinstance(partitions, list):
            partitions = [partitions]

        # get indices of partitions
        subset = Subset(
            self,
            self.df[self.df["partition"].isin(partitions)].index.to_list(),
        )

        # recompute sample weights within subset
        if sample_weighting is None:
            sample_weighting = self.sample_weighting
        if sample_weighting:
            subset.sample_weights = self.compute_sample_weights(
                self.df.iloc[subset.indices]
            )
        else:
            subset.sample_weights = torch.ones(len(subset), dtype=torch.float16)

        return subset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.load(self.files[idx], weights_only=False)
        if self.inference:
            return x
        else:
            return x + [self.labels[idx]] + [self.sample_weights[idx]]


class TCRDatasetInMemory(TCRDataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        datadict_path: Union[Path, str],
        sample_weighting: bool = False,
        inference: bool = False,
    ): 
        """In-memory style dataset class for TCR specificity data.
        Expects data points to be a list of format:
        [peptide, A1, A2, A3, B1, B2, B3], where each element is a
        NxH tensor where N is sequence length and H is embedding dimension.

        Args:
            dataset_path (Union[Path, str]): Path to the dataset .csv file.
            datadict_path (dict): Path to dictionary of tensors with keys corresponding to filenames.
            sample_weighting (bool, optional): Whether to use sample weighting.
            inference: (bool) Whether to return labels and sample weights in __getitem__.
        """
        super().__init__(
            dataset_path=dataset_path,
            datadir_path=datadict_path,
            sample_weighting=sample_weighting,
            inference=inference,
        )
        self.datadir_dict = torch.load(datadict_path, weights_only=False) # This dictionary should have keys in fmt {peptide}_{a1}_{a2}_{a3}_{b1}_{b2}_{b3}.pt

    def __getitem__(self, idx):
        file_name = self.files[idx]
        x = self.datadir_dict[file_name.stem] # file_name.stem returns the filename without extension or preceding path

        if self.inference:
            return x
        else:
            return x + [self.labels[idx]] + [self.sample_weights[idx]]
