#!/bin/csh


set PYTHON = "/home/projects2/jonnil/miniconda3/envs/nettcr_torch/bin/python"
set data = "/home/projects2/ruchir/NetTCR_Torch/data/nettcr_train_swapped_peptide_ls_3_26_peptides_full_tcr_final.csv"
# Modify to your own model output directory
set outdir = "/home/projects2/ruchir/NetTCR_Torch/models/tcrlang_swn_lr.1/output"
# Modify to your own copy if needed
set SCRIPT = "/home/projects2/ruchir/NetTCR_Torch/src/predict_kfold.py"

set name = "experiment"

$PYTHON $SCRIPT --dataset_path $data --outdir_path $outdir --run_name $name