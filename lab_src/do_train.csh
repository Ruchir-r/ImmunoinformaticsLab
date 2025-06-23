#!/bin/csh

set PYTHON = "/home/projects2/jonnil/miniconda3/envs/nettcr_torch/bin/python"
set data = "/home/projects2/ruchir/NetTCR_Torch/data/nettcr_train_swapped_peptide_ls_3_26_peptides_full_tcr_final.csv"
# Modify to your own model output directory
set outdir = "/home/projects2/ruchir/NetTCR_Torch/models/tcrlang_sampleweight_norm/output"
# Modify to your own copy if needed
set SCRIPT = "/home/projects2/ruchir/NetTCR_Torch/src/train_kfold_tcrlang_norm.py"
# Modify to your own encoding if needed
set encoding = "/home/projects2/ruchir/NetTCR_Torch/encodings/tcrlang_padded_data.pt"

# Test partitions
foreach t ( 0 1 2 3 4 )
        
        # Validation partitions
        foreach v ( 0 1 2 3 4 )

                if ( $t != $v ) then                    

                                if ( ! -e $t.$v.csh ) then      

                                        echo '#\!/bin/csh' > $t.$v.csh
                                        echo '#SBATCH --nodes=1' >> $t.$v.csh
                                        echo '#SBATCH --partition=cpu' >> $t.$v.csh
                                        echo '#SBATCH --mem=6GB' >> $t.$v.csh
                                        echo '#SBATCH --cpus-per-task=4' >> $t.$v.csh
                                        echo '#SBATCH --time=12:00:00' >> $t.$v.csh
                                        echo "$PYTHON $SCRIPT --sample_weighting --datadict_path $encoding --dataset_path $data --outdir_path $outdir --test_partition $t --valid_partition $v --epochs 300 --patience 50 --batch_size 64 > out.$t.$v" >> $t.$v.csh
                                        sbatch $t.$v.csh 
                                        sleep 0.01

                                endif
                endif
        end
end
