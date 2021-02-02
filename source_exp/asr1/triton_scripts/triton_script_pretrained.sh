#!/bin/bash
#SBATCH --time 4-12:0:0 --mem-per-cpu 32G
srun ./run_without_lm.sh --stage 5 --backend pytorch >> wer_logs/fixed_epoch_train.log

